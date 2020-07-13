
#include <iostream>
#include <math.h>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <chrono>

#define N 20
#define H 10
#define W 20
#define BLOCK_SIZE 1
#define ILP 4

std::vector<at::Tensor> add_tensors_naive(std::vector<std::vector<at::Tensor>>& tensor_lists) {
    //std::cout << "tensor list size: " << tensor_lists.size() << std::endl;
    //std::cout << "tensor list[0] size: " << tensor_lists[0].size() << std::endl;
    //std::cout << "tensor list[1] size: " << tensor_lists[1].size() << std::endl;

    std::vector<at::Tensor> result;
    for (int i = 0; i < tensor_lists[0].size(); i++) {
        auto temp = tensor_lists[0][i] + tensor_lists[1][i];
        result.push_back(temp);
    }

    return result;
}

template<typename T>
__device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (ILP*sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

// TensorListMetadata has to be < 4KB - the limit for kernel launch argument
constexpr int depth_to_max_tensors[5] = {110, 64, 48, 36, 30};
constexpr int depth_to_max_blocks[5] = {320, 320, 320, 320, 320};
template<int n> struct TensorListMetadata
{
  void* addresses[n][depth_to_max_tensors[n-1]];
  int sizes[depth_to_max_tensors[n-1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n-1]];
  int block_to_chunk[depth_to_max_blocks[n-1]];
  //int start_tensor_this_launch;
};

template<typename x_t, typename y_t, typename out_t>
struct AddFunctor
{
    __device__ void operator() (
        int chunk_size,
        TensorListMetadata<3>& tl) 
        {
            int tensor_loc = tl.block_to_tensor[blockIdx.x];
            int chunk_idx = tl.block_to_chunk[blockIdx.x];
            int n = tl.sizes[tensor_loc];
            //printf("[AxpbyFunctor] \n\tblockIdx.x = %i\n\ttensor_loc = %i\n\tchunk_idx = %i\n", blockIdx.x, tensor_loc, chunk_idx);
            
            x_t* x = (x_t*)tl.addresses[0][tensor_loc];
            x += chunk_idx*chunk_size;

            y_t* y = (y_t*)tl.addresses[1][tensor_loc];
            y += chunk_idx*chunk_size;

            out_t* out = (out_t*)tl.addresses[2][tensor_loc];
            out += chunk_idx*chunk_size;

            n -= chunk_idx*chunk_size;

            x_t r_x[ILP];
            y_t r_y[ILP];
            out_t r_out[ILP];

            // to make things simple, we put aligned case in a different code path
            if(n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x) && is_aligned(y) && is_aligned(out))
            {
                printf("[AxpbyFunctor] case 1\n");
                for(int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x)
                {
                    // load
                    load_store(r_x, x, 0 , i_start);
                    load_store(r_y, y, 0 , i_start);
#pragma unroll
                    for(int ii = 0; ii < ILP; ii++)
                    {
                        r_out[ii] = static_cast<float>(r_x[ii]) + static_cast<float>(r_y[ii]);
                    }
                    // store
                    load_store(out, r_out, i_start , 0);
                }
            }
            else
            {
                printf("[AxpbyFunctor] case 2\n");
                // Non-divergent exit condition for __syncthreads, not necessary here
                for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x*ILP)
                {
#pragma unroll
                    for(int ii = 0; ii < ILP; ii++)
                    {
                    r_x[ii] = 0;
                    r_y[ii] = 0;
                    int i = i_start + threadIdx.x + ii*blockDim.x;
                    if(i < n && i < chunk_size)
                    {
                        r_x[ii] = x[i];
                        r_y[ii] = y[i];
                    }
                    }
#pragma unroll
                    for(int ii = 0; ii < ILP; ii++)
                    {
                        r_out[ii] = static_cast<float>(r_x[ii]) + static_cast<float>(r_y[ii]);
                    }
#pragma unroll
                    for(int ii = 0; ii < ILP; ii++)
                    {
                    int i = i_start + threadIdx.x + ii*blockDim.x;
                    if(i < n && i < chunk_size)
                        out[i] = r_out[ii];
                    }
                }
            }
        }
};

template<typename T, typename U>
__global__ void multi_tensor_apply_kernel(
    int chunk_size,
    T tl,
    U callable)
{
  // Hand the chunk information to the user-supplied functor to process however it likes.
  callable(chunk_size, tl); 
}

// depth - how many tensor lists to expect
// chunk size -
// 
template<int depth, typename T>
void multi_tensor_apply(
    int block_size,
    int chunk_size, 
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable) 
    {
        //std::cout << "\n\nhello from multi_tensor_apply" << std::endl;
        //std::cout << "chunk_size: " << chunk_size << std::endl;
        //std::cout << "lists : " << tensor_lists.size() << std::endl;
        // TODO: 
        // 1. check sizes, dtypes, layouts, depth 
        // 2. 

        int n_tensors = tensor_lists[0].size();
        //std::cout << "n_tensors: " << n_tensors << std::endl;
        //std::cout << "depth: " << depth << std::endl;
        
        TensorListMetadata<depth> tl_meta;
        auto cuda_stream = at::cuda::getCurrentCUDAStream();

        //tl_meta.start_tensor_this_launch = 0;
        int loc_block_info = 0;
        int loc_tensor_info = 0;
        for(int t = 0; t < n_tensors; t++) 
        {
            //std::cout << "\n\n\n" << std::endl;
            //std::cout << "t = " << t << " loc_tensor_info " << loc_tensor_info << std::endl;
            tl_meta.sizes[loc_tensor_info] = tensor_lists[0][t].numel();
            for (int d = 0; d < depth; d++) 
            {
                tl_meta.addresses[d][loc_tensor_info] = tensor_lists[d][t].data_ptr();
                //std::cout << "[multi_tensor_apply] tl.addresses[" << d << "][" << loc_tensor_info << "] = " << tl_meta.addresses[d][loc_tensor_info] << std::endl;
            }
            loc_tensor_info++;

            int chunks = (tensor_lists[0][t].numel() + chunk_size - 1)/chunk_size;
            //std::cout << "chunks: " << chunks << std::endl;
            for (int chunk = 0; chunk < chunks; chunk++) 
            {
                tl_meta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
                tl_meta.block_to_chunk[loc_block_info] = chunk;
                loc_block_info++;

                bool tensors_full = (loc_tensor_info == depth_to_max_tensors[depth-1] &&
                    chunk == chunks - 1);
                bool blocks_full = (loc_block_info == depth_to_max_blocks[depth-1]);
                bool last_chunk = (t == n_tensors - 1 && chunk == chunks - 1);
                //std::cout << "[multi_tensor_apply] tensors_full = " << tensors_full << std::endl;
                //std::cout << "[multi_tensor_apply] blocks_full = " << blocks_full << std::endl;
                //std::cout << "[multi_tensor_apply] last_chunk = " << last_chunk << std::endl;

                if (tensors_full || blocks_full || last_chunk)
                {
                    // using accscalar_t = acc_type<scalar_t, true>;
                    //std::cout << "[multi_tensor_apply] calling apply kernel with" << std::endl;
                    //std::cout << "[                  ] loc_block_info = " << loc_block_info << std::endl;
                    //std::cout << "[                  ] block_size = " << block_size << std::endl;
                    //std::cout << "[                  ] chunk_size = " << chunk_size << std::endl;
                    multi_tensor_apply_kernel<<<loc_block_info, block_size, 0, cuda_stream>>>(
                        chunk_size,
                        tl_meta,
                        callable);
              
                    AT_CUDA_CHECK(cudaGetLastError());

                    // Reset.  The control flow possibilities here make my brain hurt.
                    loc_block_info = 0;
                    if(chunk == chunks - 1)
                    {
                        //std::cout << "Hit case 1 " << std::endl;
                        loc_tensor_info = 0; 
                        //tl_meta.start_tensor_this_launch = t + 1;
                    }
                    else
                    {
                        //std::cout << "Hit case 2 " << std::endl;
                        tl_meta.sizes[0] = tl_meta.sizes[loc_tensor_info-1];
                        for(int d = 0; d < depth; d++)
                            tl_meta.addresses[d][0] = tl_meta.addresses[d][loc_tensor_info-1];
                        loc_tensor_info = 1;
                        //tl_meta.start_tensor_this_launch = t;
                    }
                }
            }
        }

        std::cout << "size of metadata: " << sizeof(tl_meta) << std::endl;
}

int main(void)
{
  std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
  torch::Device device(torch::kCUDA);
  
  std::vector<std::vector<at::Tensor>> tensor_lists; 
  std::vector<at::Tensor> vec_a;
  std::vector<at::Tensor> vec_b;
  std::vector<at::Tensor> vec_res;

  for (int i = 0; i < N; i++) 
  {
      auto temp1 = torch::randn({H, W}, device);
      auto temp2 = torch::randn({H, W}, device);
      vec_a.push_back(temp1);
      vec_b.push_back(temp2);
      //std::cout << "temp1 data ptr: " << temp1.data_ptr() << std::endl;
      //std::cout << "temp2 data ptr: " << temp2.data_ptr() << std::endl;
      vec_res.push_back(torch::zeros({H, W}, device));
  }
  tensor_lists.push_back(vec_a);
  tensor_lists.push_back(vec_b);
  tensor_lists.push_back(vec_res);
  
  auto naive_start = std::chrono::steady_clock::now();
  auto res = add_tensors_naive(tensor_lists);
  auto naive_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> naive_elapsed_seconds = naive_end - naive_start;
  std::cout << "elapsed time: " << naive_elapsed_seconds.count() << "s\n";

  
  //for (int chunk_size = 1; chunk_size < 100; chunk_size *= 10) {
  //  auto start = std::chrono::steady_clock::now();
  //  multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, tensor_lists, AddFunctor<float, float, float>());
  //  AT_CUDA_CHECK(cudaGetLastError());
  //  AT_CUDA_CHECK(cudaDeviceSynchronize());
  //  auto end = std::chrono::steady_clock::now();
  //  std::chrono::duration<double> elapsed_seconds = end - start;
  //  std::cout << "chunk size: " << chunk_size << std::endl;
  //  std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
  //}
  int chunk_size = 1;
  multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, tensor_lists, AddFunctor<float, float, float>());
  AT_CUDA_CHECK(cudaGetLastError());
  AT_CUDA_CHECK(cudaDeviceSynchronize());
  
  bool all_matched = true;
  for(int i = 0; i < N; i++) 
  {
      all_matched = tensor_lists[2][i].equal(res[i]);
      if (!all_matched) {
        std::cout << "TEST FAILED!" << std::endl << std::endl;
        std::cout << "fail info: " << std::endl;
        std::cout << "i = " << i << std::endl;
        std::cout << "tensor_lists[2][i] = " << tensor_lists[2][i] << std::endl;
        std::cout << "res[i] = " << res[i] << std::endl;
      }
  }

  if (all_matched) {
    std::cout << "tests PASSED" << std::endl;
  }

  return 0;
}