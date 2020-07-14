
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
#include "AddListsFunctor.cu"
#include "AddScalarFunctor.cu"

#define N 2
#define H 1
#define W 2
#define BLOCK_SIZE 1

std::vector<at::Tensor> add_tensors_naive(std::vector<std::vector<at::Tensor>>& tensor_lists) {
    std::vector<at::Tensor> result;
    for (int i = 0; i < tensor_lists[0].size(); i++) {
        auto temp = tensor_lists[0][i] + tensor_lists[1][i];
        result.push_back(temp);
    }

    return result;
}

std::vector<at::Tensor> add_scalar_naive(std::vector<std::vector<at::Tensor>>& tensor_lists, float scalar) {
    std::vector<at::Tensor> result;
    for (int i = 0; i < tensor_lists[0].size(); i++) {
        auto temp = tensor_lists[0][i] + scalar;
        result.push_back(temp);
    }

    return result;
}

template<typename T, typename U, typename... ArgTypes>
__global__ void multi_tensor_apply_kernel(
    int chunk_size,
    T tl,
    U callable,
    ArgTypes... args)
{
  // Hand the chunk information to the user-supplied functor to process however it likes.
  callable(chunk_size, tl, args...); 
}

// depth - how many tensor lists to expect
// chunk size -
// 
template<int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(
    int block_size,
    int chunk_size, 
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable,
    ArgTypes... args) 
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
                        callable,
                        args...);
              
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

void test_add_scalar(std::vector<std::vector<at::Tensor>> tensor_lists) {
    std::cout << "\ntest_add_scalar test" << std::endl;
    float scalar = 2;
    auto naive_start = std::chrono::steady_clock::now();
    auto res = add_tensors_naive(tensor_lists);
    auto naive_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> naive_elapsed_seconds = naive_end - naive_start;
    std::cout << "elapsed time naive: " << naive_elapsed_seconds.count() << "s\n";

    int chunk_size = 10;
    multi_tensor_apply<2>(BLOCK_SIZE, chunk_size, tensor_lists, AddScalarFunctor<float, float>(), scalar);
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
        std::cout << "add scalar test PASSED" << std::endl;
    }
}

void test_add_lists(std::vector<std::vector<at::Tensor>> tensor_lists) {
    std::cout << "\ntest_add_lists test" << std::endl;
    auto naive_start = std::chrono::steady_clock::now();
    auto res = add_tensors_naive(tensor_lists);
    auto naive_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> naive_elapsed_seconds = naive_end - naive_start;
    std::cout << "elapsed time naive: " << naive_elapsed_seconds.count() << "s\n";

     /* for (int chunk_size = 100; chunk_size < 10000; chunk_size *= 10) {
        auto start = std::chrono::steady_clock::now();
        multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, tensor_lists, AddFunctor<float, float, float>());
        AT_CUDA_CHECK(cudaGetLastError());
        AT_CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "chunk size: " << chunk_size << std::endl;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    } */

    int chunk_size = 10;
    multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, tensor_lists, AddListsFunctor<float, float, float>());
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaDeviceSynchronize());
    
    bool all_lists_matched = true;
    for(int i = 0; i < N; i++) 
    {
        all_lists_matched = tensor_lists[2][i].equal(res[i]);
        if (!all_lists_matched) {
            std::cout << "TEST FAILED!" << std::endl << std::endl;
            std::cout << "fail info: " << std::endl;
            std::cout << "i = " << i << std::endl;
            std::cout << "tensor_lists[2][i] = " << tensor_lists[2][i] << std::endl;
            std::cout << "res[i] = " << res[i] << std::endl;
        }
    }

    if (all_lists_matched) {
        std::cout << "add lists test PASSED" << std::endl;
    }
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

  test_add_lists(tensor_lists);
  test_add_scalar(tensor_lists);

  return 0;
}