
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

#include "AddListsFunctor.cu"
#include "AddScalarFunctor.cu"
#include "AddScalar_Functor.cu"

#define N 200
#define H 100
#define W 200
#define BLOCK_SIZE 1
#define CHUNK_SIZE 10

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

template<int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(
    int block_size,
    int chunk_size, 
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable,
    ArgTypes... args) 
    {
        // TODO: 
        // 1. check sizes, dtypes, layouts, depth 
        // 2. 

        int n_tensors = tensor_lists[0].size();

        TensorListMetadata<depth> tl_meta;
        auto cuda_stream = at::cuda::getCurrentCUDAStream();

        int loc_block_info = 0;
        int loc_tensor_info = 0;
        for(int t = 0; t < n_tensors; t++) 
        {
            tl_meta.sizes[loc_tensor_info] = tensor_lists[0][t].numel();
            for (int d = 0; d < depth; d++) 
            {
                tl_meta.addresses[d][loc_tensor_info] = tensor_lists[d][t].data_ptr();
            }
            loc_tensor_info++;

            int chunks = (tensor_lists[0][t].numel() + chunk_size - 1)/chunk_size;
            for (int chunk = 0; chunk < chunks; chunk++) 
            {
                tl_meta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
                tl_meta.block_to_chunk[loc_block_info] = chunk;
                loc_block_info++;

                bool tensors_full = (loc_tensor_info == depth_to_max_tensors[depth-1] &&
                    chunk == chunks - 1);
                bool blocks_full = (loc_block_info == depth_to_max_blocks[depth-1]);
                bool last_chunk = (t == n_tensors - 1 && chunk == chunks - 1);

                if (tensors_full || blocks_full || last_chunk)
                {
                    multi_tensor_apply_kernel<<<loc_block_info, block_size, 0, cuda_stream>>>(
                        chunk_size,
                        tl_meta,
                        callable,
                        args...);
              
                    AT_CUDA_CHECK(cudaGetLastError());

                    // Reset.
                    loc_block_info = 0;
                    if(chunk == chunks - 1)
                    {
                        loc_tensor_info = 0; 
                    }
                    else
                    {
                        tl_meta.sizes[0] = tl_meta.sizes[loc_tensor_info-1];
                        for(int d = 0; d < depth; d++)
                            tl_meta.addresses[d][0] = tl_meta.addresses[d][loc_tensor_info-1];
                        loc_tensor_info = 1;
                    }
                }
            }
        }
}

void test_add_scalar(std::vector<std::vector<at::Tensor>> tensor_lists, float scalar) {
    auto res = add_scalar_naive(tensor_lists, scalar);

    multi_tensor_apply<2>(BLOCK_SIZE, CHUNK_SIZE, tensor_lists, AddScalarFunctor<float, float>(), scalar);
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaDeviceSynchronize());
    
    bool all_matched = true;
    for(int i = 0; i < N; i++) 
    {
        all_matched = tensor_lists[1][i].equal(res[i]);
        if (!all_matched) {
            std::cout << "add scalar test FAILED!" << std::endl << std::endl;
            std::cout << "fail info: " << std::endl;
            std::cout << "i = " << i << std::endl;
            std::cout << "tensor_lists[1][i] = " << tensor_lists[1][i] << std::endl;
            std::cout << "res[i] = " << res[i] << std::endl;
            break;
        }
    }

    if (all_matched) {
        std::cout << "add scalar test PASSED" << std::endl;
    }
}

void test_add_scalar_(std::vector<std::vector<at::Tensor>> tensor_lists, float scalar) {
    auto res = add_scalar_naive(tensor_lists, scalar);

    multi_tensor_apply<1>(BLOCK_SIZE, CHUNK_SIZE, tensor_lists, AddScalar_Functor<float>(), scalar);
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaDeviceSynchronize());
    
    bool all_matched = true;
    for(int i = 0; i < N; i++) 
    {
        all_matched = tensor_lists[0][i].equal(res[i]);
        if (!all_matched) {
            std::cout << "add scalar inplace test FAILED!" << std::endl << std::endl;
            std::cout << "fail info: " << std::endl;
            std::cout << "i = " << i << std::endl;
            std::cout << "tensor_lists[0][i] = " << tensor_lists[0][i] << std::endl;
            std::cout << "res[i] = " << res[i] << std::endl;
            break;
        }
    }

    if (all_matched) {
        std::cout << "add scalar inplace test PASSED" << std::endl;
    }
}

void test_add_lists(std::vector<std::vector<at::Tensor>> tensor_lists) {
    std::cout << "\ntest_add_lists test" << std::endl;
    auto res = add_tensors_naive(tensor_lists);

    multi_tensor_apply<3>(BLOCK_SIZE, CHUNK_SIZE, tensor_lists, AddListsFunctor<float, float, float>());
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaDeviceSynchronize());

    bool all_lists_matched = true;
    for(int i = 0; i < N; i++) 
    {
        all_lists_matched = tensor_lists[2][i].equal(res[i]);
        if (!all_lists_matched) {
            std::cout << "add lists test FAILED!" << std::endl << std::endl;
            std::cout << "fail info: " << std::endl;
            std::cout << "i = " << i << std::endl;
            std::cout << "tensor_lists[2][i] = " << tensor_lists[2][i] << std::endl;
            std::cout << "res[i] = " << res[i] << std::endl;
            break;
        }
    }

    if (all_lists_matched) {
        std::cout << "add lists test PASSED" << std::endl;
    }
}

int main(void)
{
  torch::Device device(torch::kCUDA);

  std::vector<std::vector<at::Tensor>> tensor_lists; 
  std::vector<at::Tensor> vec_a;
  std::vector<at::Tensor> vec_b;
  std::vector<at::Tensor> vec_res;

  for (int i = 0; i < N; i++) 
  {
      auto temp1 = torch::zeros({H, W}, device);
      auto temp2 = torch::ones({H, W}, device);
      vec_a.push_back(temp1);
      vec_b.push_back(temp2);
      vec_res.push_back(torch::zeros({H, W}, device));
  }
  tensor_lists.push_back(vec_a);
  tensor_lists.push_back(vec_b);
  tensor_lists.push_back(vec_res);

  test_add_lists(tensor_lists);
  test_add_scalar(tensor_lists, 2);
  test_add_scalar_(tensor_lists, 10);
  return 0;
}