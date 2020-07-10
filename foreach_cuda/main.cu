
#include <iostream>
#include <math.h>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 2
#define H 1
#define W 3

std::vector<at::Tensor> add_tensors_naive(std::vector<std::vector<at::Tensor>> tensor_lists) {
    std::cout << "tensor list size: " << tensor_lists.size() << std::endl;
    std::cout << "tensor list[0] size: " << tensor_lists[0].size() << std::endl;
    std::cout << "tensor list[1] size: " << tensor_lists[1].size() << std::endl;

    std::vector<at::Tensor> result;
    for (int i = 0; i < tensor_lists[0].size(); i++) {
        auto temp = tensor_lists[0][i] + tensor_lists[1][i];
        result.push_back(temp);
    }

    return result;
}

int main(void)
{
  std::cout << "CUDA available:" << torch::cuda::is_available() << std::endl;
  torch::Device device(torch::kCUDA);
  torch::Tensor tensor = torch::eye(3, device);
  std::cout << tensor << std::endl;
  
  std::vector<std::vector<at::Tensor>> tensor_lists; 
  std::vector<at::Tensor> vec_a;
  std::vector<at::Tensor> vec_b;

  for (int i = 0; i < N; i++) {
      auto temp = torch::ones({H, W}, device);
      vec_a.push_back(temp);
      vec_b.push_back(temp);
  }
  tensor_lists.push_back(vec_a);
  tensor_lists.push_back(vec_b);
  
  auto res = add_tensors_naive(tensor_lists);
  std::cout << "res.size() = " << res.size() << std::endl;
  for (int i = 0; i < res.size(); i++) {
      std::cout << res[i] << std::endl;
  }
  return 0;
}