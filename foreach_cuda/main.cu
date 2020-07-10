
#include <iostream>
#include <math.h>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main(void)
{
  //std::cout << torch.cuda.is_available() << std::endl;
  torch::Device device(torch::kCUDA);
  torch::Tensor tensor = torch::eye(3, device);
  std::cout << tensor << std::endl;
  return 0;
}