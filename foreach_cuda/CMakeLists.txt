cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(foreach_cuda)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

cuda_add_executable(foreach_cuda main.cu)

target_link_libraries(foreach_cuda "${TORCH_LIBRARIES}")
set_property(TARGET foreach_cuda PROPERTY CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")