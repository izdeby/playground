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
#include "Utils.cuh"

template<typename x_t, typename y_t, typename out_t>
struct AddListsFunctor
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
                //printf("[AxpbyFunctor] case 2\n");
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