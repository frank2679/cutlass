#ifndef COPY_KERNEL_CUH
#define COPY_KERNEL_CUH

#include <cute/tensor.hpp>
#include <cutlass/half.h>

using namespace cute;
using half_t = cutlass::half_t;

__global__ void copy_demo_kernel(half_t* gmem_a, half_t* gmem_b, int M, int N);
__global__ void custom_copy_tma_style_demo_kernel(half_t* gmem_a, half_t* gmem_b, int M, int N);

// Function to calculate shared memory size for entire matrix
inline size_t get_shared_memory_size(int M, int N) {
    // Two matrices of size MxN each
    return sizeof(half_t) * M * N * 2;
}

// Function to get grid dimensions (1x1)
inline dim3 get_grid_dims() {
    return dim3(1, 1);
}

// Function to get block dimensions (1x1)
inline dim3 get_block_dims() {
    return dim3(1, 1);
}

#endif // COPY_KERNEL_CUH