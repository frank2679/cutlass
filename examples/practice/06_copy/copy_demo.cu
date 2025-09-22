#include <iostream>
#include "copy_kernel.cuh"
#include "utils.hpp"
#include <cutlass/half.h>
#include <cuda_runtime.h>

using namespace cute;
using half_t = cutlass::half_t;

int main() {
    // Matrix dimensions
    constexpr int M = 64;
    constexpr int N = 64;
    
    // Allocate device memory
    half_t *d_a, *d_b;
    cudaMalloc(&d_a, sizeof(half_t) * M * N);
    cudaMalloc(&d_b, sizeof(half_t) * M * N);
    
    // Allocate host memory
    half_t *h_a = new half_t[M * N];
    half_t *h_b = new half_t[M * N];
    half_t *h_b_ref = new half_t[M * N];
    
    // Initialize input matrix with random values
    init_matrix(h_a, M, N);
    
    // Initialize output matrix with zeros
    for (int i = 0; i < M * N; ++i) {
        h_b[i] = half_t(0.0f);
        h_b_ref[i] = half_t(0.0f);
    }
    
    // Copy input matrix to device
    cudaMemcpy(d_a, h_a, sizeof(half_t) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(half_t) * M * N, cudaMemcpyHostToDevice);
    
    // Calculate shared memory size needed for entire matrices
    size_t shared_mem_size = get_shared_memory_size(M, N);
    std::cout << "Shared memory size required: " << shared_mem_size << " bytes" << std::endl;
    
    // Launch kernel with 1x1 grid and 1x1 block
    dim3 grid = get_grid_dims();
    dim3 block = get_block_dims();
    
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << ") and block (" 
              << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    
    copy_demo_kernel<<<grid, block, shared_mem_size>>>(d_a, d_b, M, N);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA kernel launch error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_b, d_b, sizeof(half_t) * M * N, cudaMemcpyDeviceToHost);
    
    // Create reference result (should be identical to input)
    for (int i = 0; i < M * N; ++i) {
        h_b_ref[i] = h_a[i];
    }
    
    // Print matrices for verification
    print_matrix(h_a, M, N, "Input (A)");
    print_matrix(h_b, M, N, "Output (B)");
    
    // Verify results
    bool verification_passed = compare_matrices(h_b, h_b_ref, M * N, 1e-5);
    
    std::cout << "Verification: " << (verification_passed ? "PASSED" : "FAILED") << std::endl;
    
    // Clean up memory
    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;
    delete[] h_b_ref;
    
    return verification_passed ? 0 : -1;
}
#ifndef COPY_KERNEL_CUH
#define COPY_KERNEL_CUH

#include <cute/tensor.hpp>
#include <cutlass/half.h>

using namespace cute;
using half_t = cutlass::half_t;

// Define tile sizes
using TILE_M = Int<32>;
using TILE_N = Int<32>;

// Function to get tile size (needed for main program)
inline std::pair<int, int> get_tile_size() {
    return {size(TILE_M{}), size(TILE_N{})};
}

__global__ void copy_demo_kernel(half_t* gmem_a, half_t* gmem_b, int M, int N);

#endif // COPY_KERNEL_CUH