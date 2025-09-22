#include "copy_kernel.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main() {
    const int M = 16;
    const int N = 16;
    const int size = M * N;
    
    // Host memory allocation
    std::vector<half_t> h_a(size);
    std::vector<half_t> h_b(size);
    
    // Initialize input data
    for (int i = 0; i < size; ++i) {
        h_a[i] = half_t(i % 10);
    }
    
    // Device memory allocation
    half_t* d_a;
    half_t* d_b;
    cudaMalloc(&d_a, size * sizeof(half_t));
    cudaMalloc(&d_b, size * sizeof(half_t));
    
    // Copy data to device
    cudaMemcpy(d_a, h_a.data(), size * sizeof(half_t), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 grid(1, 1);
    dim3 block(1, 1);
    size_t shared_mem_size = 2 * size * sizeof(half_t);
    
    printf("Running custom copy demo kernel\n");
    custom_copy_tma_style_demo_kernel<<<grid, block, shared_mem_size>>>(d_a, d_b, M, N);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_b.data(), d_b, size * sizeof(half_t), cudaMemcpyDeviceToHost);
    
    // Verify results
    bool success = true;
    for (int i = 0; i < size; ++i) {
        if (h_a[i] != h_b[i]) {
            std::cout << "Mismatch at index " << i << ": " << float(h_a[i]) << " != " << float(h_b[i]) << std::endl;
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "Custom copy test passed!" << std::endl;
    } else {
        std::cout << "Custom copy test failed!" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}