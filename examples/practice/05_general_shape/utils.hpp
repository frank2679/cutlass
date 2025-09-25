#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <cute/tensor.hpp>
#include <cutlass/half.h>
#include <cstdlib>
#include <ctime>

using namespace cute;
using half_t = cutlass::half_t;

// CPU GEMM implementation for verification
void cpu_gemm(const half_t* a, const half_t* b, const half_t* c, half_t* d, 
              int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = float(c[m * N + n]); // Initialize with C value
            for (int k = 0; k < K; ++k) {
                sum += float(a[m * K + k]) * float(b[k * N + n]);
            }
            d[m * N + n] = half_t(sum);
        }
    }
}

// Print matrix for debugging
void print_matrix(const half_t* matrix, int rows, int cols, const char* name) {
    std::cout << "\n" << name << " Matrix (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < rows && i < 10; ++i) { // Print first 10 rows
        for (int j = 0; j < cols && j < 10; ++j) { // Print first 10 columns
            std::cout << float(matrix[i * cols + j]) << " ";
        }
        if (cols > 10) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > 10) std::cout << "..." << std::endl;
}

// Compare GPU and CPU results
bool compare_results(const half_t* gpu_result, const half_t* cpu_result, int size, float epsilon = 1e-2) {
    bool passed = true;
    int error_count = 0;
    const int max_errors_to_show = 10;
    
    std::cout << "\nComparing GPU and CPU results (showing first " << max_errors_to_show << " errors):\n";
    
    for (int i = 0; i < size; ++i) {
        float gpu_val = float(gpu_result[i]);
        float cpu_val = float(cpu_result[i]);
        float diff = abs(gpu_val - cpu_val);
        
        if (diff > epsilon) {
            if (error_count < max_errors_to_show) {
                std::cout << "  Index " << i << ": GPU = " << gpu_val 
                          << ", CPU = " << cpu_val << ", Diff = " << diff << std::endl;
            }
            error_count++;
            passed = false;
        }
    }
    
    if (passed) {
        std::cout << "  All elements match within epsilon = " << epsilon << std::endl;
    } else {
        std::cout << "  Total errors: " << error_count << " out of " << size << " elements" << std::endl;
    }
    
    return passed;
}

#endif // UTILS_HPP