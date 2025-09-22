#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <cute/tensor.hpp>
#include <cutlass/half.h>
#include <cstdlib>
#include <ctime>

using namespace cute;
using half_t = cutlass::half_t;

// CPU function to initialize matrix with random values
void init_matrix(half_t* matrix, int rows, int cols) {
    srand(time(NULL));
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = half_t(float(rand()) / float(RAND_MAX));
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

// Compare matrices
bool compare_matrices(const half_t* matrix1, const half_t* matrix2, int size, float epsilon = 1e-2) {
    bool passed = true;
    int error_count = 0;
    const int max_errors_to_show = 10;
    
    std::cout << "\nComparing matrices (showing first " << max_errors_to_show << " errors):\n";
    
    for (int i = 0; i < size; ++i) {
        float val1 = float(matrix1[i]);
        float val2 = float(matrix2[i]);
        float diff = abs(val1 - val2);
        
        if (diff > epsilon) {
            if (error_count < max_errors_to_show) {
                std::cout << "  Index " << i << ": Matrix1 = " << val1 
                          << ", Matrix2 = " << val2 << ", Diff = " << diff << std::endl;
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