#include <cute/tensor.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include "cutlass/util/GPU_Clock.hpp"

// Naive GEMM kernel using cute::Tensor
template <size_t M, size_t N, size_t K, class TA, class TB, class TC>
__global__ void gemm_naive(TA const *A, TB const *B, TC *C)
{
    using namespace cute;

    // Create cute tensors for input and output matrices
    Tensor tA = make_tensor(make_gmem_ptr(A), make_shape(K, M)); // Tensor for A (K x M, column-major)
    Tensor tB = make_tensor(make_gmem_ptr(B), make_shape(N, K)); // Tensor for B (N x K, column-major)
    Tensor tC = make_tensor(make_gmem_ptr(C), make_shape(N, M)); // Tensor for C (N x M, column-major)

    // Thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index of C

    // Perform matrix multiplication if within bounds
    // if (row < M && col < N)
    // {
    //     TC sum = 0;
    //     for (int k = 0; k < K; ++k)
    //     {
    //         sum += tA(k, row) * tB(col, k); // Access elements using cute::Tensor
    //     }
    //     tC(col, row) = sum; // Write the result to C (column-major)
    // }
    gemm(tA, tB, tC);
}

int main(int argc, char **argv)
{
    constexpr size_t M = 1024; // Default matrix dimensions
    constexpr size_t N = 1024;
    constexpr size_t K = 1024;
    // // Matrix dimensions
    // int M = 1024;
    // if (argc >= 2)
    //     sscanf(argv[1], "%d", &M);

    // int N = 1024;
    // if (argc >= 3)
    //     sscanf(argv[2], "%d", &N);

    // int K = 1024;
    // if (argc >= 4)
    //     sscanf(argv[3], "%d", &K);

    // Host matrices
    thrust::host_vector<float> h_A(K * M, 1.0f); // Initialize A with all 1s (column-major)
    thrust::host_vector<float> h_B(N * K, 1.0f); // Initialize B with all 1s (column-major)
    thrust::host_vector<float> h_C(N * M, 0.0f); // Initialize C with all 0s (column-major)

    // Device matrices
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(N * M);

    // Launch kernel
    dim3 blockDim(16, 16);                                                              // Threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y); // Blocks per grid

    gemm_naive<M, N, K><<<gridDim, blockDim>>>(
                                      thrust::raw_pointer_cast(d_A.data()),
                                      thrust::raw_pointer_cast(d_B.data()),
                                      thrust::raw_pointer_cast(d_C.data()));

    // Copy result back to host
    thrust::copy(d_C.begin(), d_C.end(), h_C.begin());

    // Print result
    // std::cout << "Result matrix C:" << std::endl;
    // for (int i = 0; i < M; ++i)
    // {
    //     for (int j = 0; j < N; ++j)
    //     {
    //         std::cout << h_C[j * M + i] << " "; // Access column-major result
    //     }
    //     std::cout << std::endl;
    // }

    double gflops = (2.0 * M * N * K) * 1e-9;

    const int timing_iterations = 100;
    GPU_Clock timer;
    // Timing iterations
    timer.start();
    for (int i = 0; i < timing_iterations; ++i)
    {
        gemm_naive<M, N, K><<<gridDim, blockDim>>>(
                                          thrust::raw_pointer_cast(d_A.data()),
                                          thrust::raw_pointer_cast(d_B.data()),
                                          thrust::raw_pointer_cast(d_C.data()));
    }
    double cute_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();
    printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time * 1000);
    return 0;
}