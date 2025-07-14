#include <cute/tensor.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

// Naive vector addition kernel using cute::Tensor
template <class T>
__global__ void vector_add_naive(int N, T const* A, T const* B, T* C) {
    using namespace cute;

    // Create cute tensors for input and output vectors
    Tensor tA = make_tensor(make_gmem_ptr(A), make_shape(N)); // Tensor for A
    Tensor tB = make_tensor(make_gmem_ptr(B), make_shape(N)); // Tensor for B
    Tensor tC = make_tensor(make_gmem_ptr(C), make_shape(N)); // Tensor for C

    // Thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform vector addition if within bounds
    if (idx < N) {
        tC(idx) = tA(idx) + tB(idx);
    }
}

int main() {
    // Vector size
    int N = 1024;
    if (N <= 0) {
        std::cerr << "Vector size must be positive!" << std::endl;
        return -1;
    }

    // Host vectors
    thrust::host_vector<float> h_A(N, 1.0f); // Initialize A with all 1s
    thrust::host_vector<float> h_B(N, 2.0f); // Initialize B with all 2s
    thrust::host_vector<float> h_C(N, 0.0f); // Initialize C with all 0s

    // Device vectors
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(N);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add_naive<<<blocksPerGrid, threadsPerBlock>>>(N,
        thrust::raw_pointer_cast(d_A.data()),
        thrust::raw_pointer_cast(d_B.data()),
        thrust::raw_pointer_cast(d_C.data()));

    // Copy result back to host
    thrust::copy(d_C.begin(), d_C.end(), h_C.begin());

    // Print result
    std::cout << "Result vector C:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}