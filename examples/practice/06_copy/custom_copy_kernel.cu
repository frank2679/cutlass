#include "custom_copy_traits.hpp"
#include "copy_kernel.cuh"
#include <iostream>

using namespace cute;

__global__ void custom_copy_tma_style_demo_kernel(half_t* gmem_a, half_t* gmem_b, int M, int N) {
    // For 1x1 configuration, process the entire matrix as one tile
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Thread block configuration: grid(1,1), block(1,1)\n");
        printf("Processing entire %dx%d matrix as one tile\n", M, N);
    }
    
    // Shared memory for the entire matrix (since we're using 1x1 configuration)
    extern __shared__ half_t shared_mem[];
    half_t* smem_a = shared_mem;
    half_t* smem_b = shared_mem + M * N;
    
    // Create layouts for the entire matrix
    auto gmem_layout = make_layout(make_shape(Int<256>{}));
    auto smem_layout = make_layout(make_shape(Int<256>{}));
    
    // Create global tensors
    Tensor gmem_a_tensor = make_tensor(make_gmem_ptr(gmem_a), gmem_layout);
    Tensor gmem_b_tensor = make_tensor(make_gmem_ptr(gmem_b), gmem_layout);
    
    // Create shared memory tensors
    Tensor smem_a_tensor = make_tensor(make_smem_ptr(smem_a), smem_layout);
    Tensor smem_b_tensor = make_tensor(make_smem_ptr(smem_b), smem_layout);
    
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Global memory tensor A layout: "); print(gmem_a_tensor.layout()); printf("\n");
        printf("Shared memory tensor A layout: "); print(smem_a_tensor.layout()); printf("\n");
        printf("Global memory tensor B layout: "); print(gmem_b_tensor.layout()); printf("\n");
        printf("Shared memory tensor B layout: "); print(smem_b_tensor.layout()); printf("\n");
    }
    
    // Create custom copy operation - similar to TMA
    // auto custom_copy = make_custom_copy(gmem_a_tensor, smem_layout);
    using ATOM = Copy_Atom<Copy_Traits<CustomCopy>, half_t>;
    ATOM custom_copy{};
    
    // Copy from global memory to shared memory using custom copy
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Copying entire matrix from global to shared memory using custom copy\n");
    }
    
    // Use standard cute::copy interface - this is the key difference
    copy(custom_copy, gmem_a_tensor, smem_a_tensor);
    
    // Synchronize to ensure copy is complete
    __syncthreads();
    
    // Copy from shared memory A to shared memory B using custom copy
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Copying data from shared memory A to B using custom copy\n");
    }
    
    // Use standard cute::copy interface - this is the key difference
    copy(custom_copy, smem_a_tensor, smem_b_tensor);
    
    // Synchronize to ensure copy is complete
    __syncthreads();
    
    // Copy from shared memory B back to global memory using custom copy
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Copying data from shared memory back to global memory using custom copy\n");
    }
    
    // Use standard cute::copy interface - this is the key difference
    copy(custom_copy, smem_b_tensor, gmem_b_tensor);
}