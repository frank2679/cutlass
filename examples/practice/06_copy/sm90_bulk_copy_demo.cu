#include <iostream>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/half.h>

using namespace cute;
using half_t = cutlass::half_t;

template <typename T, int CTA_M, int CTA_N, class SmemLayout, class TmaLoad, class TmaStore, class GmemTensor>
__global__ void tma_load_store_kernel(__grid_constant__ const TmaLoad tma_load, 
                                      __grid_constant__ const TmaStore tma_store,
                                      GmemTensor gmem_tensor_in,
                                      GmemTensor gmem_tensor_out) {
  using namespace cute;
  constexpr int tma_transaction_bytes = CTA_M * CTA_N * sizeof(T);

  __shared__ T smem_data[CTA_M * CTA_N];
  __shared__ uint64_t tma_load_mbar;
  // Remove tma_store_mbar as TMA store doesn't use barriers in the same way

  auto smem_tensor = make_tensor(make_smem_ptr(smem_data), SmemLayout{});

  // TMA Load from gmem to smem
  if (threadIdx.x == 0) {
    auto gmem_tensor_coord_in = tma_load.get_tma_tensor(shape(gmem_tensor_in));

    auto gmem_tensor_coord_cta_in = local_tile(
        gmem_tensor_coord_in,
        make_shape(Int<CTA_M>{}, Int<CTA_N>{}),
        make_coord(blockIdx.x, blockIdx.y));

    initialize_barrier(tma_load_mbar, /* arrival count */ 1);

    set_barrier_transaction_bytes(tma_load_mbar, tma_transaction_bytes);

    auto tma_load_per_cta = tma_load.get_slice(0);
    copy(tma_load.with(tma_load_mbar),
         tma_load_per_cta.partition_S(gmem_tensor_coord_cta_in),
         tma_load_per_cta.partition_D(smem_tensor));
  }
  __syncthreads();
  wait_barrier(tma_load_mbar, /* phase */ 0);

  // After this line, the TMA load is finished
  // Now perform TMA store from smem to gmem
  // Add a fence to ensure all smem operations are complete before storing
  if (thread0()) {
    tma_store_fence();
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    auto gmem_tensor_coord_out = tma_store.get_tma_tensor(shape(gmem_tensor_out));

    auto gmem_tensor_coord_cta_out = local_tile(
        gmem_tensor_coord_out,
        make_shape(Int<CTA_M>{}, Int<CTA_N>{}),
        make_coord(blockIdx.x, blockIdx.y));

    auto tma_store_per_cta = tma_store.get_slice(0);
    // TMA store doesn't use barriers like TMA load, directly copy
    copy(tma_store,
         tma_store_per_cta.partition_S(smem_tensor),
         tma_store_per_cta.partition_D(gmem_tensor_coord_cta_out));
         
    // Commit the store operation and wait for completion
    tma_store_arrive();
  }
  __syncthreads();
  
  // Wait for all TMA store operations to complete
  if (thread0()) {
    tma_store_wait<0>();
  }
}


template <typename T, int CTA_M, int CTA_N>
void host_load_store_fn(T* data_in, T* data_out, int M, int N) {
  using namespace cute;

  // create the GMEM tensors
  auto gmem_layout = make_layout(make_shape(M, N), LayoutRight{});
  auto gmem_tensor_in = make_tensor(make_gmem_ptr(data_in), gmem_layout);
  auto gmem_tensor_out = make_tensor(make_gmem_ptr(data_out), gmem_layout);

  // create the SMEM layout - using proper CUTE types
  auto smem_layout = make_layout(make_shape(Int<CTA_M>{}, Int<CTA_N>{}), LayoutRight{});

  // create the TMA objects with proper tile shape
  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor_in, smem_layout);
  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, gmem_tensor_out, smem_layout);

  // invoke the kernel
  dim3 grid_dim(M / CTA_M, N/ CTA_N, 1);  // Only 1 block since CTA_M, CTA_N = M, N
  dim3 block_dim(32, 1, 1);  // Use 32 threads per block
  tma_load_store_kernel<T, CTA_M, CTA_N, decltype(smem_layout), decltype(tma_load), decltype(tma_store), decltype(gmem_tensor_in)>
                 <<<grid_dim, block_dim>>>
                 (tma_load, tma_store, gmem_tensor_in, gmem_tensor_out);
}


int main() {
    // Matrix dimensions - using power of 2 sizes for efficient bulk copy
    constexpr int M = 512;
    constexpr int N = 1024;
    
    // Allocate device memory
    half_t *d_a, *d_b;
    cudaMalloc(&d_a, sizeof(half_t) * M * N);
    cudaMalloc(&d_b, sizeof(half_t) * M * N);
    
    // Allocate host memory
    half_t *h_a = new half_t[M * N];
    half_t *h_b = new half_t[M * N];
    half_t *h_b_ref = new half_t[M * N];
    
    // Initialize input matrix with random values
    for (int i = 0; i < M * N; ++i) {
        h_a[i] = half_t(float(i % 100) / 10.0f);  // Simple pattern
    }
    
    // Initialize output matrix with zeros
    for (int i = 0; i < M * N; ++i) {
        h_b[i] = half_t(0.0f);
        h_b_ref[i] = half_t(0.0f);
    }
    
    // Copy input matrix to device
    cudaMemcpy(d_a, h_a, sizeof(half_t) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(half_t) * M * N, cudaMemcpyHostToDevice);
    
    // Calculate shared memory size needed for entire matrices
    size_t shared_mem_size = sizeof(half_t) * M * N * 2;
    std::cout << "Shared memory size required: " << shared_mem_size << " bytes" << std::endl;
    
    host_load_store_fn<half_t, 128, 64>(d_a, d_b, M, N);
    
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
    
    // Print first few elements for verification
    std::cout << "First 10 elements:" << std::endl;
    std::cout << "Input (A): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << float(h_a[i]) << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Output (B): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << float(h_b[i]) << " ";
    }
    std::cout << std::endl;
    
    // Verify results
    bool verification_passed = true;
    for (int i = 0; i < M * N; ++i) {
        if (float(h_b[i]) != float(h_b_ref[i])) {
            verification_passed = false;
            break;
        }
    }
    
    std::cout << "Verification: " << (verification_passed ? "PASSED" : "FAILED") << std::endl;
    
    // Clean up memory
    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;
    delete[] h_b_ref;
    
    return verification_passed ? 0 : -1;
}