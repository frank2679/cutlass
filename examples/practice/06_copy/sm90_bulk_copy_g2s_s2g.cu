#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/half.h>
#include <iostream>

using namespace cute;
using half_t = cutlass::half_t;

template <typename T, int CTA_M, int CTA_N, class SmemLayout, class GmemTensor>
__global__ void bulk_copy_kernel(GmemTensor gmem_tensor_in,
                                 GmemTensor gmem_tensor_out) {
  using namespace cute;
  constexpr int bulk_transaction_bytes = CTA_M * CTA_N * sizeof(T);
  constexpr int bulk_transaction_bits = bulk_transaction_bytes * 8;

  __shared__ T smem_data[CTA_M * CTA_N];
  __shared__ uint64_t bulk_load_mbar;

  auto smem_tensor = make_tensor(make_smem_ptr(smem_data), SmemLayout{});

  // Create bulk copy atoms
  using BulkCopyG2S =
      Copy_Atom<Copy_Traits<SM90_BULK_COPY_G2S, Int<bulk_transaction_bits>>, T>;
  using BulkCopyS2G =
      Copy_Atom<Copy_Traits<SM90_BULK_COPY_S2G, Int<bulk_transaction_bits>>, T>;

  // Bulk copy from gmem to smem
  if (threadIdx.x == 0) {
    initialize_barrier(bulk_load_mbar, /* arrival count */ 1);
    set_barrier_transaction_bytes(bulk_load_mbar, bulk_transaction_bytes);

    auto gmem_tensor_coord_cta_in =
        local_tile(gmem_tensor_in, make_shape(Int<CTA_M>{}, Int<CTA_N>{}),
                   make_coord(blockIdx.x, blockIdx.y));

    // Perform bulk copy G2S
    copy(BulkCopyG2S{}.with(bulk_load_mbar), gmem_tensor_coord_cta_in,
         smem_tensor);

    if (block0() && thread0()) {
      print("gmem_tensor_coord_cta_in: "); print(gmem_tensor_coord_cta_in); print("\n");
      print("smem_tensor: "); print(smem_tensor); print("\n");
      printf("CTA_M: %d, CTA_N: %d, sizeof(T): %zu\n", CTA_M, CTA_N, sizeof(T));
      printf("bulk_transaction_bytes: %d\n", bulk_transaction_bytes);
      printf("bulk_transaction_bits: %d\n", bulk_transaction_bits);
      printf("G2S NumValSrc: %d\n", BulkCopyG2S::NumValSrc);
      printf("G2S NumValDst: %d\n", BulkCopyG2S::NumValDst);
      printf("S2G NumValSrc: %d\n", BulkCopyS2G::NumValSrc);
      printf("S2G NumValDst: %d\n", BulkCopyS2G::NumValDst);
      printf("sizeof src_tensor: %d\n", size(gmem_tensor_coord_cta_in));
      printf("sizeof dst_tensor: %d\n", size(smem_tensor));
    }
  }

  __syncthreads();
  wait_barrier(bulk_load_mbar, /* phase */ 0);

  // After this line, the bulk load is finished
  // Add a fence to ensure all smem operations are complete before storing
  if (thread0()) {
    tma_store_fence();
  }
  __syncthreads();

  // Bulk copy from smem to gmem
  if (threadIdx.x == 0) {
    auto gmem_tensor_coord_cta_out = local_tile(
        gmem_tensor_out, make_shape(Int<CTA_M>{}, Int<CTA_N>{}),
        make_coord(blockIdx.x, blockIdx.y));

    // Perform bulk copy S2G
    copy(BulkCopyS2G{}, smem_tensor, gmem_tensor_coord_cta_out);

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
void host_bulk_copy_fn(T *data_in, T *data_out, int M, int N) {
  using namespace cute;
  // create the GMEM tensors
  auto gmem_layout = make_layout(make_shape(M, N), LayoutRight{});
  auto gmem_tensor_in = make_tensor(make_gmem_ptr(data_in), gmem_layout);
  auto gmem_tensor_out = make_tensor(make_gmem_ptr(data_out), gmem_layout);

  // Create the SMEM layout
  auto smem_layout =
      make_layout(make_shape(Int<CTA_M>{}, Int<CTA_N>{}), LayoutRight{});

  // Calculate grid and block dimensions
  dim3 grid_dim((M * N) / (CTA_M * CTA_N), 1, 1);
  dim3 block_dim(32, 1, 1);

  // Launch kernel
  bulk_copy_kernel<T, CTA_M, CTA_N, decltype(smem_layout)>
      <<<grid_dim, block_dim>>>(gmem_tensor_in, gmem_tensor_out);
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
    h_a[i] = half_t(float(i % 100) / 10.0f); // Simple pattern
  }

  // Initialize output matrix with zeros
  for (int i = 0; i < M * N; ++i) {
    h_b[i] = half_t(0.0f);
    h_b_ref[i] = half_t(0.0f);
  }

  // Copy input matrix to device
  cudaMemcpy(d_a, h_a, sizeof(half_t) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(half_t) * M * N, cudaMemcpyHostToDevice);

  // Calculate shared memory size needed
  size_t shared_mem_size = sizeof(half_t) * 128 * 64 * 2;
  std::cout << "Shared memory size required: " << shared_mem_size << " bytes"
            << std::endl;

  host_bulk_copy_fn<half_t, 128, 64>(d_a, d_b, M, N);

  // Check for kernel launch errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "CUDA kernel launch error: " << cudaGetErrorString(error)
              << std::endl;
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

  std::cout << "Verification: " << (verification_passed ? "PASSED" : "FAILED")
            << std::endl;

  // Clean up memory
  cudaFree(d_a);
  cudaFree(d_b);
  delete[] h_a;
  delete[] h_b;
  delete[] h_b_ref;

  return verification_passed ? 0 : -1;
}