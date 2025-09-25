#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "gmem_desc_general_shape.hpp"
#include <cute/tensor.hpp>
#include <cutlass/half.h>

using namespace cute;
using half_t = cutlass::half_t;

using ATOM_M = Int<64>; // 符合multiple of 64约束
using ATOM_N = Int<32>; // 符合multiple of 32约束 (减小以节省内存)
using ATOM_K = Int<32>; // 符合FP16的multiple of 32约束 (32 is valid for FP16)

__global__ void test_gemm_with_dmem_desc_final(half_t *ptr_a, half_t *ptr_b,
                                               half_t *ptr_c, half_t *ptr_d,
                                               int M, int N, int K) {
  // 计算需要多少个tile来覆盖整个矩阵
  const int M_TILES = M / size(ATOM_M{});
  const int N_TILES = N / size(ATOM_N{});
  const int K_TILES = K / size(ATOM_K{});

  // 创建MMA原子操作
  using MMA_Op = CustomMMA<half_t, ATOM_M, ATOM_N, ATOM_K>;
  // auto mma_atom = MMA_Atom<MMA_Op>{}; // 可以直接用 gemm(mma_atom, xxx)
  using MMA_ATOM = MMA_Atom<MMA_Traits<MMA_Op>>;
  using TiledMMA = TiledMMA<MMA_ATOM, Layout<Shape<_1, _1, _1>>>;
  TiledMMA tiled_mma{};

  auto thr_mma = tiled_mma.get_slice(0); // 只有一个线程

  // print("tiled_mma");
  // print(tiled_mma);

  // 定义矩阵布局 (row-major)
  auto layout_a = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
  auto layout_b = make_layout(make_shape(K, N), make_stride(N, Int<1>{}));
  auto layout_c = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));
  auto layout_d = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));

  Tensor gmem_a_tensor = make_tensor(make_gmem_ptr(ptr_a), layout_a);
  Tensor gmem_b_tensor = make_tensor(make_gmem_ptr(ptr_b), layout_b);
  Tensor gmem_c_tensor = make_tensor(make_gmem_ptr(ptr_c), layout_c);
  Tensor gmem_d_tensor = make_tensor(make_gmem_ptr(ptr_d), layout_d);

  // 分配共享内存缓冲区用于存储A、B、C和D的tile
  extern __shared__ half_t shared_mem[];
  half_t *smem_a = shared_mem;
  half_t *smem_b = smem_a + size(ATOM_M{}) * size(ATOM_K{});
  half_t *smem_d = smem_b + size(ATOM_K{}) * size(ATOM_N{});

  auto layout_a_smem = make_layout(make_shape(ATOM_M{}, ATOM_K{}),
                                   make_stride(size(ATOM_K{}), Int<1>{}));
  auto layout_b_smem = make_layout(make_shape(ATOM_K{}, ATOM_N{}),
                                   make_stride(size(ATOM_N{}), Int<1>{}));
  auto layout_d_smem = make_layout(make_shape(ATOM_M{}, ATOM_N{}),
                                   make_stride(size(ATOM_N{}), Int<1>{}));
  Tensor smem_a_tensor = make_tensor(make_smem_ptr(smem_a), layout_a_smem);
  Tensor smem_b_tensor = make_tensor(make_smem_ptr(smem_b), layout_b_smem);
  Tensor smem_d_tensor = make_tensor(make_smem_ptr(smem_d), layout_d_smem);

  // 执行分块GEMM计算
  for (int m = 0; m < M_TILES; ++m) {
    for (int n = 0; n < N_TILES; ++n) {
      for (int k = 0; k < K_TILES; ++k) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
          printf("Processing block (m=%d, n=%d, k=%d)\n", m, n, k);
        }

        // 将A和B的部分数据复制到连续的共享内存缓冲区
        // for (int i = 0; i < size(ATOM_M{}); ++i) {
        //   for (int j = 0; j < size(ATOM_K{}); ++j) {
        //     int global_i = m * size(ATOM_M{}) + i;
        //     int global_j = k * size(ATOM_K{}) + j;
        //     smem_a[i * size(ATOM_K{}) + j] = ptr_a[global_i * K + global_j];
        //   }
        // }

        // 该实现与上面等价
        // Tensor gmem_a_tile = local_tile(
        //     gmem_a_tensor, make_tile(ATOM_M{}, ATOM_K{}), make_coord(m, k));
        // copy(gmem_a_tile, smem_a_tensor);
        // print("\ngmem_a_tile\n");
        // print(gmem_a_tile);

        // 该实现与上面等价
        auto thr_a_tensor = thr_mma.partition_A(gmem_a_tensor);
        copy(thr_a_tensor(_, m, k), smem_a_tensor);
        // print("\nthr_a_tensor\n");
        // print(thr_a_tensor);

        auto thr_b_tensor = thr_mma.partition_B(gmem_b_tensor);
        copy(thr_b_tensor(_, k, n), smem_b_tensor);

        if (k == 0) {
          auto thr_c_tensor = thr_mma.partition_C(gmem_c_tensor);
          copy(thr_c_tensor(_, m, n), smem_d_tensor);
        }

        __syncthreads();


        // 当前实现：使用cute::gemm调用CustomMMA，仅实现 custom MMA 操作
        // todo: 内部实现 tile for loop
        cute::gemm(tiled_mma, smem_d_tensor, smem_a_tensor, smem_b_tensor,
                   smem_d_tensor);

        __syncthreads();
      }

      auto thr_d_tensor = thr_mma.partition_C(gmem_d_tensor);
      copy(smem_d_tensor, thr_d_tensor(_, m, n));

      __syncthreads();
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("Kernel execution completed\n");
  }
}

#endif // KERNEL_HPP