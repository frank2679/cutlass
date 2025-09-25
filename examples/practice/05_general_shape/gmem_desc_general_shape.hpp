#ifndef DMEM_DESC_EXAMPLE_HPP
#define DMEM_DESC_EXAMPLE_HPP

#include <cute/tensor.hpp>
#include <cutlass/half.h>

using half_t = cutlass::half_t;

namespace cute {

// 定义CustomMMA操作 (模板化)
template <typename T, typename M, typename N, typename K> struct CustomMMA {
  // 定义GEMM的MNK维度，符合硬件约束
  // M: 1/2/4/8/16/32/64/multiple of 64 until 4096
  // N: multiple of 32 until 4096
  // K (FP16): 32/multiple of 64 until 65536

  // 验证M值是否符合硬件约束
  static_assert((size(M{}) == 1) || (size(M{}) == 2) || (size(M{}) == 4) ||
                    (size(M{}) == 8) || (size(M{}) == 16) ||
                    (size(M{}) == 32) || (size(M{}) == 64) ||
                    (size(M{}) % 64 == 0 && size(M{}) <= 4096),
                "M value must be 1/2/4/8/16/32/64/multiple of 64 until 4096");

  // 验证N值是否符合硬件约束
  static_assert((size(N{}) % 32 == 0) && (size(N{}) <= 4096),
                "N value must be multiple of 32 until 4096");

  // 验证K值是否符合FP16硬件约束 (假设使用FP16)
  static_assert(((size(K{}) == 32) || (size(K{}) % 64 == 0)) &&
                    (size(K{}) <= 65536),
                "K value for FP16 must be 32/multiple of 64 until 65536");

  CUTE_HOST_DEVICE static void fma(smem_ptr<T *> d, smem_ptr<T *> const a,
                                   smem_ptr<T *> const b,
                                   smem_ptr<T *> const c) {
    // 获取 M, N, K 的值
    constexpr int M_VALUE = size(M{});
    constexpr int N_VALUE = size(N{});
    constexpr int K_VALUE = size(K{});

    // 执行矩阵乘法累加: D = A * B + C
    // 使用二维坐标访问简化代码
    for (int m = 0; m < M_VALUE; ++m) {
      for (int n = 0; n < N_VALUE; ++n) {
        // 初始化D值为C值
        T d_val = c[m * N_VALUE + n];

        // 执行K步累加
        for (int k = 0; k < K_VALUE; ++k) {
          T a_val = a[m * K_VALUE + k];
          T b_val = b[k * N_VALUE + n];
          d_val = d_val + a_val * b_val;
        }

        // 将结果写回D矩阵
        d[m * N_VALUE + n] = d_val;
      }
    }
  }
};

// 为CustomMMA<half_t>特化MMA_Traits
template <typename M, typename N, typename K>
struct MMA_Traits<CustomMMA<half_t, M, N, K>> {
  using ValTypeD = smem_ptr<half_t *>;
  using ValTypeA = smem_ptr<half_t *>;
  using ValTypeB = smem_ptr<half_t *>;
  using ValTypeC = smem_ptr<half_t *>;

  using FrgTypeA = smem_ptr<half_t *>;
  using FrgTypeB = smem_ptr<half_t *>;
  using FrgTypeC = smem_ptr<half_t *>;
  using FrgTypeD = smem_ptr<half_t *>;

  using Shape_MNK = Shape<M, N, K>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape<_1, Shape<M, K>>>;
  using BLayout = Layout<Shape<_1, Shape<N, K>>>;
  using CLayout = Layout<Shape<_1, Shape<M, N>>>;

  //
  // 特化mma_unpack实现，针对CustomMMA<half_t>
  //
  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend void
  mma_unpack(MMA_Traits const &traits, Tensor<TD, DLayout> &D,
             Tensor<TA, ALayout> const &A, Tensor<TB, BLayout> const &B,
             Tensor<TC, CLayout> const &C) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
      printf("%s, %s: %d\n", __FILE__, __FUNCTION__, __LINE__);
    // 获取 buffer 指针 D.data() 是 cute::smem_ptr<cutlass::half_t *>,
    // D.data().get() 拿到 half *
    auto rD = D.data();
    auto rA = A.data();
    auto rB = B.data();
    auto rC = C.data();

    // 调用CustomMMA的fma操作
    CustomMMA<half_t, M, N, K>::fma(rD, rA, rB, rC);
  }
};

template <class MMA, class TD, class DLayout, class TA, class ALayout, class TB,
          class BLayout, class TC, class CLayout,
          __CUTE_REQUIRES(DLayout::rank == 2 && is_smem<TD>::value &&
                          ALayout::rank == 2 && is_smem<TA>::value &&
                          BLayout::rank == 2 && is_smem<TB>::value &&
                          CLayout::rank == 2 && is_smem<TC>::value)>
CUTE_HOST_DEVICE void gemm(MMA_Atom<MMA> const &mma,
                           Tensor<TD, DLayout> &D,       // (M,N) Logical data
                           Tensor<TA, ALayout> const &A, // (M,K) Logical data
                           Tensor<TB, BLayout> const &B, // (N,K) Logical data
                           Tensor<TC, CLayout> const &C) // (M,N) Logical data
{
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    printf("%s, %s: %d\n", __FILE__, __FUNCTION__, __LINE__);
  CUTE_STATIC_ASSERT_V(size<0>(A) == size<0>(C)); // AM == CM
  CUTE_STATIC_ASSERT_V(size<0>(B) == size<1>(C)); // BN == CN
  CUTE_STATIC_ASSERT_V(size<1>(A) == size<1>(B)); // AK == BK
  CUTE_STATIC_ASSERT_V(size<0>(C) == size<0>(D) && size<1>(C) == size<1>(D));

  mma.call(D, A, B, C); // Call the MMA atom with global memory tensors
}

// 为CustomMMA<half_t>特化MMA_Atom
template <typename M, typename N, typename K>
struct MMA_Atom<MMA_Traits<CustomMMA<half_t, M, N, K>>>
    : MMA_Traits<CustomMMA<half_t, M, N, K>> {
  using MMA_Op = CustomMMA<half_t, M, N, K>;
  using Traits = MMA_Traits<CustomMMA<half_t, M, N, K>>;

  // Element value types from the MMA_Traits
  using ValTypeD = typename Traits::ValTypeD;
  using ValTypeA = typename Traits::ValTypeA;
  using ValTypeB = typename Traits::ValTypeB;
  using ValTypeC = typename Traits::ValTypeC;

  // Thr-Val layouts from the MMA_Traits
  using Shape_MNK = typename Traits::Shape_MNK;
  using ThrID = typename Traits::ThrID;
  using LayoutC_TV = typename Traits::CLayout;
  using LayoutA_TV = typename Traits::ALayout;
  using LayoutB_TV = typename Traits::BLayout;

  // Fragment value types from the MMA_Traits (optional, defaults to Val type)
  using FrgTypeD = typename detail::FrgTypeC_or_Default<Traits>::type;
  using FrgTypeA = typename detail::FrgTypeA_or_Default<Traits>::type;
  using FrgTypeB = typename detail::FrgTypeB_or_Default<Traits>::type;
  using FrgTypeC = typename detail::FrgTypeC_or_Default<Traits>::type;

  //
  // Tensor call interfaces
  //

  // Cast, check, and call fma
  template <class TD, class DLayout, class TA, class ALayout, class TB,
            class BLayout, class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr void
  call(Tensor<TD, DLayout> &D, Tensor<TA, ALayout> const &A,
       Tensor<TB, BLayout> const &B, Tensor<TC, CLayout> const &C) const {
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
      printf("%s, %s: %d\n", __FILE__, __FUNCTION__, __LINE__);

    mma_unpack(static_cast<Traits const &>(*this), D, A, B, C);
  }

  // Three arguments reproduces C
  template <class TA, class ALayout, class TB, class BLayout, class TC,
            class CLayout>
  CUTE_HOST_DEVICE constexpr void call(Tensor<TA, ALayout> const &A,
                                       Tensor<TB, BLayout> const &B,
                                       Tensor<TC, CLayout> &C) const {
    return call(C, A, B, C);
  }
};

} // namespace cute

#endif // DMEM_DESC_EXAMPLE_HPP