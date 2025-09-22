#ifndef CUSTOM_COPY_TMA_STYLE_HPP
#define CUSTOM_COPY_TMA_STYLE_HPP

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/atom/copy_atom.hpp>

namespace cute {

// Custom copy operation tag - similar to TMA operations
struct CustomCopy {
  // Actual copy implementation
  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_HOST_DEVICE static constexpr
  void copy(Tensor<SEngine, SLayout> const& src,
            Tensor<DEngine, DLayout>      & dst)
  {
    // Our custom copy implementation - simple element-wise copy
    // This is where we would implement our custom logic
    CUTE_STATIC_ASSERT_V(size(src) == size(dst), "CustomCopy: src and dst must have same size.");
    printf("%s, %s: %d\n", __FILE__, __FUNCTION__, __LINE__);
    
    // Simple element-wise copy - this is where you would add your custom logic
    for (int i = 0; i < size(src); ++i) {
      dst(i) = src(i);
    }
  }
};

// Traits for the custom copy operation - similar to TMA traits
template <>
struct Copy_Traits<CustomCopy>
{
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  // For 16x16 half_t matrix = 256 elements = 512 bytes = 4096 bits
  using SrcLayout = Layout<Shape<_1, Int<4096>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, Int<4096>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
  
  template <class... Args>
  CUTE_HOST_DEVICE
  static constexpr auto
  with(Args&&... args) {
    return Copy_Traits<CustomCopy>{};
  }
};

// Specialize copy_unpack for our custom copy operation
// This is similar to how TMA operations implement copy_unpack
template <class SEngine, class SLayout,
          class DEngine, class DLayout>
CUTE_HOST_DEVICE constexpr
void
copy_unpack(Copy_Traits<CustomCopy> const& traits,
            Tensor<SEngine, SLayout> const& src,
            Tensor<DEngine, DLayout>      & dst)
{
  // Delegate to the actual copy implementation in the CustomCopy struct
  CustomCopy::copy(src, dst);
}

// Factory function to create custom copy - similar to make_tma_copy
template <class GEngine, class GLayout,
          class SLayout>
CUTE_HOST_DEVICE
auto
make_custom_copy(Tensor<GEngine, GLayout> const& gtensor,
                 SLayout                 const& slayout) {
  // Create a Copy_Atom with our custom traits - similar to TMA
  using Atom = Copy_Atom<Copy_Traits<CustomCopy>, typename GEngine::value_type>;
  
  Copy_Traits<CustomCopy> traits{};
  Atom atom{traits};
  
  return atom;
}

} // end namespace cute

#endif // CUSTOM_COPY_TMA_STYLE_HPP