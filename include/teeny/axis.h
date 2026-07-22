#ifndef TNY_MD_AXIS
#define TNY_MD_AXIS
// Axis-manipulation view builders on a raw mdspan: permute, flip, unsqueeze,
// squeeze. Each reads only extents/strides/data_handle, so it works on ANY
// source layout, and returns a cs::layout_stride mdspan the tensor class wraps
// with as_tensor(). Kept out of tensor.h so the class sits near the top of it.
#include <cuda/std/mdspan>
#include <cuda/std/array>
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <teeny/defines.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

namespace _detail {

// reorder axes P... -> a layout_stride view with the permuted extents/strides.
template <class MD, cs::size_t... P>
_TNY_API auto perm_md(const MD & v, cs::index_sequence<P...>) {
    using El  = typename MD::element_type;
    using Idx = typename MD::index_type;
    using E   = typename MD::extents_type;
    using PE  = cs::extents<Idx, E::static_extent(P)...>;
    cs::layout_stride::mapping<PE> m(
        PE(static_cast<Idx>(v.extent(P))...),
        cs::array<Idx, sizeof...(P)>{ static_cast<Idx>(v.stride(P))... });
    return cs::mdspan<El, PE, cs::layout_stride>(v.data_handle(), m);
}
// reverse axis AX: keep the extents, negate that axis' stride, and shift the
// data handle to the last element along AX (so index 0 maps to the old last).
template <cs::size_t AX, class MD, cs::size_t... D>
_TNY_API auto flip_md(const MD & v, cs::index_sequence<D...>) {
    using El  = typename MD::element_type;
    using Idx = typename MD::index_type;
    using E   = typename MD::extents_type;
    static_assert(cs::is_signed<Idx>::value, "flip needs a signed index type (e.g. shape<...>)");
    cs::layout_stride::mapping<E> m(
        v.extents(),
        cs::array<Idx, sizeof...(D)>{ static_cast<Idx>(D == AX ? -static_cast<Idx>(v.stride(D)) : static_cast<Idx>(v.stride(D)))... });
    const Idx n = static_cast<Idx>(v.extent(AX));
    const Idx off = n > Idx(0) ? (n - 1) * static_cast<Idx>(v.stride(AX)) : Idx(0);   // empty axis: no shift
    return cs::mdspan<El, E, cs::layout_stride>(v.data_handle() + off, m);
}
// insert a size-1 axis at position AX (output rank = N+1). The new axis gets
// stride 1 (its index is always 0, so the value is irrelevant to the offset).
// J... = 0..N ; input axis for output j is j (j<AX) or j-1 (j>AX).
template <cs::size_t AX, class MD, cs::size_t... J>
_TNY_API auto unsqueeze_md(const MD & v, cs::index_sequence<J...>) {
    using El  = typename MD::element_type;
    using Idx = typename MD::index_type;
    using E   = typename MD::extents_type;
    using OE  = cs::extents<Idx, (J == AX ? cs::size_t(1) : E::static_extent(J < AX ? J : J - 1))...>;
    cs::layout_stride::mapping<OE> m(
        OE(static_cast<Idx>(J == AX ? Idx(1) : v.extent(J < AX ? J : J - 1))...),
        cs::array<Idx, sizeof...(J)>{ static_cast<Idx>(J == AX ? Idx(1) : v.stride(J < AX ? J : J - 1))... });
    return cs::mdspan<El, OE, cs::layout_stride>(v.data_handle(), m);
}
// drop axis AX (must have extent 1) -> output rank = N-1. J... = 0..N-2 ;
// input axis for output j is j (j<AX) or j+1 (j>=AX).
template <cs::size_t AX, class MD, cs::size_t... J>
_TNY_API auto squeeze_md(const MD & v, cs::index_sequence<J...>) {
    using El  = typename MD::element_type;
    using Idx = typename MD::index_type;
    using E   = typename MD::extents_type;
    using OE  = cs::extents<Idx, E::static_extent(J < AX ? J : J + 1)...>;
    cs::layout_stride::mapping<OE> m(
        OE(static_cast<Idx>(v.extent(J < AX ? J : J + 1))...),
        cs::array<Idx, sizeof...(J)>{ static_cast<Idx>(v.stride(J < AX ? J : J + 1))... });
    return cs::mdspan<El, OE, cs::layout_stride>(v.data_handle(), m);
}

} // namespace _detail

_TNY_NAMESPACE_END(tny)
#endif // TNY_MD_AXIS
