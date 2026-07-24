#ifndef TNY_MD_AXIS
#define TNY_MD_AXIS
// Axis-manipulation view builders on a raw mdspan: permute, flip, unsqueeze,
// squeeze. Each reads only extents/strides/data_handle, so it works on ANY
// source layout, and returns a folded teeny `strides<...>` mdspan the tensor
// class wraps with as_tensor() — so a permuted/flipped/un-squeezed static (or
// `strides<...>`) source keeps its compile-time strides, like the slice gather.
// Kept out of tensor.h so the class sits near the top of it.
#include <cuda/std/mdspan>
#include <cuda/std/array>
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <teeny/defines.h>
#include <teeny/layout.h>   // strides<...>, _src_sstride, dynamic_stride

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

namespace _detail {

// flip: negate axis AX's static source stride (a dynamic stride stays dynamic —
// `-dynamic_stride` is not the sentinel).
template <cs::size_t D, cs::size_t AX, class L, class E>
_TNY_API constexpr cs::int64_t _flip_sstride() {
    constexpr cs::int64_t s = _src_sstride<D, L, E>();
    return (D == AX) ? ((s == dynamic_stride) ? dynamic_stride : -s) : s;
}

// Build a folded `strides<Sf...>::mapping<OE>` from output extents `oe` and the
// per-output-axis runtime strides `rstr` (only the dynamic_stride slots are read).
template <class SF, class OE, class Idx>
_TNY_API auto fold_mapping(const OE & oe, const Idx * rstr) {
    using Map = typename SF::template mapping<OE>;
    if constexpr (SF::all_static()) { (void)rstr; return Map(oe); }
    else {
        cs::array<Idx, SF::ndyn()> dyn{};
        for (cs::size_t r = 0; r < OE::rank(); ++r)
            if (SF::S_[r] == dynamic_stride) dyn[SF::slot(r)] = rstr[r];
        return Map(oe, dyn);
    }
}

// Rebuild layout `L`'s mapping over NEW extents `oe`, carrying the source per-axis
// runtime strides `rstr`. This is `recast`'s primitive: it lets the mapping itself
// merge layout + extents rather than synthesising a stride layout up front.
//   - contiguous (ccontiguous/fcontiguous): re-DERIVE from the extents. A contiguous
//     source's strides ARE those products, so this preserves them (and richer static
//     extents then fold in the accessor); it is also the "reinterpret AS contiguous"
//     path when `L` is an explicit override.
//   - strides<...>: bake the static slots, fill the dynamic ones from `rstr`.
//   - layout_stride: carry every stride at run time.
template <class L, class OE, class Idx>
_TNY_API auto retype_mapping(const OE & oe, const Idx * rstr) {
    if constexpr (_contiguous_layout<L>::value) { (void)rstr; return typename L::template mapping<OE>(oe); }
    else if constexpr (_is_strides<L>::value)   return fold_mapping<L>(oe, rstr);
    else {   // layout_stride (or any runtime-strided mapping): carry all axes
        cs::array<Idx, OE::rank()> st{};
        for (cs::size_t r = 0; r < OE::rank(); ++r) st[r] = rstr[r];
        return typename L::template mapping<OE>(oe, st);
    }
}

// reorder axes P... -> a folded strides<...> view (output stride[i] = source
// stride[P[i]], compile-time where the source stride is static).
template <class MD, cs::size_t... P>
_TNY_API auto perm_md(const MD & v, cs::index_sequence<P...>) {
    using El  = typename MD::element_type; using Idx = typename MD::index_type;
    using E   = typename MD::extents_type; using L   = typename MD::layout_type;
    using PE  = cs::extents<Idx, E::static_extent(P)...>;
    using SF  = strides< _src_sstride<P, L, E>()... >;
    PE pe(static_cast<Idx>(v.extent(P))...);
    const Idx rstr[sizeof...(P) ? sizeof...(P) : 1] = { static_cast<Idx>(v.stride(P))... };
    return cs::mdspan<El, PE, SF>(v.data_handle(), fold_mapping<SF>(pe, rstr));
}
// reverse axis AX: negate that axis' stride, shift the handle to the last element
// (so index 0 maps to the old last). Folds (the negated static stride is static).
template <cs::size_t AX, class MD, cs::size_t... D>
_TNY_API auto flip_md(const MD & v, cs::index_sequence<D...>) {
    using El  = typename MD::element_type; using Idx = typename MD::index_type;
    using E   = typename MD::extents_type; using L   = typename MD::layout_type;
    static_assert(cs::is_signed<Idx>::value, "flip needs a signed index type (e.g. shape<...>)");
    using SF  = strides< _flip_sstride<D, AX, L, E>()... >;
    E e = v.extents();
    const Idx rstr[sizeof...(D) ? sizeof...(D) : 1] =
        { static_cast<Idx>(D == AX ? -static_cast<Idx>(v.stride(D)) : static_cast<Idx>(v.stride(D)))... };
    const Idx n = static_cast<Idx>(v.extent(AX));
    const Idx off = n > Idx(0) ? (n - 1) * static_cast<Idx>(v.stride(AX)) : Idx(0);   // empty axis: no shift
    return cs::mdspan<El, E, SF>(v.data_handle() + off, fold_mapping<SF>(e, rstr));
}
// insert a size-1 axis at position AX (output rank = N+1). The new axis gets a
// static stride 1 (its index is always 0, so the value is irrelevant to offsets).
// J... = 0..N ; input axis for output j is j (j<AX) or j-1 (j>AX). For j==AX the
// phantom source index is clamped to 0 so `_src_sstride`/`stride` never overrun.
template <cs::size_t AX, class MD, cs::size_t... J>
_TNY_API auto unsqueeze_md(const MD & v, cs::index_sequence<J...>) {
    using El  = typename MD::element_type; using Idx = typename MD::index_type;
    if constexpr (MD::extents_type::rank() == 0) {
        // rank-0 -> rank-1 (AX is necessarily 0): one size-1 axis over the same
        // element. A rank-0 source has no strides to read (CCCL constrains
        // `stride()` to rank > 0, and `static_extent(0)` is out of range), so the
        // general body below can't even instantiate for it — build the fixed
        // `strides<1>` view directly. #71.
        using OE = cs::extents<Idx, 1>; using SF = strides<1>;
        return cs::mdspan<El, OE, SF>(v.data_handle(), typename SF::template mapping<OE>(OE{}));
    } else {
        using E   = typename MD::extents_type; using L   = typename MD::layout_type;
        using OE  = cs::extents<Idx, (J == AX ? cs::size_t(1) : E::static_extent(J < AX ? J : J - 1))...>;
        using SF  = strides< (J == AX ? cs::int64_t(1) : _src_sstride<(J == AX ? cs::size_t(0) : (J < AX ? J : J - 1)), L, E>())... >;
        OE oe(static_cast<Idx>(J == AX ? Idx(1) : v.extent(J < AX ? J : J - 1))...);
        const Idx rstr[sizeof...(J) ? sizeof...(J) : 1] =
            { static_cast<Idx>(J == AX ? Idx(1) : v.stride(J < AX ? J : J - 1))... };
        return cs::mdspan<El, OE, SF>(v.data_handle(), fold_mapping<SF>(oe, rstr));
    }
}
// drop axis AX (must have extent 1) -> output rank = N-1. J... = 0..N-2 ;
// input axis for output j is j (j<AX) or j+1 (j>=AX).
template <cs::size_t AX, class MD, cs::size_t... J>
_TNY_API auto squeeze_md(const MD & v, cs::index_sequence<J...>) {
    using El  = typename MD::element_type; using Idx = typename MD::index_type;
    using E   = typename MD::extents_type; using L   = typename MD::layout_type;
    using OE  = cs::extents<Idx, E::static_extent(J < AX ? J : J + 1)...>;
    using SF  = strides< _src_sstride<(J < AX ? J : J + 1), L, E>()... >;
    OE oe(static_cast<Idx>(v.extent(J < AX ? J : J + 1))...);
    const Idx rstr[sizeof...(J) ? sizeof...(J) : 1] =
        { static_cast<Idx>(v.stride(J < AX ? J : J + 1))... };
    return cs::mdspan<El, OE, SF>(v.data_handle(), fold_mapping<SF>(oe, rstr));
}

} // namespace _detail

_TNY_NAMESPACE_END(tny)
#endif // TNY_MD_AXIS
