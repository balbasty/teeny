#ifndef TNY_MD_ITERATE
#define TNY_MD_ITERATE
#include <cuda/std/mdspan>
#include <cuda/std/utility>
#include <teeny/defines.h>
#include <teeny/tensor.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/* ================================================================== *
 *  nd-peel: iterate over a SUBSET of axes, yielding a lower-rank view *
 *  over the remaining axes.                                          *
 *                                                                    *
 *  Replaces the ndindex<->linear plumbing (jitfields index2offset /  *
 *  sub2offset): a linear index over the peeled axes is decoded for   *
 *  you, and the corresponding sub-view (a `tny::tensor` view into the  *
 *  original data) is returned.                                       *
 *                                                                    *
 *  Two entry points:                                                 *
 *    peel_at<Axes...>(t, i)  -> the i-th sub-view  (grid-stride loop) *
 *    peel<Axes...>(t)       -> a range of them    (range-for)       *
 * ================================================================== */

namespace _md {

// position of source axis A within the peeled set (as an index_sequence), <0 if kept.
template <cs::size_t A, class Seq> struct peel_pos;
template <cs::size_t A, cs::size_t... Axes> struct peel_pos<A, cs::index_sequence<Axes...>>
{ static constexpr int value = _pos_in<A, Axes...>(); };

// per source axis A: static output extent / stride (drop sentinels if peeled).
template <cs::size_t A, class E, class Seq>
_TNY_API constexpr cs::size_t peel_ext() { return peel_pos<A, Seq>::value >= 0 ? _drop_axis : E::static_extent(A); }
template <cs::size_t A, class L, class E, class Seq>
_TNY_API constexpr cs::int64_t peel_str() { return peel_pos<A, Seq>::value >= 0 ? _sdrop : _src_sstride<A, L, E>(); }

template <cs::size_t A, class Seq, class MD, class I>
_TNY_API void peel_axis(const MD & v, const I * idx, I & off, I * ext, I * str, cs::size_t & k) {
    constexpr int p = peel_pos<A, Seq>::value;
    const I sd = static_cast<I>(v.stride(A));
    if constexpr (p >= 0) off += idx[p] * sd;                         // peeled: bind decoded index
    else { ext[k] = static_cast<I>(v.extent(A)); str[k] = sd; ++k; }  // kept axis
}

// Build the peeled sub-view by hand (no submdspan -> works on ANY source layout,
// incl. strides<...>) and fold kept strides to compile-time values where known.
template <own OW, class MD, class I, class Seq, cs::size_t... A>
_TNY_API auto gather_peel(const MD & v, const I * idx, Seq, cs::index_sequence<A...>) {
    using El  = typename MD::element_type;
    using Idx = typename MD::index_type;
    using E   = typename MD::extents_type;
    using L   = typename MD::layout_type;
    using OE  = typename _compact<Idx, peel_ext<A, E, Seq>()...>::type;
    using SF  = typename _str_compact<peel_str<A, L, E, Seq>()...>::type;
    constexpr cs::size_t Nk = sizeof...(A) - Seq::size();
    Idx ext[Nk ? Nk : 1] = {}, str[Nk ? Nk : 1] = {}, off = 0; cs::size_t k = 0;
    ( peel_axis<A, Seq>(v, idx, off, ext, str, k), ... );
    cs::array<Idx, Nk> ea{};
    for (cs::size_t i = 0; i < Nk; ++i) ea[i] = ext[i];
    // fold the kept strides into the strides<...> mapping (shared with the slice
    // gather and axis builders); Nk == OE::rank().
    return tensor<El, OE, SF, OW>(v.data_handle() + off, _detail::fold_mapping<SF>(OE(ea), str));
}

// Space-aware peel-at over an mdspan: `OW` is the view kind to tag the result
// with (own_view_of the source: view for a host source, gpu_view for a device
// one, pinned_view/mapped_view for page-locked host memory).
template <own OW, cs::size_t... Axes, class MD>
_TNY_API auto peel_at_ow(const MD & src, typename MD::index_type i) {
    using I = typename MD::index_type;
    constexpr cs::size_t nd = sizeof...(Axes);
    const I e[nd ? nd : 1]   = { static_cast<I>(src.extent(Axes))... };
    I       idx[nd ? nd : 1] = {};
    I rem = i;
    for (int p = static_cast<int>(nd) - 1; p >= 0; --p) { idx[p] = rem % e[p]; rem /= e[p]; }
    return gather_peel<OW>(src, idx, cs::index_sequence<Axes...>{},
                           cs::make_index_sequence<MD::rank()>{});
}

} // namespace _md

/** @brief The `i`-th sub-view obtained by peeling `Axes...` (0 <= i < product
 *         of the peeled extents). Peeled axes vary in row-major order (the
 *         last listed axis fastest). Returns a `tny::tensor` view. A raw mdspan
 *         carries no memory space, so this tags the result as a host `view`; the
 *         `tny::tensor` overloads below preserve the source's space. */
template <cs::size_t... Axes, class MD>
_TNY_API auto peel_at(const MD & src, typename MD::index_type i) {
    return _md::peel_at_ow<own::view, Axes...>(src, i);
}
// convenience: peel from a tny::tensor (uses its view). Non-const -> mutable
// peel; const -> read-only peel. A device source (gpu/gpu_view) yields gpu_view.
template <long... Axes, class T, class E, class L, own O>
_TNY_API auto peel_at(tensor<T,E,L,O> & t, typename tensor<T,E,L,O>::index_type i) {
    static_assert((_axis_in_range(Axes, tensor<T,E,L,O>::rank()) && ...), "peel_at: axis out of range");
    return _md::peel_at_ow<own_view_of(O), _norm_axis(Axes, tensor<T,E,L,O>::rank())...>(t.mdspan(), i);
}
template <long... Axes, class T, class E, class L, own O>
_TNY_API auto peel_at(const tensor<T,E,L,O> & t, typename tensor<T,E,L,O>::index_type i) {
    static_assert((_axis_in_range(Axes, tensor<T,E,L,O>::rank()) && ...), "peel_at: axis out of range");
    return _md::peel_at_ow<own_view_of(O), _norm_axis(Axes, tensor<T,E,L,O>::rank())...>(t.mdspan(), i);
}

/** @brief A range of sub-views obtained by peeling `Axes...`. Supports
 *         `size()`, `operator[]`, and range-for. */
template <class MD, own OW, cs::size_t... Axes>
struct peel_range {
    using index_type = typename MD::index_type;
    MD src;

    _TNY_API index_type size() const noexcept {
        const index_type e[] = { static_cast<index_type>(src.extent(Axes))..., index_type(1) };
        index_type n = 1;
        for (cs::size_t p = 0; p < sizeof...(Axes); ++p) n *= e[p];
        return n;
    }
    _TNY_API auto operator[](index_type i) const { return _md::peel_at_ow<OW, Axes...>(src, i); }

    struct iterator {
        peel_range r;                 // by value (a single view) -> no dangle if the range is a temporary
        index_type i;
        _TNY_API auto operator*() const { return r[i]; }
        _TNY_API iterator & operator++() { ++i; return *this; }
        _TNY_API bool operator!=(const iterator & o) const { return i != o.i; }
        _TNY_API bool operator==(const iterator & o) const { return i == o.i; }
    };
    _TNY_API iterator begin() const { return { *this, 0 }; }
    _TNY_API iterator end()   const { return { *this, size() }; }
};

/** @brief Build a range of sub-views by peeling `Axes...` of `t`. Non-const `t`
 *         yields mutable peel; const `t` yields read-only peel. */
template <long... Axes, class T, class E, class L, own O>
_TNY_API auto peel(tensor<T,E,L,O> & t) {
    static_assert((_axis_in_range(Axes, tensor<T,E,L,O>::rank()) && ...), "peel: axis out of range");
    return peel_range<decltype(t.mdspan()), own_view_of(O), _norm_axis(Axes, tensor<T,E,L,O>::rank())...>{ t.mdspan() };
}
template <long... Axes, class T, class E, class L, own O>
_TNY_API auto peel(const tensor<T,E,L,O> & t) {
    static_assert((_axis_in_range(Axes, tensor<T,E,L,O>::rank()) && ...), "peel: axis out of range");
    return peel_range<decltype(t.mdspan()), own_view_of(O), _norm_axis(Axes, tensor<T,E,L,O>::rank())...>{ t.mdspan() };
}
/** @brief Build a range of sub-views over a raw mdspan. */
template <cs::size_t... Axes, class MD>
_TNY_API peel_range<MD, own::view, Axes...> peel_of(const MD & m) { return { m }; }

namespace _md {
template <class T, cs::size_t... A> _TNY_API auto sfront(T & t, cs::index_sequence<A...>) { return peel<A...>(t); }
template <class T, cs::size_t... A> _TNY_API auto sfront_at(T & t, typename T::index_type i, cs::index_sequence<A...>) { return peel_at<A...>(t, i); }
} // namespace _md

// # of FRONT axes to peel for a peel_front index N over a rank-R tensor:
//   N >= 0 -> peel the first N axes;  N < 0 -> keep the last |N| (peel R - |N|).
// N must be in [-R, R] (else R - |N| underflows into a giant index_sequence).
template <long N, cs::size_t R> constexpr cs::size_t _front_count() {
    static_assert(N >= -static_cast<long>(R) && N <= static_cast<long>(R), "peel_front: N out of range [-rank, rank]");
    return N >= 0 ? static_cast<cs::size_t>(N) : R - static_cast<cs::size_t>(-N);
}

/** @brief Peel the FIRST `N` axes -> a range of sub-views over the rest — the
 *         runtime-batch-rank half of `(*batch, *spatial, C)`. `N` is **signed**:
 *         `peel_front<3>` peels 3 leading dims; `peel_front<-1>` keeps the last
 *         axis (peels all but it), so negative = "keep the last |N|". */
template <long N, class T, class E, class L, own O>
_TNY_API auto peel_front(tensor<T,E,L,O> & t)       { return _md::sfront(t, cs::make_index_sequence<_front_count<N, tensor<T,E,L,O>::rank()>()>{}); }
template <long N, class T, class E, class L, own O>
_TNY_API auto peel_front(const tensor<T,E,L,O> & t) { return _md::sfront(t, cs::make_index_sequence<_front_count<N, tensor<T,E,L,O>::rank()>()>{}); }

/** @brief The `i`-th sub-view obtained by peeling the first `N` axes (grid-stride style). */
template <long N, class T, class E, class L, own O>
_TNY_API auto peel_front_at(tensor<T,E,L,O> & t, typename tensor<T,E,L,O>::index_type i)
{ return _md::sfront_at(t, i, cs::make_index_sequence<_front_count<N, tensor<T,E,L,O>::rank()>()>{}); }
template <long N, class T, class E, class L, own O>
_TNY_API auto peel_front_at(const tensor<T,E,L,O> & t, typename tensor<T,E,L,O>::index_type i)
{ return _md::sfront_at(t, i, cs::make_index_sequence<_front_count<N, tensor<T,E,L,O>::rank()>()>{}); }

namespace _md {
// product of the peeled FRONT extents (axes 0..sizeof...(A)); empty pack -> 1.
template <class MD, cs::size_t... A>
_TNY_API typename MD::index_type sfront_size(const MD & m, cs::index_sequence<A...>) {
    using I = typename MD::index_type;
    I n = 1;
    ( (n *= static_cast<I>(m.extent(A))), ... );
    return n;
}
} // namespace _md

/** @brief The number of sub-views `peel_front<N>(t)` would yield — the product of
 *         the peeled leading extents — computed directly, without materialising
 *         the range. Same signed `N` as `peel_front`: `size_front<3>(t)`
 *         multiplies the first 3 extents; `size_front<-2>(t)` the all-but-last-two
 *         (the flattened batch count of a `(*batch, C, C)` tensor). */
template <long N, class T, class E, class L, own O>
_TNY_API typename tensor<T,E,L,O>::index_type size_front(const tensor<T,E,L,O> & t) {
    return _md::sfront_size(t.mdspan(), cs::make_index_sequence<_front_count<N, tensor<T,E,L,O>::rank()>()>{});
}

// (removed: `channel(md,c)` was just `peel_at<0>(md,c)`, and `batch_offset` — a
//  raw F-order offset helper — was unused by any kernel. Use peel/peel_at.)

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_ITERATE
