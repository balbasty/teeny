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
template <storage OW, class MD, class I, class Seq, cs::size_t... A>
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
// with (storage_view_of the source: view for a host source, gpu_view for a device
// one, pinned_view/mapped_view for page-locked host memory).
template <storage OW, cs::size_t... Axes, class MD>
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

// Incremental mixed-radix cursor over a set of peeled axes (#110). Keeps a running
// base OFFSET so a SEQUENTIAL sweep advances by one stride-add (plus an occasional
// carry) instead of re-decoding the flat index from zero each step. `seed(i0)` starts
// at any linear index with a single decode — so a thread/block can begin mid-range
// (chunked parallelism); a grid-stride loop (`i += nthreads`) keeps the random-access
// `peel_at`, whose stride the odometer can't express. Handles ANY peeled strides
// (contiguous, permuted, negative). Nd == number of peeled axes.
template <class I, cs::size_t Nd>
struct peel_cursor {
    I ext[Nd ? Nd : 1];   // peeled extents
    I str[Nd ? Nd : 1];   // peeled SOURCE strides
    I ctr[Nd ? Nd : 1];   // odometer multi-index
    I off;                // running base offset == sum_d ctr[d]*str[d]
    I lin;                // flat index (end() compares this)
    _TNY_API void seed(I i0) {
        lin = i0; off = 0; I rem = i0;
        for (int d = static_cast<int>(Nd) - 1; d >= 0; --d) {
            const I e = ext[d]; const I k = e ? rem % e : I(0); rem = e ? rem / e : rem;
            ctr[d] = k; off += k * str[d];
        }
    }
    _TNY_API void advance() {
        ++lin;
        for (int d = static_cast<int>(Nd) - 1; d >= 0; --d) {
            if (ctr[d] + 1 < ext[d]) { ++ctr[d]; off += str[d]; return; }   // no carry
            off -= ctr[d] * str[d]; ctr[d] = 0;                             // wrap axis d, carry up
        }
    }
};

} // namespace _md

/** @brief The `i`-th sub-view obtained by peeling `Axes...` (0 <= i < product
 *         of the peeled extents). Peeled axes vary in row-major order (the
 *         last listed axis fastest). Returns a `tny::tensor` view. A raw mdspan
 *         carries no memory space, so this tags the result as a host `view`; the
 *         `tny::tensor` overloads below preserve the source's space. */
template <cs::size_t... Axes, class MD>
_TNY_API auto peel_at(const MD & src, typename MD::index_type i) {
    return _md::peel_at_ow<storage::view, Axes...>(src, i);
}
// convenience: peel from a tny::tensor (uses its view). Non-const -> mutable
// peel; const -> read-only peel. A device source (gpu/gpu_view) yields gpu_view.
template <long... Axes, class T, class E, class L, storage O>
_TNY_API auto peel_at(tensor<T,E,L,O> & t, typename tensor<T,E,L,O>::index_type i) {
    static_assert((_axis_in_range(Axes, tensor<T,E,L,O>::rank()) && ...), "peel_at: axis out of range");
    return _md::peel_at_ow<storage_view_of(O), _norm_axis(Axes, tensor<T,E,L,O>::rank())...>(t.mdspan(), i);
}
template <long... Axes, class T, class E, class L, storage O>
_TNY_API auto peel_at(const tensor<T,E,L,O> & t, typename tensor<T,E,L,O>::index_type i) {
    static_assert((_axis_in_range(Axes, tensor<T,E,L,O>::rank()) && ...), "peel_at: axis out of range");
    return _md::peel_at_ow<storage_view_of(O), _norm_axis(Axes, tensor<T,E,L,O>::rank())...>(t.mdspan(), i);
}
// value form: peel_at(t, i, axis<0,1>{}) == peel_at<0,1>(t, i). The axis selector
// is a single value tag, so a dependent receiver needs no `.template` (and it reads
// better than a trailing `Int<>` list).
template <long... Axes, class T, class E, class L, storage O>
_TNY_API auto peel_at(tensor<T,E,L,O> & t, typename tensor<T,E,L,O>::index_type i, axis<Axes...>)
{ return peel_at<Axes...>(t, i); }
template <long... Axes, class T, class E, class L, storage O>
_TNY_API auto peel_at(const tensor<T,E,L,O> & t, typename tensor<T,E,L,O>::index_type i, axis<Axes...>)
{ return peel_at<Axes...>(t, i); }

/** @brief A range of sub-views obtained by peeling `Axes...`. Supports `size()`,
 *         random-access `operator[]` (grid-stride loops), range-for (an INCREMENTAL
 *         cursor — #110 — that advances the pointer instead of re-decoding each
 *         step, and builds the loop-invariant sub-view mapping once), and
 *         `subrange(lo,hi)` for chunked/threaded sweeps. */
template <class MD, storage OW, cs::size_t... Axes>
struct peel_range {
    using index_type = typename MD::index_type;
    static constexpr cs::size_t Nd = sizeof...(Axes);
    MD src;

    _TNY_API index_type size() const noexcept {
        const index_type e[] = { static_cast<index_type>(src.extent(Axes))..., index_type(1) };
        index_type n = 1;
        for (cs::size_t p = 0; p < Nd; ++p) n *= e[p];
        return n;
    }
    // Random access — the i-th cell decoded from scratch. This is what a device
    // grid-stride loop (`i += nthreads`) needs; the range-for below is incremental.
    _TNY_API auto operator[](index_type i) const { return _md::peel_at_ow<OW, Axes...>(src, i); }

    // Every cell shares the SAME extents/strides (only the base offset moves), so the
    // iterator carries one pre-built "template" cell (its mapping is loop-invariant)
    // and rebases its pointer via the incremental cursor.
    using Cell = decltype(_md::peel_at_ow<OW, Axes...>(cs::declval<const MD &>(), index_type(0)));
    using El   = typename Cell::element_type;   // carries the source's const-ness
    struct iterator {
        Cell tmpl;                                  // cell at offset 0 -> supplies the invariant mapping
        El * base;                                  // tmpl.data() captured mutably (== src.data()+0)
        _md::peel_cursor<index_type, Nd> cur;
        _TNY_API Cell operator*() const { return Cell(base + cur.off, tmpl.mapping()); }
        _TNY_API iterator & operator++() { cur.advance(); return *this; }
        _TNY_API bool operator!=(const iterator & o) const { return cur.lin != o.cur.lin; }
        _TNY_API bool operator==(const iterator & o) const { return cur.lin == o.cur.lin; }
    };
    _TNY_API iterator _iter_at(index_type i) const {
        iterator it{ _md::peel_at_ow<OW, Axes...>(src, index_type(0)), nullptr, {} };   // template cell at offset 0
        it.base = it.tmpl.data();                   // non-const here -> El* (mutable when the source is)
        const index_type e[] = { static_cast<index_type>(src.extent(Axes))..., index_type(1) };
        const index_type s[] = { static_cast<index_type>(src.stride(Axes))..., index_type(0) };
        for (cs::size_t d = 0; d < Nd; ++d) { it.cur.ext[d] = e[d]; it.cur.str[d] = s[d]; }
        it.cur.seed(i);
        return it;
    }
    _TNY_API iterator begin() const { return _iter_at(0); }
    _TNY_API iterator end()   const { iterator it = _iter_at(0); it.cur.lin = size(); return it; }

    /** @brief A `[lo, hi)` slice of the cells for chunked/threaded sweeps: seed the
     *         incremental cursor once at `lo`, then O(1) per step within the chunk.
     *         (Split `[0,size())` across threads/blocks; each sweeps its chunk.) */
    struct subrange_t {
        iterator b, e;
        _TNY_API iterator begin() const { return b; }
        _TNY_API iterator end()   const { return e; }
    };
    _TNY_API subrange_t subrange(index_type lo, index_type hi) const {
        iterator b = _iter_at(lo);
        iterator e = b; e.cur.lin = hi;   // end sentinel: only `lin` is compared
        return { b, e };
    }
};

/** @brief Build a range of sub-views by peeling `Axes...` of `t`. Non-const `t`
 *         yields mutable peel; const `t` yields read-only peel. */
template <long... Axes, class T, class E, class L, storage O>
_TNY_API auto peel(tensor<T,E,L,O> & t) {
    static_assert((_axis_in_range(Axes, tensor<T,E,L,O>::rank()) && ...), "peel: axis out of range");
    return peel_range<decltype(t.mdspan()), storage_view_of(O), _norm_axis(Axes, tensor<T,E,L,O>::rank())...>{ t.mdspan() };
}
template <long... Axes, class T, class E, class L, storage O>
_TNY_API auto peel(const tensor<T,E,L,O> & t) {
    static_assert((_axis_in_range(Axes, tensor<T,E,L,O>::rank()) && ...), "peel: axis out of range");
    return peel_range<decltype(t.mdspan()), storage_view_of(O), _norm_axis(Axes, tensor<T,E,L,O>::rank())...>{ t.mdspan() };
}
// value form: peel(t, axis<0,1>{}) == peel<0,1>(t) (numpy-like axis selector; no
// `.template` on a type-dependent receiver, and reads better than a trailing list).
template <long... Axes, class T, class E, class L, storage O>
_TNY_API auto peel(tensor<T,E,L,O> & t, axis<Axes...>)       { return peel<Axes...>(t); }
template <long... Axes, class T, class E, class L, storage O>
_TNY_API auto peel(const tensor<T,E,L,O> & t, axis<Axes...>) { return peel<Axes...>(t); }

/** @brief Build a range of sub-views over a raw mdspan. */
template <cs::size_t... Axes, class MD>
_TNY_API peel_range<MD, storage::view, Axes...> peel_of(const MD & m) { return { m }; }

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
template <long N, class T, class E, class L, storage O>
_TNY_API auto peel_front(tensor<T,E,L,O> & t)       { return _md::sfront(t, cs::make_index_sequence<_front_count<N, tensor<T,E,L,O>::rank()>()>{}); }
template <long N, class T, class E, class L, storage O>
_TNY_API auto peel_front(const tensor<T,E,L,O> & t) { return _md::sfront(t, cs::make_index_sequence<_front_count<N, tensor<T,E,L,O>::rank()>()>{}); }

/** @brief The `i`-th sub-view obtained by peeling the first `N` axes (grid-stride style). */
template <long N, class T, class E, class L, storage O>
_TNY_API auto peel_front_at(tensor<T,E,L,O> & t, typename tensor<T,E,L,O>::index_type i)
{ return _md::sfront_at(t, i, cs::make_index_sequence<_front_count<N, tensor<T,E,L,O>::rank()>()>{}); }
template <long N, class T, class E, class L, storage O>
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
template <long N, class T, class E, class L, storage O>
_TNY_API typename tensor<T,E,L,O>::index_type size_front(const tensor<T,E,L,O> & t) {
    return _md::sfront_size(t.mdspan(), cs::make_index_sequence<_front_count<N, tensor<T,E,L,O>::rank()>()>{});
}

// (removed: `channel(md,c)` was just `peel_at<0>(md,c)`, and `batch_offset` — a
//  raw F-order offset helper — was unused by any kernel. Use peel/peel_at.)

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_ITERATE
