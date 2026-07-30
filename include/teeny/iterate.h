#ifndef TNY_ITERATE_H
#define TNY_ITERATE_H
#include <cuda/std/mdspan>
#include <cuda/std/utility>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <teeny/defines.h>
#include <teeny/tensor.h>
#include <teeny/math.h>   // peel_zip (#327) reuses math.h's broadcast rule (bc1/bc_sext/bc_ext/bc_str)

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/* ================================================================== *
 *  nd-peel: iterate over a SUBSET of axes, yielding a lower-rank view *
 *  over the remaining axes.                                          *
 *                                                                    *
 *  Replaces the hand-written ndindex<->linear-offset plumbing: a     *
 *  linear index over the peeled axes is decoded for you, and the     *
 *  matching sub-view (a `tny::tensor` view into the original data) is *
 *  returned.                                                         *
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
        // The current cell's multi-index over the PEELED axes (in `Axes...` order) —
        // the odometer the incremental cursor already maintains, surfaced for free.
        // `index(d)` is coordinate d; `index()` is the whole tuple. Lets a table-indexed
        // kernel read `axtab[d][it.index(d)]` without a separate manual decode. (#213)
        _TNY_API index_type index(cs::size_t d) const noexcept { return cur.ctr[d]; }
        _TNY_API cs::array<index_type, Nd ? Nd : 1> index() const noexcept {
            cs::array<index_type, Nd ? Nd : 1> m{};
            for (cs::size_t d = 0; d < Nd; ++d) m[d] = cur.ctr[d];
            return m;
        }
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

    // --- enumerate: range-for that ALSO yields the peeled multi-index (#213) -----
    // The default `for (auto cell : peel(...))` keeps the cell LEAN (no coordinate
    // words in the view type). When a kernel needs the coords (e.g. a per-axis table
    // `axtab[d][m[d]]`), `enumerate()` pairs each cell with the odometer's multi-index
    // WITHOUT bloating the cell — you opt in, and only then pay for the tuple:
    //   for (auto [m, cell] : peel(out, axis<...>{}).enumerate()) ...   // m[d] = coord d
    // Composes with `subrange` for chunked/threaded sweeps: `.enumerate().subrange(lo,hi)`.
    struct item { cs::array<index_type, Nd ? Nd : 1> index; Cell cell; };
    struct enum_iterator {
        iterator it;
        _TNY_API item operator*() const { return { it.index(), *it }; }
        _TNY_API enum_iterator & operator++() { ++it; return *this; }
        _TNY_API bool operator!=(const enum_iterator & o) const { return it != o.it; }
        _TNY_API bool operator==(const enum_iterator & o) const { return it == o.it; }
    };
    struct enum_range {
        peel_range r;   // by VALUE (just an mdspan) -> safe on a `peel(t).enumerate()` temporary
        _TNY_API enum_iterator begin() const { return { r.begin() }; }
        _TNY_API enum_iterator end()   const { return { r.end() }; }
        struct enum_subrange {
            enum_iterator b, e;
            _TNY_API enum_iterator begin() const { return b; }
            _TNY_API enum_iterator end()   const { return e; }
        };
        _TNY_API enum_subrange subrange(index_type lo, index_type hi) const {
            subrange_t s = r.subrange(lo, hi);
            return { { s.b }, { s.e } };
        }
    };
    _TNY_API enum_range enumerate() const { return { *this }; }
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

/* ================================================================== *
 *  peel_zip: walk 2 or 3 BROADCAST-compatible tensors in lock-step,   *
 *  yielding a cs::tuple<ViewA,ViewB[,ViewC]> per step (#327) — the    *
 *  "triangle's three vertex tensors" idiom. A distinct name from      *
 *  `peel` (not an overload): 1 tensor -> a view per step, 2+ ->       *
 *  a tuple per step is a silent return-type bifurcation on arity, a   *
 *  real footgun (mirrors python's own `zip()` being its own name      *
 *  rather than an overload of single-iterable iteration).             *
 *                                                                    *
 *  Broadcasting reuses math.h's EXISTING per-operand rule (bc1/       *
 *  bc_sext/bc_ext/bc_str) verbatim — generalized here from 2 operands *
 *  to N via a fold, not a new broadcasting RULE. Peeled/kept axes are *
 *  named in BROADCAST-rank numbering (the largest operand's rank).    *
 * ================================================================== */

namespace _md {

// max over a pack of ranks -> the broadcast RESULT rank (right-align; numpy:
// result rank = the largest operand rank) — generalizes math.h's binary bc_rank.
template <cs::size_t... Rs> constexpr cs::size_t bc_rank_n() {
    cs::size_t m = 0; ( (m = m > Rs ? m : Rs), ... ); return m;
}
// STATIC per-axis broadcast extent: fold math.h's bc1/bc_sext over N operand
// extents types, each individually right-aligned into result rank R.
template <cs::size_t D, cs::size_t R, class... Es> constexpr cs::size_t bcn_sext() {
    cs::size_t r = 1; ( (r = bc1(r, bc_sext<Es, R>(D))), ... ); return r;
}
// per-axis broadcast-compat check across N operands: every operand's own extent
// must be 1, dynamic, or agree with the FOLDED result (equivalent to bc_axis_ok's
// pairwise check for N==2; the correct generalization for N>2).
template <cs::size_t D, cs::size_t R, class... Es> constexpr bool bcn_axis_ok() {
    constexpr cs::size_t r = bcn_sext<D, R, Es...>();
    bool ok = true; ( (ok = ok && bc_axis_ok(bc_sext<Es, R>(D), r)), ... ); return ok;
}
template <cs::size_t R, class... Es, cs::size_t... D>
constexpr bool bcn_static_ok_r(cs::index_sequence<D...>) {
    bool ok = true; ( (ok = ok && bcn_axis_ok<D, R, Es...>()), ... ); return ok;
}
// RUNTIME per-axis broadcast extent, same fold, over N operand mdspans.
template <cs::size_t R, class I, class... MDs>
_TNY_API I bcn_ext_rt(cs::size_t d, const MDs & ... m) {
    I r = 1; ( (r = (static_cast<I>(bc_ext<R>(m, d)) == I(1) ? r : static_cast<I>(bc_ext<R>(m, d)))), ... ); return r;
}
// STATIC per-operand, per-broadcast-axis stride: 0 for a padded (out-of-rank) axis
// or a KNOWN static stretch axis (extent==1); the source's own folded stride
// (`_src_sstride`, layout.h) for a real axis; `dynamic_stride` if only known at
// RUNTIME whether it stretches (a dynamic extent might turn out to be 1).
template <cs::size_t D, cs::size_t R, class L, class E> constexpr cs::int64_t bc_src_sstride() {
    constexpr cs::size_t off = R - E::rank();
    if constexpr (D < off) return cs::int64_t(0);
    else {
        constexpr cs::size_t ax = D - off;
        constexpr cs::size_t se = E::static_extent(ax);
        if constexpr (se == 1) return cs::int64_t(0);
        else if constexpr (se == cs::dynamic_extent) return dynamic_stride;
        else return _src_sstride<ax, L, E>();
    }
}
// shared (broadcast) kept-axis extents type: ONE static extents type reused by
// EVERY operand's view at a zip step (peeled axes drop via `_drop_axis`, same
// sentinel `gather_peel` uses).
//
// `zip_oe_` needs the Es... pack (one extents type per operand) AND its own
// deduced A... pack (one entry per broadcast axis) at once. A function template
// declared (never defined, used only via decltype) with a TYPE pack followed by
// a further DEDUCED pack in the same parameter list hits real-MSVC C2672 ("no
// matching overloaded function found") even though GCC/Clang accept it fine
// (confirmed on Windows CI, #327) -- `es_list<Es...>` bundles the operand-extents
// pack into ONE type first, so `zip_oe_` itself keeps only the single deduced
// pack, mirroring the already-MSVC-proven shape of `reduced_ext_`/`_red_ext_v`
// (math.h/tensor.h) -- a class template's `::value`, not a function call fed a
// pack directly, is the pattern real MSVC tolerates here.
template <class... Es> struct es_list {};
template <cs::size_t A, class Seq, cs::size_t R, class EsList> struct peel_zip_ext_v;
template <cs::size_t A, class Seq, cs::size_t R, class... Es>
struct peel_zip_ext_v<A, Seq, R, es_list<Es...>> {
    static constexpr cs::size_t value = peel_pos<A, Seq>::value >= 0 ? _drop_axis : bcn_sext<A, R, Es...>();
};
template <class Idx, class Seq, cs::size_t R, class EsList, cs::size_t... A>
typename _compact<Idx, peel_zip_ext_v<A, Seq, R, EsList>::value...>::type zip_oe_(cs::index_sequence<A...>);
template <class Idx, class Seq, cs::size_t R, class... Es>
using zip_oe_t = decltype(zip_oe_<Idx, Seq, R, es_list<Es...>>(cs::make_index_sequence<R>{}));

// per (broadcast) axis, per operand: PEELED -> accumulate this operand's offset
// contribution from the shared decoded coordinate; KEPT -> record this operand's
// own runtime stride (into the dynamic slot `_str_compact` leaves open). Uses
// `bc_str` (math.h) directly — 0 for a padded or broadcast-stretched axis, same
// rule every elementwise op already uses per operand.
template <cs::size_t A, cs::size_t R, class Seq, class MD, class I>
_TNY_API void peel_zip_axis(const MD & v, const I * idx, I & off, I * str, cs::size_t & k) {
    constexpr int p = peel_pos<A, Seq>::value;
    const I sd = static_cast<I>(bc_str<R>(v, A));
    if constexpr (p >= 0) off += idx[p] * sd;
    else { str[k] = sd; ++k; }
}
// Build ONE operand's kept-axis view for a zip step: same shape as `gather_peel`,
// but the KEPT-axis extents come from the SHARED `oe` (identical for every
// operand in the zip) while the stride is THIS operand's own.
template <storage OW, cs::size_t R, class Seq, class OE, class MD, cs::size_t... A>
_TNY_API auto gather_peel_zip(const MD & v, const typename OE::index_type * idx, const OE & oe, cs::index_sequence<A...>) {
    using El  = typename MD::element_type;
    using Idx = typename OE::index_type;
    using L   = typename MD::layout_type;
    using E   = typename MD::extents_type;
    using SF  = typename _str_compact<(peel_pos<A,Seq>::value >= 0 ? _sdrop : bc_src_sstride<A,R,L,E>())...>::type;
    constexpr cs::size_t Nk = OE::rank();
    Idx str[Nk ? Nk : 1] = {}; Idx off = 0; cs::size_t k = 0;
    ( peel_zip_axis<A, R, Seq>(v, idx, off, str, k), ... );
    return tensor<El, OE, SF, OW>(v.data_handle() + off, _detail::fold_mapping<SF>(oe, str));
}

// pairs a storage-view kind with its mdspan type -- lets `peel_zip_range` hold
// ONE variadic `Ops...` pack instead of two parallel (and easy to desync) ones.
template <storage OW, class MD> struct zop { static constexpr storage ow = OW; using md_type = MD; };

// runtime broadcast extent for broadcast axis `d` (peeled OR kept -- this doesn't
// care which), folded over every operand in the tuple (same rule as `bcn_ext_rt`,
// unpacked from a `cs::tuple` via `K...`).
template <cs::size_t R, class Idx, class Tup, cs::size_t... K>
_TNY_API Idx zip_bc_ext(cs::size_t d, const Tup & srcs, cs::index_sequence<K...>) {
    return bcn_ext_rt<R, Idx>(d, cs::get<K>(srcs)...);
}
// per-axis (across every operand): each operand's own extent must be 1, dynamic,
// or agree with the folded broadcast extent `fd` for that axis (the RUNTIME twin
// of `bcn_static_ok_r` -- a static extent already proved compatible at compile
// time, so this only has teeth where an operand's extent is dynamic).
template <cs::size_t R, class Idx, class Tup, cs::size_t... K>
_TNY_API void zip_check_axis(cs::size_t d, const Tup & srcs, Idx fd, cs::index_sequence<K...>) {
    ( _TNY_CHECK(static_cast<Idx>(bc_ext<R>(cs::get<K>(srcs), d)) == fd ||
                 static_cast<Idx>(bc_ext<R>(cs::get<K>(srcs), d)) == Idx(1),
        "peel_zip: broadcast: operand extent mismatch"), ... );
}
template <cs::size_t R, class Idx, class Tup, cs::size_t... K, cs::size_t... D>
_TNY_API void zip_check_bcast(const Tup & srcs, cs::index_sequence<K...> ks, cs::index_sequence<D...>) {
    ( zip_check_axis<R, Idx>(D, srcs, zip_bc_ext<R, Idx>(D, srcs, ks), ks), ... );
}

/** @brief A range of `cs::tuple<ViewA,ViewB[,ViewC]>` obtained by zip-peeling 2 or
 *         3 broadcast-compatible tensors' `Axes...` in lock-step. `Ops...` are
 *         `zop<OW,MD>` pairs (one per operand); `Seq` is the peeled (broadcast-
 *         numbered) axis set; `R` is the broadcast rank. Supports `size()`,
 *         random-access `operator[]`, range-for, `subrange(lo,hi)`, and
 *         `.enumerate()` — the same shape as the single-tensor `peel_range`,
 *         decoding the shared linear index fresh each step (no incremental
 *         cursor yet — correctness first; a follow-up perf issue can add one, #327). */
template <class Seq, cs::size_t R, class... Ops>
struct peel_zip_range {
    using Tup = cs::tuple<typename Ops::md_type...>;
    static constexpr cs::size_t Nop = sizeof...(Ops);
    static constexpr cs::size_t Nd  = Seq::size();
    // The type this range decodes offsets in — AND the index type every cell it
    // hands out carries. It must represent every extent and stride value ANY
    // operand can produce, so it is `math.h`'s signedness-aware `_offset_int_t`
    // (the same rule `bzip_`/`zipreduce_decode_`/`scalo_`/`unaryo_`/`allclose_`
    // decode in — one rule for every multi-tensor engine, #362), NOT
    // `cs::common_type_t`: common_type applies the usual arithmetic conversions,
    // so at EQUAL width the UNSIGNED type wins (`common_type_t<int32_t,uint32_t>`
    // is `uint32_t`), and a flipped operand's stride of -1 then zero-extends to
    // 4294967295 — a cell base pointer ~4G elements past the buffer (SEGV).
    // Both halves of this line matter, and `_offset_int_t` fixes both at once:
    //   - the DECODE (`peel_zip_axis`'s `off += idx[p] * sd`, and `gather_peel_zip`'s
    //     `data_handle() + off`), exactly as in the four engines above; and
    //   - the CELL's own type. A `peel_zip` cell is a VIEW of its operand, not a
    //     fresh allocation, so unlike a broadcast RESULT (`_wider_index_t`'s
    //     documented rationale) its kept-axis strides can legitimately be NEGATIVE
    //     — an unsigned index type cannot represent them even when the base
    //     pointer happens to come out right.
    // `_offset_int_t` is already variadic (its rule is stated over a participant
    // SET), so the 2- and 3-operand `peel_zip` forms both use it unchanged, and
    // for an all-signed or all-unsigned set it is the plain widest — the same
    // type `common_type_t` picks, so every non-mixed zip is byte-identical.
    using Idx = _offset_int_t<typename Ops::md_type::index_type...>;
    using OE  = zip_oe_t<Idx, Seq, R, typename Ops::md_type::extents_type...>;
    using Ks  = cs::index_sequence_for<Ops...>;

    Tup srcs;

    // one row of the mixed-radix decode: per (broadcast) PEELED axis, the shared
    // extent every operand agrees on (folded across all of them).
    template <cs::size_t... PA>
    _TNY_API void _peeled_ext(Idx * e, cs::index_sequence<PA...>) const {
        cs::size_t p = 0; ( (e[p++] = zip_bc_ext<R, Idx>(PA, srcs, Ks{})), ... );
    }
    _TNY_API Idx size() const {
        Idx e[Nd ? Nd : 1]; _peeled_ext(e, Seq{});
        Idx n = 1; for (cs::size_t p = 0; p < Nd; ++p) n *= e[p];
        return n;
    }
    // debug-only broadcast-compat check across every operand (host-debug guard, a
    // no-op under -DNDEBUG, same convention as every other runtime shape check).
    _TNY_API void _check() const { zip_check_bcast<R, Idx>(srcs, Ks{}, cs::make_index_sequence<R>{}); }

    // the shared per-KEPT-axis broadcast extents object (identical for every
    // operand's cell) -- built once per step from all operands together. Needs
    // exactly `OE::rank()` values (one per KEPT axis; peeled axes are already
    // dropped from OE's own type), so this walks ALL R axes but only records the
    // kept ones (mirrors `gather_peel`'s own k-counting). UN-floored, matching
    // `gather_peel` (iterate.h): when every axis is peeled (a rank-0 cell), `OE`
    // is `cs::extents<Idx>` (rank 0), whose constructor only accepts an array of
    // size EXACTLY 0 -- a floored `cs::array<Idx,1>` has no viable constructor
    // there. The `ea[k]` write below is inside `if constexpr (... < 0)`, which is
    // discarded for every axis in the all-peeled case, so a size-0 array is never
    // indexed (found + fixed via review: peel_zip<Axes...> spanning every axis
    // failed to compile before this fix, #327).
    template <cs::size_t A>
    _TNY_API void _oe_axis(cs::array<Idx, OE::rank()> & ea, cs::size_t & k) const {
        if constexpr (peel_pos<A, Seq>::value < 0) { ea[k] = zip_bc_ext<R, Idx>(A, srcs, Ks{}); ++k; }
    }
    template <cs::size_t... A>
    _TNY_API OE _oe(cs::index_sequence<A...>) const {
        cs::array<Idx, OE::rank()> ea{}; cs::size_t k = 0;
        ( _oe_axis<A>(ea, k), ... );
        return OE(ea);
    }

    // The `i`-th step: decode `i` over the shared peeled-axis space, then build
    // every operand's cell against the SAME decoded coordinates + shared `oe`.
    _TNY_API auto operator[](Idx i) const { return _at(i, cs::make_index_sequence<R>{}, Ks{}); }
    template <cs::size_t... A, cs::size_t... K>
    _TNY_API auto _at(Idx i, cs::index_sequence<A...> aseq, cs::index_sequence<K...>) const {
        Idx e[Nd ? Nd : 1]; _peeled_ext(e, Seq{});
        Idx idx[Nd ? Nd : 1] = {}; Idx rem = i;
        for (int p = static_cast<int>(Nd) - 1; p >= 0; --p)
        { const Idx ee = e[p]; idx[p] = ee ? rem % ee : Idx(0); rem = ee ? rem / ee : rem; }
        OE oe = _oe(aseq);
        return cs::make_tuple(gather_peel_zip<Ops::ow, R, Seq, OE>(cs::get<K>(srcs), idx, oe, aseq)...);
    }

    // Iterators hold a COPY of the range (just a tuple of mdspans -- cheap, no
    // owning storage) rather than a pointer back to it: `peel_zip<Axes...>(a,b)`
    // is typically a TEMPORARY (`for (auto x : peel_zip<0>(a,b).enumerate())`),
    // and only the range-for's own range-expression gets lifetime-extended, not
    // sub-expressions used to build it -- a pointer-to-range iterator would dangle
    // the moment `.enumerate()`/`.subrange()` is chained straight onto a temporary
    // (ASan-confirmed stack-use-after-scope during development). Mirrors the
    // single-tensor `peel_range::enum_range`'s own "by VALUE -> safe on a
    // temporary" comment, generalized here to the whole range, not just one mdspan.
    struct iterator {
        peel_zip_range rg; Idx lin;
        _TNY_API auto operator*() const { return rg[lin]; }
        _TNY_API iterator & operator++() { ++lin; return *this; }
        _TNY_API bool operator!=(const iterator & o) const { return lin != o.lin; }
        _TNY_API bool operator==(const iterator & o) const { return lin == o.lin; }
    };
    _TNY_API iterator begin() const { return { *this, Idx(0) }; }
    _TNY_API iterator end()   const { return { *this, size() }; }

    /** @brief A `[lo, hi)` slice for chunked/threaded sweeps (same shape as the
     *         single-tensor peel's `subrange`). */
    struct subrange_t {
        iterator b, e;
        _TNY_API iterator begin() const { return b; }
        _TNY_API iterator end()   const { return e; }
    };
    _TNY_API subrange_t subrange(Idx lo, Idx hi) const { return { { *this, lo }, { *this, hi } }; }

    // --- enumerate: range-for that ALSO yields the peeled multi-index --------
    struct item { cs::array<Idx, Nd ? Nd : 1> index; decltype(cs::declval<const peel_zip_range &>()[Idx(0)]) cell; };
    _TNY_API cs::array<Idx, Nd ? Nd : 1> _index_of(Idx lin) const {
        Idx e[Nd ? Nd : 1]; _peeled_ext(e, Seq{});
        cs::array<Idx, Nd ? Nd : 1> m{}; Idx rem = lin;
        for (int p = static_cast<int>(Nd) - 1; p >= 0; --p)
        { const Idx ee = e[p]; m[p] = ee ? rem % ee : Idx(0); rem = ee ? rem / ee : rem; }
        return m;
    }
    struct enum_iterator {
        peel_zip_range rg; Idx lin;
        _TNY_API item operator*() const { return { rg._index_of(lin), rg[lin] }; }
        _TNY_API enum_iterator & operator++() { ++lin; return *this; }
        _TNY_API bool operator!=(const enum_iterator & o) const { return lin != o.lin; }
        _TNY_API bool operator==(const enum_iterator & o) const { return lin == o.lin; }
    };
    struct enum_range {
        peel_zip_range rg;
        _TNY_API enum_iterator begin() const { return { rg, Idx(0) }; }
        _TNY_API enum_iterator end()   const { return { rg, rg.size() }; }
        struct enum_subrange {
            enum_iterator b, e;
            _TNY_API enum_iterator begin() const { return b; }
            _TNY_API enum_iterator end()   const { return e; }
        };
        _TNY_API enum_subrange subrange(Idx lo, Idx hi) const { return { { rg, lo }, { rg, hi } }; }
    };
    _TNY_API enum_range enumerate() const { return { *this }; }
};

} // namespace _md

/** @brief Zip-peel 2 tensors' `Axes...` in lock-step -> a range of
 *         `cs::tuple<ViewA,ViewB>` (numpy-style broadcast: shapes may differ as
 *         long as they're broadcast-compatible; `Axes...` name axes in the
 *         BROADCAST rank's numbering — the larger of the two operands' own
 *         ranks — negatives wrap against it). A distinct name from `peel` (see
 *         the design note above `peel_zip_range`), not an overload.
 *
 *         The operands need not share an INDEX TYPE: the cells carry one wide
 *         enough — and, where the operands disagree on signedness, signed enough
 *         — to address every operand exactly, so a reversed view (`flip`, or a
 *         negative slice step) zipped against an unsigned-indexed tensor steps
 *         backwards instead of wrapping to a huge positive offset (#362). */
template <long... Axes, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto peel_zip(tensor<Ta,Ea,La,Oa> & a, tensor<Tb,Eb,Lb,Ob> & b) {
    constexpr cs::size_t R = _md::bc_rank_n<Ea::rank(), Eb::rank()>();
    static_assert((_axis_in_range(Axes, R) && ...), "peel_zip: axis out of range");
    static_assert(_all_distinct<_norm_axis(Axes, R)...>(), "peel_zip: axes must be distinct");
    static_assert(_md::bcn_static_ok_r<R, Ea, Eb>(cs::make_index_sequence<R>{}), "peel_zip: incompatible static extents");
    using Seq = cs::index_sequence<_norm_axis(Axes, R)...>;
    using Range = _md::peel_zip_range<Seq, R,
        _md::zop<storage_view_of(Oa), decltype(a.mdspan())>, _md::zop<storage_view_of(Ob), decltype(b.mdspan())>>;
    Range rg{ typename Range::Tup(a.mdspan(), b.mdspan()) };
    rg._check();
    return rg;
}
template <long... Axes, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto peel_zip(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b) {
    constexpr cs::size_t R = _md::bc_rank_n<Ea::rank(), Eb::rank()>();
    static_assert((_axis_in_range(Axes, R) && ...), "peel_zip: axis out of range");
    static_assert(_all_distinct<_norm_axis(Axes, R)...>(), "peel_zip: axes must be distinct");
    static_assert(_md::bcn_static_ok_r<R, Ea, Eb>(cs::make_index_sequence<R>{}), "peel_zip: incompatible static extents");
    using Seq = cs::index_sequence<_norm_axis(Axes, R)...>;
    using Range = _md::peel_zip_range<Seq, R,
        _md::zop<storage_view_of(Oa), decltype(a.mdspan())>, _md::zop<storage_view_of(Ob), decltype(b.mdspan())>>;
    Range rg{ typename Range::Tup(a.mdspan(), b.mdspan()) };
    rg._check();
    return rg;
}
// value form: peel_zip(a, b, axis<0,1>{}) == peel_zip<0,1>(a, b) -- trailing
// axis<...> selector (keywords AFTER positionals, per the design discussion on
// #327), unlike take_along/peel_at's LEADING axis<...> (which disambiguates a
// second variadic arg pack there; peel_zip's tensor arguments are each a single,
// fixed-arity positional, so a trailing tag is unambiguous and deducible).
template <long... Axes, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto peel_zip(tensor<Ta,Ea,La,Oa> & a, tensor<Tb,Eb,Lb,Ob> & b, axis<Axes...>) { return peel_zip<Axes...>(a, b); }
template <long... Axes, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto peel_zip(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b, axis<Axes...>) { return peel_zip<Axes...>(a, b); }

/** @brief Zip-peel 3 tensors' `Axes...` in lock-step -> a range of
 *         `cs::tuple<ViewA,ViewB,ViewC>` (same broadcast/axis-numbering rule as
 *         the 2-tensor form). The "triangle's three vertex tensors" idiom. */
template <long... Axes, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob, class Tc,class Ec,class Lc,storage Oc>
_TNY_API auto peel_zip(tensor<Ta,Ea,La,Oa> & a, tensor<Tb,Eb,Lb,Ob> & b, tensor<Tc,Ec,Lc,Oc> & c) {
    constexpr cs::size_t R = _md::bc_rank_n<Ea::rank(), Eb::rank(), Ec::rank()>();
    static_assert((_axis_in_range(Axes, R) && ...), "peel_zip: axis out of range");
    static_assert(_all_distinct<_norm_axis(Axes, R)...>(), "peel_zip: axes must be distinct");
    static_assert(_md::bcn_static_ok_r<R, Ea, Eb, Ec>(cs::make_index_sequence<R>{}), "peel_zip: incompatible static extents");
    using Seq = cs::index_sequence<_norm_axis(Axes, R)...>;
    using Range = _md::peel_zip_range<Seq, R,
        _md::zop<storage_view_of(Oa), decltype(a.mdspan())>, _md::zop<storage_view_of(Ob), decltype(b.mdspan())>,
        _md::zop<storage_view_of(Oc), decltype(c.mdspan())>>;
    Range rg{ typename Range::Tup(a.mdspan(), b.mdspan(), c.mdspan()) };
    rg._check();
    return rg;
}
template <long... Axes, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob, class Tc,class Ec,class Lc,storage Oc>
_TNY_API auto peel_zip(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b, const tensor<Tc,Ec,Lc,Oc> & c) {
    constexpr cs::size_t R = _md::bc_rank_n<Ea::rank(), Eb::rank(), Ec::rank()>();
    static_assert((_axis_in_range(Axes, R) && ...), "peel_zip: axis out of range");
    static_assert(_all_distinct<_norm_axis(Axes, R)...>(), "peel_zip: axes must be distinct");
    static_assert(_md::bcn_static_ok_r<R, Ea, Eb, Ec>(cs::make_index_sequence<R>{}), "peel_zip: incompatible static extents");
    using Seq = cs::index_sequence<_norm_axis(Axes, R)...>;
    using Range = _md::peel_zip_range<Seq, R,
        _md::zop<storage_view_of(Oa), decltype(a.mdspan())>, _md::zop<storage_view_of(Ob), decltype(b.mdspan())>,
        _md::zop<storage_view_of(Oc), decltype(c.mdspan())>>;
    Range rg{ typename Range::Tup(a.mdspan(), b.mdspan(), c.mdspan()) };
    rg._check();
    return rg;
}
template <long... Axes, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob, class Tc,class Ec,class Lc,storage Oc>
_TNY_API auto peel_zip(tensor<Ta,Ea,La,Oa> & a, tensor<Tb,Eb,Lb,Ob> & b, tensor<Tc,Ec,Lc,Oc> & c, axis<Axes...>)
{ return peel_zip<Axes...>(a, b, c); }
template <long... Axes, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob, class Tc,class Ec,class Lc,storage Oc>
_TNY_API auto peel_zip(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b, const tensor<Tc,Ec,Lc,Oc> & c, axis<Axes...>)
{ return peel_zip<Axes...>(a, b, c); }

/* ================================================================== *
 *  scan_: in-place sequential fold along ONE axis, batched (peeled)   *
 *  over every other axis (#254). The recurrence itself is inherently  *
 *  sequential; the batching over every other axis reuses `peel`'s own *
 *  incremental cursor, so it stays O(1)/step there.                   *
 * ================================================================== */

namespace _md {

// axes 0..Rank-1 EXCLUDING Axis, ascending -- the peel list that batches
// every axis except the one `scan_` walks sequentially.
template <cs::size_t Rank, cs::size_t Axis>
_TNY_API constexpr cs::array<long, Rank - 1> scan_complement() {
    cs::array<long, Rank - 1> out{}; cs::size_t k = 0;
    for (cs::size_t i = 0; i < Rank; ++i) if (i != Axis) out[k++] = static_cast<long>(i);
    return out;
}
template <cs::size_t Rank, cs::size_t Axis> struct scan_complement_t
{ static constexpr cs::array<long, Rank - 1> value = scan_complement<Rank, Axis>(); };

// Walk every peeled line (one per batch cell), threading `carry` through
// `f` in increasing order along the kept axis: `carry = f(carry, line(i))`,
// `line(i) = carry` (the new carry doubles as the new element -- a running
// fold in place, e.g. `carry = min(carry + w, line(i))` is exactly a 1-D
// Felzenszwalb L1 sweep, see examples/distance_transform.cpp's hand-written
// twin). A reverse sweep is just `scan_<Axis>(t.flip<Axis>(), init, f)`.
template <cs::size_t A, class Tn, class Carry, class F, cs::size_t... I>
_TNY_API void scan_lines(Tn & t, Carry init, F f, cs::index_sequence<I...>) {
    using Idx = typename Tn::index_type;
    for (auto line : peel<scan_complement_t<Tn::rank(), A>::value[I]...>(t)) {
        Carry carry = init;
        const Idx n = static_cast<Idx>(line.shape(0));
        for (Idx i = 0; i < n; ++i) {
            carry = f(carry, line(i));
            line(i) = static_cast<typename Tn::element_type>(carry);
        }
    }
}

} // namespace _md

/** @brief In-place sequential fold ("scan") along axis `Axis`, batched over
 *         every other axis: `carry = init`, then for each element along
 *         `Axis` (in increasing order) `carry = f(carry, x)`, `x = carry` --
 *         the new carry doubles as the new element. `f` is a device-safe
 *         functor (lambda-free engines, like `map_`/`zip_with_`): `Carry
 *         operator()(Carry carry, T x) const`. A reverse sweep composes with
 *         the existing negative-stride view, no separate "direction" flag:
 *         `scan_<Axis>(t.flip<Axis>(), init, f)` (an rvalue view binds fine --
 *         `scan_` has both lvalue and rvalue overloads, unlike `peel` this
 *         doesn't need a named temporary first).
 *         `scan_<Axis>(t, init, f)` == `scan_(t, axis<Axis>{}, init, f)`. */
template <long Axis, class T, class E, class L, storage O, class Carry, class F>
_TNY_API void scan_(tensor<T,E,L,O> & t, Carry init, F f) {
    static_assert(tensor<T,E,L,O>::rank() >= 1, "scan_: needs rank >= 1");
    static_assert(_axis_in_range(Axis, tensor<T,E,L,O>::rank()), "scan_: axis out of range");
    constexpr cs::size_t A = _norm_axis(Axis, tensor<T,E,L,O>::rank());
    _md::scan_lines<A>(t, init, f, cs::make_index_sequence<tensor<T,E,L,O>::rank() - 1>{});
}
// rvalue overload: a temporary VIEW (e.g. `t.flip<Axis>()`) mutates the same
// underlying storage as any named view would -- only the view OBJECT is a
// temporary, the data it points at is not. Forwards to the lvalue overload.
template <long Axis, class T, class E, class L, storage O, class Carry, class F>
_TNY_API void scan_(tensor<T,E,L,O> && t, Carry init, F f) { scan_<Axis>(t, init, f); }
template <long Axis, class T, class E, class L, storage O, class Carry, class F>
_TNY_API void scan_(tensor<T,E,L,O> & t, axis<Axis>, Carry init, F f) { scan_<Axis>(t, init, f); }
template <long Axis, class T, class E, class L, storage O, class Carry, class F>
_TNY_API void scan_(tensor<T,E,L,O> && t, axis<Axis>, Carry init, F f) { scan_<Axis>(t, init, f); }

/** @brief Out-of-place twin of `scan_`: a fresh dense copy of `t`, scanned.
 *         Static shape -> stack (host+device); dynamic -> heap (host only,
 *         like `clone()`, which this is built on). */
template <long Axis, class T, class E, class L, storage O, class Carry, class F,
          cs::enable_if_t<tensor<T,E,L,O>::is_static, int> = 0>
_TNY_API auto scan(const tensor<T,E,L,O> & t, Carry init, F f) {
    auto out = t.clone();
    scan_<Axis>(out, init, f);
    return out;
}
template <long Axis, class T, class E, class L, storage O, class Carry, class F,
          cs::enable_if_t<!tensor<T,E,L,O>::is_static, int> = 0>
_TNY_HOST auto scan(const tensor<T,E,L,O> & t, Carry init, F f) {
    auto out = t.clone();
    scan_<Axis>(out, init, f);
    return out;
}
/** @brief Value form: `scan(t, axis<Axis>{}, init, f)` == `scan<Axis>(t, init, f)`.
 *         SPLIT IN TWO on the same `is_static` key as the `<Axis>` pair it forwards
 *         to (#375): a static shape yields a stack result (host+device) so the
 *         forwarder is `_TNY_API`; a dynamic shape yields a heap result (host only,
 *         via `clone()`) so it is `_TNY_HOST` — else nvcc's device pass would see a
 *         `_TNY_API` forwarder call a `__host__` allocator. */
template <long Axis, class T, class E, class L, storage O, class Carry, class F,
          cs::enable_if_t<tensor<T,E,L,O>::is_static, int> = 0>
_TNY_API auto scan(const tensor<T,E,L,O> & t, axis<Axis>, Carry init, F f) { return scan<Axis>(t, init, f); }
template <long Axis, class T, class E, class L, storage O, class Carry, class F,
          cs::enable_if_t<!tensor<T,E,L,O>::is_static, int> = 0>
_TNY_HOST auto scan(const tensor<T,E,L,O> & t, axis<Axis>, Carry init, F f) { return scan<Axis>(t, init, f); }

/** @brief `into(dest)` form: write the scanned result into a preallocated
 *         `dest` (a shape matching `t`'s EXACTLY, checked -- a `static_assert`
 *         when both are static, `_TNY_CHECK` otherwise; unlike `copy_`'s own
 *         numpy-style broadcast, `dest` must match rather than merely receive
 *         a broadcast copy, since `scan_` then walks `dest`'s own axis
 *         numbering) -- one copy, no fresh allocation beyond that; device-safe.
 *         `copy_` casts INTO `dest`'s element type FIRST, so if `dest`'s dtype
 *         differs from `t`'s the whole recurrence then runs in `dest`'s own
 *         precision (unlike `index_select`/the reductions' own `into(dest)`,
 *         which only cast the FINAL result). Returns `dest&`. */
template <long Axis, class T, class E, class L, storage O, class Carry, class F, class D>
_TNY_API auto & scan(const tensor<T,E,L,O> & t, Carry init, F f, into_t<D> out) {
    using DstE = typename D::extents_type;
    static_assert(tensor<T,E,L,O>::rank() == D::rank(), "scan into(dest): rank mismatch");
    static_assert(_md::ext_static_eq<E, DstE>(cs::make_index_sequence<E::rank()>{}),
                  "scan into(dest): dest's shape must match the source exactly");
    for (cs::size_t d = 0; d < tensor<T,E,L,O>::rank(); ++d)
        _TNY_CHECK(static_cast<long>(out.dest.extent(d)) == static_cast<long>(t.extent(d)),
                   "scan into(dest): dest's shape must match the source exactly");
    out.dest.copy_(t);
    scan_<Axis>(out.dest, init, f);
    return out.dest;
}
template <long Axis, class T, class E, class L, storage O, class Carry, class F, class D>
_TNY_API auto & scan(const tensor<T,E,L,O> & t, axis<Axis>, Carry init, F f, into_t<D> out)
{ return scan<Axis>(t, init, f, out); }

_TNY_NAMESPACE_END(tny)

#endif // TNY_ITERATE_H
