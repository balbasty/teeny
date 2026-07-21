#ifndef TNY_MD_INDEXING
#define TNY_MD_INDEXING
// Free-function vocabulary for indexing & slicing a tensor: argument
// classification, python-style axis/index wrapping, the `slice(...)` spec and
// its traits, and the compile-time output-extents machinery for the range
// slicer. Kept out of tensor.h so the tensor class sits near the top of it.
// The member helpers that USE these (operator(), take_along, _slice_range, ...)
// live in the tensor class; this is the shared toolkit they build on.
#include <cuda/std/mdspan>
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <teeny/defines.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

// an "index" argument to operator() is a (static or runtime) integer; anything
// else (all / slice, i.e. a slice specifier) turns operator() into a view.
template <class T> struct _is_ic : cs::false_type {};
template <class T, T V> struct _is_ic<cs::integral_constant<T,V>> : cs::true_type {};
template <class A> struct _is_index
    : cs::integral_constant<bool, cs::is_integral<A>::value || _is_ic<A>::value> {};

// normalise a python-style AXIS index (negatives count from the back) against a
// rank. Axis template parameters are signed (`long`) so `-1` is the last axis.
_TNY_API constexpr cs::size_t _norm_axis(long a, cs::size_t rank) noexcept {
    return static_cast<cs::size_t>(a < 0 ? a + static_cast<long>(rank) : a);
}

/** @brief Open-ended slice sentinel — teeny's `None` (python `a[:n]` / `a[m:]`).
 *
 * `slice(none, n)` starts at 0, `slice(m, none)` runs to the end, and
 * `slice(none, none)` **folds** to `full_extent` — so `all == slice(none, none)`,
 * keeping the axis and its static extent (`all` is built from it). Combined with
 * runtime bounds it resolves at run time, so the one sentinel covers both. */
struct none_t {};
constexpr none_t none{};

// The one index/bound wrap used everywhere: `none` -> `dflt`; a negative value
// wraps python-style (in a SIGNED domain, so it works for an unsigned index_type).
// Folds when it can: a static (integral_constant) or unsigned arg needs no branch,
// and -DTNY_NO_NEGATIVE_INDEX drops the wrap entirely for runtime signed args
// (kernels that guarantee non-negative indices, for the tightest codegen).
template <class Idx, class V>
_TNY_API constexpr Idx _wrap_idx(V v, Idx n, Idx dflt) noexcept {
    if constexpr (cs::is_same<V, none_t>::value)     { (void)v; (void)n; return dflt; }
    else if constexpr (cs::is_unsigned<V>::value)    { (void)n; return static_cast<Idx>(v); }
#ifdef TNY_NO_NEGATIVE_INDEX
    else                                             { (void)n; return static_cast<Idx>(v); }
#else
    else { using S = cs::make_signed_t<Idx>; const S i = static_cast<S>(v);
           return static_cast<Idx>(i < S(0) ? i + static_cast<S>(n) : i); }
#endif
}

// ---- compile-time output extents for the manual range slicer --------------
// Each axis maps to: DROP (integer arg, no output axis), its static extent (an
// `all`/full_extent kept axis), or dynamic_extent (a range). `_compact` removes
// the DROPs and builds the resulting extents type (static where derivable).
inline constexpr cs::size_t _drop_axis = cs::size_t(-2);
template <cs::size_t V, class E> struct _ext_prepend;
template <cs::size_t V, class Idx, cs::size_t... E> struct _ext_prepend<V, cs::extents<Idx, E...>> { using type = cs::extents<Idx, V, E...>; };
template <class Idx, cs::size_t... V> struct _compact { using type = cs::extents<Idx>; };
template <class Idx, cs::size_t V0, cs::size_t... Vs>
struct _compact<Idx, V0, Vs...> {
    using rest = typename _compact<Idx, Vs...>::type;
    using type = cs::conditional_t<V0 == _drop_axis, rest, typename _ext_prepend<V0, rest>::type>;
};

// a python-like slice spec `[start : stop : step]`; start/stop may be `none`.
template <class A, class B, class S>
struct slice_spec { A start; B stop; S step; };
template <class T> struct _is_slice_spec : cs::false_type {};
template <class A, class B, class S> struct _is_slice_spec<slice_spec<A,B,S>> : cs::true_type {};

// step == 1 (as a static integral constant)?
template <class S> _TNY_API constexpr bool _step1() {
    if constexpr (_is_ic<S>::value) return static_cast<long>(S::value) == 1; else return false;
}
// a "full" slice is `slice(none, none)` with unit step: it is exactly `all`
// (keeps the whole axis, folds, preserves static extents). `all == slice(none,none)`.
template <class Arg> struct _is_full_slice : cs::false_type {};
template <class S> struct _is_full_slice<slice_spec<none_t,none_t,S>> : cs::integral_constant<bool, _step1<S>()> {};

// a "real" range (needs the layout_stride path) is a slice that is not a full slice
template <class Arg> struct _is_real_range
    : cs::integral_constant<bool, _is_slice_spec<Arg>::value && !_is_full_slice<Arg>::value> {};
template <class... Args> struct _any_range : cs::integral_constant<bool, (_is_real_range<Args>::value || ... || false)> {};

/**
 * @brief A python-like slice `[start : stop : step)` for `operator()` /
 *        `take_along`. `none` marks an open end; negative bounds wrap
 *        (count from the back); `step` defaults to 1 and may exceed 1.
 *
 * `slice(1, 4)` = `[1,4)`; `slice(none, 4)` = `[0,4)`; `slice(2, none)` =
 * `[2,end)`; `slice(0, none, 2)` = every other element; `slice(none, none)`
 * keeps the whole axis (== `all`, which is preferable when you want the axis
 * kept — it folds and preserves static extents). A ranged axis is resolved at
 * run time (its extent becomes dynamic); axes kept with `all` stay static.
 */
template <class A, class B>
_TNY_API auto slice(A start, B stop) { return slice_spec<A, B, cs::integral_constant<long,1>>{ start, stop, {} }; }
template <class A, class B, class S>
_TNY_API auto slice(A start, B stop, S step) { return slice_spec<A, B, S>{ start, stop, step }; }

// position of axis A within the pack Axes... (-1 if absent)
template <cs::size_t A, cs::size_t... Axes>
_TNY_API constexpr int _pos_in() {
    cs::size_t axs[] = { Axes..., static_cast<cs::size_t>(-1) };
    for (cs::size_t p = 0; p < sizeof...(Axes); ++p) if (axs[p] == A) return static_cast<int>(p);
    return -1;
}

_TNY_NAMESPACE_END(tny)
#endif // TNY_MD_INDEXING
