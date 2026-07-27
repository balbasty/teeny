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
#include <cuda/std/limits>
#include <cuda/std/cstdint>
#include <teeny/defines.h>
#include <teeny/layout.h>

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
// Whether a signed axis (negatives count from the back) is valid for `rank`, i.e.
// in [-rank, rank). Used by the view ops to reject an out-of-range axis at compile
// time instead of silently wrapping to a huge index via `_norm_axis`.
_TNY_API constexpr bool _axis_in_range(long a, cs::size_t rank) noexcept {
    return a >= -static_cast<long>(rank) && a < static_cast<long>(rank);
}
// Whether the ALREADY-NORMALISED axes `A...` are a permutation of 0..N-1 (each in
// range, no repeats). An out-of-range source axis normalises to a huge index >= N,
// so this also catches that — `permute` needs a genuine permutation or it aliases.
template <cs::size_t... A> _TNY_API constexpr bool _is_perm() noexcept {
    constexpr cs::size_t N = sizeof...(A);
    cs::size_t a[N ? N : 1] = { A... };
    bool seen[N ? N : 1] = {};
    for (cs::size_t i = 0; i < N; ++i) { if (a[i] >= N || seen[a[i]]) return false; seen[a[i]] = true; }
    return true;
}

// No repeats among A... (a SUBSET of axes, unlike `_is_perm` which needs a full
// 0..N-1 permutation). Used by `take_along`, where a repeated axis would bind two
// args to the same axis and silently drop one.
template <cs::size_t... A> _TNY_API constexpr bool _all_distinct() noexcept {
    constexpr cs::size_t N = sizeof...(A);
    cs::size_t a[N ? N : 1] = { A... };
    for (cs::size_t i = 0; i < N; ++i) for (cs::size_t j = i + 1; j < N; ++j) if (a[i] == a[j]) return false;
    return true;
}

// Whether the (already-normalised) axes are STRICTLY ascending — i.e. distinct and
// in order. The axis-LIST ops that fold one axis at a time (`unsqueeze<Ax...>` /
// `squeeze<Ax...>` in tensor.h, `normalize<Axes...>`'s keepdim fold in math.h) need
// that: each step shifts the positions of the axes on one side, so a sorted, repeat-
// free list is what makes the fold well defined. Takes runtime `long`s (not a
// template pack) so a call site can normalise first: `_axes_ascending(_norm_axis(Ax, N)...)`.
_TNY_API constexpr bool _axes_ascending() { return true; }
_TNY_API constexpr bool _axes_ascending(long) { return true; }
template <class... R> _TNY_API constexpr bool _axes_ascending(long a, long b, R... r) { return a < b && _axes_ascending(b, r...); }

/** @brief Open-ended slice sentinel — teeny's `None` (python `a[:n]` / `a[m:]`).
 *
 * `slice(none, n)` starts at 0, `slice(m, none)` runs to the end, and
 * `slice(none, none)` **folds** to `full_extent` — so `all == slice(none, none)`,
 * keeping the axis and its static extent (`all` is built from it). Combined with
 * runtime bounds it resolves at run time, so the one sentinel covers both.
 *
 * A BARE `none` **argument** to `operator()`/`uget` is a different thing: numpy
 * `newaxis` (`a[None]`), which inserts a size-1 axis — see `_is_newaxis` below.
 * `newaxis` is a named alias of `none` for that bare-argument spelling (numpy
 * calls the same value `None` when it's a slice bound and `np.newaxis` when
 * it's inserting an axis; teeny's `none`/`newaxis` mirror that with one type). */
struct none_t {};
constexpr none_t none{};
constexpr none_t newaxis = none;

// A BARE `none` argument to operator()/uget is numpy `newaxis` (`a[None]`): it
// inserts a new size-1 axis (static extent 1, stride 0) at its position while
// consuming NO source axis — the mirror of an integer arg (consumes a source
// axis, emits no output axis). A `none` INSIDE `slice(...)` stays a bound
// sentinel (it becomes a `_slice_spec` member), so only a bare `none_t` argument
// is newaxis; this trait keys on exactly that type.
template <class A> struct _is_newaxis : cs::false_type {};
template <> struct _is_newaxis<none_t> : cs::true_type {};

// The `ellipsis` / `ellipsis_t` marker (numpy `...`) is defined in alias.h — it is
// shared with `anyshape<...>`, where it is spelled `etc` (an alias). In an index
// expression it stands for "as many `all` as it takes to fill the rank": `t(1, ellipsis,
// 2)` on a rank-5 tensor is `t(1, all, all, all, 2)`. At most one per call. It expands
// to `rank - (#other args)` copies of `all` (which may be zero), then the call proceeds
// as usual — so if what remains is all integers you get an element `T&`, otherwise a
// view. `etc` works here too (same marker).
template <class A> struct _is_ellipsis : cs::false_type {};
template <> struct _is_ellipsis<ellipsis_t> : cs::true_type {};   // matches `ellipsis` and `etc` alike
template <class... Args> struct _has_ellipsis
    : cs::integral_constant<bool, (_is_ellipsis<Args>::value || ... || false)> {};

// position of the (first) ellipsis in a pack, and how many there are.
template <class... Args> _TNY_API constexpr cs::size_t _ellipsis_pos() {
    bool is[] = { _is_ellipsis<Args>::value..., false };
    for (cs::size_t i = 0; i < sizeof...(Args); ++i) if (is[i]) return i;
    return sizeof...(Args);
}
template <class... Args> _TNY_API constexpr cs::size_t _ellipsis_count() {
    return (cs::size_t(0) + ... + (_is_ellipsis<Args>::value ? cs::size_t(1) : cs::size_t(0)));
}

// The one index/bound wrap used everywhere: `none` -> `dflt`; a negative value
// wraps python-style (in a SIGNED domain, so it works for an unsigned index_type).
// Folds when it can: a static (integral_constant) or unsigned arg needs no branch,
// and -DTNY_NO_NEGATIVE_INDEX drops the wrap entirely for runtime signed args
// (kernels that guarantee non-negative indices, for the tightest codegen).
//
// `Wrap` is a PER-CALL version of that opt-out (the `uget`/`uat`
// unchecked accessors pass `Wrap=false`): it drops the wrap for RUNTIME SIGNED
// args only. `none`/unsigned are unaffected, and a STATIC (`integral_constant`)
// bound STILL wraps regardless of `Wrap` — the compile-time slice fold
// (`_static_range_len`) always wraps static bounds, so keeping them wrapped here
// is what prevents the folded static extent from diverging from the runtime one.
template <class Idx, bool Wrap = true, class V>
_TNY_API constexpr Idx _wrap_idx(V v, Idx n, Idx dflt) noexcept {
    if constexpr (cs::is_same<V, none_t>::value)     { (void)v; (void)n; return dflt; }
    else if constexpr (cs::is_unsigned<V>::value)    { (void)n; return static_cast<Idx>(v); }
#ifdef TNY_NO_NEGATIVE_INDEX
    else                                             { (void)n; return static_cast<Idx>(v); }
#else
    else if constexpr (!Wrap && !_is_ic<V>::value)   { (void)n; return static_cast<Idx>(v); }
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
struct _slice_spec { A start; B stop; S step; };
template <class T> struct _is_slice_spec : cs::false_type {};
template <class A, class B, class S> struct _is_slice_spec<_slice_spec<A,B,S>> : cs::true_type {};
template <class Arg> struct _slice_step { using type = void; };                 // step type of a slice arg
template <class A, class B, class S> struct _slice_step<_slice_spec<A,B,S>> { using type = S; };

// step == 1 (as a static integral constant)?
template <class S> _TNY_API constexpr bool _step1() {
    if constexpr (_is_ic<S>::value) return static_cast<long>(S::value) == 1; else return false;
}
// a "full" slice is `slice(none, none)` with unit step: it is exactly `all`
// (keeps the whole axis, folds, preserves static extents). `all == slice(none,none)`.
template <class Arg> struct _is_full_slice : cs::false_type {};
template <class S> struct _is_full_slice<_slice_spec<none_t,none_t,S>> : cs::integral_constant<bool, _step1<S>()> {};

// ---- output STATIC STRIDES for the gather (companion to _compact) ----------
// A gathered view's layout is teeny's strides<...>, folding each kept axis to a
// compile-time stride when derivable: (source static stride) × (static step).
// Per source axis we emit `_sdrop` (integer arg -> no output axis), that folded
// value, or `dynamic_stride` (runtime). `_str_compact` drops the `_sdrop`s.
inline constexpr cs::int64_t _sdrop = cs::numeric_limits<cs::int64_t>::min() + 1;  // != dynamic_stride
template <cs::int64_t V, class S> struct _str_prepend;
template <cs::int64_t V, cs::int64_t... S> struct _str_prepend<V, strides<S...>> { using type = strides<V, S...>; };
template <cs::int64_t... V> struct _str_compact { using type = strides<>; };
template <cs::int64_t V0, cs::int64_t... Vs>
struct _str_compact<V0, Vs...> {
    using rest = typename _str_compact<Vs...>::type;
    using type = cs::conditional_t<V0 == _sdrop, rest, typename _str_prepend<V0, rest>::type>;
};

// static output stride for source axis Ax under (Layout, Shape), given the arg.
template <class Arg, cs::size_t Ax, class Layout, class Shape>
_TNY_API constexpr cs::int64_t _out_sstride() {
    if constexpr (_is_newaxis<Arg>::value)        return 0;                      // newaxis: inserted size-1 axis, stride 0 (static)
    else {
        constexpr cs::int64_t ss = _src_sstride<Ax, Layout, Shape>();
        if constexpr (_is_index<Arg>::value)          return _sdrop;                 // dropped
        else if constexpr (_is_full_slice<Arg>::value) return ss;                    // slice(none,none) == all
        else if constexpr (_is_slice_spec<Arg>::value) {                             // range: ss × step
            using S = typename _slice_step<Arg>::type;
            if constexpr (ss == dynamic_stride)       return dynamic_stride;
            else if constexpr (_is_ic<S>::value)      return ss * static_cast<cs::int64_t>(S::value);
            else                                       return dynamic_stride;
        } else return ss;                                                            // full_extent (all): step 1
    }
}

// start / stop bound types of a slice arg (companions to _slice_step).
template <class Arg> struct _slice_start { using type = void; };
template <class A, class B, class S> struct _slice_start<_slice_spec<A,B,S>> { using type = A; };
template <class Arg> struct _slice_stop { using type = void; };
template <class A, class B, class S> struct _slice_stop<_slice_spec<A,B,S>> { using type = B; };

// a slice bound (start/stop) is STATICALLY known iff it's `none` or an integral_constant.
template <class V> struct _static_bound
    : cs::integral_constant<bool, cs::is_same<V, none_t>::value || _is_ic<V>::value> {};

// Clamp a resolved (start, stop) to python bounds for `step` over extent `n` and
// return the slice's element count. `st`/`sp` are clamped IN PLACE — the runtime
// gather needs the clamped start for the base offset, the compile-time fold uses
// only the count. This is the ONE shared body of the runtime `_sl_axis` and the
// compile-time `_static_range_len`, so the folded static extent can never diverge
// from the runtime one (that would be UB). Signed domain (`I` signed) — the
// negative-step branch reaches -1; it is only ever run for a signed index type
// (an unsigned step compares >= 0, taking the forward branch).
template <class I>
_TNY_API constexpr I _range_count(I & st, I & sp, I step, I n) noexcept {
    const I Z = I(0);
    if (step >= Z) {                                        // start, stop in [0, n]
        st = st < Z ? Z : (st > n ? n : st);
        sp = sp < Z ? Z : (sp > n ? n : sp);
        const I w = sp - st;                   return w <= Z ? Z : (w + step - 1) / step;
    } else {                                               // start, stop in [-1, n-1]
        const I ns = -step, hi = n - 1;
        st = st > hi ? hi : (st < I(-1) ? I(-1) : st);
        sp = sp > hi ? hi : (sp < I(-1) ? I(-1) : sp);
        const I w = (n <= Z) ? Z : (st - sp);  return w <= Z ? Z : (w + ns - 1) / ns;
    }
}
// Compile-time length of slice<A,B,S> over a static source extent `n`. Resolves the
// bounds with the SAME `_wrap_idx` the runtime uses (A/B are `none_t` or Int<>
// literal types) and the SAME `_range_count`, so static == runtime by construction
// (a divergence would make the folded extent disagree with the runtime one -> UB).
template <class A, class B, class S>
_TNY_API constexpr cs::size_t _static_range_len(long n) {
    static_assert(S::value != 0, "slice step cannot be 0");
    const long step = static_cast<long>(S::value);
    long st = 0, sp = 0;   // g++ requires constexpr locals initialised even though both branches assign
    if (step >= 0) { st = _wrap_idx<long>(A{}, n, long(0));     sp = _wrap_idx<long>(B{}, n, n); }
    else           { st = _wrap_idx<long>(A{}, n, n - 1);       sp = _wrap_idx<long>(B{}, n, long(-1)); }
    return static_cast<cs::size_t>(_range_count(st, sp, step, n));
}
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
_TNY_API auto slice(A start, B stop) { return _slice_spec<A, B, cs::integral_constant<long,1>>{ start, stop, {} }; }
template <class A, class B, class S>
_TNY_API auto slice(A start, B stop, S step) { return _slice_spec<A, B, S>{ start, stop, step }; }

// Compile-time slice forms (bounds baked into the type so they fold like `all`):
//   slice<1,4>()  slice<0,10,2>()                (value form; longs, step default 1)
//   slice<Int<1>, Int<4>>()                       (type form; integral_constant/none_t
//                                                  bounds — the only way to bake `none`)
template <long Start, long Stop, long Step = 1>
_TNY_API auto slice() { return slice(cs::integral_constant<long, Start>{}, cs::integral_constant<long, Stop>{}, cs::integral_constant<long, Step>{}); }
template <class Start, class Stop, class Step = cs::integral_constant<long, 1>>
_TNY_API auto slice() { return _slice_spec<Start, Stop, Step>{ Start{}, Stop{}, Step{} }; }

// position of axis A within the pack Axes... (-1 if absent)
template <cs::size_t A, cs::size_t... Axes>
_TNY_API constexpr int _pos_in() {
    cs::size_t axs[] = { Axes..., static_cast<cs::size_t>(-1) };
    for (cs::size_t p = 0; p < sizeof...(Axes); ++p) if (axs[p] == A) return static_cast<int>(p);
    return -1;
}

_TNY_NAMESPACE_END(tny)
#endif // TNY_MD_INDEXING
