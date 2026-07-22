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

/** @brief Open-ended slice sentinel — teeny's `None` (python `a[:n]` / `a[m:]`).
 *
 * `slice(none, n)` starts at 0, `slice(m, none)` runs to the end, and
 * `slice(none, none)` **folds** to `full_extent` — so `all == slice(none, none)`,
 * keeping the axis and its static extent (`all` is built from it). Combined with
 * runtime bounds it resolves at run time, so the one sentinel covers both. */
struct none_t {};
constexpr none_t none{};

/** @brief Ellipsis sentinel — teeny's `...` (python `a[..., 0]` / numpy `Ellipsis`).
 *
 * In an index expression it stands for "as many `all` as it takes to fill the
 * rank": `t(1, ellipsis, 2)` on a rank-5 tensor is `t(1, all, all, all, 2)`.
 * At most one ellipsis per call. It expands to `rank - (#other args)` copies of
 * `all` (which may be zero), then the call proceeds as usual — so if what
 * remains is all integers you get an element `T&`, otherwise a view. */
struct ellipsis_t {};
constexpr ellipsis_t ellipsis{};
template <class A> struct _is_ellipsis : cs::false_type {};
template <> struct _is_ellipsis<ellipsis_t> : cs::true_type {};
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

// start / stop bound types of a slice arg (companions to _slice_step).
template <class Arg> struct _slice_start { using type = void; };
template <class A, class B, class S> struct _slice_start<_slice_spec<A,B,S>> { using type = A; };
template <class Arg> struct _slice_stop { using type = void; };
template <class A, class B, class S> struct _slice_stop<_slice_spec<A,B,S>> { using type = B; };

// a slice bound (start/stop) is STATICALLY known iff it's `none` or an integral_constant.
template <class V> struct _static_bound
    : cs::integral_constant<bool, cs::is_same<V, none_t>::value || _is_ic<V>::value> {};

// resolve a static bound to an index at compile time, mirroring _wrap_idx EXACTLY:
// `none` -> the default; else a signed value with negative-wrap — but NOT under
// TNY_NO_NEGATIVE_INDEX, where the runtime leaves it unwrapped (so the fold must
// too, else static != runtime). Static bounds come from Int<> literals (signed).
template <class V> _TNY_API constexpr long _bound_static(long dflt, long n) {
    if constexpr (cs::is_same<V, none_t>::value) { (void)n; return dflt; }
    else {
        const long i = static_cast<long>(V::value);
#ifdef TNY_NO_NEGATIVE_INDEX
        (void)n; return i;                       // no wrap — matches _wrap_idx
#else
        return i < 0 ? i + n : i;
#endif
    }
}
// negative-step stop default is -1 (go past index 0), mirroring _stop_neg.
template <class V> _TNY_API constexpr long _stop_static(long n) {
    if constexpr (cs::is_same<V, none_t>::value) return -1;
    else {
        const long i = static_cast<long>(V::value);
#ifdef TNY_NO_NEGATIVE_INDEX
        (void)n; return i;                       // no wrap — matches _wrap_idx
#else
        return i < 0 ? i + n : i;
#endif
    }
}
// Compile-time length of slice<A,B,S> over a static source extent `n`. This MUST
// reproduce the runtime _sl_axis count EXACTLY (else the folded static extent would
// disagree with the runtime-filled value -> UB), so the clamps below mirror it 1:1.
template <class A, class B, class S>
_TNY_API constexpr cs::size_t _static_range_len(long n) {
    static_assert(S::value != 0, "slice step cannot be 0");
    const long step = static_cast<long>(S::value), Z = 0;
    long st = 0, cnt = 0;
    if (step >= Z) {
        st = _bound_static<A>(Z, n);
        long sp = _bound_static<B>(n, n);
        st = st < Z ? Z : (st > n ? n : st);
        sp = sp < Z ? Z : (sp > n ? n : sp);
        const long w = sp - st; cnt = w <= Z ? Z : (w + step - 1) / step;
    } else {
        const long ns = -step, hi = n - 1;
        st = _bound_static<A>(n - 1, n);
        long sp = _stop_static<B>(n);
        st = st > hi ? hi : (st < Z ? Z : st);
        sp = sp > hi ? hi : (sp < -1 ? -1 : sp);
        const long w = (n <= Z) ? Z : (st - sp); cnt = w <= Z ? Z : (w + ns - 1) / ns;
    }
    return static_cast<cs::size_t>(cnt);
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
