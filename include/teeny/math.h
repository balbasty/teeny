#ifndef TNY_MD_MATH
#define TNY_MD_MATH
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <cuda/std/limits>
#include <cuda/std/cmath>
#include <teeny/defines.h>
#include <teeny/alias.h>
#include <teeny/half.h>
#include <teeny/tensor.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/* ================================================================== *
 *  Type promotion for binary math.                                   *
 *                                                                    *
 *  Result type of `a + b` (and `.add`, `dot`, ...) follows the usual  *
 *  C++ arithmetic conversions, EXCEPT among floating types where the  *
 *  LOWER precision wins (float16 > float32 > float64), pytorch-style  *
 *  — keeping the compact type through a chain of ops. Opt out (revert *
 *  to plain C++ `common_type`) with `-DTNY_STD_PROMOTION`.            *
 * ================================================================== */
// floating-point "rank": -1 = not floating; else lower value = lower precision.
template <class T> struct _frank              { static constexpr int value = -1; };
template <> struct _frank<half>               { static constexpr int value = 0; };
template <> struct _frank<bfloat16>           { static constexpr int value = 0; };
template <> struct _frank<float>              { static constexpr int value = 1; };
template <> struct _frank<double>             { static constexpr int value = 2; };
template <> struct _frank<long double>        { static constexpr int value = 3; };

// Both non-floating -> `common_type` (numpy-like: int8+int8 -> int8, NOT the
// C++ integer-promoted int). common_type is only ever instantiated here, never
// for a float that would choke it (e.g. cs::common_type<half,float> is
// ill-formed — half has both an implicit `operator float` and an implicit ctor
// from float), which is also why the float ranking below is explicit.
template <class A, class B, bool Low, bool BothInt = (_frank<A>::value < 0 && _frank<B>::value < 0)>
struct _promote { using type = cs::common_type_t<A, B>; };
// At least one floating. `Low` = prefer the LOWER-precision float (teeny default,
// half>float>double); otherwise the standard wider-float-wins. A non-float
// operand yields the float; a same-rank tie between distinct 16-bit types -> float.
template <class A, class B, bool Low>
struct _promote<A, B, Low, false> {
    static constexpr int fa = _frank<A>::value, fb = _frank<B>::value;
    using type = cs::conditional_t<(fa < 0), B,
                 cs::conditional_t<(fb < 0), A,
                 cs::conditional_t<(fa == fb), cs::conditional_t<cs::is_same<A,B>::value, A, float>,
                 cs::conditional_t<(Low ? (fa < fb) : (fa > fb)), A, B>>>>;
};
#ifdef TNY_STD_PROMOTION
template <class A, class B> using promote_t = typename _promote<A, B, false>::type;  // wider float wins
#else
template <class A, class B> using promote_t = typename _promote<A, B, true>::type;   // lower float wins
#endif

/* ================================================================== *
 *  valarray-like math.                                               *
 *                                                                    *
 *  - In-place ops (`a.add_(b)`, `a.mul_(scalar)`) work on ANY tensor  *
 *    or view and mutate it in place. A tensor rhs broadcasts into     *
 *    `a`'s shape numpy-style (a dim of 1 on the rhs is stretched).    *
 *  - Out-of-place ops (`a + b`, `a.add(b)`) broadcast the two shapes  *
 *    and return a fresh tensor: a stack-owned one (host AND device)   *
 *    when the result extent is fully static, else a heap-owned one    *
 *    (HOST ONLY, since it must be allocated at run time).             *
 *                                                                    *
 *  All engines are lambda-free (index-sequence folds + tiny functors) *
 *  so they instantiate on device without `--extended-lambda`, and a   *
 *  fully-static shape folds the whole thing to straight-line code.    *
 * ================================================================== */

namespace _md {

/* ---- binary element ops (out = x OP y) --------------------------- */
struct add { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return x + static_cast<X>(y); } };
struct sub { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return x - static_cast<X>(y); } };
struct mul { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return x * static_cast<X>(y); } };
struct div { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return x / static_cast<X>(y); } };
struct zip_sqdiff { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { X d = x - static_cast<X>(y); return d * d; } };  // (x-y)² (combine for sqdist/dist's fused reduce)

/* ---- fused scaled accumulate (axpy: out = x (+/-) coeff*y) -------- *
 * The coefficient rides in the functor; `bzip` runs the op in the      *
 * destination's compute type, so `coeff` is cast to that type per      *
 * element (a `half` coeff computes in float, like every other op).     */
template <class C> struct fma_add { C coeff; template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return x + static_cast<X>(coeff) * static_cast<X>(y); } };
template <class C> struct fma_sub { C coeff; template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return x - static_cast<X>(coeff) * static_cast<X>(y); } };

/* ---- reversed scalar ops (out = scalar OP x, for scalar-on-the-left) */
struct rsub { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return static_cast<X>(y) - x; } };  // s - x
struct rdiv { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return static_cast<X>(y) / x; } };  // s / x

/* ---- assignment functors ----------------------------------------- */
struct rhs  { template <class X, class Y> _TNY_API X operator()(X, Y y) const { return static_cast<X>(y); } };  // c = b
struct nrhs { template <class X, class Y> _TNY_API X operator()(X, Y y) const { return -static_cast<X>(y); } }; // c += (-b) atomic sub
struct setc { template <class X> _TNY_API X operator()(X, X s) const { return s; } };                          // c = s

// Ops whose result never depends on their FIRST ("current value") argument at all
// (rhs/nrhs/setc — pure assignment, ignoring whatever was already at the destination).
// An engine's decode loop uses this to skip reading the destination before computing
// op(...): harmless for w_set (the read is merely redundant there), but for the w_add
// atomic writer it's more than redundant — reading c's own memory ordinarily while
// another thread may be atomically RMW-ing that same location via fetch_add's
// atomic_ref is a genuine data race, even though the loaded value is thrown away.
// Keyed on the OP, not the writer: any future op that genuinely needs the current
// value keeps the read no matter which writer commits it (default false = safe).
template <class Op> struct _ignores_lhs : cs::false_type {};
template <> struct _ignores_lhs<rhs>  : cs::true_type {};
template <> struct _ignores_lhs<nrhs> : cs::true_type {};
template <> struct _ignores_lhs<setc> : cs::true_type {};

/* ---- write policies: how an engine commits op(...) to c ---------- *
 * `w_set` overwrites; `w_add` accumulates ATOMICALLY on host AND      *
 * device (#257; the scatter/push write). In-place add_/sub_ pick the  *
 * policy via their `Atomic` flag; every other engine defaults to      *
 * w_set.                                                              */
struct w_set { template <class P, class V> _TNY_API void operator()(P * p, V v) const { *p = static_cast<P>(v); } };
struct w_add { template <class P, class V> _TNY_API void operator()(P * p, V v) const { fetch_add(p, static_cast<P>(v)); } };

/* ---- unary functors (cuda::std math -> device-callable) ---------- */
struct u_neg  { template <class X> _TNY_API X operator()(X x) const { return -x; } };
struct u_abs  { template <class X> _TNY_API X operator()(X x) const { return x < X(0) ? -x : x; } };
struct u_exp  { template <class X> _TNY_API X operator()(X x) const { return cs::exp(x); } };
struct u_log  { template <class X> _TNY_API X operator()(X x) const { return cs::log(x); } };
struct u_sin  { template <class X> _TNY_API X operator()(X x) const { return cs::sin(x); } };
struct u_cos  { template <class X> _TNY_API X operator()(X x) const { return cs::cos(x); } };
struct u_sqrt { template <class X> _TNY_API X operator()(X x) const { return cs::sqrt(x); } };
struct u_tanh { template <class X> _TNY_API X operator()(X x) const { return cs::tanh(x); } };
struct u_floor{ template <class X> _TNY_API X operator()(X x) const { return cs::floor(x); } };
struct u_ceil { template <class X> _TNY_API X operator()(X x) const { return cs::ceil(x); } };
struct u_round{ template <class X> _TNY_API X operator()(X x) const { return cs::round(x); } };
struct u_trunc{ template <class X> _TNY_API X operator()(X x) const { return cs::trunc(x); } };
struct u_sign { template <class X> _TNY_API X operator()(X x) const { return x < X(0) ? X(-1) : (x > X(0) ? X(1) : X(0)); } };
struct pw     { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return cs::pow(x, static_cast<X>(y)); } };

/* ---- binary min/max (broadcast) and clamp (functor with bounds) --- */
struct b_min  { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { X yy = static_cast<X>(y); return yy < x ? yy : x; } };
struct b_max  { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { X yy = static_cast<X>(y); return yy > x ? yy : x; } };
struct u_clamp{ double lo, hi; template <class X> _TNY_API X operator()(X x) const { X l = static_cast<X>(lo), h = static_cast<X>(hi); return x < l ? l : (x > h ? h : x); } };

/* ---- bitwise (integer element types only) ------------------------ */
struct b_and { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return x & static_cast<X>(y); } };
struct b_or  { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return x | static_cast<X>(y); } };
struct b_xor { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return x ^ static_cast<X>(y); } };
struct u_bnot{ template <class X> _TNY_API X operator()(X x) const { return ~x; } };

/* ---- comparisons (-> bool) and boolean reductions ---------------- */
struct c_eq { template <class X> _TNY_API bool operator()(X x, X y) const { return x == y; } };
struct c_ne { template <class X> _TNY_API bool operator()(X x, X y) const { return x != y; } };
struct c_lt { template <class X> _TNY_API bool operator()(X x, X y) const { return x <  y; } };
struct c_le { template <class X> _TNY_API bool operator()(X x, X y) const { return x <= y; } };
struct c_gt { template <class X> _TNY_API bool operator()(X x, X y) const { return x >  y; } };
struct c_ge { template <class X> _TNY_API bool operator()(X x, X y) const { return x >= y; } };
// compare through the compute type so half/bfloat16 don't need native host operators.
struct r_all { template <class X> _TNY_API bool operator()(bool a, X x) const { using C = compute_type_t<X>; return a && (static_cast<C>(x) != C(0)); } };
struct r_any { template <class X> _TNY_API bool operator()(bool a, X x) const { using C = compute_type_t<X>; return a || (static_cast<C>(x) != C(0)); } };

/* ---- reduce ops (acc = op(acc, x)) ------------------------------- */
struct r_add { template <class A, class X> _TNY_API A operator()(A a, X x) const { return a + static_cast<A>(x); } };
struct r_addsq { template <class A, class X> _TNY_API A operator()(A a, X x) const { A xx = static_cast<A>(x); return a + xx * xx; } };  // Σx² (for sqnorm/norm)
struct r_mul { template <class A, class X> _TNY_API A operator()(A a, X x) const { return a * static_cast<A>(x); } };
struct r_max { template <class A, class X> _TNY_API A operator()(A a, X x) const { A y = static_cast<A>(x); return y > a ? y : a; } };
struct r_min { template <class A, class X> _TNY_API A operator()(A a, X x) const { A y = static_cast<A>(x); return y < a ? y : a; } };

/* ---- numpy-style broadcasting (same rank; a dim of 1 broadcasts) - *
 * c(i) = op(a(i), b(i)), where a and b broadcast into c's shape       *
 * (stride 0 on any axis whose operand extent is 1).                   */

// one axis is broadcast-compatible if the extents are equal, one is 1, or
// either is dynamic (only known at run time -> checked by _TNY_CHECK below).
_TNY_API constexpr bool bc_axis_ok(cs::size_t a, cs::size_t b) {
    return a == cs::dynamic_extent || b == cs::dynamic_extent || a == b || a == 1 || b == 1;
}

// exact-match static check for contractions (dot) — NO broadcast: each axis must
// be EQUAL, unless either extent is dynamic (then a runtime _TNY_CHECK guards it).
// Distinct from bc_axis_ok, which also accepts extent 1 (stretch) — wrong for dot,
// where a smaller operand would be indexed past its end (OOB under NDEBUG).
_TNY_API constexpr bool ext_axis_eq(cs::size_t a, cs::size_t b) {
    return a == cs::dynamic_extent || b == cs::dynamic_extent || a == b;
}
template <class Ea, class Eb, cs::size_t... D>
_TNY_API constexpr bool ext_static_eq(cs::index_sequence<D...>) {
    bool ok = true;
    ( (ok = ok && ext_axis_eq(Ea::static_extent(D), Eb::static_extent(D))), ... );
    return ok;
}

/* ---- broadcast geometry (numpy-style, LEFT-PADDED) --------------- *
 * Operands are aligned from the RIGHT; a shorter operand's missing leading axes
 * are treated as extent 1 (stride 0). The result rank is the larger of the two.
 * Defined here (before the engine) so bzip_ can name them. */
_TNY_API constexpr cs::size_t bc1(cs::size_t A, cs::size_t B) {   // one result axis extent
    return (A == cs::dynamic_extent || B == cs::dynamic_extent) ? cs::dynamic_extent
         : (A == 1 ? B : A);
}
constexpr cs::size_t bc_rank(cs::size_t ra, cs::size_t rb) { return ra > rb ? ra : rb; }
// STATIC extent of operand extents `E` for RESULT axis `d` in result rank `R`,
// right-aligned: axes before the operand's start (`d < R - rank`) are extent 1.
template <class E, cs::size_t R> constexpr cs::size_t bc_sext(cs::size_t d) {
    constexpr cs::size_t off = R - E::rank();       // E::rank() <= R (checked at the call sites)
    return d < off ? cs::size_t(1) : E::static_extent(d - off);
}
// static broadcast-compat check, right-aligned into result rank `R`.
template <class Ea, class Eb, cs::size_t R, cs::size_t... D>
_TNY_API constexpr bool bc_static_ok_r(cs::index_sequence<D...>) {
    bool ok = true;
    ( (ok = ok && bc_axis_ok(bc_sext<Ea, R>(D), bc_sext<Eb, R>(D))), ... );
    return ok;
}
template <class Idx, class Ea, class Eb, cs::size_t R, cs::size_t... D>
cs::extents<Idx, bc1(bc_sext<Ea, R>(D), bc_sext<Eb, R>(D))...>
bcast_ext_(cs::index_sequence<D...>);
// The WIDER of two offset index TYPES (by `sizeof`; a tie keeps the first, so a
// same-width pair — the overwhelmingly common case — is unchanged). This is a pure
// WIDTH pick, used for the index type a broadcast RESULT carries: the result is a
// fresh, C-contiguous allocation, so its own extents/strides are non-negative and
// the wider of the two operands' widths holds all of them.
// (It does not — and cannot from the types alone — widen two equal-narrow operands
// whose combined SPAN overflows: that stays the caller's responsibility, guarded by
// index_fits/dispatch_index at the boundary.)
// NB it is NOT the type an engine may decode OFFSETS in — see `_offset_int_t` below,
// which is signedness-aware; a pure width pick can select an unsigned type over a
// signed participant's negative stride and turn it into a huge positive offset.
template <class Ia, class Ib>
using _wider_int_t = cs::conditional_t<(sizeof(Ib) > sizeof(Ia)), Ib, Ia>;
// The broadcast RESULT carries the wider of the two OPERANDS' index types.
template <class Ea, class Eb>
using _wider_index_t = _wider_int_t<typename Ea::index_type, typename Eb::index_type>;

/* ---- the type an ENGINE decodes its offsets in ---------------------------- *
 * It must represent EVERY extent and stride value that ANY participant (the
 * destination and both operands) can produce — otherwise a `static_cast` into it
 * is legal but VALUE-CHANGING, which is exactly how the truncation bugs in this
 * family (#342, #346) mis-addressed.
 *
 * `sizeof` alone is not enough. The widest participant can be UNSIGNED while a
 * NARROWER one is SIGNED and carries a NEGATIVE stride (a flipped/reversed view):
 * `static_cast<uint32_t>(-1)` is 4294967295, which zero-extends into the 64-bit
 * pointer offset instead of stepping backwards, and the access lands far off the
 * front of the buffer. So the pick is signedness-aware:
 *   - all participants SIGNED   -> the widest of them (the previous rule, verbatim:
 *     same type, same code, so every all-signed call site is untouched);
 *   - all participants UNSIGNED -> the widest of them (likewise untouched). A
 *     negative stride cannot arise here: teeny's negative-stride views REQUIRE a
 *     signed index type (`flip` static_asserts it, a negative slice step only folds
 *     for a signed index), so an unsigned participant only yields non-negative
 *     extents and strides, all of which its own width holds;
 *   - MIXED -> a SIGNED type wide enough for BOTH sides: at least as wide as the
 *     widest participant, AND wide enough to hold the widest UNSIGNED participant's
 *     full range (twice its size), capped at 64 bits.
 * The 64-bit cap is the one place this leans on teeny's contract instead of proving
 * containment outright: no signed type holds all of `uint64_t`. It costs nothing
 * real — for `data()[off]` to be defined at all, `off` must fit `ptrdiff_t`, so on
 * every platform teeny targets a reachable offset is within `int64_t`. teeny's own
 * reach contract is signed throughout for the same reason (see `index_fits`).
 *
 * Only the MIXED case changes behaviour, and only where the previous type would
 * have been wrong: a mixed pair with no negative stride and values inside both
 * ranges decodes to the same offsets in the wider signed type as it did before. */
template <cs::size_t N> struct _sint_of_size;
template <> struct _sint_of_size<1> { using type = cs::int8_t;  };
template <> struct _sint_of_size<2> { using type = cs::int16_t; };
template <> struct _sint_of_size<4> { using type = cs::int32_t; };
template <> struct _sint_of_size<8> { using type = cs::int64_t; };
// widest of a pack by `sizeof`, left-folded so a tie keeps the FIRST (same tie rule,
// and for the destination-first order used below the same answer, as `_wider_int_t`).
template <class... I> struct _widest_int;
template <class I0> struct _widest_int<I0> { using type = I0; };
template <class I0, class I1, class... In> struct _widest_int<I0, I1, In...>
    { using type = typename _widest_int<_wider_int_t<I0, I1>, In...>::type; };
template <class T> struct _ident { using type = T; };
// `sizeof` of the widest UNSIGNED member of the pack (0 if there is none) — the range
// a signed pick has to grow past.
template <class... I> constexpr cs::size_t _max_unsigned_size() {
    cs::size_t m = 0;
    ( (m = (!cs::is_signed<I>::value && sizeof(I) > m) ? sizeof(I) : m), ... );
    return m;
}
template <class... I> struct _offset_int {
    using _wide = typename _widest_int<I...>::type;
    static constexpr bool _mixed = ( cs::is_signed<I>::value || ...) &&
                                   (!cs::is_signed<I>::value || ...);
    static constexpr cs::size_t _u     = _max_unsigned_size<I...>();
    static constexpr cs::size_t _need  = _u * 2 > 8 ? 8 : _u * 2;           // signed: one size up
    static constexpr cs::size_t _bytes = sizeof(_wide) > _need ? sizeof(_wide) : _need;
    // `conditional_t` on the CARRIERS, so `_sint_of_size<_bytes>` is only completed
    // when it is the one selected.
    using type = typename cs::conditional_t<_mixed, _sint_of_size<_bytes>, _ident<_wide>>::type;
};
template <class... I> using _offset_int_t = typename _offset_int<I...>::type;

/* ---- the type an ENGINE runs its OP in (`Cv`) ------------------------------ *
 * Distinct from `_offset_int_t` above (that one is about ADDRESSING): `Cv` is the
 * arithmetic type each element is widened to before the op and cast back from
 * after it. Every engine takes it as a template parameter, and there are exactly
 * two suppliers:
 *   - the IN-PLACE callers (`a.add_(b)`, `a.mul_(2.0)`, `a.exp_()`, …) leave it
 *     unspecified (`void`), which resolves to the DESTINATION's compute type —
 *     and there the destination IS the lhs operand, so that is a source type;
 *   - every OUT-OF-PLACE producer names it from its OPERANDS: the comparisons
 *     pass `Rc` (`compute_type_t<promote_t<A,B>>`, so `a < b` compares the values
 *     rather than their bool cast) and the `into(dest)` entry points pass exactly
 *     the type their allocating twin's destination would have carried (#379).
 * `into(dest)` is the ONE path where the caller picks the destination's element
 * type, so it is the one place where "the destination's compute type" is NOT a
 * source type: taking it from there ran the whole computation — the operands, a
 * scalar rhs, an axpy coefficient — in the destination's type, so
 * `a.mul(0.5, into(int_y))` multiplied by `int(0.5)` == 0. The docs promise the
 * opposite (source precision throughout, the RESULT cast to `dest`), which is what
 * naming `Cv` from the operands restores: `x.op(y, into(dest))` is numerically
 * indistinguishable from `dest.copy_(x.op(y))`, just without the temporary. */
template <class Cv, class C> struct _cv_or_dest { using type = Cv; };
template <class C> struct _cv_or_dest<void, C> { using type = compute_type_t<typename C::element_type>; };
template <class Cv, class C> using _cv_or_dest_t = typename _cv_or_dest<Cv, C>::type;

template <class Ea, class Eb>
using bcast_extents = decltype(bcast_ext_<_wider_index_t<Ea, Eb>, Ea, Eb,
    bc_rank(Ea::rank(), Eb::rank())>(cs::make_index_sequence<bc_rank(Ea::rank(), Eb::rank())>{}));

// RUNTIME extent/stride of operand `x` for RESULT axis `d` in result rank `R`,
// right-aligned (missing leading axes -> extent 1, stride 0).
template <cs::size_t R, class X> _TNY_API typename X::index_type bc_ext(const X & x, cs::size_t d) {
    using I = typename X::index_type;
    // rank-0 operand: a 0-d scalar broadcasts as all-1 extents. Guarded with
    // if constexpr so x.extent(runtime) (CCCL-constrained to rank>0) is never
    // instantiated for rank 0 (same reason as is_contiguous, #55).
    if constexpr (X::rank() == 0) { (void)x; (void)d; return I(1); }
    else { constexpr cs::size_t off = R - X::rank(); return d < off ? I(1) : static_cast<I>(x.extent(d - off)); }
}
template <cs::size_t R, class X> _TNY_API typename X::index_type bc_str(const X & x, cs::size_t d) {
    using I = typename X::index_type;
    if constexpr (X::rank() == 0) { (void)x; (void)d; return I(0); }   // 0-d: stride 0 everywhere
    else {
        constexpr cs::size_t off = R - X::rank();
        if (d < off) return I(0);                    // missing (padded) axis: stride 0
        const cs::size_t xa = d - off;
        return x.extent(xa) == 1 ? I(0) : static_cast<I>(x.stride(xa));   // stretched axis: stride 0
    }
}
// runtime broadcast extents object (for the heap result), over the RESULT axes.
template <class RE, class A, class B, cs::size_t... D>
_TNY_HOST RE bcast_runtime_(const A & a, const B & b, cs::index_sequence<D...>) {
    using I = typename RE::index_type; constexpr cs::size_t R = RE::rank();
    return RE(static_cast<I>(bc_ext<R>(a, D) == 1 ? bc_ext<R>(b, D) : bc_ext<R>(a, D))...);
}

// `Cv` is the type the op runs in (`_cv_or_dest_t` above states the rule and who
// supplies it): the destination's compute type for an IN-PLACE op (where the
// destination is the lhs operand — the `bzip` wrapper's default), the operands'
// promoted compute type for an out-of-place one (`oop`/`oop_to`), or their compare
// type for a comparison whose result is a bool mask (`bcmp`) — one engine, three uses.
// `Restrict` (set only by the OUT-OF-PLACE callers, where `c` is a fresh
// allocation that provably cannot alias `a`/`b`) enables an auto-vectorizable
// linear fast path: when the plain-store writer `w_set` is in use and every
// operand has the SAME rank + extents as `c` and is C-contiguous, offsets are
// just `i`, so the write goes through a `__restrict__` destination and the
// per-element mixed-radix decode is skipped. Any mismatch falls back unchanged.
template <class W, bool Restrict, class Cv, class C, class A, class B, class Op, cs::size_t... D>
_TNY_API void bzip_(C & c, const A & a, const B & b, Op op, cs::index_sequence<D...>) {
    // Offsets are decoded in a type that holds EVERY value all THREE participants —
    // the destination and both operands — can produce (#346; `_offset_int_t` above
    // spells the rule out). Taking the DESTINATION's index type alone truncated a
    // wider-indexed operand's extents/strides: silently, since the `static_cast<I>`s
    // below suppress the narrowing diagnostic that caught the sibling
    // `zipreduce_decode_` bug (#342). An OUT-OF-PLACE `c` already carries
    // `_wider_index_t` of the two operands (#167), so it is the widest and the WIDTH
    // half of this is a no-op there; it is the IN-PLACE ops (`c` IS `a`: `a.add_(b)`,
    // `a.copy_(b)`, …) and a caller-supplied `into(dest)` that can hand us a NARROW
    // destination next to a wide operand, and there `a.stride()` of 40000 folded to an
    // int16 -25536 and read off the front of the buffer. The SIGNEDNESS half applies
    // to both: a mixed-signedness trio (an unsigned-indexed operand next to a flipped,
    // signed-indexed one) decodes in a signed type, so a stride of -1 stays -1 instead
    // of becoming 4294967295.
    //
    // This is not a promise about `c`'s own type: `c`'s own offsets are computed from
    // `c`'s own extents/strides, so they fit `c`'s index type by construction (a wider
    // decode type only carries them in a larger register, same values), and nothing
    // here is stored back into a tensor's type — `data()[off]` takes any integer.
    // Symmetrically, each operand's offsets stay inside its own tensor: the loop
    // counter `k` is bounded by `ce[d]`, and the extent checks just below pin
    // `ae[d]`/`be[d]` to `ce[d]` (or 1 -> stride 0), so no operand is ever indexed
    // past its own extent in any axis.
    using I = _offset_int_t<typename C::index_type,
                            typename A::extents_type::index_type,
                            typename B::extents_type::index_type>;
    constexpr cs::size_t R = C::rank();   // c has the result (largest) rank; a,b right-align into it
    // Array size floored to 1 (rank-0 c -> empty D..., see scal_'s comment above).
    const I ce[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(c.extent(D))... },
            sc[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(c.stride(D))... };
    const I ae[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(bc_ext<R>(a, D))... },
            sa[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(bc_str<R>(a, D))... };
    const I be[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(bc_ext<R>(b, D))... },
            sb[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(bc_str<R>(b, D))... };
    // runtime shape check: each operand extent must equal c's or be 1 (a larger
    // rhs would silently truncate — the worst failure mode in a numerics lib).
    for (cs::size_t r = 0; r < sizeof...(D); ++r) {
        _TNY_CHECK(ae[r] == ce[r] || ae[r] == 1, "broadcast: lhs extent mismatch");
        _TNY_CHECK(be[r] == ce[r] || be[r] == 1, "broadcast: rhs extent mismatch");
    }
    I n = 1; for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= ce[r];
    // Contiguous linear fast path (out-of-place only, plain store, no broadcast).
    if constexpr (Restrict && cs::is_same<W, w_set>::value && A::rank() == R && B::rank() == R) {
        bool lin = c.is_contiguous() && a.is_contiguous() && b.is_contiguous();
        for (cs::size_t r = 0; lin && r < sizeof...(D); ++r) lin = (ae[r] == ce[r]) && (be[r] == ce[r]);
        if (lin) {
            using Ce = typename C::element_type;
            Ce * _TNY_RESTRICT cp = c.data();                    // fresh dest: cannot alias a/b -> restrict
            const typename A::element_type * ap = a.data();      // a and b may alias each other (e.g. a+a): NOT restrict
            const typename B::element_type * bp = b.data();
            for (I i = 0; i < n; ++i)
                cp[i] = static_cast<Ce>(op(static_cast<Cv>(ap[i]), static_cast<Cv>(bp[i])));   // mirrors w_set exactly
            return;
        }
    }
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0, ob = 0, oc = 0;
        for (int d = (int)sizeof...(D)-1; d >= 0; --d) {
            I k = rem % ce[d]; rem /= ce[d];
            oc += k * sc[d];
            oa += (ae[d] == 1 ? I(0) : k) * sa[d];
            ob += (be[d] == 1 ? I(0) : k) * sb[d];
        }
        // Same rationale as scal_'s decode loop: rhs/nrhs (the only ops paired with
        // the atomic writer w_add, via atomic_add_(b)/atomic_sub_(b)) ignore their
        // first argument outright, and here `a` IS `c` (atomic_add_ calls
        // bzip<w_add>(*this, *this, b, rhs{})) — so reading a.data()[oa] would be an
        // ordinary (non-atomic) load racing the concurrent atomic_ref RMW that same
        // fetch_add is performing on that very object.
        if constexpr (!_ignores_lhs<Op>::value)
            W{}(&c.data()[oc], op(static_cast<Cv>(a.data()[oa]), static_cast<Cv>(b.data()[ob])));
        else
            W{}(&c.data()[oc], op(Cv{}, static_cast<Cv>(b.data()[ob])));
    }
}
// A self-overlapping DESTINATION — an axis with extent>1 AND stride 0 — makes every
// write into it ALIAS: several distinct indices name ONE element. What that spoils
// depends on the writer, and BOTH kinds get here:
//   - a read-modify-write (`v.add_(b)`, `mul_`, `iota_`, the in-place `unary`) applies
//     the update to that one element repeatedly — it DOUBLE-COUNTS;
//   - a plain store (`scalo_`/`unaryo_` writing a caller-supplied `into(dest)`, e.g.
//     `a.mul(2.0, into(y))` / `exp(a, into(y))`) DISCARDS all but the last result
//     landing in the slot (#364).
// Either way the answer is wrong in a way no later check can recover, so the guard is
// on the write itself, not on the op. Such a destination can only come from
// `wrap(ptr, e, strides-with-a-0)`; teeny's own view ops never make one, and
// clone()/to()/the allocating out-of-place producers write into a fresh DENSE
// destination. Host-debug guard (a no-op on device / under -DNDEBUG); DESTINATION
// only, so a broadcasting RHS (which legitimately stretches with stride 0) and an
// overlapping SOURCE read by `into(dest)` are never flagged.
template <class C, cs::size_t... D>
_TNY_API void check_dest_no_overlap(const C & c, cs::index_sequence<D...>) {
    ( _TNY_CHECK(!(static_cast<long long>(c.extent(D)) > 1 && c.stride(D) == 0),
        "write into a self-overlapping view (an axis has extent>1 and stride 0): several "
        "indices alias one element, so an in-place update is applied to it repeatedly and an "
        "out-of-place store into it keeps only the last result — clone() to a dense tensor "
        "first, or write into a non-overlapping destination."), ... );
}
// EXACT per-axis extent check between a destination and the single source it is
// written from — the runtime half of the guard the SINGLE-SOURCE engines
// (`scalo_`/`unaryo_`) need, and the twin of the compile-time `ext_static_eq`
// above. Those engines take their loop BOUNDS from the source and their strides
// from the destination, so a destination shorter in any axis is written past its
// end (#357): `a.mul(2.0, into(y))` with an 8x8 `a` and a 2x2 `y` stored 64
// elements through a 4-element buffer. Their allocating producers (`oops`/
// `uop_out`) build the destination from the source's own extents type, so the only
// way to reach them mis-shaped is a caller-supplied `into(dest)`.
//
// EXACT equality, unlike `bzip_`'s `ae[r] == ce[r] || ae[r] == 1`: that engine
// BROADCASTS (an extent-1 operand axis gets stride 0 and is stretched), these two
// do not — they index the source with the same counter they index the destination
// with, so an extent 1 against an extent n is a plain mismatch, not a stretch.
// Extents are non-negative by construction, so comparing them as `cs::size_t`
// (mdspan's own `static_extent` type) is lossless and sidesteps a signed/unsigned
// mismatch between two differently-indexed tensors.
template <class C, class A, cs::size_t... D>
_TNY_API void check_same_extents(const C & c, const A & a, cs::index_sequence<D...>) {
    ( _TNY_CHECK(static_cast<cs::size_t>(c.extent(D)) == static_cast<cs::size_t>(a.extent(D)),
        "into(dest): dest's shape must match the source's exactly (no broadcast here — a "
        "scalar-rhs or unary op has nothing to stretch); a shorter dest is written past its end."), ... );
}
// `Restrict` defaults to false so every IN-PLACE caller (add_/sub_/.../copy_,
// where `c` IS the destination and may alias the rhs) takes the safe decode path
// byte-for-byte unchanged; the out-of-place `oop` passes true.
// `Cv` (the op's compute type) defaults to `void` = "the destination's compute
// type", which is what every IN-PLACE caller wants (`c` IS the lhs operand there).
// The out-of-place ones name it from the OPERANDS instead — see `_cv_or_dest_t`.
template <class W = w_set, bool Restrict = false, class Cv = void, class C, class A, class B, class Op>
_TNY_API void bzip(C & c, const A & a, const B & b, Op op) {
    // C holds the RESULT (largest) rank; operands may be shorter (left-padded).
    check_dest_no_overlap(c, cs::make_index_sequence<C::rank()>{});
    static_assert(A::rank() <= C::rank() && B::rank() <= C::rank(), "broadcast: operand rank exceeds result");
    static_assert(bc_static_ok_r<typename A::extents_type, typename B::extents_type, C::rank()>(
                      cs::make_index_sequence<C::rank()>{}),
                  "broadcast: incompatible static extents");
    bzip_<W, Restrict, _cv_or_dest_t<Cv, C>>(c, a, b, op, cs::make_index_sequence<C::rank()>{});
}

/* ---- c = op(c, scalar), elementwise ------------------------------ */
template <class W, class C, class Op, cs::size_t... D>
_TNY_API void scal_(C & c, typename C::element_type s, Op op, cs::index_sequence<D...>) {
    check_dest_no_overlap(c, cs::index_sequence<D...>{});
    using I  = typename C::index_type;
    using Cv = compute_type_t<typename C::element_type>;   // compute in float for half types
    const Cv sv = static_cast<Cv>(s);
    // Array size floored to 1: a rank-0 `C` (e.g. `.at(i,j).add_(v)`) makes `D...`
    // empty, and a genuine zero-length array is a GCC/Clang extension MSVC rejects
    // (C2466). The loop below is bounded by `sizeof...(D)`, not the array size, so
    // the unused padding slot when rank is 0 is never read.
    const I e[sizeof...(D) ? sizeof...(D) : 1]  = { c.extent(D)... };
    const I sc[sizeof...(D) ? sizeof...(D) : 1] = { c.stride(D)... };
    I n = 1;
    for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    // Linear fast path. An in-place SCALAR op is a SINGLE-array read-modify-write (the
    // rhs is a register value, not a second pointer), so there is no aliasing to defeat
    // — one pointer vectorizes with NO `__restrict__`. It is also ORDER-INDEPENDENT, so
    // ANY dense view works: `is_dense()` (dense in some axis order — C/F/permuted) has
    // offsets exactly [0,numel) and excludes stride-0 overlap and negative strides, so
    // the physical block can be walked linearly. Gated to the plain store `w_set`, so
    // an atomic writer keeps the decode path.
    if constexpr (cs::is_same<W, w_set>::value) {
        if (c.is_dense()) {
            using Ce = typename C::element_type;
            Ce * cp = c.data();
            for (I i = 0; i < n; ++i) cp[i] = static_cast<Ce>(op(static_cast<Cv>(cp[i]), sv));   // explicit store, mirrors w_set
            return;
        }
    }
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oc = 0;
        for (int d = static_cast<int>(sizeof...(D)) - 1; d >= 0; --d) {
            I k = rem % e[d]; rem /= e[d]; oc += k * sc[d];
        }
        // Most ops (add/sub/mul/div/...) are a genuine read-modify-write and need
        // the current value; rhs/nrhs/setc (_ignores_lhs) don't, so skip reading
        // c.data()[oc] for them — with an atomic writer (w_add) that ordinary
        // (non-atomic) read would otherwise race the concurrent atomic_ref RMW on
        // the same object, even though the loaded value is discarded either way.
        if constexpr (!_ignores_lhs<Op>::value)
            W{}(&c.data()[oc], op(static_cast<Cv>(c.data()[oc]), sv));
        else
            W{}(&c.data()[oc], op(Cv{}, sv));
    }
}
template <class W = w_set, class C, class Op>
_TNY_API void scal(C & c, typename C::element_type s, Op op) {
    scal_<W>(c, s, op, cs::make_index_sequence<C::rank()>{});
}

/* ---- c = start, start+step, ... in row-major logical order -------- */
template <class C, cs::size_t... D>
_TNY_API void iota_(C & c, typename C::element_type start, typename C::element_type step, cs::index_sequence<D...>) {
    check_dest_no_overlap(c, cs::index_sequence<D...>{});
    using I = typename C::index_type; using Cv = compute_type_t<typename C::element_type>;
    // Array size floored to 1 (rank-0 dest -> empty D..., see scal_'s comment above).
    const I e[sizeof...(D) ? sizeof...(D) : 1]  = { c.extent(D)... };
    const I sc[sizeof...(D) ? sizeof...(D) : 1] = { c.stride(D)... };
    I n = 1; for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    // Contiguous linear fast path: a pure single-array write (no read, no second
    // pointer) — vectorizes with one pointer, no `__restrict__` needed.
    if (c.is_contiguous()) {
        typename C::element_type * cp = c.data();
        for (I i = 0; i < n; ++i) cp[i] = static_cast<Cv>(start) + static_cast<Cv>(i) * static_cast<Cv>(step);
        return;
    }
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oc = 0;
        for (int d = static_cast<int>(sizeof...(D)) - 1; d >= 0; --d) { I k = rem % e[d]; rem /= e[d]; oc += k * sc[d]; }
        c.data()[oc] = static_cast<Cv>(start) + static_cast<Cv>(lin) * static_cast<Cv>(step);
    }
}

/* ---- c(i) = op(a(i), scalar) ------------------------------------- */
template <bool Restrict, class Cv, class C, class A, class S, class Op, cs::size_t... D>
_TNY_API void scalo_(C & c, const A & a, S s, Op op, cs::index_sequence<D...>) {
    // Offsets are decoded in a type that holds every value BOTH participants — the
    // destination and the source — can produce (#353; `_offset_int_t` above spells the
    // rule out, and `bzip_` below carries the same note for its three-way version).
    // Taking the DESTINATION's index type alone truncated a wider-indexed source's
    // extents/strides. The allocating producers (`oops`) build `c` from `a`'s own
    // extents type, so `C::index_type == A::index_type` there and this is a no-op; it
    // is a caller-supplied `into(dest)` that can hand us a NARROW destination next to a
    // wide source, and there `a.stride()` of 40000 folded to an int16 -25536 and read
    // off the front of the buffer. Unlike `bzip_` this site was not even silent — the
    // initializers below lacked the `static_cast<I>`s, so g++ warned (`-Wnarrowing`) and
    // clang rejected the instantiation outright.
    // `Cv` = the op's compute type — see `_cv_or_dest_t` above for the rule (in-place:
    // the destination's, which is the lhs operand's; out-of-place: the operands' own,
    // #379; compare: `Rc`). It is NOT necessarily `C`'s own type, so the stores below
    // cast to `Ce` explicitly, exactly as `bzip_`'s `w_set`/fast path do.
    //
    // Shape guard (#357): the bounds below come from `a` and the stores go through
    // `c`'s strides, so the two must agree in EVERY axis or the write runs off the
    // end of `c`. Checked HERE, in the engine, rather than in the `scalo` wrapper
    // the way `bzip` places its own static gate — `scmp` (comparisons) calls
    // `scalo_` directly, and one guard at the single point that does the indexing
    // cannot be bypassed by a future caller. Both halves, as `dot`/`scan`'s
    // `into(dest)` do it: a `static_assert` when both shapes are fully static (the
    // documented repro is a COMPILE error, not a debug-only trip) and a per-axis
    // `_TNY_CHECK` for anything dynamic. No-ops for every non-`into` caller, whose
    // destination is built from `a`'s own extents type (`oops`/`oops_cmp`) or IS
    // `a` (`unary`).
    //
    // ...and the SELF-OVERLAP guard (#364), for the same reachability: a stride-0
    // `into(dest)` axis makes many source elements store into one slot, so all but
    // the last result is silently dropped. The other four engines that write a
    // destination (`bzip`, `scal_`, `iota_`, the in-place `unary`) have called
    // `check_dest_no_overlap` all along — these two never did, so the SAME mistake
    // aborted with a diagnostic through `a.add(a, into(y))` and passed silently
    // through `a.mul(2.0, into(y))`. Same convention as those four (destination only,
    // the full axis pack, host-debug), and a no-op for every non-`into` caller: the
    // allocating producers build a fresh dense `c`, and the in-place `unary` has
    // already run this exact check before delegating here.
    static_assert(A::rank() == C::rank(), "into(dest): dest rank must match the source's");
    static_assert(ext_static_eq<typename C::extents_type, typename A::extents_type>(
                      cs::make_index_sequence<C::rank()>{}),
                  "into(dest): dest's shape must match the source's exactly (no broadcast)");
    check_same_extents(c, a, cs::index_sequence<D...>{});
    check_dest_no_overlap(c, cs::index_sequence<D...>{});
    using I = _offset_int_t<typename C::index_type,
                            typename A::extents_type::index_type>;
    // Array size floored to 1 (rank-0 operands -> empty D..., see scal_'s comment above).
    const I e[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(a.extent(D))... },
            sa[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(a.stride(D))... },
            sc[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(c.stride(D))... };
    I n = 1; for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    using Ce = typename C::element_type;
    // Contiguous linear fast path (out-of-place only; `c` fresh, cannot alias `a`).
    if constexpr (Restrict) {
        if (c.is_contiguous() && a.is_contiguous()) {
            Ce * _TNY_RESTRICT cp = c.data();
            const typename A::element_type * ap = a.data();
            const Cv sv = static_cast<Cv>(s);
            for (I i = 0; i < n; ++i) cp[i] = static_cast<Ce>(op(static_cast<Cv>(ap[i]), sv));   // mirrors the slow store
            return;
        }
    }
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0, oc = 0;
        for (int d = (int)sizeof...(D)-1; d >= 0; --d) { I k = rem%e[d]; rem/=e[d]; oa+=k*sa[d]; oc+=k*sc[d]; }
        c.data()[oc] = static_cast<Ce>(op(static_cast<Cv>(a.data()[oa]), static_cast<Cv>(s)));
    }
}
// `Cv` as in `bzip` (`void` = the destination's compute type; the out-of-place
// producers name it from the operands — `_cv_or_dest_t`).
template <bool Restrict = false, class Cv = void, class C, class A, class S, class Op> _TNY_API void scalo(C & c, const A & a, S s, Op op)
{ scalo_<Restrict, _cv_or_dest_t<Cv, C>>(c, a, s, op, cs::make_index_sequence<C::rank()>{}); }

/* ---- c(i) = uop(a(i))  and  c(i) = uop(c(i)) (in place) ---------- */
template <bool Restrict, class Cv, class C, class A, class Uop, cs::size_t... D>
_TNY_API void unaryo_(C & c, const A & a, Uop f, cs::index_sequence<D...>) {
    // Same rule (and the same reachability) as `scalo_` just above: the offsets decode
    // in a type covering the destination AND the source, so a caller-supplied
    // `into(dest)` narrower than the source no longer truncates the source's strides
    // (#353). The allocating producer (`uop_out`) builds `c` from `a`'s extents type,
    // so that path is unchanged.
    //
    // ...and the same shape guard as `scalo_` (#357), for the same reason: bounds
    // from `a`, strides from `c`, so a mis-shaped `into(dest)` — the only way a
    // destination of the caller's own choosing gets here — writes past its end.
    // `static_assert` when both shapes are static, per-axis `_TNY_CHECK` otherwise;
    // a no-op for `uop_out` (dest built from `a`'s extents) and for the in-place
    // `unary`, which passes `c` as both arguments.
    //
    // ...and the SELF-OVERLAP guard, exactly as `scalo_` above documents it (#364):
    // `exp(a, into(y))` into a stride-0 `y` used to drop every result but the last,
    // while the tensor-rhs `a.add(a, into(y))` aborted. Redundant-but-harmless for
    // the in-place `unary`, which checks before delegating here (it needs its own
    // copy for the dense fast path it takes instead).
    //
    // `Cv` (the type `f` runs in) is a PARAMETER, not derived from `c`: `uop_to`
    // (`into(dest)`) hands us a destination of the caller's choosing, and deriving
    // the compute type from it evaluated `f` in that type — `exp(double_a,
    // into(int_y))` exponentiated the TRUNCATED input (#379). It is the SOURCE's
    // compute type there, exactly what `uop_out`'s own destination would have
    // carried; the in-place `unary` still passes `void` (= `c`'s, which is `a`'s).
    static_assert(A::rank() == C::rank(), "into(dest): dest rank must match the source's");
    static_assert(ext_static_eq<typename C::extents_type, typename A::extents_type>(
                      cs::make_index_sequence<C::rank()>{}),
                  "into(dest): dest's shape must match the source's exactly (no broadcast)");
    check_same_extents(c, a, cs::index_sequence<D...>{});
    check_dest_no_overlap(c, cs::index_sequence<D...>{});
    using I = _offset_int_t<typename C::index_type,
                            typename A::extents_type::index_type>;
    using Ce = typename C::element_type;
    // Array size floored to 1 (rank-0 operands -> empty D..., see scal_'s comment above).
    const I e[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(a.extent(D))... },
            sa[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(a.stride(D))... },
            sc[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(c.stride(D))... };
    I n = 1; for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    // Contiguous linear fast path (out-of-place only; `c` fresh, cannot alias `a`).
    if constexpr (Restrict) {
        if (c.is_contiguous() && a.is_contiguous()) {
            Ce * _TNY_RESTRICT cp = c.data();
            const typename A::element_type * ap = a.data();
            for (I i = 0; i < n; ++i) cp[i] = static_cast<Ce>(f(static_cast<Cv>(ap[i])));   // mirrors the slow store
            return;
        }
    }
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0, oc = 0;
        for (int d = (int)sizeof...(D)-1; d >= 0; --d) { I k = rem%e[d]; rem/=e[d]; oa+=k*sa[d]; oc+=k*sc[d]; }
        c.data()[oc] = static_cast<Ce>(f(static_cast<Cv>(a.data()[oa])));
    }
}
// `Restrict` is false for `unary` (in-place: c===a, must not restrict) and true
// for `uop_out` (out-of-place: fresh dest). `Cv` as in `bzip`/`scalo` (`void` = the
// destination's compute type; `uop_to` names the source's — `_cv_or_dest_t`).
template <bool Restrict = false, class Cv = void, class C, class A, class Uop> _TNY_API void unaryo(C & c, const A & a, Uop f)
{ unaryo_<Restrict, _cv_or_dest_t<Cv, C>>(c, a, f, cs::make_index_sequence<C::rank()>{}); }
// In-place unary (`a.neg_()`/`exp_()`/`map_(f)`/…) is a SINGLE-array read-modify-write:
// `a[i] = f(a[i])`. One pointer, so it vectorizes with NO `__restrict__` (nothing to
// alias), and being ORDER-INDEPENDENT it walks the physical block of ANY dense view.
// A stride-0 destination would apply `f` twice to an aliased element, so guard it like
// `scal_`/`iota_`; a genuinely strided view falls back to the general decode.
template <class C, class Uop> _TNY_API void unary(C & c, Uop f) {
    check_dest_no_overlap(c, cs::make_index_sequence<C::rank()>{});
    using I = typename C::index_type; using Cv = compute_type_t<typename C::element_type>;
    if (c.is_dense()) {
        typename C::element_type * cp = c.data();
        const I n = static_cast<I>(c.numel());
        for (I i = 0; i < n; ++i) cp[i] = static_cast<Cv>(f(static_cast<Cv>(cp[i])));   // mirrors the store
        return;
    }
    unaryo(c, c, f);
}

/* ---- out-of-place tensor (op) tensor, broadcasting --------------- *
 * static -> stack (host+device), else heap (host only).              */
template <class Op, class A, class B,
          cs::enable_if_t<bcast_extents<typename A::extents_type, typename B::extents_type>::rank_dynamic() == 0, int> = 0>
_TNY_API auto oop(const A & a, const B & b, Op op) {
    using RE = bcast_extents<typename A::extents_type, typename B::extents_type>;
    // `_uninit` (not `{}`): the `w_set` engine below fully overwrites `c`, so a
    // zero-fill here is a dead store (a `memset` per call for a static result that
    // survives DSE — #212). Every out-of-place result on this page is uninitialised
    // for the same reason (bzip/scalo/bcmp/scmp/unaryo all full-write, Restrict=true).
    tensor<promote_t<typename A::element_type, typename B::element_type>, RE, ccontiguous, storage::stack> c(_uninit);
    bzip<w_set, true>(c, a, b, op); return c;   // fresh dest -> restrict + linear fast path
}
template <class Op, class A, class B,
          cs::enable_if_t<bcast_extents<typename A::extents_type, typename B::extents_type>::rank_dynamic() != 0, int> = 0>
_TNY_HOST auto oop(const A & a, const B & b, Op op) {
    using RE = bcast_extents<typename A::extents_type, typename B::extents_type>;
    tensor<promote_t<typename A::element_type, typename B::element_type>, RE, ccontiguous, storage::heap>
        c(bcast_runtime_<RE>(a, b, cs::make_index_sequence<RE::rank()>{}));
    bzip<w_set, true>(c, a, b, op); return c;   // fresh dest -> restrict + linear fast path
}

/* ---- out-of-place tensor (op) scalar ----------------------------- */
template <class Op, class A, class S, cs::enable_if_t<A::is_static, int> = 0>
_TNY_API auto oops(const A & a, S s, Op op) {
    tensor<promote_t<typename A::element_type, S>, typename A::extents_type, ccontiguous, storage::stack> c(_uninit);
    scalo<true>(c, a, s, op); return c;   // fresh dest -> restrict + linear fast path
}
template <class Op, class A, class S, cs::enable_if_t<!A::is_static, int> = 0>
_TNY_HOST auto oops(const A & a, S s, Op op) {
    tensor<promote_t<typename A::element_type, S>, typename A::extents_type, ccontiguous, storage::heap> c(a.extents());
    scalo<true>(c, a, s, op); return c;   // fresh dest -> restrict + linear fast path
}

/* ---- comparisons -> a bool tensor (broadcast; computed in Rc) ----- *
 * A comparison is just `bzip_`/`scalo_` run in the operands' compare type `Rc`
 * (so `a < b` compares the values, not their bool cast) with a plain-store writer
 * (`w_set`) into the bool result — so they reuse those engines directly rather
 * than duplicating the broadcast decode. */
template <class Rc, bool Restrict = false, class C, class A, class B, class Op>
_TNY_API void bcmp(C & c, const A & a, const B & b, Op op) {
    static_assert(A::rank() <= C::rank() && B::rank() <= C::rank(), "compare: operand rank exceeds result");
    static_assert(bc_static_ok_r<typename A::extents_type, typename B::extents_type, C::rank()>(cs::make_index_sequence<C::rank()>{}), "compare: incompatible static extents");
    bzip_<w_set, Restrict, Rc>(c, a, b, op, cs::make_index_sequence<C::rank()>{});   // compare in Rc; op returns bool -> stored
}
template <class Rc, bool Restrict = false, class C, class A, class S, class Op> _TNY_API void scmp(C & c, const A & a, S s, Op op)
{ scalo_<Restrict, Rc>(c, a, s, op, cs::make_index_sequence<C::rank()>{}); }

// tensor (cmp) tensor -> bool tensor (static -> stack, dynamic -> heap)
template <class Op, class A, class B,
          cs::enable_if_t<bcast_extents<typename A::extents_type, typename B::extents_type>::rank_dynamic() == 0, int> = 0>
_TNY_API auto oop_cmp(const A & a, const B & b, Op op) {
    using RE = bcast_extents<typename A::extents_type, typename B::extents_type>;
    using Rc = compute_type_t<promote_t<typename A::element_type, typename B::element_type>>;
    tensor<bool, RE, ccontiguous, storage::stack> c(_uninit); bcmp<Rc, true>(c, a, b, op); return c;
}
template <class Op, class A, class B,
          cs::enable_if_t<bcast_extents<typename A::extents_type, typename B::extents_type>::rank_dynamic() != 0, int> = 0>
_TNY_HOST auto oop_cmp(const A & a, const B & b, Op op) {
    using RE = bcast_extents<typename A::extents_type, typename B::extents_type>;
    using Rc = compute_type_t<promote_t<typename A::element_type, typename B::element_type>>;
    tensor<bool, RE, ccontiguous, storage::heap> c(bcast_runtime_<RE>(a, b, cs::make_index_sequence<RE::rank()>{}));
    bcmp<Rc, true>(c, a, b, op); return c;
}
// tensor (cmp) scalar -> bool tensor
template <class Op, class A, class S, cs::enable_if_t<A::is_static, int> = 0>
_TNY_API auto oops_cmp(const A & a, S s, Op op) {
    using Rc = compute_type_t<promote_t<typename A::element_type, S>>;
    tensor<bool, typename A::extents_type, ccontiguous, storage::stack> c(_uninit); scmp<Rc, true>(c, a, s, op); return c;
}
template <class Op, class A, class S, cs::enable_if_t<!A::is_static, int> = 0>
_TNY_HOST auto oops_cmp(const A & a, S s, Op op) {
    using Rc = compute_type_t<promote_t<typename A::element_type, S>>;
    tensor<bool, typename A::extents_type, ccontiguous, storage::heap> c(a.extents()); scmp<Rc, true>(c, a, s, op); return c;
}

/* ---- out-of-place unary : static -> stack, dynamic -> heap ------- */
template <class Uop, class A, cs::enable_if_t<A::is_static, int> = 0>
_TNY_API auto uop_out(const A & a, Uop f) {
    tensor<typename A::element_type, typename A::extents_type, ccontiguous, storage::stack> c(_uninit);
    unaryo<true>(c, a, f); return c;   // fresh dest -> restrict + linear fast path
}
template <class Uop, class A, cs::enable_if_t<!A::is_static, int> = 0>
_TNY_HOST auto uop_out(const A & a, Uop f) {
    tensor<typename A::element_type, typename A::extents_type, ccontiguous, storage::heap> c(a.extents());
    unaryo<true>(c, a, f); return c;   // fresh dest -> restrict + linear fast path
}

/* ---- into(dest): run a producer's engine into a CALLER-provided dest ------ *
 * One fused pass, NO allocation. `Restrict=false` because a user dest may alias
 * an operand; the engine validates dest's extents against the result the producer
 * would have allocated and casts to dest's element type. The producer hands
 * `into_t::dest` back.
 *
 * The dest is the ONLY shape here the caller picks freely, so it is the only one
 * that can disagree: `oop_to`'s `bzip_` checks it against the BROADCAST result
 * (an operand extent of 1 stretches, the dest's does not), while `oops_to`/
 * `uop_to`'s `scalo_`/`unaryo_` require EXACT equality with the source — a
 * scalar-rhs or unary op has no stretch semantics at all (#357).
 *
 * Those two ALSO gate it at compile time (`ext_static_eq`) when both shapes are
 * fully static; `bzip_`'s is a `_TNY_CHECK` only, since `bzip`'s own static gate
 * (`bc_static_ok_r`) compares the two OPERANDS with each other, never either one
 * against `c`. Not a silent write either way — just a debug-time trip rather than
 * a compile error for a static mis-shaped tensor-rhs dest.
 *
 * The dest's ELEMENT TYPE is likewise the caller's alone, and it is the RESULT that
 * is cast to it — the arithmetic itself runs in the OPERANDS' compute type, so each
 * of these hands its engine the very `Cv` its allocating twin would have got from
 * its own `promote_t` destination (#379). Deriving `Cv` from the dest instead ran
 * the whole computation in the dest's type, scalar rhs and axpy coefficient
 * included: `a.mul(0.5, into(int_y))` multiplied by `int(0.5)` == 0, and
 * `a.add(b, 0.5, into(int_y))` left `y` == `a`. With the operands' type, each
 * `x.op(y, into(dest))` is numerically identical to `dest.copy_(x.op(y))` — the
 * `into` form just skips the temporary. (A `half`/`bfloat16` DEST over half/float
 * operands is unaffected: `compute_type_t` is `float` on both sides. Over WIDER
 * operands it now computes in the wider type and narrows on the store, which is
 * the same "cast the result" rule, not a half-specific carve-out.)
 *
 * The allocating producers (`oop`/`oops`/`uop_out`) keep the default: their
 * destination IS built as the promoted type, so `compute_type_t<C::element_type>`
 * already IS the operands' compute type there — same type, byte-identical code. */
template <class Op, class Out, class A, class B> _TNY_API void oop_to (Out & o, const A & a, const B & b, Op op)
{ bzip<w_set, false, compute_type_t<promote_t<typename A::element_type, typename B::element_type>>>(o, a, b, op); }
template <class Op, class Out, class A, class S>  _TNY_API void oops_to(Out & o, const A & a, S s, Op op)
{ scalo<false, compute_type_t<promote_t<typename A::element_type, S>>>(o, a, s, op); }
template <class Uop, class Out, class A>          _TNY_API void uop_to (Out & o, const A & a, Uop f)
{ unaryo<false, compute_type_t<typename A::element_type>>(o, a, f); }

// static element count of a fully-static extents type (moved ahead of
// zipreduce_/axreduce, #255: both static fast paths need it).
template <class E, cs::size_t... D>
constexpr cs::size_t _static_numel_(cs::index_sequence<D...>) { return (cs::size_t(1) * ... * E::static_extent(D)); }
template <class E>
constexpr cs::size_t _static_numel() { return _static_numel_<E>(cs::make_index_sequence<E::rank()>{}); }

/* ---- reduce op(a, b) elementwise into a scalar (dot: op=mul; sqdist: op=zip_sqdiff) --- *
 * One fused pass, NO intermediate tensor materialised. */
template <class R, class A, class B, class Op, cs::size_t... D>
_TNY_API R zipreduce_decode_(const A & a, const B & b, Op op, cs::index_sequence<D...>) {
    // Offsets are decoded in a type that holds EVERY value BOTH participants can
    // produce (`_offset_int_t` above spells the rule out). Here the participants are
    // just the two operands — unlike `bzip_`, this engine writes no tensor, only a
    // scalar accumulator `acc` of the caller's reduce type `R`, so there is no third
    // (destination) index type in play; the SAME rule applied to a two-element set.
    // Taking the FIRST operand's index type alone truncated the second's
    // extents/strides (#342) — a hard `-Wc++11-narrowing` error under clang, a
    // silently wrong offset under g++ once a stride overflowed it. Widening by
    // `sizeof` ALONE then still mis-addressed when the wider operand is UNSIGNED and
    // the narrower one is SIGNED with a NEGATIVE stride (a flipped/reversed view):
    // `static_cast<uint32_t>(-1)` is 4294967295, which zero-extends into the pointer
    // offset instead of stepping backwards, and `a.data()[oa]` reads far off the front
    // of the buffer (#355 — the same defect class `bzip_` carried in #346). So the
    // pick is signedness-aware. For an all-signed or all-unsigned pair — every call
    // site that is not this mixed case — `_offset_int_t` IS `_wider_index_t`
    // (`_widest_int` of two left-folds to exactly `_wider_int_t`, same tie rule), so
    // those instantiations are byte-identical to before.
    //
    // As in `bzip_`, the widening is internal only: neither operand's own type
    // changes, each operand's offsets fit its own index type by construction, and
    // `data()[off]` takes any integer, so nothing is narrowed back on the way out.
    using I = _offset_int_t<typename A::extents_type::index_type,
                            typename B::extents_type::index_type>;
    // Array size floored to 1 (rank-0 operands -> empty D..., see scal_'s comment above).
    const I e[sizeof...(D) ? sizeof...(D) : 1]  = { static_cast<I>(a.extent(D))... };
    const I be[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(b.extent(D))... };
    const I sa[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(a.stride(D))... };
    const I sb[sizeof...(D) ? sizeof...(D) : 1] = { static_cast<I>(b.stride(D))... };
    for (cs::size_t r = 0; r < sizeof...(D); ++r)
        _TNY_CHECK(e[r] == be[r], "dot/sqdist: operand extents must match exactly (no broadcast)");
    I n = 1;
    for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    R acc = R(0);
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0, ob = 0;
        for (int d = static_cast<int>(sizeof...(D)) - 1; d >= 0; --d) {
            I k = rem % e[d]; rem /= e[d]; oa += k * sa[d]; ob += k * sb[d];
        }
        acc += op(static_cast<R>(a.data()[oa]), static_cast<R>(b.data()[ob]));
    }
    return acc;
}

// STATIC fast path (#255): when BOTH operands are static-shaped (their extents
// already match exactly at compile time -- dot/sqdist each `static_assert` that
// before ever calling zipreduce_) AND both C-contiguous, linear index `Lin`
// addresses the SAME logical element in both operands directly -- no per-step
// decode. Unrolling over the (static, shared) element count then emits
// straight-line code with no loop back-edge and no runtime %/div, mirroring
// axreduce's own #218 static fast path (small fixed-rank dot/sqdist -- a
// posdef cross-channel dot, a stencil tap accumulation -- pay pure loop
// overhead otherwise).
template <cs::size_t Lin, class R, class A, class B, class Op>
_TNY_API R zipreduce_one_(const A & a, const B & b, Op op) {
    return op(static_cast<R>(a.data()[Lin]), static_cast<R>(b.data()[Lin]));
}
template <class R, class A, class B, class Op, cs::size_t... Lin>
_TNY_API R zipreduce_static_(const A & a, const B & b, Op op, cs::index_sequence<Lin...>) {
    R acc = R(0);
    ( (acc = static_cast<R>(acc + zipreduce_one_<Lin, R>(a, b, op))), ... );
    return acc;
}

template <class R, class A, class B, class Op, cs::size_t... D>
_TNY_API R zipreduce_(const A & a, const B & b, Op op, cs::index_sequence<D...> seq) {
    using EA = typename A::extents_type; using LA = typename A::layout_type;
    using EB = typename B::extents_type;
    if constexpr (EA::rank_dynamic() == 0 && EB::rank_dynamic() == 0
                  && cs::is_same<LA, ccontiguous>::value && cs::is_same<typename B::layout_type, ccontiguous>::value) {
        return zipreduce_static_<R>(a, b, op, cs::make_index_sequence<_static_numel<EA>()>{});
    } else {
        return zipreduce_decode_<R>(a, b, op, seq);
    }
}

/* ---- fold a into a scalar with `op`, starting from `init` --------- */
template <class R, class A, class Op, cs::size_t... D>
_TNY_API R reduce_(const A & a, R init, Op op, cs::index_sequence<D...>) {
    using I = typename A::index_type;
    // Array size floored to 1 (rank-0 operand -> empty D..., see scal_'s comment above).
    const I e[sizeof...(D) ? sizeof...(D) : 1]  = { a.extent(D)... };
    const I sa[sizeof...(D) ? sizeof...(D) : 1] = { a.stride(D)... };
    I n = 1;
    for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    R acc = init;
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0;
        for (int d = static_cast<int>(sizeof...(D)) - 1; d >= 0; --d) {
            I k = rem % e[d]; rem /= e[d]; oa += k * sa[d];
        }
        acc = op(acc, a.data()[oa]);
    }
    return acc;
}

/* ---- axis reductions -> a tensor (the reduced axes are removed) ---- *
 * Output extents = input extents with `Axes...` dropped (static where the
 * input is). Fully static -> stack (host+device); dynamic -> heap (HOST). */

// `red_ext` / `reduced_ext_` / `reduced_extents` (the output extents of an axis
// reduction) live in tensor.h: the reduction METHOD declarations there SFINAE on
// them to split _TNY_API (static result) from _TNY_HOST (dynamic result), exactly
// as the free reductions below do, and tensor.h is included before this header.

// the engine: init `out` to `init`, then fold each input element into its output
// cell (reduced axes contribute stride 0 to the output offset). `out` is a fresh
// contiguous tensor, so out.data()[k] is its k-th element.
template <class R, class Out, class A, class Op, cs::size_t... D>
_TNY_API void reduce_axes_(Out & out, const A & a, R init, Op op, const bool * reduced, cs::index_sequence<D...>) {
    using I = typename A::index_type; using Tout = typename Out::element_type;
    constexpr int N = sizeof...(D);
    // Array size floored to 1 (rank-0 `a` -> empty D..., see scal_'s comment above).
    const I e[N ? N : 1]  = { static_cast<I>(a.extent(D))... };
    const I sa[N ? N : 1] = { static_cast<I>(a.stride(D))... };
    I so[N ? N : 1]; int oi = 0;                           // output stride per input axis (0 if reduced)
    for (int d = 0; d < N; ++d) { if (reduced[d]) so[d] = I(0); else { so[d] = static_cast<I>(out.stride(oi)); ++oi; } }
    const I on = static_cast<I>(out.numel());
    for (I k = 0; k < on; ++k) out.data()[k] = static_cast<Tout>(init);
    I n = 1; for (int d = 0; d < N; ++d) n *= e[d];
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, ia = 0, oo = 0;
        for (int d = N - 1; d >= 0; --d) { I k = rem % e[d]; rem /= e[d]; ia += k * sa[d]; oo += k * so[d]; }
        out.data()[oo] = static_cast<Tout>(op(static_cast<R>(out.data()[oo]), a.data()[ia]));
    }
}

// STATIC fast path (#218): when the input extents are all static AND it is
// C-contiguous, the per-element decode of `reduce_axes_` folds to nothing — the input
// offset IS the linear index (contiguous), and the OUTPUT offset is a compile-time
// function of it. Unrolling over the (static) element count then emits straight-line
// FMA-style code with no loop back-edge and no runtime `%`/`/` — matching a
// hand-written nested loop (the axis reduction was 3-7x slower otherwise, since the
// runtime radix decode did not fold). `_red_oo` is the compile-time output offset for
// input linear index `lin`: decode `lin` over the static input extents, weight each
// kept axis by the (ccontiguous) output stride, drop the reduced axes.
template <class E, class OE, long... Axes>
_TNY_API constexpr typename E::index_type _red_oo(typename E::index_type lin) noexcept {
    using I = typename E::index_type;
    constexpr int N = static_cast<int>(E::rank());
    I so[N ? N : 1]{};                                         // output-offset stride per INPUT axis
    I ostr[OE::rank() ? OE::rank() : 1]{};                     // ccontiguous output strides
    { I s = 1; for (int i = static_cast<int>(OE::rank()) - 1; i >= 0; --i) { ostr[i] = s; s *= static_cast<I>(OE::static_extent(i)); } }
    int oi = 0;
    for (int d = 0; d < N; ++d) {
        const bool red = ( (d == _norm_axis(Axes, E::rank())) || ... );
        if (red) so[d] = 0; else { so[d] = ostr[oi]; ++oi; }
    }
    I rem = lin, oo = 0;
    for (int d = N - 1; d >= 0; --d) { const I ext = static_cast<I>(E::static_extent(d)); const I k = rem % ext; rem /= ext; oo += k * so[d]; }
    return oo;
}
// `_static_numel_`/`_static_numel` moved ahead of `zipreduce_` (#255) -- both
// static fast paths need it, and zipreduce_ comes first in the file.

// one unrolled step for input linear index `Lin` (a TEMPLATE arg, so the output
// offset `oo` is bound as a `constexpr` — computed once, at compile time; clang would
// otherwise re-evaluate the `_red_oo` call for the read and the write).
template <cs::size_t Lin, class E, class OE, long... Axes, class R, class Out, class A, class Op>
_TNY_API void reduce_axes_one_(Out & out, const A & a, R, Op op) {
    using Tout = typename Out::element_type;
    constexpr auto oo = _red_oo<E, OE, Axes...>(static_cast<typename E::index_type>(Lin));
    out.data()[oo] = static_cast<Tout>(op(static_cast<R>(out.data()[oo]), a.data()[Lin]));
}
template <long... Axes, class R, class Out, class A, class Op, cs::size_t... Lin>
_TNY_API void reduce_axes_static_(Out & out, const A & a, R init, Op op, cs::index_sequence<Lin...>) {
    using E = typename A::extents_type; using OE = typename Out::extents_type;
    using Tout = typename Out::element_type;
    constexpr cs::size_t ON = _static_numel<OE>();
    for (cs::size_t k = 0; k < ON; ++k) out.data()[k] = static_cast<Tout>(init);
    // fully unrolled: input offset == Lin (contiguous), output offset is compile-time.
    ( reduce_axes_one_<Lin, E, OE, Axes...>(out, a, init, op), ... );
}

// fully static result -> stack (host+device). Output element type = R (the
// accumulator), so accumulation runs in full `R` precision; the public reduction
// (`sum<0>` etc.) then casts this down to the tensor's element type via reduce_to.
template <long... Axes, class R, class Op, class T,class E,class L,storage O,
          class OE = reduced_extents<E, Axes...>, cs::enable_if_t<OE::rank_dynamic() == 0, int> = 0>
_TNY_API auto axreduce(const tensor<T,E,L,O> & a, R init, Op op) {
    static_assert((_axis_in_range(Axes, E::rank()) && ...), "reduction axis out of range");
    tensor<R, OE, ccontiguous, storage::stack> out{};
    if constexpr (E::rank_dynamic() == 0 && cs::is_same<L, ccontiguous>::value) {
        // static + C-contiguous: unroll (input offset == linear index; output offset folds)
        reduce_axes_static_<Axes...>(out, a, init, op, cs::make_index_sequence<_static_numel<E>()>{});
    } else {
        bool red[E::rank()] = {}; ( (red[_norm_axis(Axes, E::rank())] = true), ... );
        reduce_axes_<R>(out, a, init, op, red, cs::make_index_sequence<E::rank()>{});
    }
    return out;
}
// any dynamic result -> heap (HOST ONLY: it must allocate; not callable on device)
template <long... Axes, class R, class Op, class T,class E,class L,storage O,
          class OE = reduced_extents<E, Axes...>, cs::enable_if_t<OE::rank_dynamic() != 0, int> = 0>
_TNY_HOST auto axreduce(const tensor<T,E,L,O> & a, R init, Op op) {
    static_assert((_axis_in_range(Axes, E::rank()) && ...), "reduction axis out of range");
    using I = typename E::index_type;
    bool red[E::rank()] = {}; ( (red[_norm_axis(Axes, E::rank())] = true), ... );
    cs::array<I, OE::rank()> ke{}; cs::size_t oi = 0;
    for (cs::size_t d = 0; d < E::rank(); ++d) if (!red[d]) ke[oi++] = static_cast<I>(a.extent(d));
    OE oe(ke);
    tensor<R, OE, ccontiguous, storage::heap> out(oe);
    reduce_axes_<R>(out, a, init, op, red, cs::make_index_sequence<E::rank()>{});
    return out;
}

// Cast an axis-reduction RESULT (accumulated in element type `RE`) down to the
// public result element type `Ret`, preserving shape + ownership; a no-op move
// when `Ret == RE`. Two overloads keep the stack path _TNY_API (host+device) and
// the heap path _TNY_HOST, mirroring the `axreduce` overload that produced `r`.
template <class Ret, class RE, class OE>
_TNY_API auto reduce_to(tensor<RE, OE, ccontiguous, storage::stack> && r) {
    if constexpr (cs::is_same<Ret, RE>::value) return static_cast<tensor<RE,OE,ccontiguous,storage::stack>&&>(r);
    else { tensor<Ret, OE, ccontiguous, storage::stack> o{}; o.copy_(r); return o; }
}
template <class Ret, class RE, class OE>
_TNY_HOST auto reduce_to(tensor<RE, OE, ccontiguous, storage::heap> && r) {
    if constexpr (cs::is_same<Ret, RE>::value) return static_cast<tensor<RE,OE,ccontiguous,storage::heap>&&>(r);
    else { tensor<Ret, OE, ccontiguous, storage::heap> o(r.extents()); o.copy_(r); return o; }
}

/* ---- allclose: |a-b| <= atol + rtol*|b| for every (broadcast) element ---- */
template <class R, class A, class B, cs::size_t... D>
_TNY_API bool allclose_(const A & a, const B & b, R rtol, R atol, cs::index_sequence<D...>) {
    // Two read-only operands, so this is the `zipreduce_decode_` situation (#342) — but
    // the pick is the signedness-aware `_offset_int_t`, not the pure width
    // `_wider_index_t`: it is an OFFSET decode type, not the index type of a fresh
    // result (see `_offset_int_t`'s note above). Taking the FIRST operand's index type
    // alone truncated a wider-indexed `b`'s extents/strides, silently — the
    // `static_cast<I>`s below suppress the narrowing diagnostic that caught #342 — and
    // a 40000 stride folded to an int16 -25536 reads off the front of the buffer (#353).
    // A same-signedness pair still resolves to the plain widest, so every ordinary call
    // decodes exactly as before.
    using I = _offset_int_t<typename A::index_type, typename B::index_type>;
    constexpr cs::size_t Rk = sizeof...(D);   // result (broadcast) rank; a,b right-align (left-pad)
    // Array size floored to 1 (rank-0 result -> empty D..., see scal_'s comment above).
    const I ae[Rk ? Rk : 1] = { static_cast<I>(bc_ext<Rk>(a, D))... },
            sa[Rk ? Rk : 1] = { static_cast<I>(bc_str<Rk>(a, D))... };
    const I be[Rk ? Rk : 1] = { static_cast<I>(bc_ext<Rk>(b, D))... },
            sb[Rk ? Rk : 1] = { static_cast<I>(bc_str<Rk>(b, D))... };
    I ce[sizeof...(D) ? sizeof...(D) : 1], n = 1;
    for (cs::size_t r = 0; r < sizeof...(D); ++r) {
        ce[r] = ae[r] == 1 ? be[r] : ae[r];
        _TNY_CHECK(ae[r] == ce[r] || ae[r] == 1, "allclose: lhs extent mismatch");
        _TNY_CHECK(be[r] == ce[r] || be[r] == 1, "allclose: rhs extent mismatch");
        n *= ce[r];
    }
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0, ob = 0;
        for (int d = (int)sizeof...(D)-1; d >= 0; --d) {
            I k = rem % ce[d]; rem /= ce[d];
            oa += (ae[d] == 1 ? I(0) : k) * sa[d]; ob += (be[d] == 1 ? I(0) : k) * sb[d];
        }
        const R av = static_cast<R>(a.data()[oa]), bv = static_cast<R>(b.data()[ob]);
        R diff = av - bv; diff = diff < R(0) ? -diff : diff;
        R mag  = bv < R(0) ? -bv : bv;
        if (diff > atol + rtol * mag) return false;
    }
    return true;
}

} // namespace _md

/* ------------------------------------------------------------------ *
 *     In-place members                                               *
 * ------------------------------------------------------------------ */

// tensor rhs (broadcasts). add_/sub_ take an `Atomic` flag: when true the write
// is `fetch_add` (atomic on host and device, #257) — the scatter/"push"
// accumulate — so the op commits a DELTA (rhs, or -rhs for sub) rather than a
// read-modify-write.
template <class T,class E,class L,storage O> template <bool Atomic, class B, cs::enable_if_t<!cs::is_arithmetic<B>::value,int>>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::add_(const B & b) {
    if constexpr (Atomic) _md::bzip<_md::w_add>(*this,*this,b,_md::rhs{});
    else                  _md::bzip(*this,*this,b,_md::add{});
    return *this;
}
template <class T,class E,class L,storage O> template <bool Atomic, class B, cs::enable_if_t<!cs::is_arithmetic<B>::value,int>>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::sub_(const B & b) {
    if constexpr (Atomic) _md::bzip<_md::w_add>(*this,*this,b,_md::nrhs{});
    else                  _md::bzip(*this,*this,b,_md::sub{});
    return *this;
}
template <class T,class E,class L,storage O> template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value,int>>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::mul_(const B & b) { _md::bzip(*this,*this,b,_md::mul{}); return *this; }
template <class T,class E,class L,storage O> template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value,int>>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::div_(const B & b) { _md::bzip(*this,*this,b,_md::div{}); return *this; }
template <class T,class E,class L,storage O> template <bool Atomic>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::add_(T s) {
    if constexpr (Atomic) _md::scal<_md::w_add>(*this,s,_md::rhs{});
    else                  _md::scal(*this,s,_md::add{});
    return *this;
}
template <class T,class E,class L,storage O> template <bool Atomic>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::sub_(T s) {
    if constexpr (Atomic) _md::scal<_md::w_add>(*this,s,_md::nrhs{});
    else                  _md::scal(*this,s,_md::sub{});
    return *this;
}
template <class T,class E,class L,storage O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::mul_(T s) { _md::scal(*this,s,_md::mul{}); return *this; }
template <class T,class E,class L,storage O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::div_(T s) { _md::scal(*this,s,_md::div{}); return *this; }
// in-place running min/max (#325): *this = min/max(*this, b), tensor rhs broadcasts.
template <class T,class E,class L,storage O> template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value,int>>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::minimum_(const B & b) { _md::bzip(*this,*this,b,_md::b_min{}); return *this; }
template <class T,class E,class L,storage O> template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value,int>>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::maximum_(const B & b) { _md::bzip(*this,*this,b,_md::b_max{}); return *this; }
template <class T,class E,class L,storage O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::minimum_(T s) { _md::scal(*this,s,_md::b_min{}); return *this; }
template <class T,class E,class L,storage O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::maximum_(T s) { _md::scal(*this,s,_md::b_max{}); return *this; }
// fused scaled accumulate (BLAS axpy): *this += alpha*b / *this -= alpha*b (b broadcasts).
template <class T,class E,class L,storage O> template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value,int>>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::add_(const B & b, T alpha) { _md::bzip(*this,*this,b,_md::fma_add<T>{alpha}); return *this; }
template <class T,class E,class L,storage O> template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value,int>>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::sub_(const B & b, T alpha) { _md::bzip(*this,*this,b,_md::fma_sub<T>{alpha}); return *this; }
// atomic accumulate aliases: the readable spelling of add_<true>/sub_<true>
// (atomic-on-device scatter/"push"). Thin forwarders over the Atomic form.
template <class T,class E,class L,storage O> template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value,int>>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::atomic_add_(const B & b) { return add_<true>(b); }
template <class T,class E,class L,storage O> template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value,int>>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::atomic_sub_(const B & b) { return sub_<true>(b); }
template <class T,class E,class L,storage O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::atomic_add_(T s) { return add_<true>(s); }
template <class T,class E,class L,storage O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::atomic_sub_(T s) { return sub_<true>(s); }
template <class T,class E,class L,storage O> template <class B>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::copy_(const B & b) { _md::bzip(*this,*this,b,_md::rhs{}); return *this; }
template <class T,class E,class L,storage O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::fill_(T s) { _md::scal(*this,s,_md::setc{}); return *this; }
template <class T,class E,class L,storage O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::zero_() { return fill_(T(0)); }
template <class T,class E,class L,storage O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::iota_(T start, T step) { _md::iota_(*this, start, step, cs::make_index_sequence<rank()>{}); return *this; }

/* ------------------------------------------------------------------ *
 *     Out-of-place operators                                         *
 *                                                                    *
 *  - fully-static extents  -> stack-owned result, host AND device.   *
 *  - any dynamic extent    -> heap-owned result, HOST ONLY (the      *
 *                             result must be allocated at run time).  *
 * ------------------------------------------------------------------ */

// tensor (op) tensor operators -> the broadcasting out-of-place engine.
#define _TNY_MD_BINOP(SYM, OP)                                                                    \
template <class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>                   \
_TNY_API auto operator SYM (const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b)        \
{ return _md::oop(a, b, OP{}); }
_TNY_MD_BINOP(+, _md::add)
_TNY_MD_BINOP(-, _md::sub)
_TNY_MD_BINOP(*, _md::mul)
_TNY_MD_BINOP(/, _md::div)
#undef _TNY_MD_BINOP

/* --- out-of-place binary methods (tensor OR scalar rhs) -> new, OR into(dest) - */
#define _TNY_MD_METHOD(NAME, OP)                                                                 \
template <class T,class E,class L,storage O> template <class B>                                       \
_TNY_API auto tensor<T,E,L,O>::NAME(const B & b) const {                                          \
    if constexpr (cs::is_arithmetic<B>::value) return _md::oops(*this, b, OP{});                  \
    else                                       return _md::oop (*this, b, OP{});                   \
}                                                                                                 \
template <class T,class E,class L,storage O> template <class B, class D>                              \
_TNY_API auto & tensor<T,E,L,O>::NAME(const B & b, into_t<D> out) const {                          \
    if constexpr (cs::is_arithmetic<B>::value) _md::oops_to(out.dest, *this, b, OP{});             \
    else                                       _md::oop_to (out.dest, *this, b, OP{});             \
    return out.dest;                                                                               \
}
_TNY_MD_METHOD(add, _md::add)
_TNY_MD_METHOD(sub, _md::sub)
_TNY_MD_METHOD(mul, _md::mul)
_TNY_MD_METHOD(div, _md::div)
_TNY_MD_METHOD(pow, _md::pw)
#undef _TNY_MD_METHOD

// fused out-of-place axpy: a + alpha*b / a - alpha*b (b tensor, broadcasts) -> new,
// or into(dest). The in-place twin is add_(b, alpha)/sub_(b, alpha).
#define _TNY_MD_FMA(NAME, F)                                                                      \
template <class T,class E,class L,storage O> template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value,int>> \
_TNY_API auto tensor<T,E,L,O>::NAME(const B & b, T alpha) const { return _md::oop(*this, b, _md::F<T>{alpha}); }   \
template <class T,class E,class L,storage O> template <class B, class D, cs::enable_if_t<!cs::is_arithmetic<B>::value,int>> \
_TNY_API auto & tensor<T,E,L,O>::NAME(const B & b, T alpha, into_t<D> out) const { _md::oop_to(out.dest, *this, b, _md::F<T>{alpha}); return out.dest; }
_TNY_MD_FMA(add, fma_add)
_TNY_MD_FMA(sub, fma_sub)
#undef _TNY_MD_FMA

/* --- tensor (op) scalar and scalar (op) tensor operators ---------- */
#define _TNY_MD_SCALOP(SYM, OP)                                                                   \
template <class T,class E,class L,storage O, class S, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0> \
_TNY_API auto operator SYM (const tensor<T,E,L,O> & a, S s) { return _md::oops(a, s, OP{}); }
_TNY_MD_SCALOP(+, _md::add)
_TNY_MD_SCALOP(-, _md::sub)
_TNY_MD_SCALOP(*, _md::mul)
_TNY_MD_SCALOP(/, _md::div)
#undef _TNY_MD_SCALOP
// scalar (op) tensor. + and * are commutative; - and / need the reversed op
// (s - a, s / a) so `2.0 - a` and `1.0 / a` do the right thing.
template <class S, class T,class E,class L,storage O, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0>
_TNY_API auto operator+(S s, const tensor<T,E,L,O> & a) { return _md::oops(a, s, _md::add{}); }
template <class S, class T,class E,class L,storage O, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0>
_TNY_API auto operator*(S s, const tensor<T,E,L,O> & a) { return _md::oops(a, s, _md::mul{}); }
template <class S, class T,class E,class L,storage O, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0>
_TNY_API auto operator-(S s, const tensor<T,E,L,O> & a) { return _md::oops(a, s, _md::rsub{}); }
template <class S, class T,class E,class L,storage O, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0>
_TNY_API auto operator/(S s, const tensor<T,E,L,O> & a) { return _md::oops(a, s, _md::rdiv{}); }

// unary minus -> a fresh negated tensor.
template <class T,class E,class L,storage O>
_TNY_API auto operator-(const tensor<T,E,L,O> & a) { return _md::uop_out(a, _md::u_neg{}); }

/* --- bitwise operators (INTEGER element types only) --------------- *
 * Out-of-place & | ^ (tensor or scalar rhs), unary ~, and in-place
 * &= |= ^= (free compound-assignment; tensor rhs broadcasts).         */
#define _TNY_MD_BITOP(SYM, OP)                                                                     \
template <class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob,                    \
          cs::enable_if_t<cs::is_integral<Ta>::value && cs::is_integral<Tb>::value, int> = 0>      \
_TNY_API auto operator SYM (const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b)         \
{ return _md::oop(a, b, _md::OP{}); }                                                              \
template <class T,class E,class L,storage O, class S,                                                  \
          cs::enable_if_t<cs::is_integral<T>::value && cs::is_integral<S>::value, int> = 0>        \
_TNY_API auto operator SYM (const tensor<T,E,L,O> & a, S s) { return _md::oops(a, s, _md::OP{}); } \
template <class T,class E,class L,storage O, class B,                                                  \
          cs::enable_if_t<cs::is_integral<T>::value && !cs::is_arithmetic<B>::value, int> = 0>     \
_TNY_API tensor<T,E,L,O> & operator SYM##= (tensor<T,E,L,O> & a, const B & b)                      \
{ _md::bzip(a, a, b, _md::OP{}); return a; }                                                       \
template <class T,class E,class L,storage O, class S,                                                  \
          cs::enable_if_t<cs::is_integral<T>::value && cs::is_integral<S>::value, int> = 0>        \
_TNY_API tensor<T,E,L,O> & operator SYM##= (tensor<T,E,L,O> & a, S s)                              \
{ _md::scal(a, static_cast<T>(s), _md::OP{}); return a; }
_TNY_MD_BITOP(&, b_and)
_TNY_MD_BITOP(|, b_or)
_TNY_MD_BITOP(^, b_xor)
#undef _TNY_MD_BITOP
// unary bitwise NOT -> a fresh tensor.
template <class T,class E,class L,storage O, cs::enable_if_t<cs::is_integral<T>::value, int> = 0>
_TNY_API auto operator~(const tensor<T,E,L,O> & a) { return _md::uop_out(a, _md::u_bnot{}); }

/* --- comparison operators -> a bool tensor (broadcast) ------------- *
 * tensor cmp tensor, tensor cmp scalar, and scalar cmp tensor (the last via the
 * reversed op: `s < a` == `a > s`). == and != are their own reverse.            */
#define _TNY_MD_CMPOP(SYM, OP, ROP)                                                               \
template <class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>                   \
_TNY_API auto operator SYM (const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b)        \
{ return _md::oop_cmp(a, b, _md::OP{}); }                                                         \
template <class T,class E,class L,storage O, class S, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0> \
_TNY_API auto operator SYM (const tensor<T,E,L,O> & a, S s) { return _md::oops_cmp(a, s, _md::OP{}); }   \
template <class S, class T,class E,class L,storage O, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0> \
_TNY_API auto operator SYM (S s, const tensor<T,E,L,O> & a) { return _md::oops_cmp(a, s, _md::ROP{}); }
_TNY_MD_CMPOP(==, c_eq, c_eq)
_TNY_MD_CMPOP(!=, c_ne, c_ne)
_TNY_MD_CMPOP(<,  c_lt, c_gt)
_TNY_MD_CMPOP(<=, c_le, c_ge)
_TNY_MD_CMPOP(>,  c_gt, c_lt)
_TNY_MD_CMPOP(>=, c_ge, c_le)
#undef _TNY_MD_CMPOP

/* --- in-place unary methods --------------------------------------- */
#define _TNY_MD_UNARY_(NAME, F)                                                                   \
template <class T,class E,class L,storage O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::NAME()        \
{ _md::unary(*this, F{}); return *this; }
_TNY_MD_UNARY_(neg_,   _md::u_neg)
_TNY_MD_UNARY_(abs_,   _md::u_abs)
_TNY_MD_UNARY_(exp_,   _md::u_exp)
_TNY_MD_UNARY_(log_,   _md::u_log)
_TNY_MD_UNARY_(sin_,   _md::u_sin)
_TNY_MD_UNARY_(cos_,   _md::u_cos)
_TNY_MD_UNARY_(sqrt_,  _md::u_sqrt)
_TNY_MD_UNARY_(tanh_,  _md::u_tanh)
_TNY_MD_UNARY_(floor_, _md::u_floor)
_TNY_MD_UNARY_(ceil_,  _md::u_ceil)
_TNY_MD_UNARY_(round_, _md::u_round)
_TNY_MD_UNARY_(trunc_, _md::u_trunc)
_TNY_MD_UNARY_(sign_,  _md::u_sign)
#undef _TNY_MD_UNARY_
template <class T,class E,class L,storage O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::pow_(T e)
{ _md::scal(*this, e, _md::pw{}); return *this; }
template <class T,class E,class L,storage O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::clamp_(T lo, T hi)
{ _md::unary(*this, _md::u_clamp{ static_cast<double>(lo), static_cast<double>(hi) }); return *this; }

/* --- generic elementwise with a user functor --------------------- */
template <class T,class E,class L,storage O> template <class F>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::map_(F f) { _md::unary(*this, f); return *this; }
template <class T,class E,class L,storage O> template <class G, class B>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::zip_with_(G g, const B & b) { _md::bzip(*this, *this, b, g); return *this; }
template <class T,class E,class L,storage O> template <class F>
_TNY_API auto tensor<T,E,L,O>::map(F f) const { return _md::uop_out(*this, f); }

/* ------------------------------------------------------------------ *
 *     Reductions                                                      *
 * ------------------------------------------------------------------ */

/**
 * @brief Default accumulator type for a reduction over element type `T`.
 *
 * `double` for floating-point types narrower than `double` (`float`, `half`,
 * `bfloat16`) — enough headroom that summing many low-precision values doesn't
 * lose catastrophically; a floating type already **at least** as wide as
 * `double` (`double` itself, or `long double`) keeps itself — `long double` is
 * `double`-sized on some ABIs (e.g. MSVC, arm64 macOS), where widening it would
 * be a no-op precision-wise but a needless type change. Integer types narrower
 * than 8 bytes accumulate in 64-bit (`int64_t` if
 * signed, `uint64_t` if unsigned — `bool` counts as unsigned) so that summing /
 * multiplying many small integers can't overflow mid-accumulation (signed
 * overflow is UB); integers already ≥8 bytes keep their own type. The RESULT is
 * still cast back to the element type `T` (accumulate wide, cast down); a caller
 * who wants the untruncated wide value uses the explicit accumulator (`sum<int64_t>(a)`).
 * Half types are spotted via `compute_type` (the only `T` whose compute type
 * differs from itself). Override per call, e.g. `sum<float>(a)`.
 */
template <class T>
using reduce_type_t = cs::conditional_t<
    (cs::is_floating_point<T>::value || !cs::is_same<compute_type_t<T>, T>::value),
    cs::conditional_t<(sizeof(T) >= sizeof(double)), T, double>,
    // integer T: widen a narrow item to 64-bit (signed/unsigned to match), keep
    // an already-wide integer as-is. `is_signed` is false for `bool` -> uint64_t.
    cs::conditional_t<(sizeof(T) >= 8), T,
        cs::conditional_t<cs::is_signed<T>::value, cs::int64_t, cs::uint64_t>>>;
// resolve an explicitly-requested accumulator (`void` -> the default above).
template <class Acc, class T>
using _acc_t = cs::conditional_t<cs::is_same<Acc, void>::value, reduce_type_t<T>, Acc>;
// the RESULT element type of a reduction: the tensor's own element type `T` by
// default (accumulate wide, then cast back down — pytorch-like), or the explicit
// accumulator `Acc` when one is given (that IS the requested output dtype).
template <class Acc, class T>
using _reduce_result_t = cs::conditional_t<cs::is_same<Acc, void>::value, T, Acc>;
// `mean`'s default result element type: an INTEGER `T` yields `double` (numpy: the
// mean of an integer array is float64, and it must divide in floating point rather
// than truncate); a floating `T` (incl. half/bfloat16) keeps `T`. An explicit
// accumulator (`mean<float>(a)`) overrides this and is honoured by `_reduce_result_t`.
template <class T>
using _mean_result_t = cs::conditional_t<cs::is_integral<T>::value, double, T>;

// Seeds for max/min reductions. `cs::numeric_limits` is NOT specialized for
// teeny's software half/bfloat16, so `numeric_limits<half>::lowest()` returns the
// value-initialized `half{}` == 0 — a WRONG max seed (a max over all-negative
// halves would return 0, a min over all-positive would return 0). Route the small
// floats through `float`'s ±infinity (a valid identity for max/min: every real
// value beats ±inf, and the software half represents inf), uniform across the
// native (nvcc __half) and portable half types. `R` is the accumulator type.
template <class R> _TNY_API R _reduce_seed_lowest() {
    if constexpr (cs::is_same<R, half>::value || cs::is_same<R, bfloat16>::value)
        return static_cast<R>(-cs::numeric_limits<float>::infinity());
    else
        return cs::numeric_limits<R>::lowest();
}
template <class R> _TNY_API R _reduce_seed_highest() {
    if constexpr (cs::is_same<R, half>::value || cs::is_same<R, bfloat16>::value)
        return static_cast<R>(cs::numeric_limits<float>::infinity());
    else
        return (cs::numeric_limits<R>::max)();
}

// Reductions ACCUMULATE in the accumulator type (`double` by default for small
// floats — see reduce_type_t, so precision holds), then CAST the result back to
// the tensor's element type `T`. Pass an explicit accumulator to make it BOTH the
// accumulation and the result dtype: `sum<float>(a)`, `mean<double, 0>(a)`.

/** @brief Sum of all elements (empty -> 0). Accumulates in the reduce type
 *         (`double` for small floats), result cast to `T`; `sum<Acc>(a)` returns
 *         `Acc`. */
template <class Acc = void, class T, class E, class L, storage O>
_TNY_API auto sum(const tensor<T,E,L,O> & a) {
    using R = _acc_t<Acc, T>;
    return static_cast<_reduce_result_t<Acc,T>>(
        _md::reduce_<R>(a, R(0), _md::r_add{}, cs::make_index_sequence<tensor<T,E,L,O>::rank()>{}));
}
/** @brief Product of all elements (empty -> 1). Accumulates in the reduce type,
 *         result cast to `T`; `prod<Acc>(a)` returns `Acc`. */
template <class Acc = void, class T, class E, class L, storage O>
_TNY_API auto prod(const tensor<T,E,L,O> & a) {
    using R = _acc_t<Acc, T>;
    return static_cast<_reduce_result_t<Acc,T>>(
        _md::reduce_<R>(a, R(1), _md::r_mul{}, cs::make_index_sequence<tensor<T,E,L,O>::rank()>{}));
}
/** @brief Maximum element. Requires a non-empty tensor. Result type `T`
 *         (`max<Acc>(a)` returns `Acc`). */
template <class Acc = void, class T, class E, class L, storage O>
_TNY_API auto max(const tensor<T,E,L,O> & a) {
    using R = _acc_t<Acc, T>;
    return static_cast<_reduce_result_t<Acc,T>>(
        _md::reduce_<R>(a, _reduce_seed_lowest<R>(), _md::r_max{}, cs::make_index_sequence<tensor<T,E,L,O>::rank()>{}));
}
/** @brief Minimum element. Requires a non-empty tensor. Result type `T`
 *         (`min<Acc>(a)` returns `Acc`). */
template <class Acc = void, class T, class E, class L, storage O>
_TNY_API auto min(const tensor<T,E,L,O> & a) {
    using R = _acc_t<Acc, T>;
    return static_cast<_reduce_result_t<Acc,T>>(
        _md::reduce_<R>(a, _reduce_seed_highest<R>(), _md::r_min{}, cs::make_index_sequence<tensor<T,E,L,O>::rank()>{}));
}

// The value-tag accumulator form (`sum(a, dtype<Acc>{})` == `sum<Acc>(a)`), its
// composition with an axis list (`sum(a, axis<0>{}, dtype<Acc>{})`), `keepdims`,
// and `into(dest)` — in any subset, any order — are all handled generically by
// `_TNY_RED_TAGGED` further below (right after the axis-reduction engines it
// dispatches into), for `sum`/`prod`/`max`/`min`/`sqnorm`/`mean`/`norm`.

/* --- boolean reductions (members; chain after a comparison) -------- */
template <class T,class E,class L,storage O> _TNY_API bool tensor<T,E,L,O>::all() const
{ return _md::reduce_<bool>(*this, true,  _md::r_all{}, cs::make_index_sequence<rank()>{}); }
template <class T,class E,class L,storage O> _TNY_API bool tensor<T,E,L,O>::any() const
{ return _md::reduce_<bool>(*this, false, _md::r_any{}, cs::make_index_sequence<rank()>{}); }

/* --- axis reductions: reduce over the named axes -> a lower-rank tensor ---- *
 * `sum<0>(a)`, `mean<0,2>(a)`, ... remove those axes. Static result -> stack
 * (host+device); dynamic -> heap (host only, since it allocates). Reduce over
 * every axis (`sum(a)`) -> the scalar overloads above.                         */
// Two overloads per reduction: a static result is stack-owned (host+device) so
// the wrapper is _TNY_API; a dynamic result is heap-owned (host only) so it is
// _TNY_HOST — matching the `axreduce` overload each resolves to (else nvcc would
// see a _TNY_API wrapper call a __host__ allocator).
// Axis reductions: `NAME<Axes...>(a)` accumulates in the reduce type (double for
// small floats) and returns the tensor's element type `T`; `NAME<Acc, Axes...>(a)`
// makes `Acc` both the accumulator AND the result element type. The two forms are
// told apart by template-arg KIND: a leading non-type (`0`) is an axis, a leading
// type (`double`) is the accumulator. Each splits static (stack, host+device) /
// dynamic (heap, host-only) to match `axreduce`; the result is accumulated in `R`
// then cast to the public element type by `reduce_to`.
// keepdim view fold: insert a size-1 axis at each (ascending, already-normalised)
// position — used by both the generic "finish" step below and axis `normalize`.
// `_axes_ascending(...)` lives in indexing.h (next to `_norm_axis`) — tensor.h's
// multi-axis `unsqueeze<Ax...>`/`squeeze<Ax...>` folds need it too, and tensor.h
// cannot include math.h.
template <class Tn> _TNY_API auto _keepdims(const Tn & t) { return t; }
template <long A0, long... Rest, class Tn> _TNY_API auto _keepdims(const Tn & t) { return _keepdims<Rest...>(t.template unsqueeze<A0>()); }

namespace _md {
/** @brief Shared "finish" step for every axis reduction's generic trailing
 *  keyword bag (`_TNY_RED_TAGGED` below): given the already-computed reduced
 *  tensor `r`, apply `keepdims_t` if present in `Tags...` — re-`unsqueeze` the
 *  named axes (normalised against `SrcRank`, the SOURCE tensor's rank) back in
 *  and materialise into a freshly-owned tensor, exactly as the old hand-written
 *  `_TNY_RED_KEEPDIMS` macro did (the view from `_keepdims`'s recursive
 *  `unsqueeze` fold bakes in a `const` element type via its CONST overload, so
 *  `.clone()` can't be reused — the target is built explicitly with `r`'s own
 *  element type) — then write into `into_t<D>` if present, else return the
 *  (possibly keepdims-wrapped) tensor by value. Two overloads matching the SAME
 *  static(stack,_TNY_API)/dynamic(heap,_TNY_HOST) split as `axreduce` itself. */
template <long SrcRank, long... Axes, class R, class... Tags>
_TNY_API decltype(auto) _red_finish_static(R && r, Tags... tags) {
    auto out = _kw::get<_is_into_tag>(_kw::unset{}, tags...);
    if constexpr (_kw::has<_is_keepdims_tag, Tags...>()) {
        static_assert(_axes_ascending(_norm_axis(Axes, SrcRank)...), "keepdims: axes must be distinct and ascending");
        auto kv = _keepdims<_norm_axis(Axes, SrcRank)...>(r);
        tensor<typename cs::remove_reference_t<R>::element_type, typename decltype(kv)::extents_type, ccontiguous, storage::stack> c{};
        c.copy_(kv);
        if constexpr (!cs::is_same<decltype(out), _kw::unset>::value) { out.dest.copy_(c); return out.dest; }
        else return c;
    } else {
        if constexpr (!cs::is_same<decltype(out), _kw::unset>::value) { out.dest.copy_(r); return out.dest; }
        else return static_cast<cs::remove_reference_t<R>>(static_cast<R&&>(r));  // force a prvalue (remove_reference_t defends against R ever deducing as a reference -- e.g. if a future caller passed an lvalue): decltype(auto) would otherwise deduce R&& from the xvalue cast and dangle once r's temporary is destroyed
    }
}
template <long SrcRank, long... Axes, class R, class... Tags>
_TNY_HOST decltype(auto) _red_finish_dynamic(R && r, Tags... tags) {
    auto out = _kw::get<_is_into_tag>(_kw::unset{}, tags...);
    if constexpr (_kw::has<_is_keepdims_tag, Tags...>()) {
        static_assert(_axes_ascending(_norm_axis(Axes, SrcRank)...), "keepdims: axes must be distinct and ascending");
        auto kv = _keepdims<_norm_axis(Axes, SrcRank)...>(r);
        tensor<typename cs::remove_reference_t<R>::element_type, typename decltype(kv)::extents_type, ccontiguous, storage::heap> c(kv.extents());
        c.copy_(kv);
        if constexpr (!cs::is_same<decltype(out), _kw::unset>::value) { out.dest.copy_(c); return out.dest; }
        else return c;
    } else {
        if constexpr (!cs::is_same<decltype(out), _kw::unset>::value) { out.dest.copy_(r); return out.dest; }
        else return static_cast<cs::remove_reference_t<R>>(static_cast<R&&>(r));  // force a prvalue (remove_reference_t defends against R ever deducing as a reference -- e.g. if a future caller passed an lvalue): decltype(auto) would otherwise deduce R&& from the xvalue cast and dangle once r's temporary is destroyed
    }
}
} // namespace _md

// Axis-reduction "core": `NAME<Axes...>(a, Tags...)` accumulates in the reduce
// type (double for small floats) and returns the tensor's element type `T`;
// `NAME<Acc, Axes...>(a, Tags...)` makes `Acc` both the accumulator AND the
// result element type. The two forms are told apart by template-arg KIND: a
// leading non-type (`0`) is an axis, a leading type (`double`) is the
// accumulator. Each splits static (stack, host+device) / dynamic (heap,
// host-only) to match `axreduce`. `Tags...` is `keepdims_t`/`into_t<D>` in any
// subset/order (`_red_finish_static`/`_dynamic` above apply them); the value-tag
// forms (`NAME(a, axis<Axes...>{})`, `NAME(a, dtype<Acc>{})`, and their
// composition with `keepdims`/`into`) are handled generically by
// `_TNY_RED_TAGGED` further below, which dispatches into this core.
// `Tags...` here is ONLY ever `keepdims_t`/`into_t<D>` (in any subset/order) --
// never `dtype`/`axis`, even when this overload is reached via the tag-driven
// dispatcher below (`_TNY_RED_TAGGED`'s `_NAME_axed` helper), which reconstructs
// a CLEAN forward (keepdims/into only) rather than blindly relaying its own
// original tag pack -- so a stray `dtype<...>`/`axis<...>` (or anything else)
// reaching HERE is always a caller mistake, not legitimate passthrough, and the
// `accepts` guard below can safely reject it instead of silently ignoring it.
#define _TNY_RED_AXIS_CORE(NAME, INIT, OP)                                                              \
template <long... Axes, class T,class E,class L,storage O, class... Tags, class R = reduce_type_t<T>,   \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0> \
_TNY_API  decltype(auto) NAME(const tensor<T,E,L,O> & a, Tags... tags) {                                \
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template known<Tags...>(), #NAME ": unrecognized keyword argument"); \
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template unique<Tags...>(), #NAME ": a keyword was given more than once"); \
    return _md::_red_finish_static<(long)E::rank(), Axes...>(_md::reduce_to<T>(_md::axreduce<Axes...>(a, INIT, _md::OP{})), tags...); } \
template <long... Axes, class T,class E,class L,storage O, class... Tags, class R = reduce_type_t<T>,   \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0> \
_TNY_HOST decltype(auto) NAME(const tensor<T,E,L,O> & a, Tags... tags) {                                \
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template known<Tags...>(), #NAME ": unrecognized keyword argument"); \
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template unique<Tags...>(), #NAME ": a keyword was given more than once"); \
    return _md::_red_finish_dynamic<(long)E::rank(), Axes...>(_md::reduce_to<T>(_md::axreduce<Axes...>(a, INIT, _md::OP{})), tags...); } \
template <class Acc, long... Axes, class T,class E,class L,storage O, class... Tags, class R = Acc,     \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0> \
_TNY_API  decltype(auto) NAME(const tensor<T,E,L,O> & a, Tags... tags) {                                \
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template known<Tags...>(), #NAME ": unrecognized keyword argument"); \
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template unique<Tags...>(), #NAME ": a keyword was given more than once"); \
    return _md::_red_finish_static<(long)E::rank(), Axes...>(_md::reduce_to<Acc>(_md::axreduce<Axes...>(a, INIT, _md::OP{})), tags...); } \
template <class Acc, long... Axes, class T,class E,class L,storage O, class... Tags, class R = Acc,     \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0> \
_TNY_HOST decltype(auto) NAME(const tensor<T,E,L,O> & a, Tags... tags) {                                \
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template known<Tags...>(), #NAME ": unrecognized keyword argument"); \
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template unique<Tags...>(), #NAME ": a keyword was given more than once"); \
    return _md::_red_finish_dynamic<(long)E::rank(), Axes...>(_md::reduce_to<Acc>(_md::axreduce<Axes...>(a, INIT, _md::OP{})), tags...); }
_TNY_RED_AXIS_CORE(sum,    R(0),                 r_add)
_TNY_RED_AXIS_CORE(prod,   R(1),                 r_mul)
_TNY_RED_AXIS_CORE(max,    _reduce_seed_lowest<R>(),  r_max)
_TNY_RED_AXIS_CORE(min,    _reduce_seed_highest<R>(), r_min)
_TNY_RED_AXIS_CORE(sqnorm, R(0),                 r_addsq)   // Σaᵢ² over the named axes (result type = T, like sum)
#undef _TNY_RED_AXIS_CORE

/** @brief Generic trailing keyword-bag entry point, shared by every reduction
 *  with this axis shape (`sum`/`prod`/`max`/`min`/`sqnorm`/`mean`/`norm`; `dot`
 *  has its own, being binary/axis-less): `NAME(a, dtype<Acc>{})`,
 *  `NAME(a, axis<Axes...>{})`, `NAME(a, into(d))`, `NAME<Acc>(a, axis<Axes...>{},
 *  keepdims, into(d))`, ... any SUBSET of `dtype`/`axis`/`keepdims`/`into`, in any
 *  ORDER — replacing the old one-hand-written-overload-per-arrangement approach.
 *  Requires at least one trailing tag (`Tag0`) so it never competes with the
 *  plain `NAME(a)`/`NAME<Acc>(a)` bare overload, or with the explicit-`Axes...`
 *  template overloads above (`_TNY_RED_AXIS_CORE`) — those stay told apart by
 *  template-argument KIND exactly as before (`NAME<0>(a, ...)` binds the
 *  `long... Axes` template; `NAME<double>(a, ...)`/`NAME(a, ...)` bind this one).
 *  Dispatches to the explicit-`Axes...` core via a small per-NAME helper
 *  (`_NAME_axed`) that pattern-matches the discovered `axis<Axes...>` TAG back
 *  into a real non-type template-argument pack (the only way to turn a value tag
 *  into `Axes...` for an explicit-template call) — `_md::_red_dyn` (tensor.h)
 *  computes the same static/dynamic split from that same tag. */
// `_NAME_axed::call` forwards into `_TNY_RED_AXIS_CORE`'s explicit-`Axes...`
// overloads, which only ever accept `keepdims_t`/`into_t<D>` (see the comment
// above `_TNY_RED_AXIS_CORE`) -- so `call` is handed an ALREADY-CLEANED
// `Tags...` (reconstructed by `NAME`'s own body below from just those two
// keywords, never the original `dtype`/`axis` tags), not a blind relay of the
// call site's full pack.
#define _TNY_RED_TAGGED(NAME)                                                                            \
namespace _md {                                                                                          \
template <class Acc, class AxisTag> struct _##NAME##_axed;                                               \
template <class Acc, long A0, long... Rest> struct _##NAME##_axed<Acc, axis<A0, Rest...>> {              \
    template <class T,class E,class L,storage O, class... Tags>                                          \
    static _TNY_API decltype(auto) call(const tensor<T,E,L,O> & a, Tags... tags) {                       \
        if constexpr (cs::is_void<Acc>::value) return NAME<A0,Rest...>(a, tags...);                      \
        else return NAME<Acc,A0,Rest...>(a, tags...);                                                    \
    }                                                                                                     \
};                                                                                                        \
}                                                                                                         \
template <class Acc = void, class T,class E,class L,storage O, class Tag0, class... Tags,                \
          class AxisTag = _kw::find_t<_is_axis_tag, axis<>, Tag0, Tags...>,                              \
          cs::enable_if_t<_md::_red_dyn<E,AxisTag>::value==0, int> = 0>                                  \
_TNY_API  decltype(auto) NAME(const tensor<T,E,L,O> & a, Tag0 tag0, Tags... tags) {                      \
    static_assert(_kw::accepts<_is_dtype,_is_axis_tag,_is_into_tag,_is_keepdims_tag>::template known<Tag0,Tags...>(), \
                  #NAME ": unrecognized keyword argument");                                              \
    static_assert(_kw::accepts<_is_dtype,_is_axis_tag,_is_into_tag,_is_keepdims_tag>::template unique<Tag0,Tags...>(), \
                  #NAME ": a keyword was given more than once");                                         \
    using RAcc = dtype_arg_t<Acc, void, Tag0, Tags...>;                                                  \
    auto out = _kw::get<_is_into_tag>(_kw::unset{}, tag0, tags...);                                      \
    constexpr bool hasInto = !cs::is_same<decltype(out), _kw::unset>::value;                             \
    if constexpr (AxisTag::rank == 0) {                                                                  \
        static_assert(!_kw::has<_is_keepdims_tag, Tag0, Tags...>(),                                      \
                      #NAME ": keepdims requires naming axes (axis<...>{}) -- a full (all-axes) "        \
                      "reduction has no axis left to keep");                                             \
        if constexpr (hasInto) {                                                                         \
            static_assert(cs::remove_reference_t<decltype(out.dest)>::rank() == 0,                       \
                          #NAME ": full reduction into(dest): dest must be rank-0 (a scalar cell); "      \
                          "name the axes -- " #NAME "(a, axis<...>{}, into(dest)) -- for a lower-rank destination"); \
            out.dest.fill_(static_cast<typename cs::remove_reference_t<decltype(out.dest)>::element_type>(NAME<RAcc>(a))); \
            return out.dest;                                                                             \
        } else return NAME<RAcc>(a);                                                                     \
    } else if constexpr (_kw::has<_is_keepdims_tag, Tag0, Tags...>()) {                                  \
        if constexpr (hasInto) return _md::_##NAME##_axed<RAcc, AxisTag>::call(a, keepdims, out);        \
        else                   return _md::_##NAME##_axed<RAcc, AxisTag>::call(a, keepdims);             \
    } else {                                                                                              \
        if constexpr (hasInto) return _md::_##NAME##_axed<RAcc, AxisTag>::call(a, out);                  \
        else                   return _md::_##NAME##_axed<RAcc, AxisTag>::call(a);                       \
    }                                                                                                      \
}                                                                                                          \
template <class Acc = void, class T,class E,class L,storage O, class Tag0, class... Tags,                \
          class AxisTag = _kw::find_t<_is_axis_tag, axis<>, Tag0, Tags...>,                              \
          cs::enable_if_t<_md::_red_dyn<E,AxisTag>::value!=0, int> = 0>                                  \
_TNY_HOST decltype(auto) NAME(const tensor<T,E,L,O> & a, Tag0 tag0, Tags... tags) {                      \
    static_assert(_kw::accepts<_is_dtype,_is_axis_tag,_is_into_tag,_is_keepdims_tag>::template known<Tag0,Tags...>(), \
                  #NAME ": unrecognized keyword argument");                                              \
    static_assert(_kw::accepts<_is_dtype,_is_axis_tag,_is_into_tag,_is_keepdims_tag>::template unique<Tag0,Tags...>(), \
                  #NAME ": a keyword was given more than once");                                         \
    using RAcc = dtype_arg_t<Acc, void, Tag0, Tags...>;                                                  \
    auto out = _kw::get<_is_into_tag>(_kw::unset{}, tag0, tags...);                                      \
    constexpr bool hasInto = !cs::is_same<decltype(out), _kw::unset>::value;                             \
    if constexpr (_kw::has<_is_keepdims_tag, Tag0, Tags...>()) {                                         \
        if constexpr (hasInto) return _md::_##NAME##_axed<RAcc, AxisTag>::call(a, keepdims, out);        \
        else                   return _md::_##NAME##_axed<RAcc, AxisTag>::call(a, keepdims);             \
    } else {                                                                                              \
        if constexpr (hasInto) return _md::_##NAME##_axed<RAcc, AxisTag>::call(a, out);                  \
        else                   return _md::_##NAME##_axed<RAcc, AxisTag>::call(a);                       \
    }                                                                                                      \
}
_TNY_RED_TAGGED(sum) _TNY_RED_TAGGED(prod) _TNY_RED_TAGGED(max) _TNY_RED_TAGGED(min) _TNY_RED_TAGGED(sqnorm)
// _TNY_RED_TAGGED(mean)/(norm) are invoked after their own axis-core definitions
// below (same shape, so the macro applies unchanged); #undef after norm's.

/** @brief Mean over the named axes -> a lower-rank tensor (sum / reduced count).
 *         For a floating `T`, accumulates in the reduce type and the result is cast
 *         to `T`. For an INTEGER `T` the result element type is `double` (numpy:
 *         integer mean is float64; divides in `double`, not truncating). `mean<Acc,
 *         Axes...>(a)` makes `Acc` both the accumulator and result type. */
// `Tags...` (keepdims/into, any subset/order) via the shared `_red_finish_*`
// helpers, exactly like `_TNY_RED_AXIS_CORE`; mean's own int-vs-float branching
// stays hand-written since it isn't shared by any other reduction.
template <long... Axes, class T,class E,class L,storage O, class... Tags, class R = reduce_type_t<T>,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0>
_TNY_API  decltype(auto) mean(const tensor<T,E,L,O> & a, Tags... tags) {
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template known<Tags...>(), "mean: unrecognized keyword argument");
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template unique<Tags...>(), "mean: a keyword was given more than once");
    auto s = sum<R, Axes...>(a);                                          // sum in the (wide) reduce type
    const auto cnt = a.numel() / s.numel();
    if constexpr (cs::is_integral<T>::value) {                           // integer -> divide in double, return double
        auto o = _md::reduce_to<double>(static_cast<decltype(s)&&>(s));
        o.div_(static_cast<double>(cnt));
        return _md::_red_finish_static<(long)E::rank(), Axes...>(static_cast<decltype(o)&&>(o), tags...);
    } else {
        s.div_(static_cast<R>(cnt));
        auto o = _md::reduce_to<T>(static_cast<decltype(s)&&>(s));
        return _md::_red_finish_static<(long)E::rank(), Axes...>(static_cast<decltype(o)&&>(o), tags...);
    }
}
template <long... Axes, class T,class E,class L,storage O, class... Tags, class R = reduce_type_t<T>,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0>
_TNY_HOST decltype(auto) mean(const tensor<T,E,L,O> & a, Tags... tags) {
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template known<Tags...>(), "mean: unrecognized keyword argument");
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template unique<Tags...>(), "mean: a keyword was given more than once");
    auto s = sum<R, Axes...>(a);                                          // sum in the (wide) reduce type
    const auto cnt = a.numel() / s.numel();
    if constexpr (cs::is_integral<T>::value) {                           // integer -> divide in double, return double
        auto o = _md::reduce_to<double>(static_cast<decltype(s)&&>(s));
        o.div_(static_cast<double>(cnt));
        return _md::_red_finish_dynamic<(long)E::rank(), Axes...>(static_cast<decltype(o)&&>(o), tags...);
    } else {
        s.div_(static_cast<R>(cnt));
        auto o = _md::reduce_to<T>(static_cast<decltype(s)&&>(s));
        return _md::_red_finish_dynamic<(long)E::rank(), Axes...>(static_cast<decltype(o)&&>(o), tags...);
    }
}
template <class Acc, long... Axes, class T,class E,class L,storage O, class... Tags,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0>
_TNY_API  decltype(auto) mean(const tensor<T,E,L,O> & a, Tags... tags) {
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template known<Tags...>(), "mean: unrecognized keyword argument");
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template unique<Tags...>(), "mean: a keyword was given more than once");
    auto s = sum<Acc, Axes...>(a); s.div_(static_cast<Acc>(a.numel() / s.numel()));
    return _md::_red_finish_static<(long)E::rank(), Axes...>(static_cast<decltype(s)&&>(s), tags...);
}
template <class Acc, long... Axes, class T,class E,class L,storage O, class... Tags,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0>
_TNY_HOST decltype(auto) mean(const tensor<T,E,L,O> & a, Tags... tags) {
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template known<Tags...>(), "mean: unrecognized keyword argument");
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template unique<Tags...>(), "mean: a keyword was given more than once");
    auto s = sum<Acc, Axes...>(a); s.div_(static_cast<Acc>(a.numel() / s.numel()));
    return _md::_red_finish_dynamic<(long)E::rank(), Axes...>(static_cast<decltype(s)&&>(s), tags...);
}
_TNY_RED_TAGGED(mean)

/** @brief Inner product over matching extents. Accumulates in the reduce type of
 *         the promoted element type (`double` for small floats), result cast to
 *         `promote(Ta,Tb)`; `dot<Acc>(a, b)` returns `Acc`. */
template <class Acc = void, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto dot(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b) {
    static_assert(tensor<Ta,Ea,La,Oa>::rank() == tensor<Tb,Eb,Lb,Ob>::rank(), "dot: rank mismatch");
    static_assert(_md::ext_static_eq<Ea, Eb>(cs::make_index_sequence<Ea::rank()>{}),
                  "dot: operand extents must match exactly (no broadcast)");   // both-static, unequal -> compile error
    using R = _acc_t<Acc, promote_t<Ta,Tb>>;
    return static_cast<_reduce_result_t<Acc, promote_t<Ta,Tb>>>(
        _md::zipreduce_<R>(a, b, _md::mul{}, cs::make_index_sequence<tensor<Ta,Ea,La,Oa>::rank()>{}));
}
/** @brief Generic trailing keyword bag for `dot` (no axis concept, being binary):
 *  `dot(a, b, dtype<Acc>{})`, `dot(a, b, into(d))`, or both composed in either
 *  order — `dot`'s own small twin of `_TNY_RED_TAGGED` above (skips the
 *  axis-tag/`_red_dyn` machinery entirely, since dot always reduces every
 *  matching axis). Requires at least one trailing tag so it never competes with
 *  the plain `dot<Acc=void>(a, b)` above. */
template <class Acc = void, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob,
          class Tag0, class... Tags>
_TNY_API decltype(auto) dot(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b, Tag0 tag0, Tags... tags) {
    static_assert(_kw::accepts<_is_dtype,_is_into_tag>::template known<Tag0,Tags...>(), "dot: unrecognized keyword argument");
    static_assert(_kw::accepts<_is_dtype,_is_into_tag>::template unique<Tag0,Tags...>(), "dot: a keyword was given more than once");
    using RAcc = dtype_arg_t<Acc, void, Tag0, Tags...>;
    auto out = _kw::get<_is_into_tag>(_kw::unset{}, tag0, tags...);
    if constexpr (!cs::is_same<decltype(out), _kw::unset>::value) {
        static_assert(cs::remove_reference_t<decltype(out.dest)>::rank() == 0, "dot into(dest): dest must be rank-0 (a scalar cell)");
        out.dest.fill_(static_cast<typename cs::remove_reference_t<decltype(out.dest)>::element_type>(dot<RAcc>(a, b)));
        return out.dest;
    } else return dot<RAcc>(a, b);
}

/* ------------------------------------------------------------------ *
 *     Vector algebra & geometry (contained exact math)               *
 * ------------------------------------------------------------------ */

/** @brief Squared Euclidean norm — the sum of squares `Σ aᵢ²`, over ALL axes.
 *         Just `dot(a, a)`: accumulates in the reduce type, result cast to the
 *         element type (`sqnorm<Acc>(a)` accumulates AND returns `Acc`). The
 *         value-tag/axis/`keepdims`/`into` composition (`sqnorm(a, dtype<Acc>{})`,
 *         `sqnorm(a, axis<0>{})`, ...) is handled generically by
 *         `_TNY_RED_TAGGED` (invoked further below, right after `sqnorm`'s own
 *         axis core — see `_TNY_RED_AXIS_CORE(sqnorm, ...)` above). */
template <class Acc = void, class T, class E, class L, storage O>
_TNY_API auto sqnorm(const tensor<T,E,L,O> & a) { return dot<Acc>(a, a); }

/** @brief Euclidean (L2) norm `√Σ aᵢ²`, over ALL axes. Accumulates the squares in
 *         the reduce type and takes the root there, then casts to the result type:
 *         a floating element type keeps its type, an INTEGER one yields `double`
 *         (numpy/`mean` rule). `norm<Acc>(a)` makes `Acc` accumulator AND result. */
template <class Acc = void, class T, class E, class L, storage O>
_TNY_API auto norm(const tensor<T,E,L,O> & a) {
    using Res = _reduce_result_t<Acc, _mean_result_t<T>>;   // floating result (double for integer T)
    using R   = _acc_t<Acc, T>;                             // accumulate the squares in the reduce type
    using D   = cs::conditional_t<cs::is_floating_point<R>::value, R,
                cs::conditional_t<cs::is_floating_point<Res>::value, Res, double>>;   // take the root in a float type
    return static_cast<Res>(cs::sqrt(static_cast<D>(sqnorm<R>(a))));
}

/* --- axis norm: √(Σaᵢ² over the named axes) -> a lower-rank tensor. Floating result
 *     (integer -> double, mean rule); norm<Acc,Axes...> makes Acc accumulator+result.
 *     Accumulates the squares in a floating type and takes the root there.
 *     `Tags...` (keepdims/into, any subset/order) via `_red_finish_*`, exactly
 *     like `_TNY_RED_AXIS_CORE` — hand-written since norm's deduced Res/R/D
 *     differ from every other reduction's. Reducing (sqrt, cast to Res) BEFORE
 *     applying `_red_finish_*` rather than after is equivalent: `unsqueeze` only
 *     reshapes (no data change), so it commutes with the elementwise sqrt/cast
 *     that follow it either way. --------- */
template <long... Axes, class T,class E,class L,storage O, class... Tags,
          class Res = _mean_result_t<T>, class R = reduce_type_t<T>,
          class D   = cs::conditional_t<cs::is_floating_point<R>::value, R, double>,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0>
_TNY_API  decltype(auto) norm(const tensor<T,E,L,O> & a, Tags... tags) {
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template known<Tags...>(), "norm: unrecognized keyword argument");
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template unique<Tags...>(), "norm: a keyword was given more than once");
    auto s = sqnorm<D, Axes...>(a); s.sqrt_();
    auto r = _md::reduce_to<Res>(static_cast<decltype(s)&&>(s));
    return _md::_red_finish_static<(long)E::rank(), Axes...>(static_cast<decltype(r)&&>(r), tags...);
}
template <long... Axes, class T,class E,class L,storage O, class... Tags,
          class Res = _mean_result_t<T>, class R = reduce_type_t<T>,
          class D   = cs::conditional_t<cs::is_floating_point<R>::value, R, double>,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0>
_TNY_HOST decltype(auto) norm(const tensor<T,E,L,O> & a, Tags... tags) {
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template known<Tags...>(), "norm: unrecognized keyword argument");
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template unique<Tags...>(), "norm: a keyword was given more than once");
    auto s = sqnorm<D, Axes...>(a); s.sqrt_();
    auto r = _md::reduce_to<Res>(static_cast<decltype(s)&&>(s));
    return _md::_red_finish_dynamic<(long)E::rank(), Axes...>(static_cast<decltype(r)&&>(r), tags...);
}
template <class Acc, long... Axes, class T,class E,class L,storage O, class... Tags,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0>
_TNY_API  decltype(auto) norm(const tensor<T,E,L,O> & a, Tags... tags) {
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template known<Tags...>(), "norm: unrecognized keyword argument");
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template unique<Tags...>(), "norm: a keyword was given more than once");
    auto s = sqnorm<Acc, Axes...>(a); s.sqrt_();
    return _md::_red_finish_static<(long)E::rank(), Axes...>(static_cast<decltype(s)&&>(s), tags...);
}
template <class Acc, long... Axes, class T,class E,class L,storage O, class... Tags,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0>
_TNY_HOST decltype(auto) norm(const tensor<T,E,L,O> & a, Tags... tags) {
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template known<Tags...>(), "norm: unrecognized keyword argument");
    static_assert(_kw::accepts<_is_into_tag,_is_keepdims_tag>::template unique<Tags...>(), "norm: a keyword was given more than once");
    auto s = sqnorm<Acc, Axes...>(a); s.sqrt_();
    return _md::_red_finish_dynamic<(long)E::rank(), Axes...>(static_cast<decltype(s)&&>(s), tags...);
}
_TNY_RED_TAGGED(norm)
#undef _TNY_RED_TAGGED

/** @brief Squared Euclidean distance `Σ(aᵢ-bᵢ)²` between two same-shape tensors —
 *         mathematically `sqnorm(a-b)`, computed as one fused pass with no `a-b`
 *         intermediate (mirrors `dot`'s convenience-wrapper status over a manual
 *         `sum(a*b)`). Each difference is formed and squared directly in the
 *         accumulator type, so the result can be MORE accurate than the un-fused
 *         `sqnorm(a-b)` spelling for a narrow element type (`a-b` there rounds to
 *         the operands' own type before `sqnorm` widens it) — not necessarily
 *         bit-identical, only for `double` operands are the two guaranteed equal.
 *         Binary only (no axis-list form, like `dot`); `sqdist<Acc>(a,b)` makes
 *         `Acc` accumulator AND result. */
template <class Acc = void, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto sqdist(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b) {
    static_assert(tensor<Ta,Ea,La,Oa>::rank() == tensor<Tb,Eb,Lb,Ob>::rank(), "sqdist: rank mismatch");
    static_assert(_md::ext_static_eq<Ea, Eb>(cs::make_index_sequence<Ea::rank()>{}),
                  "sqdist: operand extents must match exactly (no broadcast)");   // both-static, unequal -> compile error
    using R = _acc_t<Acc, promote_t<Ta,Tb>>;
    return static_cast<_reduce_result_t<Acc, promote_t<Ta,Tb>>>(
        _md::zipreduce_<R>(a, b, _md::zip_sqdiff{}, cs::make_index_sequence<tensor<Ta,Ea,La,Oa>::rank()>{}));
}

/** @brief Euclidean distance `√Σ(aᵢ-bᵢ)²` — mathematically `norm(a-b)`, one fused
 *         pass (see `sqdist`'s doc comment for the accuracy note). Floating result
 *         (integer operands -> `double`, the `norm`/`mean` rule); `dist<Acc>(a,b)`
 *         makes `Acc` accumulator AND result. */
template <class Acc = void, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto dist(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b) {
    using P   = promote_t<Ta,Tb>;
    using Res = _reduce_result_t<Acc, _mean_result_t<P>>;   // floating result (double for integer P)
    using R   = _acc_t<Acc, P>;                             // accumulate the squares in the reduce type
    using D   = cs::conditional_t<cs::is_floating_point<R>::value, R,
                cs::conditional_t<cs::is_floating_point<Res>::value, Res, double>>;   // take the root in a float type
    return static_cast<Res>(cs::sqrt(static_cast<D>(sqdist<R>(a, b))));
}

// dtype/into trailing-bag form (dot's shape: binary, no axis, so no _TNY_RED_TAGGED).
template <class Acc = void, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob,
          class Tag0, class... Tags>
_TNY_API decltype(auto) sqdist(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b, Tag0 tag0, Tags... tags) {
    static_assert(_kw::accepts<_is_dtype,_is_into_tag>::template known<Tag0,Tags...>(), "sqdist: unrecognized keyword argument");
    static_assert(_kw::accepts<_is_dtype,_is_into_tag>::template unique<Tag0,Tags...>(), "sqdist: a keyword was given more than once");
    using RAcc = dtype_arg_t<Acc, void, Tag0, Tags...>;
    auto out = _kw::get<_is_into_tag>(_kw::unset{}, tag0, tags...);
    if constexpr (!cs::is_same<decltype(out), _kw::unset>::value) {
        static_assert(cs::remove_reference_t<decltype(out.dest)>::rank() == 0, "sqdist into(dest): dest must be rank-0 (a scalar cell)");
        out.dest.fill_(static_cast<typename cs::remove_reference_t<decltype(out.dest)>::element_type>(sqdist<RAcc>(a, b)));
        return out.dest;
    } else return sqdist<RAcc>(a, b);
}
template <class Acc = void, class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob,
          class Tag0, class... Tags>
_TNY_API decltype(auto) dist(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b, Tag0 tag0, Tags... tags) {
    static_assert(_kw::accepts<_is_dtype,_is_into_tag>::template known<Tag0,Tags...>(), "dist: unrecognized keyword argument");
    static_assert(_kw::accepts<_is_dtype,_is_into_tag>::template unique<Tag0,Tags...>(), "dist: a keyword was given more than once");
    using RAcc = dtype_arg_t<Acc, void, Tag0, Tags...>;
    auto out = _kw::get<_is_into_tag>(_kw::unset{}, tag0, tags...);
    if constexpr (!cs::is_same<decltype(out), _kw::unset>::value) {
        static_assert(cs::remove_reference_t<decltype(out.dest)>::rank() == 0, "dist into(dest): dest must be rank-0 (a scalar cell)");
        out.dest.fill_(static_cast<typename cs::remove_reference_t<decltype(out.dest)>::element_type>(dist<RAcc>(a, b)));
        return out.dest;
    } else return dist<RAcc>(a, b);
}

/** @brief Out-of-place unit vector `a / norm(a)` -> a NEW dense tensor (static
 *         shape -> stack, dynamic -> heap). The result element type is floating
 *         (integer input -> `double`, like `norm`). A zero vector yields NaNs
 *         (no epsilon — exact math; add one at the call site if you need it). */
template <class T, class E, class L, storage O>
_TNY_API auto normalize(const tensor<T,E,L,O> & a) {
    // Divide by a real ARITHMETIC scalar so `.div` takes its scalar path (a `half`
    // divisor is not `is_arithmetic` and would look like a tensor rhs); the compute
    // type of the float result is that scalar (`half`->`float`), and promotion then
    // gives back the intended element type (half/float -> that float, int -> double).
    // `.div` itself splits static->stack / dynamic->heap.
    using S = compute_type_t<_mean_result_t<T>>;
    return a.div(static_cast<S>(norm(a)));
}
/** @brief `normalize(a, into(y))` — the unit vector into a caller buffer `y`. */
template <class T, class E, class L, storage O, class D>
_TNY_API auto & normalize(const tensor<T,E,L,O> & a, into_t<D> out) {
    using S = compute_type_t<_mean_result_t<T>>;
    return a.div(static_cast<S>(norm(a)), out);   // .div's into overload writes out & returns out.dest
}

/* --- axis normalize: divide each sub-vector by its norm over the named axes ------ *
 * `n = norm<Axes...>(a)` removes the reduced axes; restore them as size-1 (keepdim)
 * so it broadcasts back over `a`. Inserting size-1 axes at ascending positions (each
 * unsqueeze grows the rank for the next), so the axes must be distinct & ascending.
 * `_keepdims` itself (used here and by the reduction `keepdims` overloads) lives
 * earlier in this file, right before the axis-reduction section that needs it first. */

/** @brief `normalize<Axes...>(a)` — unit vectors along the named axes: each element
 *         divided by the L2 norm over those axes (keepdim broadcast). Floating result
 *         (integer -> double). Axes distinct & ascending (numpy-normalised). */
template <long... Axes, class T, class E, class L, storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0), int> = 0>
_TNY_API auto normalize(const tensor<T,E,L,O> & a) {
    static_assert(_axes_ascending(_norm_axis(Axes, (long)E::rank())...), "normalize: axes must be distinct and ascending");
    auto n = norm<Axes...>(a);                                          // reduced norm (floating tensor)
    return a.div(_keepdims<_norm_axis(Axes, (long)E::rank())...>(n));   // broadcast-divide (keepdim)
}
// value form: normalize(a, axis<Axes...>{}) == normalize<Axes...>(a)
template <long... Axes, class T, class E, class L, storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0), int> = 0>
_TNY_API auto normalize(const tensor<T,E,L,O> & a, axis<Axes...>) { return normalize<Axes...>(a); }

// internal: 3D cross product a × b into `out` (all rank-1, length 3). Computes the
// three components into temporaries FIRST, so `out` may alias `a` or `b` (this is
// what makes both `cross` and the in-place `cross_` — where out IS a — safe). Runs
// in the compute type (`half` in float). Not public: use `cross`/`cross_` below.
template <class To,class Eo,class Lo,storage Oo,
          class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>
_TNY_API void _cross3(tensor<To,Eo,Lo,Oo> & out,
                      const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b) {
    static_assert(Eo::rank() == 1 && Ea::rank() == 1 && Eb::rank() == 1, "cross: operands must be rank-1 3-vectors");
    static_assert(Ea::static_extent(0) == 3 || Ea::static_extent(0) == cs::dynamic_extent, "cross: a must have length 3");
    static_assert(Eb::static_extent(0) == 3 || Eb::static_extent(0) == cs::dynamic_extent, "cross: b must have length 3");
    static_assert(Eo::static_extent(0) == 3 || Eo::static_extent(0) == cs::dynamic_extent, "cross: out must have length 3");
    _TNY_CHECK(a.extent(0) == 3 && b.extent(0) == 3 && out.extent(0) == 3, "cross: 3-vectors required");
    using C = compute_type_t<promote_t<Ta,Tb>>;
    const C a0 = static_cast<C>(a(0)), a1 = static_cast<C>(a(1)), a2 = static_cast<C>(a(2));
    const C b0 = static_cast<C>(b(0)), b1 = static_cast<C>(b(1)), b2 = static_cast<C>(b(2));
    out(0) = static_cast<To>(a1 * b2 - a2 * b1);
    out(1) = static_cast<To>(a2 * b0 - a0 * b2);
    out(2) = static_cast<To>(a0 * b1 - a1 * b0);
}

/** @brief 3D cross product `a × b` -> a NEW stack 3-vector of `promote(Ta,Tb)`.
 *         Both operands are rank-1, length 3. In place: the member `a.cross_(b)`
 *         (`a` becomes `a × b`). Into a preallocated slot: `cross(a, b, into(y))`. */
template <class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto cross(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b) {
    tensor<promote_t<Ta,Tb>, shape<3>, ccontiguous, storage::stack> c(_uninit);
    _cross3(c, a, b);
    return c;
}
/** @brief `cross(a, b, into(y))` — the cross product into a caller buffer `y`
 *         (rank-1, length 3); `y` may alias `a` or `b`. This is ff's "crossto".
 *         `y` may be a SLICE of a bigger output, written with no named
 *         intermediate: `cross(a, b, into(N(i, all)))` fills row `i` of a matrix
 *         of 3-vectors (`into()` binds such a temporary view — tensor.h). */
template <class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob, class D>
_TNY_API auto & cross(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b, into_t<D> out) {
    _cross3(out.dest, a, b);
    return out.dest;
}

// in-place 3D cross product: *this becomes (*this) × b (rank-1, length 3). Safe
// because _cross3 buffers the three components before storing (out aliases a).
template <class T,class E,class L,storage O> template <class Tb,class Eb,class Lb,storage Ob>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::cross_(const tensor<Tb,Eb,Lb,Ob> & b) {
    _cross3(*this, *this, b);
    return *this;
}

// in-place unit vector: *this /= norm(*this). Floating element types only (an
// integer element would truncate the division). Defined here so `norm` (above)
// is visible. A zero vector yields NaNs (no epsilon; add one at the call site).
template <class T,class E,class L,storage O>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::normalize_() {
    static_assert(cs::is_floating_point<compute_type_t<T>>::value,
                  "normalize_: requires a floating-point element type (integer division would truncate)");
    return div_(static_cast<T>(tny::norm(*this)));   // tny:: — the member norm() now shadows the free one here
}
// in-place unit vectors along the named axes: *this /= norm(*this over Axes) (keepdim).
template <class T,class E,class L,storage O> template <long... Axes>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::normalize_() {
    static_assert(cs::is_floating_point<compute_type_t<T>>::value,
                  "normalize_: requires a floating-point element type (integer division would truncate)");
    static_assert(sizeof...(Axes) > 0, "normalize_<Axes...>: need at least one axis");
    static_assert(_axes_ascending(_norm_axis(Axes, (long)rank())...), "normalize_: axes must be distinct and ascending");
    auto n = tny::norm<Axes...>(*this);                                          // reduced norm (floating tensor)
    return div_(_keepdims<_norm_axis(Axes, (long)rank())...>(n));                // broadcast-divide (keepdim)
}

/** @brief True if every element satisfies `|a-b| <= atol + rtol*|b|` (numpy
 *         `allclose`; broadcasts, computes in the compute type). */
template <class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>
_TNY_API bool allclose(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b,
                       double rtol = 1e-5, double atol = 1e-8) {
    constexpr cs::size_t Rk = _md::bc_rank(Ea::rank(), Eb::rank());   // broadcast (left-pad)
    static_assert(_md::bc_static_ok_r<Ea, Eb, Rk>(cs::make_index_sequence<Rk>{}), "allclose: incompatible static extents");
    using R = compute_type_t<promote_t<Ta,Tb>>;
    return _md::allclose_<R>(a, b, static_cast<R>(rtol), static_cast<R>(atol),
                            cs::make_index_sequence<Rk>{});
}

/* --- out-of-place unary free functions ---------------------------- */
#define _TNY_MD_UNARY(NAME, F)                                                                    \
template <class T,class E,class L,storage O> _TNY_API auto NAME(const tensor<T,E,L,O> & a)             \
{ return _md::uop_out(a, F{}); }                                                                   \
template <class T,class E,class L,storage O, class D>                                                  \
_TNY_API auto & NAME(const tensor<T,E,L,O> & a, into_t<D> out) { _md::uop_to(out.dest, a, F{}); return out.dest; }
_TNY_MD_UNARY(neg,   _md::u_neg)
_TNY_MD_UNARY(abs,   _md::u_abs)
_TNY_MD_UNARY(exp,   _md::u_exp)
_TNY_MD_UNARY(log,   _md::u_log)
_TNY_MD_UNARY(sin,   _md::u_sin)
_TNY_MD_UNARY(cos,   _md::u_cos)
_TNY_MD_UNARY(sqrt,  _md::u_sqrt)
_TNY_MD_UNARY(tanh,  _md::u_tanh)
_TNY_MD_UNARY(floor, _md::u_floor)
_TNY_MD_UNARY(ceil,  _md::u_ceil)
_TNY_MD_UNARY(round, _md::u_round)
_TNY_MD_UNARY(trunc, _md::u_trunc)
_TNY_MD_UNARY(sign,  _md::u_sign)
#undef _TNY_MD_UNARY

/* --- binary minimum/maximum (broadcast) + clamp -> new tensor ------ */
template <class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto minimum(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b) { return _md::oop(a, b, _md::b_min{}); }
template <class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto maximum(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b) { return _md::oop(a, b, _md::b_max{}); }
template <class T,class E,class L,storage O, class S, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0>
_TNY_API auto minimum(const tensor<T,E,L,O> & a, S s) { return _md::oops(a, s, _md::b_min{}); }
template <class T,class E,class L,storage O, class S, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0>
_TNY_API auto maximum(const tensor<T,E,L,O> & a, S s) { return _md::oops(a, s, _md::b_max{}); }
// ... into(dest): minimum/maximum, tensor rhs (broadcasts) or scalar rhs.
template <class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob, class D>
_TNY_API auto & minimum(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b, into_t<D> out) { _md::oop_to(out.dest, a, b, _md::b_min{}); return out.dest; }
template <class Ta,class Ea,class La,storage Oa, class Tb,class Eb,class Lb,storage Ob, class D>
_TNY_API auto & maximum(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b, into_t<D> out) { _md::oop_to(out.dest, a, b, _md::b_max{}); return out.dest; }
template <class T,class E,class L,storage O, class S, class D, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0>
_TNY_API auto & minimum(const tensor<T,E,L,O> & a, S s, into_t<D> out) { _md::oops_to(out.dest, a, s, _md::b_min{}); return out.dest; }
template <class T,class E,class L,storage O, class S, class D, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0>
_TNY_API auto & maximum(const tensor<T,E,L,O> & a, S s, into_t<D> out) { _md::oops_to(out.dest, a, s, _md::b_max{}); return out.dest; }
/** @brief `clamp(a, lo, hi)` -> a new tensor with each element clamped; `clamp(a,
 *         lo, hi, into(y))` writes into `y`. */
template <class T,class E,class L,storage O>
_TNY_API auto clamp(const tensor<T,E,L,O> & a, T lo, T hi) { return a.map(_md::u_clamp{ static_cast<double>(lo), static_cast<double>(hi) }); }
template <class T,class E,class L,storage O, class D>
_TNY_API auto & clamp(const tensor<T,E,L,O> & a, T lo, T hi, into_t<D> out) { _md::uop_to(out.dest, a, _md::u_clamp{ static_cast<double>(lo), static_cast<double>(hi) }); return out.dest; }

/** @brief Arithmetic mean of all elements. For a floating `T`, accumulates in the
 *         reduce type (`double` for small floats) and the result is cast to `T`.
 *         For an INTEGER `T` the result is `double` (numpy: integer mean is
 *         float64; the division runs in `double`, not truncating integer division).
 *         `mean<Acc>(a)` makes `Acc` both the accumulator and the result type. */
template <class Acc = void, class T, class E, class L, storage O>
_TNY_API auto mean(const tensor<T,E,L,O> & a) {
    using Res = _reduce_result_t<Acc, _mean_result_t<T>>;   // result dtype (double for integer T)
    using R   = _acc_t<Acc, T>;                             // sum accumulator (wide for narrow ints)
    // Divide in a floating type whenever the result is floating (so integer means
    // keep their fractional part); an explicit integer `Acc` still divides in `Acc`.
    using D = cs::conditional_t<cs::is_floating_point<R>::value, R,
              cs::conditional_t<cs::is_floating_point<Res>::value, Res, R>>;
    const D m = static_cast<D>(sum<R>(a)) / static_cast<D>(a.numel());
    return static_cast<Res>(m);
}
// `mean(a, dtype<Acc>{})`, `into(dest)`, and their composition with an axis list
// / `keepdims` are handled generically by `_TNY_RED_TAGGED(mean)` above (right
// after mean's own axis core) — same as sum/prod/max/min/sqnorm/norm. `dot`'s
// own bare `into(dest)`/`dtype<Acc>` composition is handled by its dedicated
// Tags... overload just above (`dot`'s twin of `_TNY_RED_TAGGED`, since it has
// no axis concept).

/* ------------------------------------------------------------------ *
 *     Reductions AS METHODS (parity with the free sum(a)/dot(a,b))   *
 *     — thin forwarders to the free forms above; declared in         *
 *     tensor.h. Same overload shapes (full / axis / Acc / value /    *
 *     into), told apart the same way the free functions are.         *
 * ------------------------------------------------------------------ */
// The AXIS forms come in the same _TNY_API (static result -> stack) / _TNY_HOST
// (dynamic result -> heap) pairs as the free functions they forward to, keyed on
// the identical condition (declared in tensor.h) — so a device-callable method
// never forwards to a host-only allocator. `_TNY_RED_AXIS_IF` repeats the
// declaration's key WITHOUT the `= 0` default (out-of-line definitions may not
// restate default template arguments).
#define _TNY_RED_AXIS_IF(E, CMP)                                                                           \
    cs::enable_if_t<(sizeof...(Ax) > 0) && _md::reduced_extents<E,Ax...>::rank_dynamic() CMP 0, int>
#define _TNY_RED_TAGGED_IF(E, CMP)                                                                         \
    class AxisTag, cs::enable_if_t<_md::_red_dyn<E,AxisTag>::value CMP 0, int>
#define _TNY_RED_AXIS_DEF(NAME, API, CMP)                                                                  \
template <class T,class E,class L,storage O> template <long... Ax, class... Tags, _TNY_RED_AXIS_IF(E, CMP)> \
API decltype(auto) tensor<T,E,L,O>::NAME(Tags... tags) const { return tny::NAME<Ax...>(*this, tags...); }  \
template <class T,class E,class L,storage O> template <class Acc, long... Ax, class... Tags, _TNY_RED_AXIS_IF(E, CMP)> \
API decltype(auto) tensor<T,E,L,O>::NAME(Tags... tags) const { return tny::NAME<Acc, Ax...>(*this, tags...); }
#define _TNY_RED_TAGGED_DEF(NAME, API, CMP)                                                                \
template <class T,class E,class L,storage O> template <class Acc, class Tag0, class... Tags, _TNY_RED_TAGGED_IF(E, CMP)> \
API decltype(auto) tensor<T,E,L,O>::NAME(Tag0 tag0, Tags... tags) const { return tny::NAME<Acc>(*this, tag0, tags...); }
#define _TNY_RED_METHOD_DEF(NAME)                                                                          \
template <class T,class E,class L,storage O> template <class Acc>                                          \
_TNY_API auto tensor<T,E,L,O>::NAME() const { return tny::NAME<Acc>(*this); }                              \
_TNY_RED_AXIS_DEF(NAME, _TNY_API,  ==)                                                                     \
_TNY_RED_AXIS_DEF(NAME, _TNY_HOST, !=)                                                                     \
_TNY_RED_TAGGED_DEF(NAME, _TNY_API,  ==)                                                                   \
_TNY_RED_TAGGED_DEF(NAME, _TNY_HOST, !=)
_TNY_RED_METHOD_DEF(sum)    _TNY_RED_METHOD_DEF(prod)  _TNY_RED_METHOD_DEF(max)
_TNY_RED_METHOD_DEF(min)    _TNY_RED_METHOD_DEF(mean)  _TNY_RED_METHOD_DEF(sqnorm)
_TNY_RED_METHOD_DEF(norm)
#undef _TNY_RED_METHOD_DEF
#undef _TNY_RED_TAGGED_DEF
#undef _TNY_RED_AXIS_DEF
#undef _TNY_RED_TAGGED_IF
#undef _TNY_RED_AXIS_IF
// dot (binary, no axis form): m.dot(b) / m.dot<Acc>(b) / m.dot(b, dtype<Acc>{}, into(cell)).
template <class T,class E,class L,storage O> template <class Acc, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto tensor<T,E,L,O>::dot(const tensor<Tb,Eb,Lb,Ob> & b) const { return tny::dot<Acc>(*this, b); }
template <class T,class E,class L,storage O> template <class Acc, class Tb,class Eb,class Lb,storage Ob, class Tag0, class... Tags>
_TNY_API decltype(auto) tensor<T,E,L,O>::dot(const tensor<Tb,Eb,Lb,Ob> & b, Tag0 tag0, Tags... tags) const { return tny::dot<Acc>(*this, b, tag0, tags...); }
// sqdist/dist: same binary (no axis) shape as dot.
template <class T,class E,class L,storage O> template <class Acc, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto tensor<T,E,L,O>::sqdist(const tensor<Tb,Eb,Lb,Ob> & b) const { return tny::sqdist<Acc>(*this, b); }
template <class T,class E,class L,storage O> template <class Acc, class Tb,class Eb,class Lb,storage Ob, class Tag0, class... Tags>
_TNY_API decltype(auto) tensor<T,E,L,O>::sqdist(const tensor<Tb,Eb,Lb,Ob> & b, Tag0 tag0, Tags... tags) const { return tny::sqdist<Acc>(*this, b, tag0, tags...); }
template <class T,class E,class L,storage O> template <class Acc, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto tensor<T,E,L,O>::dist(const tensor<Tb,Eb,Lb,Ob> & b) const { return tny::dist<Acc>(*this, b); }
template <class T,class E,class L,storage O> template <class Acc, class Tb,class Eb,class Lb,storage Ob, class Tag0, class... Tags>
_TNY_API decltype(auto) tensor<T,E,L,O>::dist(const tensor<Tb,Eb,Lb,Ob> & b, Tag0 tag0, Tags... tags) const { return tny::dist<Acc>(*this, b, tag0, tags...); }

/* ------------------------------------------------------------------ *
 *     Out-of-place producers AS METHODS (parity with a.add(b))       *
 *     — thin forwarders to the free forms / engines above.           *
 * ------------------------------------------------------------------ */
#define _TNY_OOP_UNARY_M(NAME, F)                                                                  \
template <class T,class E,class L,storage O> _TNY_API auto tensor<T,E,L,O>::NAME() const            \
{ return _md::uop_out(*this, _md::F{}); }                                                           \
template <class T,class E,class L,storage O> template <class D>                                     \
_TNY_API auto & tensor<T,E,L,O>::NAME(into_t<D> out) const { _md::uop_to(out.dest, *this, _md::F{}); return out.dest; }
_TNY_OOP_UNARY_M(neg,   u_neg)   _TNY_OOP_UNARY_M(abs,   u_abs)   _TNY_OOP_UNARY_M(exp,   u_exp)
_TNY_OOP_UNARY_M(log,   u_log)   _TNY_OOP_UNARY_M(sin,   u_sin)   _TNY_OOP_UNARY_M(cos,   u_cos)
_TNY_OOP_UNARY_M(sqrt,  u_sqrt)  _TNY_OOP_UNARY_M(tanh,  u_tanh)  _TNY_OOP_UNARY_M(floor, u_floor)
_TNY_OOP_UNARY_M(ceil,  u_ceil)  _TNY_OOP_UNARY_M(round, u_round) _TNY_OOP_UNARY_M(trunc, u_trunc)
_TNY_OOP_UNARY_M(sign,  u_sign)
#undef _TNY_OOP_UNARY_M

template <class T,class E,class L,storage O> template <class B>
_TNY_API auto tensor<T,E,L,O>::minimum(const B & b) const { return tny::minimum(*this, b); }
template <class T,class E,class L,storage O> template <class B>
_TNY_API auto tensor<T,E,L,O>::maximum(const B & b) const { return tny::maximum(*this, b); }
template <class T,class E,class L,storage O> template <class B, class D>
_TNY_API auto & tensor<T,E,L,O>::minimum(const B & b, into_t<D> out) const { return tny::minimum(*this, b, out); }
template <class T,class E,class L,storage O> template <class B, class D>
_TNY_API auto & tensor<T,E,L,O>::maximum(const B & b, into_t<D> out) const { return tny::maximum(*this, b, out); }
template <class T,class E,class L,storage O>
_TNY_API auto tensor<T,E,L,O>::clamp(T lo, T hi) const { return tny::clamp(*this, lo, hi); }
template <class T,class E,class L,storage O> template <class D>
_TNY_API auto & tensor<T,E,L,O>::clamp(T lo, T hi, into_t<D> out) const { return tny::clamp(*this, lo, hi, out); }
template <class T,class E,class L,storage O>
_TNY_API auto tensor<T,E,L,O>::normalize() const { return tny::normalize(*this); }
template <class T,class E,class L,storage O> template <class D>
_TNY_API auto & tensor<T,E,L,O>::normalize(into_t<D> out) const { return tny::normalize(*this, out); }
template <class T,class E,class L,storage O> template <class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto tensor<T,E,L,O>::cross(const tensor<Tb,Eb,Lb,Ob> & b) const { return tny::cross(*this, b); }
template <class T,class E,class L,storage O> template <class Tb,class Eb,class Lb,storage Ob, class D>
_TNY_API auto & tensor<T,E,L,O>::cross(const tensor<Tb,Eb,Lb,Ob> & b, into_t<D> out) const { return tny::cross(*this, b, out); }

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_MATH
