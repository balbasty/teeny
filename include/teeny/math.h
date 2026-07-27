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

/* ---- write policies: how an engine commits op(...) to c ---------- *
 * `w_set` overwrites; `w_add` accumulates ATOMICALLY on device (the   *
 * scatter/push write). In-place add_/sub_ pick the policy via their   *
 * `Atomic` flag; every other engine defaults to w_set.                */
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
// The broadcast RESULT carries the WIDER of the two operands' offset index types
// (by `sizeof`; a tie keeps the first operand's, so same-width mixes are unchanged).
// The engine `bzip_` casts both operands' extents/strides to the result index type,
// so broadening is lossless AND fixes the truncation that a narrow result would cause
// to a wide operand's strides. (It does not — and cannot from the types alone — widen
// two equal-narrow operands whose broadcast SPAN overflows: that stays the caller's
// responsibility, guarded by index_fits/dispatch_index at the boundary.)
template <class Ea, class Eb>
using _wider_index_t = cs::conditional_t<
    (sizeof(typename Eb::index_type) > sizeof(typename Ea::index_type)),
    typename Eb::index_type, typename Ea::index_type>;
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

// `Cv` is the type the op runs in: the destination's compute type for arithmetic
// (default via the `bzip` wrapper), or the operands' compare type for a comparison
// whose result is a bool mask (the `bcmp` wrapper passes it) — one engine, two uses.
// `Restrict` (set only by the OUT-OF-PLACE callers, where `c` is a fresh
// allocation that provably cannot alias `a`/`b`) enables an auto-vectorizable
// linear fast path: when the plain-store writer `w_set` is in use and every
// operand has the SAME rank + extents as `c` and is C-contiguous, offsets are
// just `i`, so the write goes through a `__restrict__` destination and the
// per-element mixed-radix decode is skipped. Any mismatch falls back unchanged.
template <class W, bool Restrict, class Cv, class C, class A, class B, class Op, cs::size_t... D>
_TNY_API void bzip_(C & c, const A & a, const B & b, Op op, cs::index_sequence<D...>) {
    using I = typename C::index_type;
    constexpr cs::size_t R = C::rank();   // c has the result (largest) rank; a,b right-align into it
    const I ce[] = { c.extent(D)... }, sc[] = { c.stride(D)... };
    const I ae[] = { static_cast<I>(bc_ext<R>(a, D))... }, sa[] = { static_cast<I>(bc_str<R>(a, D))... };
    const I be[] = { static_cast<I>(bc_ext<R>(b, D))... }, sb[] = { static_cast<I>(bc_str<R>(b, D))... };
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
        W{}(&c.data()[oc], op(static_cast<Cv>(a.data()[oa]), static_cast<Cv>(b.data()[ob])));
    }
}
// A self-overlapping DESTINATION — an axis with extent>1 AND stride 0 — makes an
// in-place write ALIAS: the update lands on the same element repeatedly, so
// `v.add_(b)` double-counts (and mul_/iota_/... are likewise wrong). Such a view
// can only come from `wrap(ptr, e, strides-with-a-0)`; teeny's own view ops never
// make one, and clone()/to() write into a fresh DENSE destination. Host-debug guard
// (a no-op on device / under -DNDEBUG); DESTINATION only, so a broadcasting RHS
// (which legitimately stretches with stride 0) is never flagged.
template <class C, cs::size_t... D>
_TNY_API void check_dest_no_overlap(const C & c, cs::index_sequence<D...>) {
    ( _TNY_CHECK(!(static_cast<long long>(c.extent(D)) > 1 && c.stride(D) == 0),
        "in-place write into a self-overlapping view (an axis has extent>1 and stride 0): "
        "the update aliases and is applied multiple times to the same element — clone() to a "
        "dense tensor first, or write into a non-overlapping destination."), ... );
}
// `Restrict` defaults to false so every IN-PLACE caller (add_/sub_/.../copy_,
// where `c` IS the destination and may alias the rhs) takes the safe decode path
// byte-for-byte unchanged; the out-of-place `oop` passes true.
template <class W = w_set, bool Restrict = false, class C, class A, class B, class Op>
_TNY_API void bzip(C & c, const A & a, const B & b, Op op) {
    // C holds the RESULT (largest) rank; operands may be shorter (left-padded).
    check_dest_no_overlap(c, cs::make_index_sequence<C::rank()>{});
    static_assert(A::rank() <= C::rank() && B::rank() <= C::rank(), "broadcast: operand rank exceeds result");
    static_assert(bc_static_ok_r<typename A::extents_type, typename B::extents_type, C::rank()>(
                      cs::make_index_sequence<C::rank()>{}),
                  "broadcast: incompatible static extents");
    bzip_<W, Restrict, compute_type_t<typename C::element_type>>(c, a, b, op, cs::make_index_sequence<C::rank()>{});
}

/* ---- c = op(c, scalar), elementwise ------------------------------ */
template <class W, class C, class Op, cs::size_t... D>
_TNY_API void scal_(C & c, typename C::element_type s, Op op, cs::index_sequence<D...>) {
    check_dest_no_overlap(c, cs::index_sequence<D...>{});
    using I  = typename C::index_type;
    using Cv = compute_type_t<typename C::element_type>;   // compute in float for half types
    const Cv sv = static_cast<Cv>(s);
    const I e[]  = { c.extent(D)... };
    const I sc[] = { c.stride(D)... };
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
        W{}(&c.data()[oc], op(static_cast<Cv>(c.data()[oc]), sv));
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
    const I e[]  = { c.extent(D)... };
    const I sc[] = { c.stride(D)... };
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
    using I = typename C::index_type;   // `Cv` = the op's compute type (arithmetic: dest compute type; compare: Rc)
    const I e[] = { a.extent(D)... }, sa[] = { a.stride(D)... }, sc[] = { c.stride(D)... };
    I n = 1; for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    // Contiguous linear fast path (out-of-place only; `c` fresh, cannot alias `a`).
    if constexpr (Restrict) {
        if (c.is_contiguous() && a.is_contiguous()) {
            typename C::element_type * _TNY_RESTRICT cp = c.data();
            const typename A::element_type * ap = a.data();
            const Cv sv = static_cast<Cv>(s);
            for (I i = 0; i < n; ++i) cp[i] = op(static_cast<Cv>(ap[i]), sv);   // mirrors the slow store
            return;
        }
    }
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0, oc = 0;
        for (int d = (int)sizeof...(D)-1; d >= 0; --d) { I k = rem%e[d]; rem/=e[d]; oa+=k*sa[d]; oc+=k*sc[d]; }
        c.data()[oc] = op(static_cast<Cv>(a.data()[oa]), static_cast<Cv>(s));
    }
}
template <bool Restrict = false, class C, class A, class S, class Op> _TNY_API void scalo(C & c, const A & a, S s, Op op)
{ scalo_<Restrict, compute_type_t<typename C::element_type>>(c, a, s, op, cs::make_index_sequence<C::rank()>{}); }

/* ---- c(i) = uop(a(i))  and  c(i) = uop(c(i)) (in place) ---------- */
template <bool Restrict, class C, class A, class Uop, cs::size_t... D>
_TNY_API void unaryo_(C & c, const A & a, Uop f, cs::index_sequence<D...>) {
    using I = typename C::index_type; using Cv = compute_type_t<typename C::element_type>;
    const I e[] = { a.extent(D)... }, sa[] = { a.stride(D)... }, sc[] = { c.stride(D)... };
    I n = 1; for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    // Contiguous linear fast path (out-of-place only; `c` fresh, cannot alias `a`).
    if constexpr (Restrict) {
        if (c.is_contiguous() && a.is_contiguous()) {
            typename C::element_type * _TNY_RESTRICT cp = c.data();
            const typename A::element_type * ap = a.data();
            for (I i = 0; i < n; ++i) cp[i] = static_cast<Cv>(f(static_cast<Cv>(ap[i])));   // mirrors the slow store
            return;
        }
    }
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0, oc = 0;
        for (int d = (int)sizeof...(D)-1; d >= 0; --d) { I k = rem%e[d]; rem/=e[d]; oa+=k*sa[d]; oc+=k*sc[d]; }
        c.data()[oc] = static_cast<Cv>(f(static_cast<Cv>(a.data()[oa])));
    }
}
// `Restrict` is false for `unary` (in-place: c===a, must not restrict) and true
// for `uop_out` (out-of-place: fresh dest).
template <bool Restrict = false, class C, class A, class Uop> _TNY_API void unaryo(C & c, const A & a, Uop f)
{ unaryo_<Restrict>(c, a, f, cs::make_index_sequence<C::rank()>{}); }
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
 * an operand; the engine validates dest's extents against the (broadcast) result
 * and casts to dest's element type. The producer hands `into_t::dest` back. */
template <class Op, class Out, class A, class B> _TNY_API void oop_to (Out & o, const A & a, const B & b, Op op) { bzip<w_set, false>(o, a, b, op); }
template <class Op, class Out, class A, class S>  _TNY_API void oops_to(Out & o, const A & a, S s, Op op) { scalo<false>(o, a, s, op); }
template <class Uop, class Out, class A>          _TNY_API void uop_to (Out & o, const A & a, Uop f) { unaryo<false>(o, a, f); }

/* ---- reduce a . b elementwise into a scalar (for dot) ------------ */
template <class R, class A, class B, cs::size_t... D>
_TNY_API R zipreduce_(const A & a, const B & b, cs::index_sequence<D...>) {
    using I = typename A::index_type;
    const I e[]  = { a.extent(D)... };
    const I be[] = { b.extent(D)... };
    const I sa[] = { a.stride(D)... };
    const I sb[] = { b.stride(D)... };
    for (cs::size_t r = 0; r < sizeof...(D); ++r)
        _TNY_CHECK(e[r] == be[r], "dot: operand extents must match exactly (no broadcast)");
    I n = 1;
    for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    R acc = R(0);
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0, ob = 0;
        for (int d = static_cast<int>(sizeof...(D)) - 1; d >= 0; --d) {
            I k = rem % e[d]; rem /= e[d]; oa += k * sa[d]; ob += k * sb[d];
        }
        acc += static_cast<R>(a.data()[oa]) * static_cast<R>(b.data()[ob]);
    }
    return acc;
}

/* ---- fold a into a scalar with `op`, starting from `init` --------- */
template <class R, class A, class Op, cs::size_t... D>
_TNY_API R reduce_(const A & a, R init, Op op, cs::index_sequence<D...>) {
    using I = typename A::index_type;
    const I e[]  = { a.extent(D)... };
    const I sa[] = { a.stride(D)... };
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
    const I e[]  = { static_cast<I>(a.extent(D))... };
    const I sa[] = { static_cast<I>(a.stride(D))... };
    I so[N]; int oi = 0;                                   // output stride per input axis (0 if reduced)
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
template <class E, cs::size_t... D>
constexpr cs::size_t _static_numel_(cs::index_sequence<D...>) { return (cs::size_t(1) * ... * E::static_extent(D)); }
template <class E>
constexpr cs::size_t _static_numel() { return _static_numel_<E>(cs::make_index_sequence<E::rank()>{}); }

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
    using I = typename A::index_type;
    constexpr cs::size_t Rk = sizeof...(D);   // result (broadcast) rank; a,b right-align (left-pad)
    const I ae[] = { static_cast<I>(bc_ext<Rk>(a, D))... }, sa[] = { static_cast<I>(bc_str<Rk>(a, D))... };
    const I be[] = { static_cast<I>(bc_ext<Rk>(b, D))... }, sb[] = { static_cast<I>(bc_str<Rk>(b, D))... };
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
// is `fetch_add` (atomic on device) — the scatter/"push" accumulate — so the op
// commits a DELTA (rhs, or -rhs for sub) rather than a read-modify-write.
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
 * `double` for floating-point types of at most 8 bytes (`float`, `double`,
 * `half`, `bfloat16`) — enough headroom that summing many low-precision values
 * doesn't lose catastrophically; a *wider* floating type (`long double`) keeps
 * itself. Integer types narrower than 8 bytes accumulate in 64-bit (`int64_t` if
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
    cs::conditional_t<(sizeof(T) > 8), T, double>,
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
        return cs::numeric_limits<R>::max();
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
#define _TNY_MD_AXRED(NAME, INIT, OP)                                                              \
template <long... Axes, class T,class E,class L,storage O, class R = reduce_type_t<T>,                 \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0> \
_TNY_API  auto NAME(const tensor<T,E,L,O> & a) { return _md::reduce_to<T>(_md::axreduce<Axes...>(a, INIT, _md::OP{})); } \
template <long... Axes, class T,class E,class L,storage O, class R = reduce_type_t<T>,                 \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0> \
_TNY_HOST auto NAME(const tensor<T,E,L,O> & a) { return _md::reduce_to<T>(_md::axreduce<Axes...>(a, INIT, _md::OP{})); } \
template <class Acc, long... Axes, class T,class E,class L,storage O, class R = Acc,                   \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0> \
_TNY_API  auto NAME(const tensor<T,E,L,O> & a) { return _md::reduce_to<Acc>(_md::axreduce<Axes...>(a, INIT, _md::OP{})); } \
template <class Acc, long... Axes, class T,class E,class L,storage O, class R = Acc,                   \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0> \
_TNY_HOST auto NAME(const tensor<T,E,L,O> & a) { return _md::reduce_to<Acc>(_md::axreduce<Axes...>(a, INIT, _md::OP{})); } \
/* value forms: NAME(a, axis<Axes...>{}) == NAME<Axes...>(a) — numpy's `axis=` spelling  \
   (no `.template` on a dependent receiver). Forward to the template axis form, keeping   \
   the static(_TNY_API)/dynamic(_TNY_HOST) split so a device path never calls host code. */ \
template <long... Axes, class T,class E,class L,storage O,                                              \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0> \
_TNY_API  auto NAME(const tensor<T,E,L,O> & a, axis<Axes...>) { return NAME<Axes...>(a); }              \
template <long... Axes, class T,class E,class L,storage O,                                              \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0> \
_TNY_HOST auto NAME(const tensor<T,E,L,O> & a, axis<Axes...>) { return NAME<Axes...>(a); }              \
template <class Acc, long... Axes, class T,class E,class L,storage O,                                   \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0> \
_TNY_API  auto NAME(const tensor<T,E,L,O> & a, axis<Axes...>) { return NAME<Acc, Axes...>(a); }         \
template <class Acc, long... Axes, class T,class E,class L,storage O,                                   \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0> \
_TNY_HOST auto NAME(const tensor<T,E,L,O> & a, axis<Axes...>) { return NAME<Acc, Axes...>(a); }
_TNY_MD_AXRED(sum,    R(0),                        r_add)
_TNY_MD_AXRED(prod,   R(1),                        r_mul)
_TNY_MD_AXRED(max,    _reduce_seed_lowest<R>(),  r_max)
_TNY_MD_AXRED(min,    _reduce_seed_highest<R>(), r_min)
_TNY_MD_AXRED(sqnorm, R(0),                        r_addsq)   // Σaᵢ² over the named axes (result type = T, like sum)
#undef _TNY_MD_AXRED

/** @brief Mean over the named axes -> a lower-rank tensor (sum / reduced count).
 *         For a floating `T`, accumulates in the reduce type and the result is cast
 *         to `T`. For an INTEGER `T` the result element type is `double` (numpy:
 *         integer mean is float64; divides in `double`, not truncating). `mean<Acc,
 *         Axes...>(a)` makes `Acc` both the accumulator and result type. */
template <long... Axes, class T,class E,class L,storage O, class R = reduce_type_t<T>,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0>
_TNY_API  auto mean(const tensor<T,E,L,O> & a) {
    auto s = sum<R, Axes...>(a);                                          // sum in the (wide) reduce type
    const auto cnt = a.numel() / s.numel();
    if constexpr (cs::is_integral<T>::value) {                           // integer -> divide in double, return double
        auto o = _md::reduce_to<double>(static_cast<decltype(s)&&>(s));
        o.div_(static_cast<double>(cnt)); return o;
    } else { s.div_(static_cast<R>(cnt)); return _md::reduce_to<T>(static_cast<decltype(s)&&>(s)); }
}
template <long... Axes, class T,class E,class L,storage O, class R = reduce_type_t<T>,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0>
_TNY_HOST auto mean(const tensor<T,E,L,O> & a) {
    auto s = sum<R, Axes...>(a);                                          // sum in the (wide) reduce type
    const auto cnt = a.numel() / s.numel();
    if constexpr (cs::is_integral<T>::value) {                           // integer -> divide in double, return double
        auto o = _md::reduce_to<double>(static_cast<decltype(s)&&>(s));
        o.div_(static_cast<double>(cnt)); return o;
    } else { s.div_(static_cast<R>(cnt)); return _md::reduce_to<T>(static_cast<decltype(s)&&>(s)); }
}
template <class Acc, long... Axes, class T,class E,class L,storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0>
_TNY_API  auto mean(const tensor<T,E,L,O> & a) {
    auto s = sum<Acc, Axes...>(a); s.div_(static_cast<Acc>(a.numel() / s.numel())); return s;
}
template <class Acc, long... Axes, class T,class E,class L,storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0>
_TNY_HOST auto mean(const tensor<T,E,L,O> & a) {
    auto s = sum<Acc, Axes...>(a); s.div_(static_cast<Acc>(a.numel() / s.numel())); return s;
}
// value forms: mean(a, axis<Axes...>{}) == mean<Axes...>(a) (numpy `axis=`); forward to
// the template form, keeping the static/dynamic split (integer->double rule preserved).
template <long... Axes, class T,class E,class L,storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0>
_TNY_API  auto mean(const tensor<T,E,L,O> & a, axis<Axes...>) { return mean<Axes...>(a); }
template <long... Axes, class T,class E,class L,storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0>
_TNY_HOST auto mean(const tensor<T,E,L,O> & a, axis<Axes...>) { return mean<Axes...>(a); }
template <class Acc, long... Axes, class T,class E,class L,storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0>
_TNY_API  auto mean(const tensor<T,E,L,O> & a, axis<Axes...>) { return mean<Acc, Axes...>(a); }
template <class Acc, long... Axes, class T,class E,class L,storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0>
_TNY_HOST auto mean(const tensor<T,E,L,O> & a, axis<Axes...>) { return mean<Acc, Axes...>(a); }

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
        _md::zipreduce_<R>(a, b, cs::make_index_sequence<tensor<Ta,Ea,La,Oa>::rank()>{}));
}

/* ------------------------------------------------------------------ *
 *     Vector algebra & geometry (contained exact math)               *
 * ------------------------------------------------------------------ */

/** @brief Squared Euclidean norm — the sum of squares `Σ aᵢ²`, over ALL axes.
 *         Just `dot(a, a)`: accumulates in the reduce type, result cast to the
 *         element type (`sqnorm<Acc>(a)` accumulates AND returns `Acc`). */
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
 *     Accumulates the squares in a floating type and takes the root there. --------- */
template <long... Axes, class T,class E,class L,storage O,
          class Res = _mean_result_t<T>, class R = reduce_type_t<T>,
          class D   = cs::conditional_t<cs::is_floating_point<R>::value, R, double>,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0>
_TNY_API  auto norm(const tensor<T,E,L,O> & a) { auto s = sqnorm<D, Axes...>(a); s.sqrt_(); return _md::reduce_to<Res>(static_cast<decltype(s)&&>(s)); }
template <long... Axes, class T,class E,class L,storage O,
          class Res = _mean_result_t<T>, class R = reduce_type_t<T>,
          class D   = cs::conditional_t<cs::is_floating_point<R>::value, R, double>,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0>
_TNY_HOST auto norm(const tensor<T,E,L,O> & a) { auto s = sqnorm<D, Axes...>(a); s.sqrt_(); return _md::reduce_to<Res>(static_cast<decltype(s)&&>(s)); }
template <class Acc, long... Axes, class T,class E,class L,storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0>
_TNY_API  auto norm(const tensor<T,E,L,O> & a) { auto s = sqnorm<Acc, Axes...>(a); s.sqrt_(); return s; }
template <class Acc, long... Axes, class T,class E,class L,storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0>
_TNY_HOST auto norm(const tensor<T,E,L,O> & a) { auto s = sqnorm<Acc, Axes...>(a); s.sqrt_(); return s; }
// value forms: norm(a, axis<Axes...>{}) == norm<Axes...>(a)
template <long... Axes, class T,class E,class L,storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0>
_TNY_API  auto norm(const tensor<T,E,L,O> & a, axis<Axes...>) { return norm<Axes...>(a); }
template <long... Axes, class T,class E,class L,storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0>
_TNY_HOST auto norm(const tensor<T,E,L,O> & a, axis<Axes...>) { return norm<Axes...>(a); }
template <class Acc, long... Axes, class T,class E,class L,storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0>
_TNY_API  auto norm(const tensor<T,E,L,O> & a, axis<Axes...>) { return norm<Acc, Axes...>(a); }
template <class Acc, long... Axes, class T,class E,class L,storage O,
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0>
_TNY_HOST auto norm(const tensor<T,E,L,O> & a, axis<Axes...>) { return norm<Acc, Axes...>(a); }

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
 * unsqueeze grows the rank for the next), so the axes must be distinct & ascending. */
// `_axes_ascending(...)` lives in indexing.h (next to `_norm_axis`) — tensor.h's
// multi-axis `unsqueeze<Ax...>`/`squeeze<Ax...>` folds need it too, and tensor.h
// cannot include math.h.
// keepdim view fold: insert a size-1 axis at each (ascending, already-normalised) position.
template <class Tn> _TNY_API auto _keepdims(const Tn & t) { return t; }
template <long A0, long... Rest, class Tn> _TNY_API auto _keepdims(const Tn & t) { return _keepdims<Rest...>(t.template unsqueeze<A0>()); }

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
 *         (rank-1, length 3); `y` may alias `a` or `b`. This is ff's "crossto". */
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

/* ------------------------------------------------------------------ *
 *     into(dest) on reductions — numpy/pytorch `out=`                 *
 * ------------------------------------------------------------------ *
 * A FULL reduction writes its scalar into a rank-0 destination
 * (`local<T>{}`, or `wrap(&x, shape<>{})` over an address) — one store,
 * no allocation; a non-rank-0 dest is a `static_assert` (so forgetting the
 * axes fails loudly instead of splatting the grand total everywhere). An
 * AXIS reduction copies its lower-rank result into `dest` (via `copy_`, so
 * `dest` is broadcast-compatible with the reduced shape). `dest` may differ
 * in dtype (result cast) / be a strided slot. Returns `dest&`. The three
 * template shapes mirror
 * the reductions themselves: bare (`sum(a, into(d))`) and leading-type
 * (`sum<double>(a, into(d))`) are the FULL forms; a leading axis
 * (`sum<0>(a, into(d))`) or type+axis (`sum<double,0>(a, into(d))`)
 * selects the AXIS form (static result -> _TNY_API, dynamic -> _TNY_HOST,
 * so a device path never calls the host allocator). */
#define _TNY_RED_INTO(NAME)                                                                          \
template <class Acc = void, class T,class E,class L,storage O, class D>                              \
_TNY_API auto & NAME(const tensor<T,E,L,O> & a, into_t<D> out) {                                     \
    static_assert(D::rank() == 0, "full reduction into(dest): dest must be rank-0 (a scalar cell); " \
                  "name the axes — NAME<Axes...>(a, into(dest)) — for a lower-rank destination");     \
    out.dest.fill_(static_cast<typename D::element_type>(NAME<Acc>(a))); return out.dest; }          \
template <long... Axes, class T,class E,class L,storage O, class D,                                  \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0> \
_TNY_API  auto & NAME(const tensor<T,E,L,O> & a, into_t<D> out) { out.dest.copy_(NAME<Axes...>(a)); return out.dest; } \
template <long... Axes, class T,class E,class L,storage O, class D,                                  \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0> \
_TNY_HOST auto & NAME(const tensor<T,E,L,O> & a, into_t<D> out) { out.dest.copy_(NAME<Axes...>(a)); return out.dest; } \
template <class Acc, long... Axes, class T,class E,class L,storage O, class D,                       \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0> \
_TNY_API  auto & NAME(const tensor<T,E,L,O> & a, into_t<D> out) { out.dest.copy_(NAME<Acc, Axes...>(a)); return out.dest; } \
template <class Acc, long... Axes, class T,class E,class L,storage O, class D,                       \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0> \
_TNY_HOST auto & NAME(const tensor<T,E,L,O> & a, into_t<D> out) { out.dest.copy_(NAME<Acc, Axes...>(a)); return out.dest; } \
/* value forms take into too: NAME(a, axis<Axes...>{}, into(d)) == NAME<Axes...>(a, into(d)) */      \
template <long... Axes, class T,class E,class L,storage O, class D,                                  \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0> \
_TNY_API  auto & NAME(const tensor<T,E,L,O> & a, axis<Axes...>, into_t<D> out) { return NAME<Axes...>(a, out); } \
template <long... Axes, class T,class E,class L,storage O, class D,                                  \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0> \
_TNY_HOST auto & NAME(const tensor<T,E,L,O> & a, axis<Axes...>, into_t<D> out) { return NAME<Axes...>(a, out); } \
template <class Acc, long... Axes, class T,class E,class L,storage O, class D,                       \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()==0, int> = 0> \
_TNY_API  auto & NAME(const tensor<T,E,L,O> & a, axis<Axes...>, into_t<D> out) { return NAME<Acc, Axes...>(a, out); } \
template <class Acc, long... Axes, class T,class E,class L,storage O, class D,                       \
          cs::enable_if_t<(sizeof...(Axes) > 0) && _md::reduced_extents<E,Axes...>::rank_dynamic()!=0, int> = 0> \
_TNY_HOST auto & NAME(const tensor<T,E,L,O> & a, axis<Axes...>, into_t<D> out) { return NAME<Acc, Axes...>(a, out); }
_TNY_RED_INTO(sum)  _TNY_RED_INTO(prod)  _TNY_RED_INTO(max)  _TNY_RED_INTO(min)
_TNY_RED_INTO(mean) _TNY_RED_INTO(sqnorm) _TNY_RED_INTO(norm)
#undef _TNY_RED_INTO

/** @brief `dot(a, b, into(d))` — inner product written into a rank-0 dest. */
template <class Acc = void, class Ta,class Ea,class La,storage Oa,
          class Tb,class Eb,class Lb,storage Ob, class D>
_TNY_API auto & dot(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b, into_t<D> out) {
    static_assert(D::rank() == 0, "dot into(dest): dest must be rank-0 (a scalar cell)");
    out.dest.fill_(static_cast<typename D::element_type>(dot<Acc>(a, b))); return out.dest;
}

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
#define _TNY_RED_AXIS_DEF(NAME, API, CMP)                                                                  \
template <class T,class E,class L,storage O> template <long... Ax, _TNY_RED_AXIS_IF(E, CMP)>               \
API auto tensor<T,E,L,O>::NAME() const { return tny::NAME<Ax...>(*this); }                                 \
template <class T,class E,class L,storage O> template <class Acc, long... Ax, _TNY_RED_AXIS_IF(E, CMP)>    \
API auto tensor<T,E,L,O>::NAME() const { return tny::NAME<Acc, Ax...>(*this); }                            \
template <class T,class E,class L,storage O> template <long... Ax, _TNY_RED_AXIS_IF(E, CMP)>               \
API auto tensor<T,E,L,O>::NAME(axis<Ax...>) const { return tny::NAME<Ax...>(*this); }                      \
template <class T,class E,class L,storage O> template <class Acc, long... Ax, _TNY_RED_AXIS_IF(E, CMP)>    \
API auto tensor<T,E,L,O>::NAME(axis<Ax...>) const { return tny::NAME<Acc, Ax...>(*this); }                 \
template <class T,class E,class L,storage O> template <long... Ax, class D, _TNY_RED_AXIS_IF(E, CMP)>      \
API auto & tensor<T,E,L,O>::NAME(into_t<D> out) const { return tny::NAME<Ax...>(*this, out); }             \
template <class T,class E,class L,storage O> template <class Acc, long... Ax, class D, _TNY_RED_AXIS_IF(E, CMP)> \
API auto & tensor<T,E,L,O>::NAME(into_t<D> out) const { return tny::NAME<Acc, Ax...>(*this, out); }        \
template <class T,class E,class L,storage O> template <long... Ax, class D, _TNY_RED_AXIS_IF(E, CMP)>      \
API auto & tensor<T,E,L,O>::NAME(axis<Ax...>, into_t<D> out) const { return tny::NAME<Ax...>(*this, out); } \
template <class T,class E,class L,storage O> template <class Acc, long... Ax, class D, _TNY_RED_AXIS_IF(E, CMP)> \
API auto & tensor<T,E,L,O>::NAME(axis<Ax...>, into_t<D> out) const { return tny::NAME<Acc, Ax...>(*this, out); }
#define _TNY_RED_METHOD_DEF(NAME)                                                                          \
template <class T,class E,class L,storage O> template <class Acc>                                          \
_TNY_API auto tensor<T,E,L,O>::NAME() const { return tny::NAME<Acc>(*this); }                              \
template <class T,class E,class L,storage O> template <class Acc, class D>                                 \
_TNY_API auto & tensor<T,E,L,O>::NAME(into_t<D> out) const { return tny::NAME<Acc>(*this, out); }          \
_TNY_RED_AXIS_DEF(NAME, _TNY_API,  ==)                                                                     \
_TNY_RED_AXIS_DEF(NAME, _TNY_HOST, !=)
_TNY_RED_METHOD_DEF(sum)    _TNY_RED_METHOD_DEF(prod)  _TNY_RED_METHOD_DEF(max)
_TNY_RED_METHOD_DEF(min)    _TNY_RED_METHOD_DEF(mean)  _TNY_RED_METHOD_DEF(sqnorm)
_TNY_RED_METHOD_DEF(norm)
#undef _TNY_RED_METHOD_DEF
#undef _TNY_RED_AXIS_DEF
#undef _TNY_RED_AXIS_IF
// dot (binary): m.dot(b) / m.dot<Acc>(b) / m.dot(b, into(cell)).
template <class T,class E,class L,storage O> template <class Acc, class Tb,class Eb,class Lb,storage Ob>
_TNY_API auto tensor<T,E,L,O>::dot(const tensor<Tb,Eb,Lb,Ob> & b) const { return tny::dot<Acc>(*this, b); }
template <class T,class E,class L,storage O> template <class Acc, class Tb,class Eb,class Lb,storage Ob, class D>
_TNY_API auto & tensor<T,E,L,O>::dot(const tensor<Tb,Eb,Lb,Ob> & b, into_t<D> out) const { return tny::dot<Acc>(*this, b, out); }

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
