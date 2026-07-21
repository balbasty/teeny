#ifndef TNY_MD_MATH
#define TNY_MD_MATH
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <cuda/std/limits>
#include <cuda/std/cmath>
#include <teeny/_core/defines.h>
#include <teeny/half.h>
#include <teeny/tensor.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

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

/* ---- assignment functors ----------------------------------------- */
struct rhs  { template <class X, class Y> _TNY_API X operator()(X, Y y) const { return static_cast<X>(y); } };  // c = b
struct setc { template <class X> _TNY_API X operator()(X, X s) const { return s; } };                          // c = s

/* ---- unary functors (cuda::std math -> device-callable) ---------- */
struct u_neg  { template <class X> _TNY_API X operator()(X x) const { return -x; } };
struct u_abs  { template <class X> _TNY_API X operator()(X x) const { return x < X(0) ? -x : x; } };
struct u_exp  { template <class X> _TNY_API X operator()(X x) const { return cs::exp(x); } };
struct u_log  { template <class X> _TNY_API X operator()(X x) const { return cs::log(x); } };
struct u_sin  { template <class X> _TNY_API X operator()(X x) const { return cs::sin(x); } };
struct u_cos  { template <class X> _TNY_API X operator()(X x) const { return cs::cos(x); } };
struct u_sqrt { template <class X> _TNY_API X operator()(X x) const { return cs::sqrt(x); } };
struct u_tanh { template <class X> _TNY_API X operator()(X x) const { return cs::tanh(x); } };
struct pw     { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return cs::pow(x, static_cast<X>(y)); } };

/* ---- reduce ops (acc = op(acc, x)) ------------------------------- */
struct r_add { template <class A, class X> _TNY_API A operator()(A a, X x) const { return a + static_cast<A>(x); } };
struct r_mul { template <class A, class X> _TNY_API A operator()(A a, X x) const { return a * static_cast<A>(x); } };
struct r_max { template <class A, class X> _TNY_API A operator()(A a, X x) const { A y = static_cast<A>(x); return y > a ? y : a; } };
struct r_min { template <class A, class X> _TNY_API A operator()(A a, X x) const { A y = static_cast<A>(x); return y < a ? y : a; } };

/* ---- c = op(a, b), elementwise over matching extents ------------- */
template <class C, class A, class B, class Op, cs::size_t... D>
_TNY_API void zip_(C & c, const A & a, const B & b, Op op, cs::index_sequence<D...>) {
    using I  = typename C::index_type;
    using Cv = compute_type_t<typename C::element_type>;  // compute in float for half types
    const I e[]  = { a.extent(D)... };
    const I sa[] = { a.stride(D)... };
    const I sb[] = { b.stride(D)... };
    const I sc[] = { c.stride(D)... };
    I n = 1;
    for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0, ob = 0, oc = 0;
        for (int d = static_cast<int>(sizeof...(D)) - 1; d >= 0; --d) {
            I k = rem % e[d]; rem /= e[d];
            oa += k * sa[d]; ob += k * sb[d]; oc += k * sc[d];
        }
        c.data()[oc] = op(static_cast<Cv>(a.data()[oa]), static_cast<Cv>(b.data()[ob]));
    }
}
template <class C, class A, class B, class Op>
_TNY_API void zip(C & c, const A & a, const B & b, Op op) {
    static_assert(A::rank() == C::rank() && B::rank() == C::rank(), "zip: rank mismatch");
    zip_(c, a, b, op, cs::make_index_sequence<C::rank()>{});
}

/* ---- numpy-style broadcasting (same rank; a dim of 1 broadcasts) - *
 * c(i) = op(a(i), b(i)), where a and b broadcast into c's shape       *
 * (stride 0 on any axis whose operand extent is 1).                   */

// one axis is broadcast-compatible if the extents are equal, one is 1, or
// either is dynamic (only known at run time -> checked by _TNY_CHECK below).
_TNY_API constexpr bool bc_axis_ok(cs::size_t a, cs::size_t b) {
    return a == cs::dynamic_extent || b == cs::dynamic_extent || a == b || a == 1 || b == 1;
}
template <class Ea, class Eb, cs::size_t... D>
_TNY_API constexpr bool bc_static_ok(cs::index_sequence<D...>) {
    bool ok = true;
    ( (ok = ok && bc_axis_ok(Ea::static_extent(D), Eb::static_extent(D))), ... );
    return ok;
}

template <class C, class A, class B, class Op, cs::size_t... D>
_TNY_API void bzip_(C & c, const A & a, const B & b, Op op, cs::index_sequence<D...>) {
    using I = typename C::index_type; using Cv = compute_type_t<typename C::element_type>;
    const I ce[] = { c.extent(D)... }, sc[] = { c.stride(D)... };
    const I ae[] = { a.extent(D)... }, sa[] = { a.stride(D)... };
    const I be[] = { b.extent(D)... }, sb[] = { b.stride(D)... };
    // runtime shape check: each operand extent must equal c's or be 1 (a larger
    // rhs would silently truncate — the worst failure mode in a numerics lib).
    for (cs::size_t r = 0; r < sizeof...(D); ++r) {
        _TNY_CHECK(ae[r] == ce[r] || ae[r] == 1, "broadcast: lhs extent mismatch");
        _TNY_CHECK(be[r] == ce[r] || be[r] == 1, "broadcast: rhs extent mismatch");
    }
    I n = 1; for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= ce[r];
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0, ob = 0, oc = 0;
        for (int d = (int)sizeof...(D)-1; d >= 0; --d) {
            I k = rem % ce[d]; rem /= ce[d];
            oc += k * sc[d];
            oa += (ae[d] == 1 ? I(0) : k) * sa[d];
            ob += (be[d] == 1 ? I(0) : k) * sb[d];
        }
        c.data()[oc] = op(static_cast<Cv>(a.data()[oa]), static_cast<Cv>(b.data()[ob]));
    }
}
template <class C, class A, class B, class Op>
_TNY_API void bzip(C & c, const A & a, const B & b, Op op) {
    static_assert(A::rank() == C::rank() && B::rank() == C::rank(), "broadcast: rank mismatch");
    static_assert(bc_static_ok<typename A::extents_type, typename B::extents_type>(
                      cs::make_index_sequence<C::rank()>{}),
                  "broadcast: incompatible static extents");
    bzip_(c, a, b, op, cs::make_index_sequence<C::rank()>{});
}

/* ---- c = op(c, scalar), elementwise ------------------------------ */
template <class C, class Op, cs::size_t... D>
_TNY_API void scal_(C & c, typename C::element_type s, Op op, cs::index_sequence<D...>) {
    using I  = typename C::index_type;
    using Cv = compute_type_t<typename C::element_type>;   // compute in float for half types
    const Cv sv = static_cast<Cv>(s);
    const I e[]  = { c.extent(D)... };
    const I sc[] = { c.stride(D)... };
    I n = 1;
    for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oc = 0;
        for (int d = static_cast<int>(sizeof...(D)) - 1; d >= 0; --d) {
            I k = rem % e[d]; rem /= e[d]; oc += k * sc[d];
        }
        c.data()[oc] = op(static_cast<Cv>(c.data()[oc]), sv);
    }
}
template <class C, class Op>
_TNY_API void scal(C & c, typename C::element_type s, Op op) {
    scal_(c, s, op, cs::make_index_sequence<C::rank()>{});
}

/* ---- c(i) = op(a(i), scalar) ------------------------------------- */
template <class C, class A, class S, class Op, cs::size_t... D>
_TNY_API void scalo_(C & c, const A & a, S s, Op op, cs::index_sequence<D...>) {
    using I = typename C::index_type; using Cv = compute_type_t<typename C::element_type>;
    const I e[] = { a.extent(D)... }, sa[] = { a.stride(D)... }, sc[] = { c.stride(D)... };
    I n = 1; for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0, oc = 0;
        for (int d = (int)sizeof...(D)-1; d >= 0; --d) { I k = rem%e[d]; rem/=e[d]; oa+=k*sa[d]; oc+=k*sc[d]; }
        c.data()[oc] = op(static_cast<Cv>(a.data()[oa]), static_cast<Cv>(s));
    }
}
template <class C, class A, class S, class Op> _TNY_API void scalo(C & c, const A & a, S s, Op op)
{ scalo_(c, a, s, op, cs::make_index_sequence<C::rank()>{}); }

/* ---- c(i) = uop(a(i))  and  c(i) = uop(c(i)) (in place) ---------- */
template <class C, class A, class Uop, cs::size_t... D>
_TNY_API void unaryo_(C & c, const A & a, Uop f, cs::index_sequence<D...>) {
    using I = typename C::index_type; using Cv = compute_type_t<typename C::element_type>;
    const I e[] = { a.extent(D)... }, sa[] = { a.stride(D)... }, sc[] = { c.stride(D)... };
    I n = 1; for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0, oc = 0;
        for (int d = (int)sizeof...(D)-1; d >= 0; --d) { I k = rem%e[d]; rem/=e[d]; oa+=k*sa[d]; oc+=k*sc[d]; }
        c.data()[oc] = static_cast<Cv>(f(static_cast<Cv>(a.data()[oa])));
    }
}
template <class C, class A, class Uop> _TNY_API void unaryo(C & c, const A & a, Uop f)
{ unaryo_(c, a, f, cs::make_index_sequence<C::rank()>{}); }
template <class C, class Uop> _TNY_API void unary(C & c, Uop f) { unaryo(c, c, f); }

/* ---- broadcast result extents ------------------------------------ *
 * one axis: dynamic if either operand is dynamic, else the non-1 one. */
_TNY_API constexpr cs::size_t bc1(cs::size_t A, cs::size_t B) {
    return (A == cs::dynamic_extent || B == cs::dynamic_extent) ? cs::dynamic_extent
         : (A == 1 ? B : A);
}
template <class Idx, class Ea, class Eb, cs::size_t... D>
cs::extents<Idx, bc1(Ea::static_extent(D), Eb::static_extent(D))...>
bcast_ext_(cs::index_sequence<D...>);
template <class Ea, class Eb>
using bcast_extents = decltype(bcast_ext_<typename Ea::index_type, Ea, Eb>(
    cs::make_index_sequence<Ea::rank()>{}));
// runtime broadcast extents object (for the heap result)
template <class RE, class A, class B, cs::size_t... D>
_TNY_API RE bcast_runtime_(const A & a, const B & b, cs::index_sequence<D...>) {
    using I = typename RE::index_type;
    return RE(static_cast<I>(a.extent(D) == 1 ? b.extent(D) : a.extent(D))...);
}

/* ---- out-of-place tensor (op) tensor, broadcasting --------------- *
 * static -> stack (host+device), else heap (host only).              */
template <class Op, class A, class B,
          cs::enable_if_t<bcast_extents<typename A::extents_type, typename B::extents_type>::rank_dynamic() == 0, int> = 0>
_TNY_API auto oop(const A & a, const B & b, Op op) {
    using RE = bcast_extents<typename A::extents_type, typename B::extents_type>;
    tensor<cs::common_type_t<typename A::element_type, typename B::element_type>, RE, cs::layout_right, own::stack> c{};
    bzip(c, a, b, op); return c;
}
template <class Op, class A, class B,
          cs::enable_if_t<bcast_extents<typename A::extents_type, typename B::extents_type>::rank_dynamic() != 0, int> = 0>
_TNY_HOST auto oop(const A & a, const B & b, Op op) {
    using RE = bcast_extents<typename A::extents_type, typename B::extents_type>;
    tensor<cs::common_type_t<typename A::element_type, typename B::element_type>, RE, cs::layout_right, own::heap>
        c(bcast_runtime_<RE>(a, b, cs::make_index_sequence<A::rank()>{}));
    bzip(c, a, b, op); return c;
}

/* ---- out-of-place tensor (op) scalar ----------------------------- */
template <class Op, class A, class S, cs::enable_if_t<A::is_static, int> = 0>
_TNY_API auto oops(const A & a, S s, Op op) {
    tensor<cs::common_type_t<typename A::element_type, S>, typename A::extents_type, cs::layout_right, own::stack> c{};
    scalo(c, a, s, op); return c;
}
template <class Op, class A, class S, cs::enable_if_t<!A::is_static, int> = 0>
_TNY_HOST auto oops(const A & a, S s, Op op) {
    tensor<cs::common_type_t<typename A::element_type, S>, typename A::extents_type, cs::layout_right, own::heap> c(a.extents());
    scalo(c, a, s, op); return c;
}

/* ---- out-of-place unary : static -> stack, dynamic -> heap ------- */
template <class Uop, class A, cs::enable_if_t<A::is_static, int> = 0>
_TNY_API auto uop_out(const A & a, Uop f) {
    tensor<typename A::element_type, typename A::extents_type, cs::layout_right, own::stack> c{};
    unaryo(c, a, f); return c;
}
template <class Uop, class A, cs::enable_if_t<!A::is_static, int> = 0>
_TNY_HOST auto uop_out(const A & a, Uop f) {
    tensor<typename A::element_type, typename A::extents_type, cs::layout_right, own::heap> c(a.extents());
    unaryo(c, a, f); return c;
}

/* ---- reduce a . b elementwise into a scalar (for dot) ------------ */
template <class R, class A, class B, cs::size_t... D>
_TNY_API R zipreduce_(const A & a, const B & b, cs::index_sequence<D...>) {
    using I = typename A::index_type;
    const I e[]  = { a.extent(D)... };
    const I sa[] = { a.stride(D)... };
    const I sb[] = { b.stride(D)... };
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

} // namespace _md

/* ------------------------------------------------------------------ *
 *     In-place members                                               *
 * ------------------------------------------------------------------ */

template <class T,class E,class L,own O> template <class B>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::add_(const B & b) { _md::bzip(*this,*this,b,_md::add{}); return *this; }
template <class T,class E,class L,own O> template <class B>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::sub_(const B & b) { _md::bzip(*this,*this,b,_md::sub{}); return *this; }
template <class T,class E,class L,own O> template <class B>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::mul_(const B & b) { _md::bzip(*this,*this,b,_md::mul{}); return *this; }
template <class T,class E,class L,own O> template <class B>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::div_(const B & b) { _md::bzip(*this,*this,b,_md::div{}); return *this; }
template <class T,class E,class L,own O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::add_(T s) { _md::scal(*this,s,_md::add{}); return *this; }
template <class T,class E,class L,own O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::sub_(T s) { _md::scal(*this,s,_md::sub{}); return *this; }
template <class T,class E,class L,own O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::mul_(T s) { _md::scal(*this,s,_md::mul{}); return *this; }
template <class T,class E,class L,own O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::div_(T s) { _md::scal(*this,s,_md::div{}); return *this; }
template <class T,class E,class L,own O> template <class B>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::copy_(const B & b) { _md::bzip(*this,*this,b,_md::rhs{}); return *this; }
template <class T,class E,class L,own O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::fill_(T s) { _md::scal(*this,s,_md::setc{}); return *this; }
template <class T,class E,class L,own O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::zero_() { return fill_(T(0)); }

/* ------------------------------------------------------------------ *
 *     Out-of-place operators                                         *
 *                                                                    *
 *  - fully-static extents  -> stack-owned result, host AND device.   *
 *  - any dynamic extent    -> heap-owned result, HOST ONLY (the      *
 *                             result must be allocated at run time).  *
 * ------------------------------------------------------------------ */

// tensor (op) tensor operators -> the broadcasting out-of-place engine.
#define _TNY_MD_BINOP(SYM, OP)                                                                    \
template <class Ta,class Ea,class La,own Oa, class Tb,class Eb,class Lb,own Ob>                   \
_TNY_API auto operator SYM (const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b)        \
{ return _md::oop(a, b, OP{}); }
_TNY_MD_BINOP(+, _md::add)
_TNY_MD_BINOP(-, _md::sub)
_TNY_MD_BINOP(*, _md::mul)
_TNY_MD_BINOP(/, _md::div)
#undef _TNY_MD_BINOP

/* --- out-of-place binary methods (tensor OR scalar rhs) ----------- */
#define _TNY_MD_METHOD(NAME, OP)                                                                 \
template <class T,class E,class L,own O> template <class B>                                       \
_TNY_API auto tensor<T,E,L,O>::NAME(const B & b) const {                                          \
    if constexpr (cs::is_arithmetic<B>::value) return _md::oops(*this, b, OP{});                  \
    else                                       return _md::oop (*this, b, OP{});                   \
}
_TNY_MD_METHOD(add, _md::add)
_TNY_MD_METHOD(sub, _md::sub)
_TNY_MD_METHOD(mul, _md::mul)
_TNY_MD_METHOD(div, _md::div)
_TNY_MD_METHOD(pow, _md::pw)
#undef _TNY_MD_METHOD

/* --- tensor (op) scalar and scalar (op) tensor operators ---------- */
#define _TNY_MD_SCALOP(SYM, OP)                                                                   \
template <class T,class E,class L,own O, class S, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0> \
_TNY_API auto operator SYM (const tensor<T,E,L,O> & a, S s) { return _md::oops(a, s, OP{}); }
_TNY_MD_SCALOP(+, _md::add)
_TNY_MD_SCALOP(-, _md::sub)
_TNY_MD_SCALOP(*, _md::mul)
_TNY_MD_SCALOP(/, _md::div)
#undef _TNY_MD_SCALOP
// scalar (+)/(*) tensor  (commutative)
template <class S, class T,class E,class L,own O, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0>
_TNY_API auto operator+(S s, const tensor<T,E,L,O> & a) { return _md::oops(a, s, _md::add{}); }
template <class S, class T,class E,class L,own O, cs::enable_if_t<cs::is_arithmetic<S>::value, int> = 0>
_TNY_API auto operator*(S s, const tensor<T,E,L,O> & a) { return _md::oops(a, s, _md::mul{}); }

/* --- in-place unary methods --------------------------------------- */
#define _TNY_MD_UNARY_(NAME, F)                                                                   \
template <class T,class E,class L,own O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::NAME()        \
{ _md::unary(*this, F{}); return *this; }
_TNY_MD_UNARY_(neg_,  _md::u_neg)
_TNY_MD_UNARY_(abs_,  _md::u_abs)
_TNY_MD_UNARY_(exp_,  _md::u_exp)
_TNY_MD_UNARY_(log_,  _md::u_log)
_TNY_MD_UNARY_(sin_,  _md::u_sin)
_TNY_MD_UNARY_(cos_,  _md::u_cos)
_TNY_MD_UNARY_(sqrt_, _md::u_sqrt)
_TNY_MD_UNARY_(tanh_, _md::u_tanh)
#undef _TNY_MD_UNARY_
template <class T,class E,class L,own O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::pow_(T e)
{ _md::scal(*this, e, _md::pw{}); return *this; }

/* ------------------------------------------------------------------ *
 *     Reductions                                                      *
 * ------------------------------------------------------------------ */

// reductions accumulate in compute_type<T> (float for half types), then cast
// back to T -- avoids catastrophic precision loss when summing many 16-bit values.

/** @brief Sum of all elements (empty -> 0). */
template <class T, class E, class L, own O>
_TNY_API T sum(const tensor<T,E,L,O> & a) {
    using R = compute_type_t<T>;
    return static_cast<T>(_md::reduce_<R>(a, R(0), _md::r_add{}, cs::make_index_sequence<tensor<T,E,L,O>::rank()>{}));
}
/** @brief Product of all elements (empty -> 1). */
template <class T, class E, class L, own O>
_TNY_API T prod(const tensor<T,E,L,O> & a) {
    using R = compute_type_t<T>;
    return static_cast<T>(_md::reduce_<R>(a, R(1), _md::r_mul{}, cs::make_index_sequence<tensor<T,E,L,O>::rank()>{}));
}
/** @brief Maximum element. Requires a non-empty tensor. */
template <class T, class E, class L, own O>
_TNY_API T max(const tensor<T,E,L,O> & a) {
    using R = compute_type_t<T>;
    return static_cast<T>(_md::reduce_<R>(a, cs::numeric_limits<R>::lowest(), _md::r_max{}, cs::make_index_sequence<tensor<T,E,L,O>::rank()>{}));
}
/** @brief Minimum element. Requires a non-empty tensor. */
template <class T, class E, class L, own O>
_TNY_API T min(const tensor<T,E,L,O> & a) {
    using R = compute_type_t<T>;
    return static_cast<T>(_md::reduce_<R>(a, cs::numeric_limits<R>::max(), _md::r_min{}, cs::make_index_sequence<tensor<T,E,L,O>::rank()>{}));
}

/** @brief Inner product over matching extents (accumulated in the compute type). */
template <class Ta,class Ea,class La,own Oa, class Tb,class Eb,class Lb,own Ob>
_TNY_API auto dot(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b) {
    static_assert(tensor<Ta,Ea,La,Oa>::rank() == tensor<Tb,Eb,Lb,Ob>::rank(), "dot: rank mismatch");
    using R = compute_type_t<cs::common_type_t<Ta,Tb>>;
    return _md::zipreduce_<R>(a, b, cs::make_index_sequence<tensor<Ta,Ea,La,Oa>::rank()>{});
}

/* --- out-of-place unary free functions ---------------------------- */
#define _TNY_MD_UNARY(NAME, F)                                                                    \
template <class T,class E,class L,own O> _TNY_API auto NAME(const tensor<T,E,L,O> & a)             \
{ return _md::uop_out(a, F{}); }
_TNY_MD_UNARY(neg,  _md::u_neg)
_TNY_MD_UNARY(abs,  _md::u_abs)
_TNY_MD_UNARY(exp,  _md::u_exp)
_TNY_MD_UNARY(log,  _md::u_log)
_TNY_MD_UNARY(sin,  _md::u_sin)
_TNY_MD_UNARY(cos,  _md::u_cos)
_TNY_MD_UNARY(sqrt, _md::u_sqrt)
_TNY_MD_UNARY(tanh, _md::u_tanh)
#undef _TNY_MD_UNARY

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_MATH
