#ifndef TNY_MD_MATH
#define TNY_MD_MATH
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <cuda/std/limits>
#include <cuda/std/cmath>
#include <teeny/_core/defines.h>
#include <teeny/tensor.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/* ================================================================== *
 *  valarray-like math.                                               *
 *                                                                    *
 *  - In-place ops (`a.add_(b)`, `a.mul_(scalar)`) work on ANY tensor  *
 *    or view and mutate it in place.                                  *
 *  - Out-of-place ops (`a + b`) are enabled ONLY when the result      *
 *    extent is statically known, so the result can be a stack-owned   *
 *    tensor; a dynamic-extent operand makes them ill-formed.          *
 *                                                                    *
 *  All engines are lambda-free (index-sequence folds + tiny functors) *
 *  so they instantiate on device without `--extended-lambda`, and a   *
 *  fully-static shape folds the whole thing to straight-line code.    *
 * ================================================================== */

namespace _md {

// binary element ops (out = x OP y)
struct add { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return x + static_cast<X>(y); } };
struct sub { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return x - static_cast<X>(y); } };
struct mul { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return x * static_cast<X>(y); } };
struct div { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return x / static_cast<X>(y); } };

// c = op(a, b), elementwise over matching extents (any layouts).
template <class C, class A, class B, class Op, cs::size_t... D>
_TNY_API void zip_(C & c, const A & a, const B & b, Op op, cs::index_sequence<D...>) {
    using I  = typename C::index_type;
    using Cv = typename C::element_type;       // compute in the destination type
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

// c = op(c, scalar), elementwise.
template <class C, class Op, cs::size_t... D>
_TNY_API void scal_(C & c, typename C::element_type s, Op op, cs::index_sequence<D...>) {
    using I = typename C::index_type;
    const I e[]  = { c.extent(D)... };
    const I sc[] = { c.stride(D)... };
    I n = 1;
    for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oc = 0;
        for (int d = static_cast<int>(sizeof...(D)) - 1; d >= 0; --d) {
            I k = rem % e[d]; rem /= e[d]; oc += k * sc[d];
        }
        c.data()[oc] = op(c.data()[oc], s);
    }
}
template <class C, class Op>
_TNY_API void scal(C & c, typename C::element_type s, Op op) {
    scal_(c, s, op, cs::make_index_sequence<C::rank()>{});
}

// reduce a . b elementwise into a scalar (for dot).
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

// reduce ops (acc = op(acc, x))
struct r_add { template <class A, class X> _TNY_API A operator()(A a, X x) const { return a + static_cast<A>(x); } };
struct r_mul { template <class A, class X> _TNY_API A operator()(A a, X x) const { return a * static_cast<A>(x); } };
struct r_max { template <class A, class X> _TNY_API A operator()(A a, X x) const { A y = static_cast<A>(x); return y > a ? y : a; } };
struct r_min { template <class A, class X> _TNY_API A operator()(A a, X x) const { A y = static_cast<A>(x); return y < a ? y : a; } };

// fold a into a scalar with `op`, starting from `init`.
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
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::add_(const B & b) { _md::zip(*this,*this,b,_md::add{}); return *this; }
template <class T,class E,class L,own O> template <class B>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::sub_(const B & b) { _md::zip(*this,*this,b,_md::sub{}); return *this; }
template <class T,class E,class L,own O> template <class B>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::mul_(const B & b) { _md::zip(*this,*this,b,_md::mul{}); return *this; }
template <class T,class E,class L,own O> template <class B>
_TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::div_(const B & b) { _md::zip(*this,*this,b,_md::div{}); return *this; }
template <class T,class E,class L,own O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::add_(T s) { _md::scal(*this,s,_md::add{}); return *this; }
template <class T,class E,class L,own O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::sub_(T s) { _md::scal(*this,s,_md::sub{}); return *this; }
template <class T,class E,class L,own O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::mul_(T s) { _md::scal(*this,s,_md::mul{}); return *this; }
template <class T,class E,class L,own O> _TNY_API tensor<T,E,L,O> & tensor<T,E,L,O>::div_(T s) { _md::scal(*this,s,_md::div{}); return *this; }

/* ------------------------------------------------------------------ *
 *     Out-of-place operators                                         *
 *                                                                    *
 *  - fully-static extents  -> stack-owned result, host AND device.   *
 *  - any dynamic extent    -> heap-owned result, HOST ONLY (the      *
 *                             result must be allocated at run time).  *
 * ------------------------------------------------------------------ */

#define _TNY_MD_BINOP(SYM, OP)                                                                    \
template <class Ta,class Ea,class La,own Oa, class Tb,class Eb,class Lb,own Ob,                   \
          cs::enable_if_t<tensor<Ta,Ea,La,Oa>::is_static && tensor<Tb,Eb,Lb,Ob>::is_static, int> = 0> \
_TNY_API auto operator SYM (const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b) {      \
    static_assert(cs::is_same<Ea,Eb>::value, "operator " #SYM ": extents must match");           \
    tensor<cs::common_type_t<Ta,Tb>, Ea, cs::layout_right, own::stack> c{};                       \
    _md::zip(c, a, b, OP{});                                                                      \
    return c;                                                                                     \
}                                                                                                \
template <class Ta,class Ea,class La,own Oa, class Tb,class Eb,class Lb,own Ob,                   \
          cs::enable_if_t<!(tensor<Ta,Ea,La,Oa>::is_static && tensor<Tb,Eb,Lb,Ob>::is_static), int> = 0> \
_TNY_HOST auto operator SYM (const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b) {     \
    static_assert(cs::is_same<Ea,Eb>::value, "operator " #SYM ": extents must match");           \
    auto c = owned<cs::common_type_t<Ta,Tb>, Ea>(a.extents());                                   \
    _md::zip(c, a, b, OP{});                                                                      \
    return c;                                                                                     \
}
_TNY_MD_BINOP(+, _md::add)
_TNY_MD_BINOP(-, _md::sub)
_TNY_MD_BINOP(*, _md::mul)
_TNY_MD_BINOP(/, _md::div)
#undef _TNY_MD_BINOP

/* ------------------------------------------------------------------ *
 *     Reductions                                                      *
 * ------------------------------------------------------------------ */

/** @brief Sum of all elements (empty -> 0). */
template <class T, class E, class L, own O>
_TNY_API T sum(const tensor<T,E,L,O> & a) {
    return _md::reduce_<T>(a, T(0), _md::r_add{}, cs::make_index_sequence<tensor<T,E,L,O>::rank()>{});
}
/** @brief Product of all elements (empty -> 1). */
template <class T, class E, class L, own O>
_TNY_API T prod(const tensor<T,E,L,O> & a) {
    return _md::reduce_<T>(a, T(1), _md::r_mul{}, cs::make_index_sequence<tensor<T,E,L,O>::rank()>{});
}
/** @brief Maximum element. Requires a non-empty tensor. */
template <class T, class E, class L, own O>
_TNY_API T max(const tensor<T,E,L,O> & a) {
    return _md::reduce_<T>(a, cs::numeric_limits<T>::lowest(), _md::r_max{}, cs::make_index_sequence<tensor<T,E,L,O>::rank()>{});
}
/** @brief Minimum element. Requires a non-empty tensor. */
template <class T, class E, class L, own O>
_TNY_API T min(const tensor<T,E,L,O> & a) {
    return _md::reduce_<T>(a, cs::numeric_limits<T>::max(), _md::r_min{}, cs::make_index_sequence<tensor<T,E,L,O>::rank()>{});
}

/** @brief Inner product over matching extents. */
template <class Ta,class Ea,class La,own Oa, class Tb,class Eb,class Lb,own Ob>
_TNY_API auto dot(const tensor<Ta,Ea,La,Oa> & a, const tensor<Tb,Eb,Lb,Ob> & b) {
    static_assert(tensor<Ta,Ea,La,Oa>::rank() == tensor<Tb,Eb,Lb,Ob>::rank(), "dot: rank mismatch");
    using R = cs::common_type_t<Ta,Tb>;
    return _md::zipreduce_<R>(a, b, cs::make_index_sequence<tensor<Ta,Ea,La,Oa>::rank()>{});
}

/* ================================================================== *
 *  Out-of-place elementwise ops + unary math (items 4 & 6)           *
 * ================================================================== */

namespace _md {

// c(i) = op(a(i), scalar)
template <class C, class A, class S, class Op, cs::size_t... D>
_TNY_API void scalo_(C & c, const A & a, S s, Op op, cs::index_sequence<D...>) {
    using I = typename C::index_type; using Cv = typename C::element_type;
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

// c(i) = uop(a(i))
template <class C, class A, class Uop, cs::size_t... D>
_TNY_API void unaryo_(C & c, const A & a, Uop f, cs::index_sequence<D...>) {
    using I = typename C::index_type; using Cv = typename C::element_type;
    const I e[] = { a.extent(D)... }, sa[] = { a.stride(D)... }, sc[] = { c.stride(D)... };
    I n = 1; for (cs::size_t r = 0; r < sizeof...(D); ++r) n *= e[r];
    for (I lin = 0; lin < n; ++lin) {
        I rem = lin, oa = 0, oc = 0;
        for (int d = (int)sizeof...(D)-1; d >= 0; --d) { I k = rem%e[d]; rem/=e[d]; oa+=k*sa[d]; oc+=k*sc[d]; }
        c.data()[oc] = static_cast<Cv>(f(a.data()[oa]));
    }
}
template <class C, class A, class Uop> _TNY_API void unaryo(C & c, const A & a, Uop f)
{ unaryo_(c, a, f, cs::make_index_sequence<C::rank()>{}); }

// c(i) = uop(c(i))  (in place)
template <class C, class Uop> _TNY_API void unary(C & c, Uop f) { unaryo(c, c, f); }

// unary functors (cuda::std math -> device-callable)
struct u_neg  { template <class X> _TNY_API X operator()(X x) const { return -x; } };
struct u_abs  { template <class X> _TNY_API X operator()(X x) const { return x < X(0) ? -x : x; } };
struct u_exp  { template <class X> _TNY_API X operator()(X x) const { return cs::exp(x); } };
struct u_log  { template <class X> _TNY_API X operator()(X x) const { return cs::log(x); } };
struct u_sin  { template <class X> _TNY_API X operator()(X x) const { return cs::sin(x); } };
struct u_cos  { template <class X> _TNY_API X operator()(X x) const { return cs::cos(x); } };
struct u_sqrt { template <class X> _TNY_API X operator()(X x) const { return cs::sqrt(x); } };
struct u_tanh { template <class X> _TNY_API X operator()(X x) const { return cs::tanh(x); } };
struct pw     { template <class X, class Y> _TNY_API X operator()(X x, Y y) const { return cs::pow(x, static_cast<X>(y)); } };

// out-of-place tensor (+) tensor : static -> stack, dynamic -> heap (host)
template <class Op, class A, class B, cs::enable_if_t<A::is_static && B::is_static, int> = 0>
_TNY_API auto oop(const A & a, const B & b, Op op) {
    static_assert(cs::is_same<typename A::extents_type, typename B::extents_type>::value, "extents must match");
    tensor<cs::common_type_t<typename A::element_type, typename B::element_type>, typename A::extents_type, cs::layout_right, own::stack> c{};
    zip(c, a, b, op); return c;
}
template <class Op, class A, class B, cs::enable_if_t<!(A::is_static && B::is_static), int> = 0>
_TNY_HOST auto oop(const A & a, const B & b, Op op) {
    static_assert(cs::is_same<typename A::extents_type, typename B::extents_type>::value, "extents must match");
    tensor<cs::common_type_t<typename A::element_type, typename B::element_type>, typename A::extents_type, cs::layout_right, own::heap> c(a.extents());
    zip(c, a, b, op); return c;
}
// out-of-place tensor (+) scalar
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
// out-of-place unary : static -> stack, dynamic -> heap (host)
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
} // namespace _md

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

/* --- tensor (+) scalar and scalar (+) tensor operators ------------ */
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
