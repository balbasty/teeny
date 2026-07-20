#ifndef TNY_MD_MATH
#define TNY_MD_MATH
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <cuda/std/limits>
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

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_MATH
