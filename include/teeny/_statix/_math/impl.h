/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 **     This file declares metaprogramming math.                            **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/

/*
 * FIXME
 *      For reductions, I currently convert to carray first and then
 *      apply the reduction (with values being compacted from then end).
 *      It would be nicer to move through the input in a container-agnostic
 *      way, similarly to what I do for element-wise operations.
 */

#ifndef TNY__STATIX__MATH_IMPL
#define TNY__STATIX__MATH_IMPL

#include <cuda/std/type_traits>
// #include <cuda/std/limits>
#include <limits>

#include <teeny/_core/defines.h>
#include <teeny/_statix/_math/decl.h>
#include <teeny/_statix/_carray/decl.h>
#include <teeny/_statix/_packapi/decl.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

using cuda::std::conditional_t;
using std::numeric_limits;

#define _TNY_PROD(A,B)      (A * B)
#define _TNY_SUM(A,B)       (A + B)
#define _TNY_SUB(A,B)       (A - B)
#define _TNY_DIV(A,B)       (A / B)
#define _TNY_MIN(A,B)       (A <= B ? A : B)
#define _TNY_MAX(A,B)       (A >= B ? A : B)
#define _TNY_OR(A,B)        (A | B)
#define _TNY_AND(A,B)       (A & B)
#define _TNY_LSH(A,B)       (A << B)
#define _TNY_RSH(A,B)       (A >> B)
#define _TNY_GT(A,B)        (A > B)
#define _TNY_GE(A,B)        (A >= B)
#define _TNY_LT(A,B)        (A < B)
#define _TNY_LE(A,B)        (A <= B)
#define _TNY_EQ(A,B)        (A == B)
#define _TNY_NE(A,B)        (A != B)
#define _TNY_NEG(A)         (-A)
#define _TNY_NOT(A)         (~A)
#define _TNY_ABS(A)         (A < 0 ? -A : A)
#define _TNY_BOOLOR(A,B)    (A || B)
#define _TNY_BOOLAND(A,B)   (A && B)
#define _TNY_BOOLXOR(A,B)   (!(A && B) && (A || B))
#define _TNY_BOOLNOT(A)     (!A)

/* ================================================================== *
 *     Comparisons                                                    *
 * ================================================================== */

#define _TNY_DEFINE(NAME,TEST) \
    template <class T, T... X>       struct _is_##NAME##_value                {}; \
    template <class T>               struct _is_##NAME##_value<T>             { using type = cbool<>; }; \
    template <class T, T X>          struct _is_##NAME##_value<T, X>          { using type = conditional_t<_TNY_##TEST(X,0), ctrue, cfalse>; }; \
    template <class T, T X0, T... X> struct _is_##NAME##_value<T, X0, X...>   { using type = cat<is_##NAME##_value<T,X0>, is_##NAME##_value<T,X...>>; }; \
    template <class X>               struct _is_##NAME                        { using type = is_##NAME<as_carray<X>>; }; \
    template <class T, T... X>       struct _is_##NAME<carray<T, X...>>       { using type = is_##NAME##_value<T, X...>; };


_TNY_DEFINE(positive,     GT)
_TNY_DEFINE(negative,     LT)
_TNY_DEFINE(zero,         EQ)
_TNY_DEFINE(nonzero,      NE)
_TNY_DEFINE(nonpositive,  LE)
_TNY_DEFINE(nonnegative,  GE)
#undef _TNY_DEFINE

#define _TNY_DEFINE(NAME,TEST) \
    template <class T, T X, T... Y>        struct _is_##NAME##_value                 {}; \
    template <class T, T X>                struct _is_##NAME##_value<T, X>           { using type = cbool<>; }; \
    template <class T, T X, T Y>           struct _is_##NAME##_value<T, X, Y>        { using type = conditional_t<_TNY_##TEST(X,Y), ctrue, cfalse>; }; \
    template <class T, T X, T Y0, T... Y>  struct _is_##NAME##_value<T, X, Y0, Y...> { using type = cat<is_##NAME##_value<T,X,Y0>, is_##NAME##_value<T,X,Y...>>; }; \
    template <class X, class Y>            struct _is_##NAME                         { \
    private: \
        static constexpr bool LX1 = size<X>::value > 1; \
        static constexpr bool LY1 = size<Y>::value > 1; \
        using first  = is_##NAME<front<X>, front<Y>>; \
        using next_x = conditional_t<LX1, erase_head<X>, X>; \
        using next_y = conditional_t<LY1, erase_head<Y>, Y>; \
        using next   = conditional_t<LX1 || LY1, is_##NAME<next_x, next_y>, empty_like<first>>; \
    public: \
        using type = cat<first, next>; \
    }; \
    template <class U, U X, class T, T Y>  struct _is_##NAME<carray<U,X>, carray<T,Y>> { using type = conditional_t<_TNY_##TEST(X,Y), ctrue, cfalse>; };

_TNY_DEFINE(greater,       GT)
_TNY_DEFINE(lower,         LT)
_TNY_DEFINE(equal,         EQ)
_TNY_DEFINE(not_equal,     NE)
_TNY_DEFINE(greater_equal, GE)
_TNY_DEFINE(lower_equal,   LE)
#undef _TNY_DEFINE

/* ================================================================== *
 *     Operations                                                     *
 * ================================================================== */

#define _TNY_DEFINE(NAME, OP) \
    template <class T, T... X>       struct _##NAME##_value              {}; \
    template <class T, T... X>       using     NAME##_value              = typename _##NAME##_value<T, X...>::type; \
    template <class T, T X0, T... X> struct _##NAME##_value<T, X0, X...> { using type = cat<carray<T, _TNY_##OP(X0)>, NAME<carray<T, X...>>>; }; \
    template <class T>               struct _##NAME##_value<T>           { using type = carray<T>; }; \
    template <class X, size_t L>     struct _##NAME                      { using type = cat<NAME<front<X>>, NAME<erase_head<X>>>; }; \
    template <class X>               struct _##NAME<X,0>                 { using type = empty_like<as_carray<X>>; }; \
    template <class T, T X>          struct _##NAME<carray<T,X>,1>       { using type = NAME##_value<T, X>; }; \

_TNY_DEFINE(abs,         ABS)
_TNY_DEFINE(neg,         NEG)
_TNY_DEFINE(boolean_not, BOOLNOT)
_TNY_DEFINE(bitwise_not, NOT)
#undef _TNY_DEFINE

#define _TNY_DEFINE(NAME,OP) \
    template <class T, T X, T... Y>          struct _##NAME##_value                   {}; \
    template <class T, T X>                  struct _##NAME##_value<T, X>             { using type = carray<T>; }; \
    template <class T, T X, T Y>             struct _##NAME##_value<T, X, Y>          { using type = carray<T, _TNY_##OP(X ,Y)>; }; \
    template <class T, T X, T Y0, T... Y>    struct _##NAME##_value<T, X, Y0, Y...>   { using type = cat<typename _##NAME##_value<T,X,Y0>::type, typename _##NAME##_value<T,X,Y...>::type>; }; \
    template <class X, class... Y>           struct _##NAME                           {}; \
    template <class X>                       struct _##NAME<X>                        { using type = X; }; \
    template <class X, class Y, class... Z>  struct _##NAME<X, Y, Z...>               { using type = NAME<NAME<X,Y>,Z...>; }; \
    template <class X, class Y>              struct _##NAME<X, Y>                     { \
    private: \
        static constexpr bool LX1 = size<X>::value > 1; \
        static constexpr bool LY1 = size<Y>::value > 1; \
        using first  = NAME<front<X>, front<Y>>; \
        using next_x = conditional_t<LX1, erase_head<X>, X>; \
        using next_y = conditional_t<LY1, erase_head<Y>, Y>; \
        using next   = conditional_t<LX1 || LY1, NAME<next_x, next_y>, empty_like<first>>; \
    public: \
        using type = cat<first, next>; \
    }; \
    template <class U, U X, class T, T Y>    struct _##NAME<carray<U,X>, carray<T,Y>> { using type = carray<decltype(_TNY_##OP(X,Y)), _TNY_##OP(X,Y)>; };

_TNY_DEFINE(add,         SUM)
_TNY_DEFINE(sub,         SUB)
_TNY_DEFINE(mul,         PROD)
_TNY_DEFINE(div,         DIV)
_TNY_DEFINE(boolean_and, BOOLAND)
_TNY_DEFINE(boolean_or,  BOOLOR)
_TNY_DEFINE(boolean_xor, BOOLXOR)
_TNY_DEFINE(bitwise_and, AND)
_TNY_DEFINE(bitwise_or,  OR)
_TNY_DEFINE(bitwise_xor, XOR)
_TNY_DEFINE(lshift,      LS)
_TNY_DEFINE(rshift,      RS)
_TNY_DEFINE(minimum,     MIN)
_TNY_DEFINE(maximum,     MAX)
#undef _TNY_DEFINE

/* ================================================================== *
 *     Reductions                                                     *
 * ================================================================== */

#define _TNY_DEFINE(NAME, OP, INIT) \
    template <class T, T... X>       struct _##NAME##_values               {}; \
    template <class T, T X0, T... X> struct _##NAME##_values<T, X0, X...> { using type = OP<carray<T,X0>, NAME##_values<T, X...>>; }; \
    template <class T, T X0>         struct _##NAME##_values<T, X0>       { using type = OP<carray<T,static_cast<T>(INIT)>, carray<T,X0>>; }; \
    template <class T>               struct _##NAME##_values<T>           { using _OT  = value_type<OP<carray<T,0>, carray<T,0>>>; \
                                                                            using type = carray<_OT, static_cast<_OT>(INIT)>; }; \
    template <class A>               struct _##NAME                       { using type = NAME<as_carray<A>>; }; \
    template <class T, T... X>       struct _##NAME<carray<T, X...>>      { using type = NAME##_values<T, X...>; };

_TNY_DEFINE(prod, mul,          1)
_TNY_DEFINE(sum,  add,          0)
_TNY_DEFINE(min,  minimum,      numeric_limits<T>::max())
_TNY_DEFINE(max,  maximum,      numeric_limits<T>::min())
_TNY_DEFINE(any,  boolean_or,   false)
_TNY_DEFINE(all,  boolean_and,  true)
#undef _TNY_DEFINE

/* ================================================================== */
/*     Cumulative Reductions                                          */
/* ================================================================== */

#define _TNY_DEFINE(NAME, OP) \
    template <class T, T... X>             struct _##NAME##_values                   {}; \
    template <class T, T X0, T X1, T... X> struct _##NAME##_values<T, X0, X1, X...>  { \
    private:\
        using First       = carray<T,X0>; \
        using Second      = OP<carray<T,X0>, carray<T,X1>>; \
        using Third       = carray<T, X...>; \
        using SecondThird = NAME<cat<Second, Third>>; \
    public: \
        using type = cat<First, SecondThird>; \
    }; \
    template <class T, T X0>         struct _##NAME##_values<T, X0>      { using _OT  = value_type<OP<carray<T,0>, carray<T,0>>>; \
                                                                           using type = carray<_OT, static_cast<_OT>(X0)>; }; \
    template <class T>               struct _##NAME##_values<T>          { using type = carray<T>; }; \
    template <class A>               struct _##NAME                      { using type = NAME<as_carray<A>>; }; \
    template <class T, T... X>       struct _##NAME<carray<T, X...>>     { using type = NAME##_values<T, X...>; };

_TNY_DEFINE(cumprod, mul)
_TNY_DEFINE(cumsum,  add)
_TNY_DEFINE(cummin,  minimum)
_TNY_DEFINE(cummax,  maximum)
_TNY_DEFINE(cumany,  boolean_or)
_TNY_DEFINE(cumall,  boolean_and)
#undef _TNY_DEFINE

#define _TNY_DEFINE(NAME, OP, INIT) \
    template <class T, T... X>       struct _##NAME##_values              {}; \
    template <class T, T X0, T... X> struct _##NAME##_values<T, X0, X...> { using type = cat<carray<T,static_cast<T>(INIT)>, OP<carray<T,X0>,  NAME##_values<T, X...>>>; }; \
    template <class T, T X0, T X1>   struct _##NAME##_values<T, X0, X1>   { using _OT  = value_type<OP<carray<T,0>, carray<T,0>>>; \
                                                                            using type = carray<_OT, static_cast<_OT>(INIT), static_cast<_OT>(X0)>; }; \
    template <class T, T X0>         struct _##NAME##_values<T, X0>       { using _OT  = value_type<OP<carray<T,0>, carray<T,0>>>; \
                                                                            using type = carray<_OT, static_cast<_OT>(INIT)>; }; \
    template <class T>               struct _##NAME##_values<T>           { using type = carray<T>; }; \
    template <class A>               struct _##NAME                       { using type = NAME<as_carray<A>>; }; \
    template <class T, T... X>       struct _##NAME<carray<T, X...>>      { using type = NAME##_values<T, X...>; };
_TNY_DEFINE(shifted_cumprod, mul,         1)
_TNY_DEFINE(shifted_cumsum,  add,         0)
_TNY_DEFINE(shifted_cummin,  minimum,     numeric_limits<T>::max())
_TNY_DEFINE(shifted_cummax,  maximum,     numeric_limits<T>::min())
_TNY_DEFINE(shifted_cumany,  boolean_or,  false)
_TNY_DEFINE(shifted_cumall,  boolean_and, true)
#undef _TNY_DEFINE

/* ================================================================== *
 *     Count                                                          *
 * ================================================================== */

template <class... T>               struct __countT              {};
template <class T0, class... T>     struct __countT<T0, T...>    { static constexpr size_t value = 1 + __countT<T...>::value; };
template <class T0>                 struct __countT<T0>          { static constexpr size_t value = 1; };
template <>                         struct __countT<>            { static constexpr size_t value = 0; };
template <class... T>               struct  _countT              { using type = csize<__countT<T...>::value>; };

template <class T, T... X>          struct __countV              {};
template <class T, T X0, T... X>    struct __countV<T, X0, X...> { static constexpr size_t value = 1 + __countV<T, X...>::value; };
template <class T, T X0>            struct __countV<T, X0>       { static constexpr size_t value = 1; };
template <class T>                  struct __countV<T>           { static constexpr size_t value = 0; };
template <class T, T... X>          struct  _countV              { using type = csize<__countV<T, X...>::value>; };

/* ================================================================== *
 *     IsIn                                                           *
 * ================================================================== */

template <class A, class B>
struct _isin {};

template <class B, class T, T X>
struct _isin<carray<T, X>, B> {
    using type = any<is_equal<B, carray<T, X>>>;
};

template <class B, class T, T X0, T... X>
struct _isin<carray<T, X0, X...>, B> {
    using type = cat<
        _isin<carray<T, X0>,   B>,
        _isin<carray<T, X...>, B>
    >;
};

/* ================================================================== *
 *     NonZero                                                        *
 * ================================================================== */

template <class A>
struct _nonzero {};

template <class T>
struct _nonzero<carray<T>> {
    using type = csize<>;
};

template <class T, T X>
struct _nonzero<carray<T, X>> {
    using type = csize<0>;
};

template <class T>
struct _nonzero<carray<T, 0>> {
    using type = csize<>;
};

template <class T, T X0, T... X>
struct _nonzero<carray<T, X0, X...>> {
    using type = cat<
        nonzero<carray<T, X0>>,
        add<nonzero<carray<T, X...>>, csize<1>>
    >;
};

#undef _TNY_PROD
#undef _TNY_SUM
#undef _TNY_SUB
#undef _TNY_DIV
#undef _TNY_MIN
#undef _TNY_MAX
#undef _TNY_OR
#undef _TNY_AND
#undef _TNY_LSH
#undef _TNY_RSH
#undef _TNY_GT
#undef _TNY_GE
#undef _TNY_LT
#undef _TNY_LE
#undef _TNY_EQ
#undef _TNY_NE
#undef _TNY_NEG
#undef _TNY_NOT
#undef _TNY_ABS
#undef _TNY_BOOLOR
#undef _TNY_BOOLAND
#undef _TNY_BOOLXOR
#undef _TNY_BOOLNOT

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif // TNY__STATIX__MATH_IMPL
