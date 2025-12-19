/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 **     This file declares metaprogramming math.                            **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/

/*
 * FIXME
 *      For reductions, I currently convert to vector first and then
 *      apply the reduction (with values being compacted from then end).
 *      It would be nicer to move through the input in a container-agnostic
 *      way, similarly to what I do for element-wise operations.
 */

#ifndef MINITEN__META__MATH_IMPL
#define MINITEN__META__MATH_IMPL

#include <miniten/_core/defines.h>
#include <miniten/_meta/traits.h>
#include <miniten/_meta/_math/decl.h>
#include <miniten/_meta/_vector/decl.h>
#include <miniten/_meta/_packapi/decl.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

#define _MINITEN_PROD(A,B)      (A * B)
#define _MINITEN_SUM(A,B)       (A + B)
#define _MINITEN_SUB(A,B)       (A - B)
#define _MINITEN_DIV(A,B)       (A / B)
#define _MINITEN_MIN(A,B)       (A <= B ? A : B)
#define _MINITEN_MAX(A,B)       (A >= B ? A : B)
#define _MINITEN_OR(A,B)        (A | B)
#define _MINITEN_AND(A,B)       (A & B)
#define _MINITEN_LSH(A,B)       (A << B)
#define _MINITEN_RSH(A,B)       (A >> B)
#define _MINITEN_GT(A,B)        (A > B)
#define _MINITEN_GE(A,B)        (A >= B)
#define _MINITEN_LT(A,B)        (A < B)
#define _MINITEN_LE(A,B)        (A <= B)
#define _MINITEN_EQ(A,B)        (A == B)
#define _MINITEN_NE(A,B)        (A != B)
#define _MINITEN_NEG(A)         (-A)
#define _MINITEN_NOT(A)         (~A)
#define _MINITEN_ABS(A)         (A < 0 ? -A : A)
#define _MINITEN_BOOLOR(A,B)    (A || B)
#define _MINITEN_BOOLAND(A,B)   (A && B)
#define _MINITEN_BOOLXOR(A,B)   (!(A && B) && (A || B))
#define _MINITEN_BOOLNOT(A)     (!A)

/* ================================================================== *
 *     Comparisons                                                    *
 * ================================================================== */

#define MINITEN_DEFINE(NAME,TEST) \
    template <class T, T... X>       struct _Is##NAME##Value                {}; \
    template <class T>               struct _Is##NAME##Value<T>             { using Type = Bool<>; }; \
    template <class T, T X>          struct _Is##NAME##Value<T, X>          { using Type = Conditional<_MINITEN_##TEST(X,0), True, False>; }; \
    template <class T, T X0, T... X> struct _Is##NAME##Value<T, X0, X...>   { using Type = Cat<Is##NAME##Value<T,X0>, Is##NAME##Value<T,X...>>; }; \
    template <class X>               struct _Is##NAME                       { using Type = Is##NAME<AsVector<X>>; }; \
    template <class T, T... X>       struct _Is##NAME<Vector<T, X...>>      { using Type = Is##NAME##Value<T, X...>; };

MINITEN_DEFINE(Positive,     GT)
MINITEN_DEFINE(Negative,     LT)
MINITEN_DEFINE(Zero,         EQ)
MINITEN_DEFINE(NonZero,      NE)
MINITEN_DEFINE(NonPositive,  LE)
MINITEN_DEFINE(NonNegative,  GE)
#undef MINITEN_DEFINE

#define MINITEN_DEFINE(NAME,TEST) \
    template <class T, T X, T... Y>             struct _Is##NAME##Value                         {}; \
    template <class T, T X>                     struct _Is##NAME##Value<T, X>                   { using Type = Bool<>; }; \
    template <class T, T X, T Y>                struct _Is##NAME##Value<T, X, Y>                { using Type = Conditional<_MINITEN_##TEST(X,Y), True, False>; }; \
    template <class T, T X, T Y0, T... Y>       struct _Is##NAME##Value<T, X, Y0, Y...>         { using Type = Cat<Is##NAME##Value<T,X,Y0>, Is##NAME##Value<T,X,Y...>>; }; \
    template <class X, class Y>                 struct _Is##NAME                                { \
    private: \
        static constexpr bool LX1 = Length<X>::Value > 1; \
        static constexpr bool LY1 = Length<Y>::Value > 1; \
        using First = Is##NAME<GetFirstValue<X>, GetFirstValue<Y>>; \
        using NextX = Conditional<LX1, DelFirst<X>, X>; \
        using NextY = Conditional<LY1, DelFirst<Y>, Y>; \
        using Next  = Conditional<LX1 || LY1, Is##NAME<NextX, NextY>, EmptyLike<First>>; \
    public: \
        using Type = Cat<First, Next>; \
    }; \
    template <class U, U X, class T, T Y>       struct _Is##NAME<Vector<U,X>, Vector<T,Y>>      { using Type = Conditional<_MINITEN_##TEST(X,Y), True, False>; };

MINITEN_DEFINE(Greater,      GT)
MINITEN_DEFINE(Lower,        LT)
MINITEN_DEFINE(Equal,        EQ)
MINITEN_DEFINE(NotEqual,     NE)
MINITEN_DEFINE(GreaterEqual, GE)
MINITEN_DEFINE(LowerEqual,   LE)
#undef MINITEN_DEFINE

/* ================================================================== *
 *     Operations                                                     *
 * ================================================================== */

#define MINITEN_DEFINE(NAME, OP) \
    template <class T, T... X>       struct _##NAME##Value              {}; \
    template <class T, T... X>       using     NAME##Value              = typename _##NAME##Value<T, X...>::Type; \
    template <class T, T X0, T... X> struct _##NAME##Value<T, X0, X...> { using Type = Cat<Vector<T, _MINITEN_##OP(X0)>, NAME<Vector<T, X...>>>; }; \
    template <class T>               struct _##NAME##Value<T>           { using Type = Vector<T>; }; \
    template <class X, size_t L>     struct _##NAME                     { using Type = Cat<NAME<GetFirstValue<X>>, NAME<DelFirst<X>>>; }; \
    template <class X>               struct _##NAME<X,0>                { using Type = EmptyLike<AsVector<X>>; }; \
    template <class T, T X>          struct _##NAME<Vector<T,X>,1>      { using Type = NAME##Value<T, X>; }; \

MINITEN_DEFINE(Abs,        ABS)
MINITEN_DEFINE(Neg,        NEG)
MINITEN_DEFINE(Not,        BOOLNOT)
MINITEN_DEFINE(BitwiseNot, NOT)
#undef MINITEN_DEFINE

#define MINITEN_DEFINE(NAME,OP) \
    template <class T, T X, T... Y>             struct _##NAME##Value                         {}; \
    template <class T, T X>                     struct _##NAME##Value<T, X>                   { using Type = Vector<T>; }; \
    template <class T, T X, T Y>                struct _##NAME##Value<T, X, Y>                { using Type = Vector<T, _MINITEN_##OP(X ,Y)>; }; \
    template <class T, T X, T Y0, T... Y>       struct _##NAME##Value<T, X, Y0, Y...>         { using Type = Cat<typename _##NAME##Value<T,X,Y0>::Type, typename _##NAME##Value<T,X,Y...>::Type>; }; \
    template <class X, class... Y>              struct _##NAME                                {}; \
    template <class X>                          struct _##NAME<X>                             { using Type = X; }; \
    template <class X, class Y, class... Z>     struct _##NAME<X, Y, Z...>                    { using Type = NAME<NAME<X,Y>,Z...>; }; \
    template <class X, class Y>                 struct _##NAME<X, Y>                          { \
    private: \
        static constexpr bool LX1 = Length<X>::Value > 1; \
        static constexpr bool LY1 = Length<Y>::Value > 1; \
        using First = NAME<GetFirstValue<X>, GetFirstValue<Y>>; \
        using NextX = Conditional<LX1, DelFirst<X>, X>; \
        using NextY = Conditional<LY1, DelFirst<Y>, Y>; \
        using Next  = Conditional<LX1 || LY1, NAME<NextX, NextY>, EmptyLike<First>>; \
    public: \
        using Type = Cat<First, Next>; \
    }; \
    template <class U, U X, class T, T Y>       struct _##NAME<Vector<U,X>, Vector<T,Y>>      { using Type = Vector<decltype(_MINITEN_##OP(X,Y)), _MINITEN_##OP(X,Y)>; };

MINITEN_DEFINE(Add,        SUM)
MINITEN_DEFINE(Sub,        SUB)
MINITEN_DEFINE(Mul,        PROD)
MINITEN_DEFINE(Div,        DIV)
MINITEN_DEFINE(And,        BOOLAND)
MINITEN_DEFINE(Or,         BOOLOR)
MINITEN_DEFINE(Xor,        BOOLXOR)
MINITEN_DEFINE(BitwiseAnd, AND)
MINITEN_DEFINE(BitwiseOr,  OR)
MINITEN_DEFINE(BitwiseXor, XOR)
MINITEN_DEFINE(LShift,     LS)
MINITEN_DEFINE(RShift,     RS)
MINITEN_DEFINE(Minimum,    MIN)
MINITEN_DEFINE(Maximum,    MAX)
#undef MINITEN_DEFINE

/* ================================================================== *
 *     Reductions                                                     *
 * ================================================================== */

#define MINITEN_DEFINE(NAME, OP, INIT) \
    template <class T, T... X>       struct _##NAME##Values              {}; \
    template <class T, T X0, T... X> struct _##NAME##Values<T, X0, X...> { using Type = OP<Vector<T,X0>, NAME##Values<T, X...>>; }; \
    template <class T, T X0>         struct _##NAME##Values<T, X0>       { using Type = OP<Vector<T,static_cast<T>(INIT)>, Vector<T,X0>>; }; \
    template <class T>               struct _##NAME##Values<T>           { using _OT  = ItemType<OP<Vector<T,0>, Vector<T,0>>>; \
                                                                           using Type = Vector<_OT, static_cast<_OT>(INIT)>; }; \
    template <class A>               struct _##NAME                      { using Type = NAME<AsVector<A>>; }; \
    template <class T, T... X>       struct _##NAME<Vector<T, X...>>     { using Type = NAME##Values<T, X...>; };

MINITEN_DEFINE(Prod, Mul,          1)
MINITEN_DEFINE(Sum,  Add,          0)
MINITEN_DEFINE(Min,  Minimum,      TypeInfo<T>::Max)
MINITEN_DEFINE(Max,  Maximum,      TypeInfo<T>::Min)
MINITEN_DEFINE(Any,  Or,           false)
MINITEN_DEFINE(All,  And,          true)
#undef MINITEN_DEFINE

/* ================================================================== */
/*     Cumulative Reductions                                          */
/* ================================================================== */

#define MINITEN_DEFINE(NAME, OP) \
    template <class T, T... X>             struct _##NAME##Values                   {}; \
    template <class T, T X0, T X1, T... X> struct _##NAME##Values<T, X0, X1, X...>  { \
    private:\
        using First  = Vector<T,X0>; \
        using Second = OP<Vector<T,X0>, Vector<T,X1>>; \
        using Third  = Vector<T, X...>; \
        using SecondThird = NAME<Cat<Second, Third>>; \
    public: \
        using Type = Cat<First, SecondThird>; \
    }; \
    template <class T, T X0>         struct _##NAME##Values<T, X0>       { using _OT  = ItemType<OP<Vector<T,0>, Vector<T,0>>>; \
                                                                           using Type = Vector<_OT, static_cast<_OT>(X0)>; }; \
    template <class T>               struct _##NAME##Values<T>           { using Type = Vector<T>; }; \
    template <class A>               struct _##NAME                      { using Type = NAME<AsVector<A>>; }; \
    template <class T, T... X>       struct _##NAME<Vector<T, X...>>     { using Type = NAME##Values<T, X...>; };

MINITEN_DEFINE(CumProd, Mul)
MINITEN_DEFINE(CumSum,  Add)
MINITEN_DEFINE(CumMin,  Minimum)
MINITEN_DEFINE(CumMax,  Maximum)
MINITEN_DEFINE(CumAny,  Or)
MINITEN_DEFINE(CumAll,  And)
#undef MINITEN_DEFINE

#define MINITEN_DEFINE(NAME, OP, INIT) \
    template <class T, T... X>       struct _##NAME##Values              {}; \
    template <class T, T X0, T... X> struct _##NAME##Values<T, X0, X...> { using Type = Cat<Vector<T,static_cast<T>(INIT)>, OP<Vector<T,X0>,  NAME##Values<T, X...>>>; }; \
    template <class T, T X0, T X1>   struct _##NAME##Values<T, X0, X1>   { using _OT  = ItemType<OP<Vector<T,0>, Vector<T,0>>>; \
                                                                           using Type = Vector<_OT, static_cast<_OT>(INIT), static_cast<_OT>(X0)>; }; \
    template <class T, T X0>         struct _##NAME##Values<T, X0>       { using _OT  = ItemType<OP<Vector<T,0>, Vector<T,0>>>; \
                                                                           using Type = Vector<_OT, static_cast<_OT>(INIT)>; }; \
    template <class T>               struct _##NAME##Values<T>           { using Type = Vector<T>; }; \
    template <class A>               struct _##NAME                      { using Type = NAME<AsVector<A>>; }; \
    template <class T, T... X>       struct _##NAME<Vector<T, X...>>     { using Type = NAME##Values<T, X...>; };

MINITEN_DEFINE(ShiftedCumProd, Mul,      1)
MINITEN_DEFINE(ShiftedCumSum,  Add,      0)
MINITEN_DEFINE(ShiftedCumMin,  Minimum,  TypeInfo<T>::Max)
MINITEN_DEFINE(ShiftedCumMax,  Maximum,  TypeInfo<T>::Min)
MINITEN_DEFINE(ShiftedCumAny,  Or,       false)
MINITEN_DEFINE(ShiftedCumAll,  And,      true)
#undef MINITEN_DEFINE

/* ================================================================== *
 *     Count                                                          *
 * ================================================================== */

template <class... T>               struct __CountT              {};
template <class T0, class... T>     struct __CountT<T0, T...>    { static constexpr size_t Value = 1 + __CountT<T...>::Value; };
template <class T0>                 struct __CountT<T0>          { static constexpr size_t Value = 1; };
template <>                         struct __CountT<>            { static constexpr size_t Value = 0; };
template <class... T>               struct  _CountT              { using Type = SizeT<__CountT<T...>::Value>; };

template <class T, T... X>          struct __CountV              {};
template <class T, T X0, T... X>    struct __CountV<T, X0, X...> { static constexpr size_t Value = 1 + __CountV<T, X...>::Value; };
template <class T, T X0>            struct __CountV<T, X0>       { static constexpr size_t Value = 1; };
template <class T>                  struct __CountV<T>           { static constexpr size_t Value = 0; };
template <class T, T... X>          struct  _CountV              { using Type = SizeT<__CountV<T, X...>::Value>; };

/* ================================================================== *
 *     IsIn                                                           *
 * ================================================================== */

template <class A, class B>
struct _IsIn {};

template <class B, class T, T X>
struct _IsIn<Vector<T, X>, B> {
    using Type = Any<IsEqual<B, Vector<T, X>>>;
};

template <class B, class T, T X0, T... X>
struct _IsIn<Vector<T, X0, X...>, B> {
    using Type = Cat<
        IsIn<Vector<T, X0>,   B>,
        IsIn<Vector<T, X...>, B>
    >;
};

/* ================================================================== *
 *     NonZero                                                        *
 * ================================================================== */

template <class A>
struct _NonZero {};

template <class T>
struct _NonZero<Vector<T>> {
    using Type = SizeT<>;
};

template <class T, T X>
struct _NonZero<Vector<T, X>> {
    using Type = SizeT<0>;
};

template <class T>
struct _NonZero<Vector<T, 0>> {
    using Type = SizeT<>;
};

template <class T, T X0, T... X>
struct _NonZero<Vector<T, X0, X...>> {
    using Type = Cat<
        NonZero<Vector<T, X0>>,
        Add<NonZero<Vector<T, X...>>, SizeT<1>>
    >;
};

#undef _MINITEN_PROD
#undef _MINITEN_SUM
#undef _MINITEN_SUB
#undef _MINITEN_DIV
#undef _MINITEN_MIN
#undef _MINITEN_MAX
#undef _MINITEN_OR
#undef _MINITEN_AND
#undef _MINITEN_LSH
#undef _MINITEN_RSH
#undef _MINITEN_GT
#undef _MINITEN_GE
#undef _MINITEN_LT
#undef _MINITEN_LE
#undef _MINITEN_EQ
#undef _MINITEN_NE
#undef _MINITEN_NEG
#undef _MINITEN_NOT
#undef _MINITEN_ABS
#undef _MINITEN_BOOLOR
#undef _MINITEN_BOOLAND
#undef _MINITEN_BOOLXOR
#undef _MINITEN_BOOLNOT

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__MATH_IMPL
