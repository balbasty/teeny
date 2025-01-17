/***********************************************************************
 * This file implements metaprogramming math
 *
 * IsPositiveValue<T,X>         IsPositive<X>           -> Bool
 * IsNegativeValue<T,X>         IsNegative<X>           -> Bool
 * IsZeroValue<T,X>             IsZero<X>               -> Bool
 * IsNonPositiveValue<T, X>     IsNonPositive<X>        -> Bool
 * IsNonNegativeValue<T, X>     IsNonNegative<X>        -> Bool
 * IsNonZeroValue<T, X>         IsNonZero<X>            -> Bool
 *
 * IsEqualValue<T, X, Y>        IsEqual<X, Y>           -> Bool
 * IsNotEqualValue<T, X, Y>     IsNotEqual<X, Y>        -> Bool
 * IsGreaterValue<T, X, Y>      IsGreater<X, Y>         -> Bool
 * IsLowerValue<T, X, Y>        IsLower<X, Y>           -> Bool
 * IsGreaterEqualValue<T, X, Y> IsGreaterEqual<X, Y>    -> Bool
 * IsLowerEqualValue<T, X, Y>   IsLowerEqual<X, Y>      -> Bool
 *
 * ProdValues<T, X...>          Prod<Vector>            -> Scalar
 * SumValues<T, X...>           Sum<Vector>             -> Scalar
 * MinValues<T, X...>           Min<Vector>             -> Scalar
 * MaxValues<T, X...>           Max<Vector>             -> Scalar
 * AnyValues<T, X...>           Any<Vector>             -> Scalar
 * AllValues<T, X...>           All<Vector>             -> Scalar
 * CountValues<T, X...>                                 -> SizeT
 * CountTypes<T...>                                     -> SizeT
 *
 * CumProdValues<T, X...>       CumProd<Vector>         -> Vector
 * CumSumValues<T, X...>        CumSum<Vector>          -> Vector
 * CumMinValues<T, X...>        CumMin<Vector>          -> Vector
 * CumMaxValues<T, X...>        CumMax<Vector>          -> Vector
 * CumAnyValues<T, X...>        CumAny<Vector>          -> Vector
 * CumAllValues<T, X...>        CumAll<Vector>          -> Vector
 *
 *                              Add<X, Y, ...>          -> Vector
 *                              Sub<X, Y>               -> Vector
 *                              Mul<X, Y, ...>          -> Vector
 *                              Div<X, Y>               -> Vector
 *                              Minimum<X, Y, ...>      -> Vector
 *                              Maximum<X, Y, ...>      -> Vector
 *                              And<X, Y, ...>          -> Vector
 *                              Or<X, Y, ...>           -> Vector
 *                              Xor<X, Y>               -> Vector
 *                              LShift<X, Y>            -> Vector
 *                              RShift<X, Y>            -> Vector
 *
 *                              Not<X>                  -> Bool
 *                              Neg<X>                  -> Vector
 *                              Abs<X>                  -> Vector
 ***********************************************************************/

// FIXME
//      For reductions, I currently convert to vector first and then
//      apply the reduction (with values being compacted from then end).
//      It would be nicer to move through the input in a container-agnostic
//      way, similarly to what I do for element-wise operations.

#ifndef MINITEN_META_MATH_IMPL_H
#define MINITEN_META_MATH_IMPL_H

#include "math.h"
#include "traits.h"
#include "vector.h"
#include "packapi.h"

namespace miniten {
namespace  meta {

#define _PROD(A,B) (A * B)
#define _SUM(A,B)  (A + B)
#define _SUB(A,B)  (A - B)
#define _DIV(A,B)  (A / B)
#define _MIN(A,B)  (A <= B ? A : B)
#define _MAX(A,B)  (A >= B ? A : B)
#define _OR(A,B)   (A | B)
#define _AND(A,B)  (A & B)
#define _LSH(A,B)  (A << B)
#define _RSH(A,B)  (A >> B)
#define _GT(A,B)   (A > B)
#define _GE(A,B)   (A >= B)
#define _LT(A,B)   (A < B)
#define _LE(A,B)   (A <= B)
#define _EQ(A,B)   (A == B)
#define _NE(A,B)   (A != B)

#define _NEG(A) (-A)
#define _NOT(A) (~A)
#define _ABS(A) (A < 0 ? -A : A)

#define _BOOLOR(A,B)   (A || B)
#define _BOOLAND(A,B)  (A && B)
#define _BOOLXOR(A,B)  (!(A && B) && (A || B))
#define _BOOLNOT(A)    (!A)

/// ---------------------------------------------------------------- ///
///     Comparisons                                                  ///
/// ---------------------------------------------------------------- ///

#define MAKE_MATHTESTUNARY(NAME,TEST) \
    template <class T, T... X>       struct _Is##NAME##Value                {}; \
    template <class T>               struct _Is##NAME##Value<T>             { using Type = Bool<>; }; \
    template <class T, T X>          struct _Is##NAME##Value<T, X>          { using Type = Conditional<_##TEST(X,0), True, False>; }; \
    template <class T, T X0, T... X> struct _Is##NAME##Value<T, X0, X...>   { using Type = Cat<Is##NAME##Value<T,X0>, Is##NAME##Value<T,X...>>; }; \
    template <class X>               struct _Is##NAME                       { using Type = Is##NAME<AsVector<X>>; }; \
    template <class T, T... X>       struct _Is##NAME<Vector<T, X...>>      { using Type = Is##NAME##Value<T, X...>; };

MAKE_MATHTESTUNARY(Positive,     GT)
MAKE_MATHTESTUNARY(Negative,     LT)
MAKE_MATHTESTUNARY(Zero,         EQ)
MAKE_MATHTESTUNARY(NonZero,      NE)
MAKE_MATHTESTUNARY(NonPositive,  LE)
MAKE_MATHTESTUNARY(NonNegative,  GE)

#define MAKE_MATHTESTBINARY(NAME,TEST) \
    template <class T, T X, T... Y>             struct _Is##NAME##Value                         {}; \
    template <class T, T X>                     struct _Is##NAME##Value<T, X>                   { using Type = Bool<>; }; \
    template <class T, T X, T Y>                struct _Is##NAME##Value<T, X, Y>                { using Type = Conditional<_##TEST(X,Y), True, False>; }; \
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
    template <class U, U X, class T, T Y>       struct _Is##NAME<Vector<U,X>, Vector<T,Y>>      { using Type = Conditional<_##TEST(X,Y), True, False>; };

MAKE_MATHTESTBINARY(Greater,      GT)
MAKE_MATHTESTBINARY(Lower,        LT)
MAKE_MATHTESTBINARY(Equal,        EQ)
MAKE_MATHTESTBINARY(NotEqual,     NE)
MAKE_MATHTESTBINARY(GreaterEqual, GE)
MAKE_MATHTESTBINARY(LowerEqual,   LE)

/// ---------------------------------------------------------------- ///
///     Operations                                                   ///
/// ---------------------------------------------------------------- ///

#define MAKE_MATHOPUNARY(NAME, OP) \
    template <class T, T... X>       struct _##NAME##Value              {}; \
    template <class T, T X0, T... X> struct _##NAME##Value<T, X0, X...> { using Type = Cat<Vector<T, _##OP(X0)>, NAME<Vector<T, X...>>>; }; \
    template <class T>               struct _##NAME##Value<T>           { using Type = Vector<T>; }; \

MAKE_MATHOPUNARY(Abs,        ABS)
MAKE_MATHOPUNARY(Neg,        NEG)
MAKE_MATHOPUNARY(Not,        BOOLNOT)
MAKE_MATHOPUNARY(BitwiseNot, NOT)

#define MAKE_MATHOPBINARY(NAME,OP) \
    template <class T, T X, T... Y>             struct _##NAME##Value                         {}; \
    template <class T, T X>                     struct _##NAME##Value<T, X>                   { using Type = Vector<T>; }; \
    template <class T, T X, T Y>                struct _##NAME##Value<T, X, Y>                { using Type = Vector<T, _##OP(X ,Y)>; }; \
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
    template <class U, U X, class T, T Y>       struct _##NAME<Vector<U,X>, Vector<T,Y>>      { using Type = Vector<decltype(_##OP(X,Y)), _##OP(X,Y)>; };

MAKE_MATHOPBINARY(Add,        SUM)
MAKE_MATHOPBINARY(Sub,        SUB)
MAKE_MATHOPBINARY(Mul,        PROD)
MAKE_MATHOPBINARY(Div,        DIV)
MAKE_MATHOPBINARY(And,        BOOLAND)
MAKE_MATHOPBINARY(Or,         BOOLOR)
MAKE_MATHOPBINARY(Xor,        BOOLXOR)
MAKE_MATHOPBINARY(BitwiseAnd, AND)
MAKE_MATHOPBINARY(BitwiseOr,  OR)
MAKE_MATHOPBINARY(BitwiseXor, XOR)
MAKE_MATHOPBINARY(LShift,     LS)
MAKE_MATHOPBINARY(RShift,     RS)
MAKE_MATHOPBINARY(Minimum,    MIN)
MAKE_MATHOPBINARY(Maximum,    MAX)

/// ---------------------------------------------------------------- ///
///     Reductions                                                   ///
/// ---------------------------------------------------------------- ///

#define MAKE_MATHREDUCE(NAME, OP, INIT) \
    template <class T, T... X>       struct _##NAME##Values              {}; \
    template <class T, T X0, T... X> struct _##NAME##Values<T, X0, X...> { using Type = OP<Vector<T,X0>, NAME##Values<T, X...>>; }; \
    template <class T, T X0>         struct _##NAME##Values<T, X0>       { using Type = OP<Vector<T,static_cast<T>(INIT)>, Vector<T,X0>>; }; \
    template <class T>               struct _##NAME##Values<T>           { using _OT  = ItemType<OP<Vector<T,0>, Vector<T,0>>>; \
                                                                           using Type = Vector<_OT, static_cast<_OT>(INIT)>; }; \
    template <class A>               struct _##NAME                      { using Type = NAME<AsVector<A>>; }; \
    template <class T, T... X>       struct _##NAME<Vector<T, X...>>     { using Type = NAME##Values<T, X...>; };

MAKE_MATHREDUCE(Prod, Mul,          1)
MAKE_MATHREDUCE(Sum,  Add,          0)
MAKE_MATHREDUCE(Min,  Minimum,      TypeInfo<T>::Max)
MAKE_MATHREDUCE(Max,  Maximum,      TypeInfo<T>::Min)
MAKE_MATHREDUCE(Any,  Or,           false)
MAKE_MATHREDUCE(All,  And,          true)

/// ---------------------------------------------------------------- ///
///     Cumulative Reductions                                        ///
/// ---------------------------------------------------------------- ///

#define MAKE_MATHREDUCECUM(NAME, OP) \
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

MAKE_MATHREDUCECUM(CumProd, Mul)
MAKE_MATHREDUCECUM(CumSum,  Add)
MAKE_MATHREDUCECUM(CumMin,  Minimum)
MAKE_MATHREDUCECUM(CumMax,  Maximum)
MAKE_MATHREDUCECUM(CumAny,  Or)
MAKE_MATHREDUCECUM(CumAll,  And)

#define MAKE_MATHREDUCECUMSHIFTED(NAME, OP, INIT) \
    template <class T, T... X>       struct _##NAME##Values              {}; \
    template <class T, T X0, T... X> struct _##NAME##Values<T, X0, X...> { using Type = Cat<Vector<T,static_cast<T>(INIT)>, OP<Vector<T,X0>,  NAME##Values<T, X...>>>; }; \
    template <class T, T X0, T X1>   struct _##NAME##Values<T, X0, X1>   { using _OT  = ItemType<OP<Vector<T,0>, Vector<T,0>>>; \
                                                                           using Type = Vector<_OT, static_cast<_OT>(INIT), static_cast<_OT>(X0)>; }; \
    template <class T, T X0>         struct _##NAME##Values<T, X0>       { using _OT  = ItemType<OP<Vector<T,0>, Vector<T,0>>>; \
                                                                           using Type = Vector<_OT, static_cast<_OT>(INIT)>; }; \
    template <class T>               struct _##NAME##Values<T>           { using Type = Vector<T>; }; \
    template <class A>               struct _##NAME                      { using Type = NAME<AsVector<A>>; }; \
    template <class T, T... X>       struct _##NAME<Vector<T, X...>>     { using Type = NAME##Values<T, X...>; };

MAKE_MATHREDUCECUMSHIFTED(ShiftedCumProd, Mul,      1)
MAKE_MATHREDUCECUMSHIFTED(ShiftedCumSum,  Add,      0)
MAKE_MATHREDUCECUMSHIFTED(ShiftedCumMin,  Minimum,  TypeInfo<T>::Max)
MAKE_MATHREDUCECUMSHIFTED(ShiftedCumMax,  Maximum,  TypeInfo<T>::Min)
MAKE_MATHREDUCECUMSHIFTED(ShiftedCumAny,  Or,       false)
MAKE_MATHREDUCECUMSHIFTED(ShiftedCumAll,  And,      true)

/// ---------------------------------------------------------------- ///
///     Count                                                        ///
/// ---------------------------------------------------------------- ///

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



} // namespace meta
} // namespace miniten

#endif // MINITEN_META_MATH_IMPL_H
