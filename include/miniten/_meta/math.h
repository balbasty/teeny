/***********************************************************************
 * This file declares metaprogramming math
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
#ifndef MINITEN_META_MATH_H
#define MINITEN_META_MATH_H

#include "traits.h"
#include "vector.h"

namespace miniten {
namespace  meta {

// ------------------------------------------------------------------ //
//     Sign                                                           //
// ------------------------------------------------------------------ //

#define DECLARE_MATHTESTUNARY(NAME) \
    template <class T, T... X> struct _Is##NAME##Value ; \
    template <class T, T... X> using   Is##NAME##Value = typename _Is##NAME##Value<T, X...>::Type; \
    template <class X>         struct _Is##NAME        ; \
    template <class X>         using   Is##NAME        = typename _Is##NAME<X>::Type;

DECLARE_MATHTESTUNARY(Positive)
DECLARE_MATHTESTUNARY(Negative)
DECLARE_MATHTESTUNARY(Zero)
DECLARE_MATHTESTUNARY(NonZero)
DECLARE_MATHTESTUNARY(NonPositive)
DECLARE_MATHTESTUNARY(NonNegative)

#define DECLARE_MATHTESTBINARY(NAME) \
    template <class T, T X, T... Y> struct _Is##NAME##Value ; \
    template <class T, T X, T... Y> using   Is##NAME##Value = typename _Is##NAME##Value<T, X, Y...>::Type; \
    template <class X, class Y>     struct _Is##NAME        ; \
    template <class X, class Y>     using   Is##NAME        = typename _Is##NAME<X,Y>::Type;

DECLARE_MATHTESTBINARY(Greater)
DECLARE_MATHTESTBINARY(Lower)
DECLARE_MATHTESTBINARY(Equal)
DECLARE_MATHTESTBINARY(NotEqual)
DECLARE_MATHTESTBINARY(GreaterEqual)
DECLARE_MATHTESTBINARY(LowerEqual)

// ------------------------------------------------------------------ //
//     Reduction                                                      //
// ------------------------------------------------------------------ //

#define DECLARE_MATHREDUCE(NAME) \
    template <class T, T... X>          struct _##NAME##Values ; \
    template <class T, T... X>          using     NAME##Values = typename _##NAME##Values<T, X...>::Type; \
    template <class A>                  struct _##NAME         ; \
    template <class A>                  using     NAME         = typename _##NAME<A>::Type;

DECLARE_MATHREDUCE(Prod)
DECLARE_MATHREDUCE(Sum)
DECLARE_MATHREDUCE(Min)
DECLARE_MATHREDUCE(Max)
DECLARE_MATHREDUCE(Any)
DECLARE_MATHREDUCE(All)

DECLARE_MATHREDUCE(CumProd)
DECLARE_MATHREDUCE(CumSum)
DECLARE_MATHREDUCE(CumMin)
DECLARE_MATHREDUCE(CumMax)
DECLARE_MATHREDUCE(CumAny)
DECLARE_MATHREDUCE(CumAll)

// These variants do not include the initial value in the result.
// Their results are the same as before, but shifted by one element to
// the right. The first element in the result is therefore the
// operation's neutral element.
//
// Ex:
//         CumSum<Int<1, 2, 3>> == Int<1, 3, 6>
//  ShiftedCumSum<Int<1, 2, 3>> == Int<0, 1, 3>

DECLARE_MATHREDUCE(ShiftedCumProd)
DECLARE_MATHREDUCE(ShiftedCumSum)
DECLARE_MATHREDUCE(ShiftedCumMin)
DECLARE_MATHREDUCE(ShiftedCumMax)
DECLARE_MATHREDUCE(ShiftedCumAny)
DECLARE_MATHREDUCE(ShiftedCumAll)

// ------------------------------------------------------------------ //
//     Operations                                                     //
// ------------------------------------------------------------------ //

#define DECLARE_MATHBINARYOP(NAME) \
    template <class X, class... Y>  struct _##NAME  ; \
    template <class X, class... Y>  using     NAME  = typename _##NAME<X, Y...>::Type;

DECLARE_MATHBINARYOP(Add)
DECLARE_MATHBINARYOP(Sub)
DECLARE_MATHBINARYOP(Mul)
DECLARE_MATHBINARYOP(Div)
DECLARE_MATHBINARYOP(And)
DECLARE_MATHBINARYOP(Or)
DECLARE_MATHBINARYOP(Xor)
DECLARE_MATHBINARYOP(BitwiseAnd)
DECLARE_MATHBINARYOP(BitwiseOr)
DECLARE_MATHBINARYOP(BitwiseXor)
DECLARE_MATHBINARYOP(LShift)
DECLARE_MATHBINARYOP(RShift)
DECLARE_MATHBINARYOP(Minimum)
DECLARE_MATHBINARYOP(Maximum)

#define DECLARE_MATHUNARYOP(NAME) \
    template <class X>  struct _##NAME  ; \
    template <class X>  using     NAME  = typename _##NAME<X>::Type;

DECLARE_MATHUNARYOP(Abs)
DECLARE_MATHUNARYOP(Neg)
DECLARE_MATHUNARYOP(Not)
DECLARE_MATHUNARYOP(BitwiseNot)

// ------------------------------------------------------------------ //
//     Count                                                          //
// ------------------------------------------------------------------ //

template <class... T>           struct _CountT;
template <class T, T... X>      struct _CountV;

// @brief Count the number of elements in a parameter pack (of types)
template <class... T>
using CountTypes = typename _CountT<T...>::Type;

// @brief Count the number of elements in a parameter pack (of objects)
template <class T, T... X>
using CountValues = typename _CountV<T, X...>::Type;

// @brief      Alias for integer count
template <int... X>
using CountInt = CountValues<int, X...>;

// @brief      Alias for long integer count
template <long... X>
using CountLong = CountValues<long, X...>;

} // namespace meta
} // namespace miniten

#endif // MINITEN_META_MATH_H
