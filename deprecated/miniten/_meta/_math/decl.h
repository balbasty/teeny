/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file declares metaprogramming math.                                **
 **                                                                         **
 **|----------------------------|-----------------------|------------------|**
 **| Applies to pack of values  | Applies to pack type  | Returns          |**
 **|----------------------------|-----------------------|------------------|**
 **  IsPositiveValue<T,X>         IsPositive<X>           -> Bool           **
 **  IsNegativeValue<T,X>         IsNegative<X>           -> Bool           **
 **  IsZeroValue<T,X>             IsZero<X>               -> Bool           **
 **  IsNonPositiveValue<T, X>     IsNonPositive<X>        -> Bool           **
 **  IsNonNegativeValue<T, X>     IsNonNegative<X>        -> Bool           **
 **  IsNonZeroValue<T, X>         IsNonZero<X>            -> Bool           **
 **                                                                         **
 **  IsEqualValue<T, X, Y>        IsEqual<X, Y>           -> Bool           **
 **  IsNotEqualValue<T, X, Y>     IsNotEqual<X, Y>        -> Bool           **
 **  IsGreaterValue<T, X, Y>      IsGreater<X, Y>         -> Bool           **
 **  IsLowerValue<T, X, Y>        IsLower<X, Y>           -> Bool           **
 **  IsGreaterEqualValue<T, X, Y> IsGreaterEqual<X, Y>    -> Bool           **
 **  IsLowerEqualValue<T, X, Y>   IsLowerEqual<X, Y>      -> Bool           **
 **                                                                         **
 **  ProdValues<T, X...>          Prod<Vector>            -> Scalar         **
 **  SumValues<T, X...>           Sum<Vector>             -> Scalar         **
 **  MinValues<T, X...>           Min<Vector>             -> Scalar         **
 **  MaxValues<T, X...>           Max<Vector>             -> Scalar         **
 **  AnyValues<T, X...>           Any<Vector>             -> Scalar         **
 **  AllValues<T, X...>           All<Vector>             -> Scalar         **
 **  CountValues<T, X...>                                 -> SizeT          **
 **  CountTypes<T...>                                     -> SizeT          **
 **                                                                         **
 **  CumProdValues<T, X...>       CumProd<Vector>         -> Vector         **
 **  CumSumValues<T, X...>        CumSum<Vector>          -> Vector         **
 **  CumMinValues<T, X...>        CumMin<Vector>          -> Vector         **
 **  CumMaxValues<T, X...>        CumMax<Vector>          -> Vector         **
 **  CumAnyValues<T, X...>        CumAny<Vector>          -> Vector         **
 **  CumAllValues<T, X...>        CumAll<Vector>          -> Vector         **
 **                                                                         **
 **                               Add<X, Y, ...>          -> Vector         **
 **                               Sub<X, Y>               -> Vector         **
 **                               Mul<X, Y, ...>          -> Vector         **
 **                               Div<X, Y>               -> Vector         **
 **                               Minimum<X, Y, ...>      -> Vector         **
 **                               Maximum<X, Y, ...>      -> Vector         **
 **                               And<X, Y, ...>          -> Vector         **
 **                               Or<X, Y, ...>           -> Vector         **
 **                               Xor<X, Y>               -> Vector         **
 **                               LShift<X, Y>            -> Vector         **
 **                               RShift<X, Y>            -> Vector         **
 **                                                                         **
 **                               Not<X>                  -> Bool           **
 **                               Neg<X>                  -> Vector         **
 **                               Abs<X>                  -> Vector         **
 **                                                                         **
 **                               IsIn<X, Y>              -> Bool           **
 **                               NonZero<X>              -> SizeT          **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef MINITEN__META__MATH_DECL
#define MINITEN__META__MATH_DECL

#include <miniten/_core/defines.h>
#include <miniten/_meta/traits.h>
#include <miniten/_meta/_vector/decl.h>
#include <miniten/_meta/_packapi/decl.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

/* ================================================================== *
 *     Sign                                                           *
 * ================================================================== */

#define MINITEN_DECLARE(NAME) \
    template <class T, T... X> struct _Is##NAME##Value ; \
    template <class T, T... X> using   Is##NAME##Value = typename _Is##NAME##Value<T, X...>::Type; \
    template <class X>         struct _Is##NAME        ; \
    template <class X>         using   Is##NAME        = typename _Is##NAME<X>::Type;

MINITEN_DECLARE(Positive)
MINITEN_DECLARE(Negative)
MINITEN_DECLARE(Zero)
MINITEN_DECLARE(NonZero)
MINITEN_DECLARE(NonPositive)
MINITEN_DECLARE(NonNegative)
#undef MINITEN_DECLARE

#define MINITEN_DECLARE(NAME) \
    template <class T, T X, T... Y> struct _Is##NAME##Value ; \
    template <class T, T X, T... Y> using   Is##NAME##Value = typename _Is##NAME##Value<T, X, Y...>::Type; \
    template <class X, class Y>     struct _Is##NAME        ; \
    template <class X, class Y>     using   Is##NAME        = typename _Is##NAME<X,Y>::Type;

MINITEN_DECLARE(Greater)
MINITEN_DECLARE(Lower)
MINITEN_DECLARE(Equal)
MINITEN_DECLARE(NotEqual)
MINITEN_DECLARE(GreaterEqual)
MINITEN_DECLARE(LowerEqual)
#undef MINITEN_DECLARE

/* ================================================================== *
 *     Reduction                                                      *
 * ================================================================== */

#define MINITEN_DECLARE(NAME) \
    template <class T, T... X>          struct _##NAME##Values ; \
    template <class T, T... X>          using     NAME##Values = typename _##NAME##Values<T, X...>::Type; \
    template <class A>                  struct _##NAME         ; \
    template <class A>                  using     NAME         = typename _##NAME<A>::Type;

MINITEN_DECLARE(Prod)
MINITEN_DECLARE(Sum)
MINITEN_DECLARE(Min)
MINITEN_DECLARE(Max)
MINITEN_DECLARE(Any)
MINITEN_DECLARE(All)

MINITEN_DECLARE(CumProd)
MINITEN_DECLARE(CumSum)
MINITEN_DECLARE(CumMin)
MINITEN_DECLARE(CumMax)
MINITEN_DECLARE(CumAny)
MINITEN_DECLARE(CumAll)

/*
 * These variants do not include the initial value in the result.
 * Their results are the same as before, but shifted by one element to
 * the right. The first element in the result is therefore the
 * operation's neutral element.
 *
 * Ex:
 *         CumSum<Int<1, 2, 3>> == Int<1, 3, 6>
 *  ShiftedCumSum<Int<1, 2, 3>> == Int<0, 1, 3>
 */

MINITEN_DECLARE(ShiftedCumProd)
MINITEN_DECLARE(ShiftedCumSum)
MINITEN_DECLARE(ShiftedCumMin)
MINITEN_DECLARE(ShiftedCumMax)
MINITEN_DECLARE(ShiftedCumAny)
MINITEN_DECLARE(ShiftedCumAll)

#undef MINITEN_DECLARE

/* ================================================================== *
 *     Operations                                                     *
 * ================================================================== */

#define MINITEN_DECLARE(NAME) \
    template <class X, class... Y>  struct _##NAME  ; \
    template <class X, class... Y>  using     NAME  = typename _##NAME<X, Y...>::Type;

MINITEN_DECLARE(Add)
MINITEN_DECLARE(Sub)
MINITEN_DECLARE(Mul)
MINITEN_DECLARE(Div)
MINITEN_DECLARE(And)
MINITEN_DECLARE(Or)
MINITEN_DECLARE(Xor)
MINITEN_DECLARE(BitwiseAnd)
MINITEN_DECLARE(BitwiseOr)
MINITEN_DECLARE(BitwiseXor)
MINITEN_DECLARE(LShift)
MINITEN_DECLARE(RShift)
MINITEN_DECLARE(Minimum)
MINITEN_DECLARE(Maximum)
#undef MINITEN_DECLARE

#define MINITEN_DECLARE(NAME) \
    template <class X, size_t L = Length<X>::Value>  struct _##NAME  ; \
    template <class X>  using     NAME  = typename _##NAME<X>::Type;

MINITEN_DECLARE(Abs)
MINITEN_DECLARE(Neg)
MINITEN_DECLARE(Not)
MINITEN_DECLARE(BitwiseNot)
#undef MINITEN_DECLARE

/* ================================================================== *
 *     Count                                                          *
 * ================================================================== */

template <class... T>           struct _CountT;
template <class T, T... X>      struct _CountV;

/** @brief Count the number of elements in a parameter pack (of types) */
template <class... T>
using CountTypes = typename _CountT<T...>::Type;

/** @brief Count the number of elements in a parameter pack (of objects) */
template <class T, T... X>
using CountValues = typename _CountV<T, X...>::Type;

/** @brief Alias for integer count */
template <int... X>
using CountInt = CountValues<int, X...>;

/** @brief Alias for long integer count */
template <long... X>
using CountLong = CountValues<long, X...>;

/* ================================================================== *
 *     IsIn                                                           *
 * ================================================================== */

template <class A, class B> struct _IsIn;
template <class A, class B> using   IsIn = typename _IsIn<A, B>::Type;

/* ================================================================== *
 *     NonZero                                                        *
 * ================================================================== */

template <class A> struct _NonZero;
template <class A> using   NonZero = typename _NonZero<A>::Type;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN_META_MATH_H
