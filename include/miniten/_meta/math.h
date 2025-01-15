/***********************************************************************
 * This file declares metaprogramming math
 *
 * IsPositiveValue<T,X>         IsPositive<X>           -> True | False
 * IsNegativeValue<T,X>         IsNegative<X>           -> True | False
 * IsZeroValue<T,X>             IsZero<X>               -> True | False
 * IsNonPositiveValue<T, X>     IsNonPositive<X>        -> True | False
 * IsNonNegativeValue<T, X>     IsNonNegative<X>        -> True | False
 * IsNonZeroValue<T, X>         IsNonZero<X>            -> True | False
 *
 * ProdValues<T, X...>          Prod<Vector>            -> Scalar<T,Result>
 * SumValues<T, X...>           Sum<Vector>             -> Scalar<T,Result>
 * MinValues<T, X...>           Min<Vector>             -> Scalar<T,Result>
 * MaxValues<T, X...>           Max<Vector>             -> Scalar<T,Result>
 * CountValues<T, X...>                                 -> SizeT<Result>
 * CountTypes<T...>                                     -> SizeT<Result>
 ***********************************************************************/
#ifndef MINITEN_META_MATH_H
#define MINITEN_META_MATH_H

#include "traits.h"
#include "vector.h"

namespace miniten {
namespace  meta {

/// ---------------------------------------------------------------- ///
///     Sign                                                         ///
/// ---------------------------------------------------------------- ///

#define DECLARE_MATHTESTUNARY(NAME) \
    template <class T, T... X> struct _Is##NAME##Value ; \
    template <class T, T... X> using   Is##NAME##Value = typename _Is##NAME##Value<T, X...>::Type; \
    template <class X>         struct _Is##NAME        ; \
    template <class X>         using   Is##NAME        = typename _Is##NAME<X>::Type;

DECLARE_MATHTESTUNARY(Positive)
DECLARE_MATHTESTUNARY(Negative)
DECLARE_MATHTESTUNARY(Zero)
DECLARE_MATHTESTUNARY(NonZero)
DECLARE_MATHTESTUNARY(NonePositive)
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

/// ---------------------------------------------------------------- ///
///     Reduction                                                    ///
/// ---------------------------------------------------------------- ///

#define DECLARE_MATHREDUCE(NAME) \
    template <class T, T... X>          struct _##NAME##V    ; \
    template <class T, T... X>          using  NAME##Values  = typename _##NAME##V<T, X...>::Type; \
    template <class A>                  struct _##NAME       ; \
    template <class A>                  using  NAME          = typename _##NAME<A>::Type;

DECLARE_MATHREDUCE(Prod)
DECLARE_MATHREDUCE(Sum)
DECLARE_MATHREDUCE(Min)
DECLARE_MATHREDUCE(Max)
DECLARE_MATHREDUCE(Or)

/// ---------------------------------------------------------------- ///
///     Count                                                        ///
/// ---------------------------------------------------------------- ///

template <class... T>           struct _CountT;
template <class T, T... X>      struct _CountV;

/// @brief Count the number of elements in a parameter pack (of types)
template <class... T>
using CountTypes = typename _CountT<T...>::Type;

/// @brief Count the number of elements in a parameter pack (of objects)
template <class T, T... X>
using CountValues = typename _CountV<T, X...>::Type;

/// @brief      Alias for integer count
template <int... X>
using CountInt = CountValues<int, X...>;

/// @brief      Alias for long integer count
template <long... X>
using CountLong = CountValues<long, X...>;

} // namespace meta
} // namespace miniten

#endif // MINITEN_META_MATH_H
