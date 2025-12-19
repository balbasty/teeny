/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file declares metaprogramming math.                                **
 **                                                                         **
 ** +---------------------------------+------------------------+----------+ **
 ** | Applies to pack of values       | Applies to pack type   | Returns  | **
 ** +---------------------------------+------------------------+----------+ **
 ** | is_positive_value<T,X>          | is_positive<X>         | cbool    | **
 ** | is_negative_value<T,X>          | is_negative<X>         | cbool    | **
 ** | is_zero_value<T,X>              | is_zero<X>             | cbool    | **
 ** | is_nonpositive_value<T, X>      | is_nonpositive<X>      | cbool    | **
 ** | is_nonnegative_value<T, X>      | is_nonnegative<X>      | cbool    | **
 ** | is_nonzero_value<T, X>          | is_nonzero<X>          | cbool    | **
 ** +---------------------------------+------------------------+----------+ **
 ** | is_equal_value<T, X, Y>         | is_equal<X, Y>         | cbool    | **
 ** | is_not_equal_value<T, X, Y>     | is_not_equal<X, Y>     | cbool    | **
 ** | is_greater_value<T, X, Y>       | is_greater<X, Y>       | cbool    | **
 ** | is_lower_value<T, X, Y>         | is_lower<X, Y>         | cbool    | **
 ** | is_greater_equal_value<T, X, Y> | is_greater_equal<X, Y> | cbool    | **
 ** | is_lower_equal_value<T, X, Y>   | is_lower_equal<X, Y>   | cbool    | **
 ** +---------------------------------+------------------------+----------+ **
 ** | prod_values<T, X...>            | prod<carray>           | cscalar  | **
 ** | sum_values<T, X...>             | sum<carray>            | cscalar  | **
 ** | min_values<T, X...>             | min<carray>            | cscalar  | **
 ** | max_values<T, X...>             | max<carray>            | cscalar  | **
 ** | any_values<T, X...>             | any<carray>            | cscalar  | **
 ** | all_values<T, X...>             | all<carray>            | cscalar  | **
 ** | count_values<T, X...>           |                        | csize    | **
 ** | count_types<T...>               |                        | csize    | **
 ** +---------------------------------+------------------------+----------+ **
 ** | cumprod_values<T, X...>         | cumprod<carray>        | carray   | **
 ** | cumsum_values<T, X...>          | cumsum<carray>         | carray   | **
 ** | cummin_values<T, X...>          | cummin<carray>         | carray   | **
 ** | cummax_values<T, X...>          | cummax<carray>         | carray   | **
 ** | cumany_values<T, X...>          | cumany<carray>         | carray   | **
 ** | cumall_values<T, X...>          | cumall<carray>         | carray   | **
 ** +---------------------------------+------------------------+----------+ **
 ** |                                 | add<X, Y, ...>         | carray   | **
 ** |                                 | sub<X, Y>              | carray   | **
 ** |                                 | mul<X, Y, ...>         | carray   | **
 ** |                                 | div<X, Y>              | carray   | **
 ** |                                 | minimum<X, Y, ...>     | carray   | **
 ** |                                 | maximum<X, Y, ...>     | carray   | **
 ** |                                 | and<X, Y, ...>         | carray   | **
 ** |                                 | or<X, Y, ...>          | carray   | **
 ** |                                 | xor<X, Y>              | carray   | **
 ** |                                 | lshift<X, Y>           | carray   | **
 ** |                                 | rshift<X, Y>           | carray   | **
 ** +---------------------------------+------------------------+----------+ **
 ** |                                 | not<X>                 | cbool    | **
 ** |                                 | neg<X>                 | carray   | **
 ** |                                 | abs<X>                 | carray   | **
 ** +---------------------------------+------------------------+----------+ **
 ** |                                 | isin<X, Y>             | cbool    | **
 ** |                                 | nonzero<X>             | csize    | **
 ** +---------------------------------+------------------------+----------+ **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef TNY__STATIX__MATH_DECL
#define TNY__STATIX__MATH_DECL

#include <teeny/_core/defines.h>
#include <teeny/_statix/_carray/decl.h>
#include <teeny/_statix/_packapi/decl.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

/* ================================================================== *
 *     Sign                                                           *
 * ================================================================== */

#define _TNY_DECLARE(NAME) \
    template <class T, T... X> struct _is_##NAME##_value ; \
    template <class T, T... X> using   is_##NAME##_value = typename _is_##NAME##_value<T, X...>::type; \
    template <class X>         struct _is_##NAME         ; \
    template <class X>         using   is_##NAME         = typename _is_##NAME<X>::type;

_TNY_DECLARE(positive)
_TNY_DECLARE(negative)
_TNY_DECLARE(zero)
_TNY_DECLARE(nonzero)
_TNY_DECLARE(nonpositive)
_TNY_DECLARE(nonnegative)
#undef _TNY_DECLARE

#define _TNY_DECLARE(NAME) \
    template <class T, T X, T... Y> struct _is_##NAME##_value ; \
    template <class T, T X, T... Y> using   is_##NAME##_value = typename _is_##NAME##_value<T, X, Y...>::type; \
    template <class X, class Y>     struct _is_##NAME         ; \
    template <class X, class Y>     using   is_##NAME         = typename _is_##NAME<X,Y>::type;

_TNY_DECLARE(greater)
_TNY_DECLARE(lower)
_TNY_DECLARE(equal)
_TNY_DECLARE(not_equal)
_TNY_DECLARE(greater_equal)
_TNY_DECLARE(lower_equal)
#undef _TNY_DECLARE

/* ================================================================== *
 *     Reduction                                                      *
 * ================================================================== */

#define _TNY_DECLARE(NAME) \
    template <class T, T... X> struct _##NAME##_values ; \
    template <class T, T... X> using     NAME##_values = typename _##NAME##_values<T, X...>::type; \
    template <class A>         struct _##NAME          ; \
    template <class A>         using     NAME          = typename _##NAME<A>::type;

_TNY_DECLARE(prod)
_TNY_DECLARE(sum)
_TNY_DECLARE(min)
_TNY_DECLARE(max)
_TNY_DECLARE(any)
_TNY_DECLARE(all)

_TNY_DECLARE(cumprod)
_TNY_DECLARE(cumsum)
_TNY_DECLARE(cummin)
_TNY_DECLARE(cummax)
_TNY_DECLARE(cumany)
_TNY_DECLARE(cumall)

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

_TNY_DECLARE(shifted_cumprod)
_TNY_DECLARE(shifted_cumsum)
_TNY_DECLARE(shifted_cummin)
_TNY_DECLARE(shifted_cummax)
_TNY_DECLARE(shifted_cumany)
_TNY_DECLARE(shifted_cumall)

#undef _TNY_DECLARE

/* ================================================================== *
 *     Operations                                                     *
 * ================================================================== */

#define _TNY_DECLARE(NAME) \
    template <class X, class... Y>  struct _##NAME  ; \
    template <class X, class... Y>  using     NAME  = typename _##NAME<X, Y...>::type;

_TNY_DECLARE(add)
_TNY_DECLARE(sub)
_TNY_DECLARE(mul)
_TNY_DECLARE(div)
_TNY_DECLARE(boolean_and)
_TNY_DECLARE(boolean_or)
_TNY_DECLARE(boolean_xor)
_TNY_DECLARE(bitwise_and)
_TNY_DECLARE(bitwise_or)
_TNY_DECLARE(bitwise_xor)
_TNY_DECLARE(lshift)
_TNY_DECLARE(rshift)
_TNY_DECLARE(minimum)
_TNY_DECLARE(maximum)
#undef _TNY_DECLARE

#define _TNY_DECLARE(NAME) \
    template <class X, size_t L = size<X>::value>  struct _##NAME  ; \
    template <class X>  using     NAME  = typename _##NAME<X>::type;

_TNY_DECLARE(abs)
_TNY_DECLARE(neg)
_TNY_DECLARE(boolean_not)
_TNY_DECLARE(bitwise_not)
#undef _TNY_DECLARE

/* ================================================================== *
 *     count                                                          *
 * ================================================================== */

template <class... T>           struct _countT;
template <class T, T... X>      struct _countV;

/** @brief Count the number of elements in a parameter pack (of types) */
template <class... T>
using count_types = typename _countT<T...>::type;

/** @brief Count the number of elements in a parameter pack (of objects) */
template <class T, T... X>
using count_values = typename _countV<T, X...>::type;

/* ================================================================== *
 *     isin                                                           *
 * ================================================================== */

template <class A, class B> struct _isin;
template <class A, class B> using   isin = typename _isin<A, B>::type;

/* ================================================================== *
 *     nonzero                                                        *
 * ================================================================== */

template <class A> struct _nonzero;
template <class A> using   nonzero = typename _nonzero<A>::type;

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif // TNY__STATIX__MATH_DECL
