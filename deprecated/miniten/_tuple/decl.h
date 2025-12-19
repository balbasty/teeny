/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file declares a compile-time "tuple of types"                      **
 **                                                                         **
 ** Tuple<T...>                                                             **
 ** Element<T>   = Tuple<T>                                                 **
 ** Pair<T, U>   = Tuple<T, U>                                              **
 ** NTuple<N, T> = Tuple<T... (N times)>                                    **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef MINITEN__TUPLE_DECL_H
#define MINITEN__TUPLE_DECL_H
#include <miniten/_core/defines.h>

NAMESPACE_BEGIN(miniten)

/* ------------------------------------------------------------------ *
 *     Tuple                                                          *
 * ------------------------------------------------------------------ */

template <class... T>
struct Tuple;

/* ------------------------------------------------------------------ *
 *     Aliases                                                        *
 * ------------------------------------------------------------------ */

/** @brief A compile-time single element */
template <class T>
using Element = Tuple<T>;

/** @brief A compile-time pair of elements */
template <class T, class U = T>
using Pair = Tuple<T, U>;

/* ------------------------------------------------------------------ *
 *     Repeat                                                         *
 * ------------------------------------------------------------------ */

template <long N, class T> struct _NTuple;

/**
 * @brief Create a Tuple of N times the same type
 *
 * NTuple<N, T> = Tuple<T...>
 */
template <long N, class T>
using NTuple = typename _NTuple<N,T>::Type;

/* ------------------------------------------------------------------ *
 *     Helpers                                                        *
 * ------------------------------------------------------------------ */

template <class... X>
Tuple<X...> tuple(const X &...);

// template <class X0, class... X>
// Cat<X0, X...> cat(const X0 &, const X &...);

NAMESPACE_END(miniten)

#endif // MINITEN__TUPLE_DECL_H
