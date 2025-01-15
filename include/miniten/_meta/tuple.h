/***********************************************************************
 * This file declares a compile-time "tuple of types"
 *
 * Tuple<T...>
 * Element<T>   = Tuple<T>
 * Pair<T, U>   = Tuple<T, U>
 * NTuple<N, T> = Tuple<T... (N times)>
 ***********************************************************************/
#ifndef MINITEN_META_TUPLE_H
#define MINITEN_META_TUPLE_H
#include "../_core/defines.h"
#include "packapi.h"

namespace miniten {
namespace meta {

/// ---------------------------------------------------------------- ///
///     Tuple                                                        ///
/// ---------------------------------------------------------------- ///

template <class... T>
struct Tuple;

/// ---------------------------------------------------------------- ///
///     Aliases                                                      ///
/// ---------------------------------------------------------------- ///

/// A compile-time single element
template <class T>
using Element = Tuple<T>;

/// A compile-time pair of elements
template <class T, class U = T>
using Pair = Tuple<T, U>;

/// ---------------------------------------------------------------- ///
///     Repeat                                                       ///
/// ---------------------------------------------------------------- ///

template <long N, class T> struct _NTuple;

/// Create a Tuple of N times the same type
///
/// NTuple<N, T>
///   Type = Tuple<T...>
template <long N, class T>
using NTuple = typename _NTuple<N,T>::Type;

/// ---------------------------------------------------------------- ///
///     Helpers                                                      ///
/// ---------------------------------------------------------------- ///

template <class... X>
Tuple<X...> tuple(const X &...);

// template <class X0, class... X>
// Cat<X0, X...> cat(const X0 &, const X &...);

} /// namespace meta
} /// namespace miniten

#endif /// MINITEN_META_TUPLE_H
