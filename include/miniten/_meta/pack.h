#ifndef MINITEN_META_PACK_H
#define MINITEN_META_PACK_H

namespace miniten {
namespace meta {

template <class... T> struct Pack;

/// ---------------------------------------------------------------- ///
///     Repeat                                                       ///
/// ---------------------------------------------------------------- ///

template <long N, class T> struct _NPack;

/// Create a Pack of N times the same type
///
/// NPack<N, T>
///   Type = Pack<T...>
template <long N, class T>
using NPack = typename _NPack<N,T>::Type;

} // namespace meta
} // namespace miniten

#endif // MINITEN_META_PACK_H
