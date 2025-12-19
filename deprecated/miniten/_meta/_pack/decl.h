/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** The most basic possible "pack of types" metatype.                       **
 **                                                                         **
 ** This type **cannot** be used to store data. For a "tuple of types"      **
 ** that can store objects of each of these types, use `Tuple`.             **
 **                                                                         **
 ** Classes                                                                 **
 ** --------                                                                **
 ** Pack<T...>                                                              **
 **                                                                         **
 ** Aliases                                                                 **
 ** -------                                                                 **
 ** Type<T>     = Pack<T>                                                   **
 ** NPack<N, T> = Pack<T... (N times)>                                      **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef MINITEN__META__PACK_DECL
#define MINITEN__META__PACK_DECL
#include <miniten/_core/defines.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

/**
 * @brief Pack of types.
 *
 * Pack<T...>
 *
 * @tparam T  Wrapped Type(s).
 */
template <class... T> struct Pack;

/**
 * @brief A single type.
 *
 * Type<T>
 *
 * @tparam T  Wrapped Type.
 */
template <class T>    using  Type = Pack<T>;

/* ------------------------------------------------------------------ *
 *     Repeat                                                         *
 * ------------------------------------------------------------------ */

template <long N, class T> struct _NPack;

/**
 * @brief Create a Pack of N times the same type.
 *
 * NPack<N, T> = Pack<T...>
 *
 * @tparam N  Number of repetitions
 * @tparam T  Type to repeat
 */
template <long N, class T>
using NPack = typename _NPack<N,T>::Type;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__PACK_DECL
