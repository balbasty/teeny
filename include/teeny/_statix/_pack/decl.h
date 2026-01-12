/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** The most basic possible "pack of types" metatype.                       **
 **                                                                         **
 ** This type **cannot** be used to store data. For a "tuple of types"      **
 ** that can store objects of each of these types, use `cuda::std::tuple`.  **
 **                                                                         **
 ** Classes                                                                 **
 ** --------                                                                **
 ** pack<T...>                                                              **
 **                                                                         **
 ** Aliases                                                                 **
 ** -------                                                                 **
 ** type<T>     = pack<T>                                                   **
 ** npack<N, T> = pack<T... (N times)>                                      **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef TNY__STATIX__PACK_DECL
#define TNY__STATIX__PACK_DECL
#include <teeny/core.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

/**
 * @brief Pack of types.
 *
 * pack<T...>
 *
 * @tparam T...  Wrapped Type(s).
 */
template <class... T> struct pack;

/**
 * @brief A single type.
 *
 * type<T>
 *
 * @tparam T  Wrapped Type.
 */
template <class T> using type = pack<T>;

/* ------------------------------------------------------------------ *
 *     Repeat                                                         *
 * ------------------------------------------------------------------ */

template <size_t N, class T> struct _npack;

/**
 * @brief Create a Pack of N times the same type.
 *
 * npack<N, T> = pack<T...>
 *
 * @tparam N    Number of repetitions
 * @tparam T... Type to repeat
 */
template <size_t N, class T>
using npack = typename _npack<N,T>::type;

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif // TNY__STATIX__PACK_DECL
