/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file declares a compile-time "array of values" and aliases.        **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef TNY__STATIX__CARRAY_DECL
#define TNY__STATIX__CARRAY_DECL
#include <cuda/std/type_traits>
#include <teeny/core.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

using cuda::std::is_same;

/* ------------------------------------------------------------------ *
 *     carray                                                         *
 * ------------------------------------------------------------------ */

/**
 * @brief A compile-time array of values with the same type
 *
 * @tparam T   Item type
 * @tparam X   Items
 */
template <class T, T... X> struct carray;

/* ------------------------------------------------------------------ *
 *     crepeat                                                        *
 * ------------------------------------------------------------------ */

template <size_t N, typename T, T X>
struct _crepeat;

/**
 * @brief Create a compile-time array with N times the same value
 *
 * crepeat<N, T, V> = carray<T, V, ..., V >
 *
 * @tparam N   Number of repetitions
 * @tparam T   Item type
 * @tparam X   Item to repeat
 */
template <size_t N, typename T, T Val>
using crepeat = typename _crepeat<N, T, Val>::type;

/* ------------------------------------------------------------------ *
 *     crange                                                         *
 * ------------------------------------------------------------------ */

template <ptrdiff_t Stop, ptrdiff_t Start = 0, ptrdiff_t Step = 1,
          bool IsEmpty = (Step > 0 ? Start >= Stop : Start <= Stop)>
struct _crange;

template <ptrdiff_t Stop, ptrdiff_t Start = 0, ptrdiff_t Step = 1>
using crange = typename _crange<Stop, Start, Step>::type;

/* ------------------------------------------------------------------ *
 *     aliases                                                        *
 * ------------------------------------------------------------------ */

/** @brief A compile-time value */
template <typename T, T N>
using cvalue = carray<T, N>;

/** @brief An empty array */
template <class T>
using cempty = carray<T>;

/** @brief A compile-time bool */
template <bool... X>
using cbool = carray<bool, X...>;

/** @brief A compile-time signed char */
template <signed char... X>
using cschar = carray<signed char, X...>;

/** @brief A compile-time short integer */
template <short... X>
using cshort = carray<short, X...>;

/** @brief A compile-time integer */
template <int... X>
using cint = carray<int, X...>;

/** @brief A compile-time long integer */
template <long... X>
using clong = carray<long, X...>;

/** @brief A compile-time long long integer */
template <long long... X>
using clonglong = carray<long long, X...>;

/** @brief A compile-time unsigned char */
template <unsigned char... X>
using cuchar = carray<unsigned char, X...>;

/** @brief A compile-time unsigned short integer */
template <unsigned short... X>
using cushort = carray<unsigned short, X...>;

/** @brief A compile-time unsigned integer */
template <unsigned int... X>
using cuint = carray<unsigned int, X...>;

/** @brief A compile-time unsigned long integer */
template <unsigned long... X>
using culong = carray<unsigned long, X...>;

/** @brief A compile-time unsigned long long integer */
template <unsigned long long... X>
using culonglong = carray<unsigned long long, X...>;

/** @brief A compile-time signed char */
template <int8_t... X>
using cint8 = carray<int8_t, X...>;

/** @brief Alias for cint8 */
template <int8_t... X>
using ci1 = cint8<X...>;

/** @brief A compile-time short integer */
template <int16_t... X>
using cint16 = carray<int16_t, X...>;

/** @brief Alias for cint16 */
template <int16_t... X>
using ci2 = cint16<X...>;

/** @brief A compile-time integer */
template <int32_t... X>
using cint32 = carray<int32_t, X...>;

/** @brief Alias for cint32 */
template <int32_t... X>
using ci4 = cint32<X...>;

/** @brief A compile-time long integer */
template <int64_t... X>
using cint64 = carray<int64_t, X...>;

/** @brief Alias for cint64 */
template <int64_t... X>
using ci8 = cint64<X...>;

/** @brief A compile-time unsigned char */
template <uint8_t... X>
using cuint8 = carray<uint8_t, X...>;

/** @brief Alias for cuint8 */
template <uint8_t... X>
using cu1 = cuint8<X...>;

/** @brief A compile-time unsigned short integer */
template <uint16_t... X>
using cuint16 = carray<uint16_t, X...>;

/** @brief Alias for cuint16 */
template <uint16_t... X>
using cu2 = cuint16<X...>;

/** @brief A compile-time unsigned integer */
template <uint32_t... X>
using cuint32 = carray<uint32_t, X...>;

/** @brief Alias for cuint32 */
template <uint32_t... X>
using cu4 = cuint32<X...>;

/** @brief A compile-time unsigned long integer */
template <uint64_t... X>
using cuint64 = carray<uint64_t, X...>;

/** @brief Alias for cuint64 */
template <uint64_t... X>
using cu8 = cuint64<X...>;

/** @brief A compile-time size_t */
template <size_t... X>
using csize = carray<size_t, X...>;

/** @brief Alias for csize */
template <size_t... X>
using cuz = csize<X...>;

/** @brief A compile-time ptrdiff_t */
template <ptrdiff_t... X>
using cptrdiff = carray<ptrdiff_t, X...>;

/** @brief Alias for cptrdiff */
template <ptrdiff_t... X>
using ciz = cptrdiff<X...>;

/** @brief A compile-time uintptr_t */
template <uintptr_t... X>
using cuintptr = carray<uintptr_t, X...>;

/** @brief Alias for cuintptr */
template <uintptr_t... X>
using cuv = cuintptr<X...>;

/** @brief A compile-time intptr_t */
template <intptr_t... X>
using cintptr = carray<intptr_t, X...>;

/** @brief Alias for cintptr */
template <intptr_t... X>
using civ = cintptr<X...>;

using ctrue  = cbool<true>;
using cfalse = cbool<false>;
using cerror = cempty<bool>;
using cnone  = cempty<void>;

template <class T> using is_ctrue  = is_same<T, ctrue>;
template <class T> using is_cfalse = is_same<T, cfalse>;
template <class T> using is_cerror = is_same<T, cerror>;
template <class T> using is_cnone  = is_same<T, cnone>;

using ci1_0  = ci1<0>;
using ci2_0  = ci2<0>;
using ci4_0  = ci4<0>;
using ci8_0  = ci8<0>;
using ciz_0  = ciz<0>;
using civ_0  = civ<0>;

using cu1_0  = cu1<0>;
using cu2_0  = cu2<0>;
using cu4_0  = cu4<0>;
using cu8_0  = cu8<0>;
using cuz_0  = cuz<0>;
using cuv_0  = cuv<0>;

using ci1_1  = ci1<1>;
using ci2_1  = ci2<1>;
using ci4_1  = ci4<1>;
using ci8_1  = ci8<1>;
using ciz_1  = ciz<1>;
using civ_1  = civ<1>;

using cu1_1  = cu1<1>;
using cu2_1  = cu2<1>;
using cu4_1  = cu4<1>;
using cu8_1  = cu8<1>;
using cuz_1  = cuz<1>;
using cuv_1  = cuv<1>;

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif // TNY__STATIX__CARRAY_DECL
