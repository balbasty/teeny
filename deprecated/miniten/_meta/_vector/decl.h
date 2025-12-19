/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file declares a compile-time "vector of values" and aliases.       **
 **                                                                         **
 ** Classes                                                                 **
 ** -------                                                                 **
 ** Vector<T, X...>                                                         **
 **                                                                         **
 ** Aliases                                                                 **
 ** -------                                                                 **
 ** Scalar<T, X>           = Vector<T, X>                                   **
 ** Empty<T>               = Vector<T>                                      **
 **                                                                         **
 ** Bool<X...>             =    B<X...> = Vector<bool,           X...>      **
 ** SignedChar<X...>       =    C<X...> = Vector<signed char,    X...>      **
 ** Short<X...>            =    S<X...> = Vector<short,          X...>      **
 ** Int<X...>              =    I<X...> = Vector<int,            X...>      **
 ** Long<X...>             =    L<X...> = Vector<long,           X...>      **
 ** LongLong<X...>         =   LL<X...> = Vector<long long,      X...>      **
 ** UnsignedChar<X...>     =   UC<X...> = Vector<unsigned char,  X...>      **
 ** UnsignedShort<X...>    =   US<X...> = Vector<unsigned short, X...>      **
 ** UnsignedInt<X...>      =   UI<X...> = Vector<unsigned int,   X...>      **
 ** UnsignedLong<X...>     =   UL<X...> = Vector<unsigned long,  X...>      **
 ** UnsignedLongLong<X...> =  ULL<X...> = Vector<unsigned long long, X...>  **
 ** Int8<X...>             =   I1<X...> = Vector<int8_t,    X...>           **
 ** Int16<X...>            =   I2<X...> = Vector<int16_t,   X...>           **
 ** Int32<X...>            =   I4<X...> = Vector<int32_t,   X...>           **
 ** Int64<X...>            =   I8<X...> = Vector<int64_t,   X...>           **
 ** UInt8<X...>            =   U1<X...> = Vector<uint8_t,   X...>           **
 ** UInt16<X...>           =   U2<X...> = Vector<uint16_t,  X...>           **
 ** UInt32<X...>           =   U4<X...> = Vector<uint32_t,  X...>           **
 ** UInt64<X...>           =   U8<X...> = Vector<uint64_t,  X...>           **
 ** PtrDiff<X...>          =    Z<X...> = Vector<ptrdiff_t, X...>           **
 ** SizeT<X...>            =   UZ<X...> = Vector<size_t,    X...>           **
 ** UIntPtr<X...>          =   UP<X...> = Vector<uintptr_t, X...>           **
 **                                                                         **
 ** True                   = Bool<true>                                     **
 ** False                  = Bool<false>                                    **
 ** Error                  = Empty<bool>                                    **
 ** None                   = Empty<void>                                    **
 **                                                                         **
 ** IsTrue<T>              = IsSame<T, True>                                **
 ** IsFalse<T>             = IsSame<T, False>                               **
 ** IsError<T>             = IsSame<T, Error>                               **
 ** IsNone<T>              = IsSame<T, None>                                **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef MINITEN__META__VECTOR_DECL
#define MINITEN__META__VECTOR_DECL
#include <miniten/_core/defines.h>
#include <miniten/_core/types.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

/* ------------------------------------------------------------------ *
 *     Forward declarations                                           *
 * ------------------------------------------------------------------ */

/**
 * @brief A compile-time tuple of values with the same type
 */
template <class T, T... N> struct Vector;

/* ------------------------------------------------------------------ *
 *     Repeat                                                         *
 * ------------------------------------------------------------------ */

template <long N, typename T, T Val>
struct _NVector;

/**
 * @brief Create a Vector of N times the same value
 *
 * NVector<N, T, V> = Vector<T, V...>
 */
template <long N, typename T, T Val>
using NVector = typename _NVector<N, T, Val>::Type;

/* ------------------------------------------------------------------ *
 *     Aliases                                                        *
 * ------------------------------------------------------------------ */

/** @brief A compile-time value */
template <typename T, T N>
using Scalar = Vector<T, N>;

template <class T>
using Empty = Vector<T>;

/** @brief A compile-time bool */
template <bool... N>
using Bool = Vector<bool, N...>;

template <bool... N>
using B = Bool<N...>;

/** @brief A compile-time signed char */
template <signed char... N>
using SignedChar = Vector<signed char, N...>;

/** @brief A compile-time signed char */
template <signed char... N>
using SC = SignedChar<N...>;

/** @brief A compile-time short integer */
template <short... N>
using Short = Vector<short, N...>;

template <short... N>
using S = Short<N...>;

/** @brief A compile-time integer */
template <int... N>
using Int = Vector<int, N...>;

template <int... N>
using I = Int<N...>;

/** @brief A compile-time long integer */
template <long... N>
using Long = Vector<long, N...>;

template <long... N>
using L = Long<N...>;

/** @brief A compile-time long long integer */
template <long long... N>
using LongLong = Vector<long long, N...>;

template <long long... N>
using LL = LongLong<N...>;

/** @brief A compile-time unsigned char */
template <unsigned char... N>
using UnsignedChar = Vector<unsigned char, N...>;

template <unsigned char... N>
using UC = UnsignedChar<N...>;

/** @brief A compile-time unsigned short integer */
template <unsigned short... N>
using UnsignedShort = Vector<unsigned short, N...>;

template <unsigned short... N>
using US = UnsignedShort<N...>;

/** @brief A compile-time unsigned integer */
template <unsigned int... N>
using UnsignedInt = Vector<unsigned int, N...>;

template <unsigned int... N>
using U = UnsignedInt<N...>;

/** @brief A compile-time unsigned long integer */
template <unsigned long... N>
using UnsignedLong = Vector<unsigned long, N...>;

template <unsigned long... N>
using UL = UnsignedLong<N...>;

/** @brief A compile-time unsigned long long integer */
template <unsigned long long... N>
using UnsignedLongLong = Vector<unsigned long long, N...>;

template <unsigned long long... N>
using ULL = UnsignedLongLong<N...>;

/** @brief A compile-time signed char */
template <int8_t... N>
using Int8 = Vector<int8_t, N...>;

template <int8_t... N>
using I1 = Int8<N...>;

/** @brief A compile-time short integer */
template <int16_t... N>
using Int16 = Vector<int16_t, N...>;

template <int16_t... N>
using I2 = Int16<N...>;

/** @brief A compile-time integer */
template <int32_t... N>
using Int32 = Vector<int32_t, N...>;

template <int32_t... N>
using I4 = Int32<N...>;

/** @brief A compile-time long integer */
template <int64_t... N>
using Int64 = Vector<int64_t, N...>;

template <int64_t... N>
using I8 = Int64<N...>;

/** @brief A compile-time unsigned char */
template <uint8_t... N>
using UInt8 = Vector<uint8_t, N...>;

template <uint8_t... N>
using U1 = UInt8<N...>;

/** @brief A compile-time unsigned short integer */
template <uint16_t... N>
using UInt16 = Vector<uint16_t, N...>;

template <uint16_t... N>
using U2 = UInt16<N...>;

/** @brief A compile-time unsigned integer */
template <uint32_t... N>
using UInt32 = Vector<uint32_t, N...>;

template <uint32_t... N>
using U4 = UInt32<N...>;

/** @brief A compile-time unsigned long integer */
template <uint64_t... N>
using UInt64 = Vector<uint64_t, N...>;

template <uint64_t... N>
using U8 = UInt64<N...>;

/** @brief A compile-time size_t */
template <size_t... N>
using SizeT = Vector<size_t, N...>;

template <size_t... N>
using UZ = SizeT<N...>;

/** @brief A compile-time ptrdiff_t */
template <ptrdiff_t... N>
using PtrDiff = Vector<ptrdiff_t, N...>;

template <ptrdiff_t... N>
using Z = PtrDiff<N...>;

/** @brief A compile-time uintptr_t */
template <uintptr_t... N>
using UIntPtr = Vector<uintptr_t, N...>;

template <uintptr_t... N>
using UP = UIntPtr<N...>;

using True  = Bool<true>;
using False = Bool<false>;
using Error = Empty<bool>;
using None  = Empty<void>;

using SC_0   = SC<0>;
using S_0    = S<0>;
using I_0    = I<0>;
using L_0    = L<0>;
using LL_0   = LL<0>;
using Z_0    = Z<0>;

using UC_0   = UC<0>;
using US_0   = US<0>;
using UI_0   = U<0>;
using UL_0   = UL<0>;
using ULL_0  = ULL<0>;
using UZ_0   = UZ<0>;
using UP_0   = UP<0>;

using I1_0   = I1<0>;
using I2_0   = I2<0>;
using I4_0   = I4<0>;
using I8_0   = I8<0>;

using U1_0   = U1<0>;
using U2_0   = U2<0>;
using U4_0   = U4<0>;
using U8_0   = U8<0>;

using SC_1   = SC<1>;
using S_1    = S<1>;
using I_1    = I<1>;
using L_1    = L<1>;
using LL_1   = LL<1>;
using Z_1    = Z<1>;

using UC_1   = UC<1>;
using US_1   = US<1>;
using UI_1   = U<1>;
using UL_1   = UL<1>;
using ULL_1  = ULL<1>;
using UZ_1   = UZ<1>;
using UP_1   = UP<1>;

using I1_1   = I1<1>;
using I2_1   = I2<1>;
using I4_1   = I4<1>;
using I8_1   = I8<1>;

using U1_1   = U1<1>;
using U2_1   = U2<1>;
using U4_1   = U4<1>;
using U8_1   = U8<1>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__VECTOR_DECL
