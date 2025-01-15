/***********************************************************************
 * This file declares a compile-time "vector of values" and aliases.
 *
 * Vector<T, X...>
 * Scalar<T, X>                                       = Vector<T, X>
 * Empty<T>                                           = Vector<T>
 * Bool<X...>             =   B<X...>                 = Vector<bool, X...>
 * SignedChar<X...>       =   C<X...> = Int8<X...>    = Vector<signed char, X...>
 * Short<X...>            =   S<X...> = Int16<X...>   = Vector<short, X...>
 * Int<X...>              =   I<X...> = Int32<X...>   = Vector<int, X...>
 * Long<X...>             =   L<X...> = Int64<X...>   = Vector<long, X...>
 * LongLong<X...>         =  LL<X...> = Int128<X...>  = Vector<long long, X...>
 * UnsignedChar<X...>     =  UC<X...> = UInt8<X...>   = Vector<unsigned char, X...>
 * UnsignedShort<X...>    =  US<X...> = UInt16<X...>  = Vector<unsigned short, X...>
 * UnsignedInt<X...>      =  US<X...> = UInt32<X...>  = Vector<unsigned int, X...>
 * UnsignedLong<X...>     =  UL<X...> = UInt64<X...>  = Vector<unsigned long, X...>
 * UnsignedLongLong<X...> = ULL<X...> = UInt128<X...> = Vector<unsigned long long, X...>
 * PtrDiff<X...>          =   Z<X...>                 = Vector<ptrdiff_t, X...>
 * SizeT<X...>            =  UZ<X...>                 = Vector<size_t, X...>
 *
 * True  = Bool<true>
 * False = Bool<false>
 * Error = Empty<bool>
 * None  = Empty<void>
 *
 * IsTrue<T>  = IsSame<T, True>;
 * IsFalse<T> = IsSame<T, False>;
 * IsError<T> = IsSame<T, Error>;
 * IsNone<T>  = IsSame<T, None>;
 ***********************************************************************/
#ifndef MINITEN_META_VECTOR_H
#define MINITEN_META_VECTOR_H
#include "../_core/types.h"

namespace miniten {
namespace meta {

/// ---------------------------------------------------------------- ///
///     Forward declarations                                         ///
/// ---------------------------------------------------------------- ///

template <class T, T... N> struct Vector;

/// ---------------------------------------------------------------- ///
///     Aliases                                                      ///
/// ---------------------------------------------------------------- ///

/// A compile-time value
template <typename T, T N>
using Scalar = Vector<T, N>;

template <class T>
using Empty = Vector<T>;

/// A compile-time bool
template <bool... N>
using Bool = Vector<bool, N...>;

template <bool... N>
using B = Bool<N...>;

/// A compile-time signed char
template <signed char... N>
using SignedChar = Vector<signed char, N...>;

/// A compile-time signed char
template <signed char... N>
using C = SignedChar<N...>;

/// A compile-time short integer
template <short... N>
using Short = Vector<short, N...>;

template <short... N>
using S = Short<N...>;

/// A compile-time integer
template <int... N>
using Int = Vector<int, N...>;

template <int... N>
using I = Int<N...>;

/// A compile-time long integer
template <long... N>
using Long = Vector<long, N...>;

template <long... N>
using L = Long<N...>;

/// A compile-time long long integer
template <long long... N>
using LongLong = Vector<long long, N...>;

template <long long... N>
using LL = LongLong<N...>;

/// A compile-time unsigned char
template <unsigned char... N>
using UnsignedChar = Vector<unsigned char, N...>;

template <unsigned char... N>
using UC = UnsignedChar<N...>;

/// A compile-time unsigned short integer
template <unsigned short... N>
using UnsignedShort = Vector<unsigned short, N...>;

template <unsigned short... N>
using US = UnsignedShort<N...>;

/// A compile-time unsigned integer
template <unsigned int... N>
using UnsignedInt = Vector<unsigned int, N...>;

template <unsigned int... N>
using U = UnsignedInt<N...>;

/// A compile-time unsigned long integer
template <unsigned long... N>
using UnsignedLong = Vector<unsigned long, N...>;

template <unsigned long... N>
using UL = UnsignedLong<N...>;

/// A compile-time unsigned long long integer
template <unsigned long long... N>
using UnsignedLongLong = Vector<unsigned long long, N...>;

template <unsigned long long... N>
using ULL = UnsignedLongLong<N...>;

/// A compile-time signed char
template <int8_t... N>
using Int8 = Vector<int8_t, N...>;

/// A compile-time short integer
template <int16_t... N>
using Int16 = Vector<int16_t, N...>;

/// A compile-time integer
template <int32_t... N>
using Int32 = Vector<int32_t, N...>;

/// A compile-time long integer
template <int64_t... N>
using Int64 = Vector<int64_t, N...>;

/// A compile-time long long integer
template <int128_t... N>
using Int128 = Vector<int128_t, N...>;

/// A compile-time unsigned char
template <uint8_t... N>
using UInt8 = Vector<uint8_t, N...>;

/// A compile-time unsigned short integer
template <uint16_t... N>
using UInt16 = Vector<uint16_t, N...>;

/// A compile-time unsigned integer
template <uint32_t... N>
using UInt32 = Vector<uint32_t, N...>;

/// A compile-time unsigned long integer
template <uint64_t... N>
using UInt64 = Vector<uint64_t, N...>;

/// A compile-time unsigned long long integer
template <uint128_t... N>
using UInt128 = Vector<uint128_t, N...>;

/// A compile-time size_t
template <size_t... N>
using SizeT = Vector<size_t, N...>;

template <size_t... N>
using UZ = SizeT<N...>;

/// A compile-time ptrdiff_t
template <ptrdiff_t... N>
using PtrDiff = Vector<ptrdiff_t, N...>;

template <ptrdiff_t... N>
using Z = PtrDiff<N...>;

using True  = Bool<true>;
using False = Bool<false>;
using Error = Empty<bool>;
using None  = Empty<void>;

using C0    = C<0>;
using S0    = S<0>;
using I0    = I<0>;
using L0    = L<0>;
using LL0   = LL<0>;
using Z0    = Z<0>;

using UC0   = UC<0>;
using US0   = US<0>;
using UI0   = U<0>;
using UL0   = UL<0>;
using ULL0  = ULL<0>;
using UZ0   = UZ<0>;

using C1    = C<1>;
using S1    = S<1>;
using I1    = I<1>;
using L1    = L<1>;
using LL1   = LL<1>;
using Z1    = Z<1>;

using UC1   = UC<1>;
using US1   = US<1>;
using UI1   = U<1>;
using UL1   = UL<1>;
using ULL1  = ULL<1>;
using UZ1   = UZ<1>;

/// ---------------------------------------------------------------- ///
///     Repeat                                                       ///
/// ---------------------------------------------------------------- ///

template <long N, typename T, T Val>
struct _NVector;

/// Create a Vector of N times the same value
///
/// NVector<N, T, V>
///   Type = Vector<T, V...>
template <long N, typename T, T Val>
using NVector = typename _NVector<N, T, Val>::Type;

} // namespace meta
} // namespace miniten

#endif // MINITEN_META_VECTOR_H
