#ifndef MINITEN_TYPES_H
#define MINITEN_TYPES_H
#include <cstddef>  // size_t, ptrdiff_t
#include <climits>  // INT MIN/MAX
#include <cfloat>   // FLT MIN/MAX
#include "defines.h"
// #include "half.h"

namespace miniten {

using uint8_t    = unsigned char;
using  int8_t    = signed char;
using uint16_t   = unsigned short;
using  int16_t   = short;
using uint32_t   = unsigned int;
using  int32_t   = int;
using uint64_t   = unsigned long;
using  int64_t   = long;
using uint128_t  = unsigned long long;
using  int128_t  = long long;
// using float16_t  = half;
using float32_t  = float;
using float64_t  = double;
using float128_t = long double;

template <typename T>
struct TypeInfo
{
    static constexpr uint8_t Min = static_cast<uint8_t>(0);
    static constexpr uint8_t Max = static_cast<uint8_t>(0);
};

template <>
struct TypeInfo<bool>
{
    static constexpr bool Min = false;
    static constexpr bool Max = true;
};

template <>
struct TypeInfo<uint8_t>
{
    static constexpr uint8_t Min = static_cast<uint8_t>(0);
    static constexpr uint8_t Max = UCHAR_MAX;
};

template <>
struct TypeInfo<int8_t>
{
    static constexpr int8_t Min = SCHAR_MIN;
    static constexpr int8_t Max = SCHAR_MAX;
};

template <>
struct TypeInfo<uint16_t>
{
    static constexpr uint16_t Min = static_cast<uint16_t>(0);
    static constexpr uint16_t Max = USHRT_MAX;
};

template <>
struct TypeInfo<int16_t>
{
    static constexpr int16_t Min = SHRT_MIN;
    static constexpr int16_t Max = SHRT_MAX;
};

template <>
struct TypeInfo<uint32_t>
{
    static constexpr uint32_t Min = static_cast<uint32_t>(0);
    static constexpr uint32_t Max = UINT_MAX;
};

template <>
struct TypeInfo<int32_t>
{
    static constexpr int32_t Min = INT_MIN;
    static constexpr int32_t Max = INT_MAX;
};

template <>
struct TypeInfo<uint64_t>
{
    static constexpr uint64_t Min = static_cast<uint64_t>(0);
    static constexpr uint64_t Max = ULONG_MAX;
};

template <>
struct TypeInfo<int64_t>
{
    static constexpr int64_t Min = LONG_MIN;
    static constexpr int64_t Max = LONG_MAX;
};

template <>
struct TypeInfo<uint128_t>
{
    static constexpr uint128_t Min = static_cast<uint128_t>(0);
    static constexpr uint128_t Max = ULLONG_MAX;
};

template <>
struct TypeInfo<int128_t>
{
    static constexpr int128_t Min = LLONG_MIN;
    static constexpr int128_t Max = LLONG_MAX;
};

template <>
struct TypeInfo<float32_t>
{
    static constexpr float32_t Min = FLT_MIN;
    static constexpr float32_t Max = FLT_MIN;
    static constexpr uint32_t  Dig = FLT_DIG;
};

template <>
struct TypeInfo<float64_t>
{
    static constexpr float64_t Min = DBL_MIN;
    static constexpr float64_t Max = DBL_MIN;
    static constexpr uint64_t  Dig = DBL_DIG;
};

template <>
struct TypeInfo<float128_t>
{
    static constexpr float128_t Min = LDBL_MIN;
    static constexpr float128_t Max = LDBL_MIN;
    static constexpr uint128_t  Dig = LDBL_DIG;
};

} // namespace miniten

#endif // MINITEN_TYPES_H
