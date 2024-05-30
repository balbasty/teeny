#ifndef MINITEN_TYPES_H
#define MINITEN_TYPES_H
#include <cstdio>
#include <climits>
#include <cfloat>
#include "defines.h"
#include "half.h"
#include "show.h"

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
using float16_t  = half;
using float32_t  = float;
using float64_t  = double;
using float128_t = long double;

namespace traits {

template <typename T>
struct type_info
{
    static constexpr uint8_t Min = static_cast<uint8_t>(0);
    static constexpr uint8_t Max = static_cast<uint8_t>(0);
};

template <>
struct type_info<bool>
{
    static constexpr bool Min = false;
    static constexpr bool Max = true;
};

template <>
struct type_info<uint8_t>
{
    static constexpr uint8_t Min = static_cast<uint8_t>(0);
    static constexpr uint8_t Max = UCHAR_MAX;
};

template <>
struct type_info<int8_t>
{
    static constexpr int8_t Min = SCHAR_MIN;
    static constexpr int8_t Max = SCHAR_MAX;
};

template <>
struct type_info<uint16_t>
{
    static constexpr uint16_t Min = static_cast<uint16_t>(0);
    static constexpr uint16_t Max = USHRT_MAX;
};

template <>
struct type_info<int16_t>
{
    static constexpr int16_t Min = SHRT_MIN;
    static constexpr int16_t Max = SHRT_MAX;
};

template <>
struct type_info<uint32_t>
{
    static constexpr uint32_t Min = static_cast<uint32_t>(0);
    static constexpr uint32_t Max = UINT_MAX;
};

template <>
struct type_info<int32_t>
{
    static constexpr int32_t Min = INT_MIN;
    static constexpr int32_t Max = INT_MAX;
};

template <>
struct type_info<uint64_t>
{
    static constexpr uint64_t Min = static_cast<uint64_t>(0);
    static constexpr uint64_t Max = ULONG_MAX;
};

template <>
struct type_info<int64_t>
{
    static constexpr int64_t Min = LONG_MIN;
    static constexpr int64_t Max = LONG_MAX;
};

template <>
struct type_info<uint128_t>
{
    static constexpr uint128_t Min = static_cast<uint128_t>(0);
    static constexpr uint128_t Max = ULLONG_MAX;
};

template <>
struct type_info<int128_t>
{
    static constexpr int128_t Min = LLONG_MIN;
    static constexpr int128_t Max = LLONG_MAX;
};

template <>
struct type_info<float32_t>
{
    static constexpr float32_t Min = FLT_MIN;
    static constexpr float32_t Max = FLT_MIN;
    static constexpr uint32_t  Dig = FLT_DIG;
};

template <>
struct type_info<float64_t>
{
    static constexpr float64_t Min = DBL_MIN;
    static constexpr float64_t Max = DBL_MIN;
    static constexpr uint64_t  Dig = DBL_DIG;
};

template <>
struct type_info<float128_t>
{
    static constexpr float128_t Min = LDBL_MIN;
    static constexpr float128_t Max = LDBL_MIN;
    static constexpr uint128_t  Dig = LDBL_DIG;
};

} // namespace traits

// -----------------------------------------------------------------------------------------
//      PRETTY PRINTING
// -----------------------------------------------------------------------------------------

template <>
struct Show<bool> {
    MINITEN_HOSTDEVICE static inline void show() { printf("bool"); }
    MINITEN_HOSTDEVICE static inline void show(bool value) { printf(value ? "true" :"false"); }
};

template <>
struct Show<uint8_t>
{
    MINITEN_HOSTDEVICE static inline void show() { printf("uint8"); }
    MINITEN_HOSTDEVICE static inline void show(uint8_t value) { printf("%u", value); }
};

template <>
struct Show<int8_t>
{
    MINITEN_HOSTDEVICE static inline void show() { printf("int8"); }
    MINITEN_HOSTDEVICE static inline void show(int8_t value) { printf("%d", value); }
};

template <>
struct Show<uint16_t>
{
    MINITEN_HOSTDEVICE static inline void show() { printf("uint16"); }
    MINITEN_HOSTDEVICE static inline void show(uint16_t value) { printf("%u", value); }
};

template <>
struct Show<int16_t>
{
    MINITEN_HOSTDEVICE static inline void show() { printf("int16"); }
    MINITEN_HOSTDEVICE static inline void show(uint16_t value) { printf("%d", value); }
};

template <>
struct Show<uint32_t>
{
    MINITEN_HOSTDEVICE static inline void show() { printf("uint32"); }
    MINITEN_HOSTDEVICE static inline void show(uint32_t value) { printf("%u", value); }
};

template <>
struct Show<int32_t>
{
    MINITEN_HOSTDEVICE static inline void show() { printf("int32"); }
    MINITEN_HOSTDEVICE static inline void show(int32_t value) { printf("%d", value); }
};

template <>
struct Show<uint64_t>
{
    MINITEN_HOSTDEVICE static inline void show() { printf("uint64"); }
    MINITEN_HOSTDEVICE static inline void show(const uint64_t & value) { printf("%lu", value); }
};

template <>
struct Show<int64_t>
{
    MINITEN_HOSTDEVICE static inline void show() { printf("int64"); }
    MINITEN_HOSTDEVICE static inline void show(const int64_t & value) { printf("%ld", value); }
};

template <>
struct Show<uint128_t>
{
    MINITEN_HOSTDEVICE static inline void show() { printf("uint128"); }
    MINITEN_HOSTDEVICE static inline void show(const uint128_t & value) { printf("%llu", value); }
};

template <>
struct Show<int128_t>
{
    MINITEN_HOSTDEVICE static inline void show() { printf("int128"); }
    MINITEN_HOSTDEVICE static inline void show(const int128_t & value) { printf("%lld", value); }
};

template <>
struct Show<float32_t>
{
    MINITEN_HOSTDEVICE static inline void show() { printf("float32"); }
    MINITEN_HOSTDEVICE static inline void show(float32_t value) { printf("%f", value); }
};

template <>
struct Show<float64_t>
{
    MINITEN_HOSTDEVICE static inline void show() { printf("float64"); }
    MINITEN_HOSTDEVICE static inline void show(const float64_t & value) { printf("%f", value); }
};

template <>
struct Show<float128_t>
{
    MINITEN_HOSTDEVICE static inline void show() { printf("float128"); }
    MINITEN_HOSTDEVICE static inline void show(const float128_t & value) { printf("%Lf", value); }
};

} // namespace miniten

#endif // MINITEN_TYPES_H
