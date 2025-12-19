#ifndef MINITEN__CORE_TYPES
#define MINITEN__CORE_TYPES
#include <cstddef>  // size_t, ptrdiff_t
#include <cstdint>  // (u)int*_t, uintptr_t
#include <climits>  // INT MIN/MAX
#include <cfloat>   // FLT MIN/MAX
#include <miniten/_core/defines.h>
// #include <miniten/_core/half.h>

NAMESPACE_BEGIN(miniten)

using std::uint8_t;
using std::int8_t;
using std::uint16_t;
using std::int16_t;
using std::uint32_t;
using std::int32_t;
using std::uint64_t;
using std::int64_t;
using std::uintptr_t;
using std::size_t;
using float32_t = float;
using float64_t = double;

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
    static constexpr uint8_t Max = UINT8_MAX;
};

template <>
struct TypeInfo<int8_t>
{
    static constexpr int8_t Min = INT8_MIN;
    static constexpr int8_t Max = INT8_MAX;
};

template <>
struct TypeInfo<uint16_t>
{
    static constexpr uint16_t Min = static_cast<uint16_t>(0);
    static constexpr uint16_t Max = UINT16_MAX;
};

template <>
struct TypeInfo<int16_t>
{
    static constexpr int16_t Min = INT16_MIN;
    static constexpr int16_t Max = INT16_MAX;
};

template <>
struct TypeInfo<uint32_t>
{
    static constexpr uint32_t Min = static_cast<uint32_t>(0);
    static constexpr uint32_t Max = UINT32_MAX;
};

template <>
struct TypeInfo<int32_t>
{
    static constexpr int32_t Min = INT32_MIN;
    static constexpr int32_t Max = INT32_MAX;
};

template <>
struct TypeInfo<uint64_t>
{
    static constexpr uint64_t Min = static_cast<uint64_t>(0);
    static constexpr uint64_t Max = UINT64_MAX;
};

template <>
struct TypeInfo<int64_t>
{
    static constexpr int64_t Min = INT64_MIN;
    static constexpr int64_t Max = INT64_MAX;
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

NAMESPACE_END(miniten)

#endif // MINITEN__CORE_TYPES
