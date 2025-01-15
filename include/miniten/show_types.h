#ifndef MINITEN_SHOW_TYPES_H
#define MINITEN_SHOW_TYPES_H
#include "_core/defines.h"
#include "_core/types.h"
#include "show.h"

namespace miniten {

template <>
struct Show<bool> {
    MINITEN_HOSTDEVICE static inline void show()    { printf("bool"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Bool"); }
    MINITEN_HOSTDEVICE static inline void show(bool value) { printf(value ? "true" :"false"); }
};

template <>
struct Show<uint8_t>
{
    MINITEN_HOSTDEVICE static inline void show()    { printf("uint8"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("UInt8"); }
    MINITEN_HOSTDEVICE static inline void show(uint8_t value) { printf("%u", value); }
};

template <>
struct Show<int8_t>
{
    MINITEN_HOSTDEVICE static inline void show()    { printf("int8"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Int8"); }
    MINITEN_HOSTDEVICE static inline void show(int8_t value) { printf("%d", value); }
};

template <>
struct Show<uint16_t>
{
    MINITEN_HOSTDEVICE static inline void show()    { printf("uint16"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("UInt16"); }
    MINITEN_HOSTDEVICE static inline void show(uint16_t value) { printf("%u", value); }
};

template <>
struct Show<int16_t>
{
    MINITEN_HOSTDEVICE static inline void show()    { printf("int16"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Int16"); }
    MINITEN_HOSTDEVICE static inline void show(uint16_t value) { printf("%d", value); }
};

template <>
struct Show<uint32_t>
{
    MINITEN_HOSTDEVICE static inline void show()    { printf("uint32"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("UInt32"); }
    MINITEN_HOSTDEVICE static inline void show(uint32_t value) { printf("%u", value); }
};

template <>
struct Show<int32_t>
{
    MINITEN_HOSTDEVICE static inline void show()    { printf("int32"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Int32"); }
    MINITEN_HOSTDEVICE static inline void show(int32_t value) { printf("%d", value); }
};

template <>
struct Show<uint64_t>
{
    MINITEN_HOSTDEVICE static inline void show()    { printf("uint64"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("UInt64"); }
    MINITEN_HOSTDEVICE static inline void show(const uint64_t & value) { printf("%lu", value); }
};

template <>
struct Show<int64_t>
{
    MINITEN_HOSTDEVICE static inline void show()    { printf("int64"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Int64"); }
    MINITEN_HOSTDEVICE static inline void show(const int64_t & value) { printf("%ld", value); }
};

template <>
struct Show<uint128_t>
{
    MINITEN_HOSTDEVICE static inline void show()    { printf("uint128"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("UInt128"); }
    MINITEN_HOSTDEVICE static inline void show(const uint128_t & value) { printf("%llu", value); }
};

template <>
struct Show<int128_t>
{
    MINITEN_HOSTDEVICE static inline void show()    { printf("int128"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Int128"); }
    MINITEN_HOSTDEVICE static inline void show(const int128_t & value) { printf("%lld", value); }
};

template <>
struct Show<float32_t>
{
    MINITEN_HOSTDEVICE static inline void show()    { printf("float32"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Float32"); }
    MINITEN_HOSTDEVICE static inline void show(float32_t value) { printf("%f", value); }
};

template <>
struct Show<float64_t>
{
    MINITEN_HOSTDEVICE static inline void show()    { printf("float64"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Float64"); }
    MINITEN_HOSTDEVICE static inline void show(const float64_t & value) { printf("%f", value); }
};

template <>
struct Show<float128_t>
{
    MINITEN_HOSTDEVICE static inline void show()    { printf("float128"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Float128"); }
    MINITEN_HOSTDEVICE static inline void show(const float128_t & value) { printf("%Lf", value); }
};

} // namespace miniten

#endif // MINITEN_SHOW_TYPES_H
