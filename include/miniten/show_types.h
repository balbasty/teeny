#ifndef MINITEN_SHOW_TYPES_H
#define MINITEN_SHOW_TYPES_H
#include "_core/defines.h"
#include "_core/types.h"
#include "show.h"

namespace miniten {

////////////////////////////////////////////////////////////////////////

template <class T, size_t L = sizeof(T), int64_t MIN = static_cast<int64_t>(TypeInfo<T>::Min)>
struct _ShowInt {};

template <class T, int64_t MIN>
struct _ShowInt<T, 1, MIN> {
    MINITEN_HOSTDEVICE static inline void show()    { printf("int8"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Int8"); }
};

template <class T>
struct _ShowInt<T, 1, 0> {
    MINITEN_HOSTDEVICE static inline void show()    { printf("uint8"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("UInt8"); }
};

template <class T, int64_t MIN>
struct _ShowInt<T, 2, MIN> {
    MINITEN_HOSTDEVICE static inline void show()    { printf("int16"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Int16"); }
};

template <class T>
struct _ShowInt<T, 2, 0> {
    MINITEN_HOSTDEVICE static inline void show()    { printf("uint16"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("UInt16"); }
};

template <class T, int64_t MIN>
struct _ShowInt<T, 4, MIN> {
    MINITEN_HOSTDEVICE static inline void show()    { printf("int32"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Int32"); }
};

template <class T>
struct _ShowInt<T, 4, 0> {
    MINITEN_HOSTDEVICE static inline void show()    { printf("uint32"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("UInt32"); }
};

template <class T, int64_t MIN>
struct _ShowInt<T, 8, MIN> {
    MINITEN_HOSTDEVICE static inline void show()    { printf("int64"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Int64"); }
};

template <class T>
struct _ShowInt<T, 8, 0> {
    MINITEN_HOSTDEVICE static inline void show()    { printf("uint64"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("UInt64"); }
};

template <class T, int64_t MIN>
struct _ShowInt<T, 16, MIN> {
    MINITEN_HOSTDEVICE static inline void show()    { printf("int128"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Int128"); }
};

template <class T>
struct _ShowInt<T, 16, 0> {
    MINITEN_HOSTDEVICE static inline void show()    { printf("uint128"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("UInt128"); }
};

////////////////////////////////////////////////////////////////////////

template <>
struct Show<bool> {
    MINITEN_HOSTDEVICE static inline void show()    { printf("bool"); }
    MINITEN_HOSTDEVICE static inline void showCap() { printf("Bool"); }
    MINITEN_HOSTDEVICE static inline void show(bool value) { printf(value ? "true" :"false"); }
};

////////////////////////////////////////////////////////////////////////

template <>
struct Show<unsigned char>
{
    using T = unsigned char;
    MINITEN_HOSTDEVICE static inline void show()        { _ShowInt<T>::show(); }
    MINITEN_HOSTDEVICE static inline void showCap()     { _ShowInt<T>::showCap(); }
    MINITEN_HOSTDEVICE static inline void show(T value) { printf("%u", value); }
};

template <>
struct Show<signed char>
{
    using T = signed char;
    MINITEN_HOSTDEVICE static inline void show()        { _ShowInt<T>::show(); }
    MINITEN_HOSTDEVICE static inline void showCap()     { _ShowInt<T>::showCap(); }
    MINITEN_HOSTDEVICE static inline void show(T value) { printf("%d", value); }
};

template <>
struct Show<unsigned short>
{
    using T = unsigned short;
    MINITEN_HOSTDEVICE static inline void show()        { _ShowInt<T>::show(); }
    MINITEN_HOSTDEVICE static inline void showCap()     { _ShowInt<T>::showCap(); }
    MINITEN_HOSTDEVICE static inline void show(T value) { printf("%u", value); }
};

template <>
struct Show<short>
{
    using T = short;
    MINITEN_HOSTDEVICE static inline void show()        { _ShowInt<T>::show(); }
    MINITEN_HOSTDEVICE static inline void showCap()     { _ShowInt<T>::showCap(); }
    MINITEN_HOSTDEVICE static inline void show(T value) { printf("%d", value); }
};

template <>
struct Show<unsigned int>
{
    using T = unsigned;
    MINITEN_HOSTDEVICE static inline void show()        { _ShowInt<T>::show(); }
    MINITEN_HOSTDEVICE static inline void showCap()     { _ShowInt<T>::showCap(); }
    MINITEN_HOSTDEVICE static inline void show(T value) { printf("%u", value); }
};

template <>
struct Show<int>
{
    using T = int;
    MINITEN_HOSTDEVICE static inline void show()        { _ShowInt<T>::show(); }
    MINITEN_HOSTDEVICE static inline void showCap()     { _ShowInt<T>::showCap(); }
    MINITEN_HOSTDEVICE static inline void show(T value) { printf("%d", value); }
};

template <>
struct Show<unsigned long>
{
    using T = unsigned long;
    MINITEN_HOSTDEVICE static inline void show()        { _ShowInt<T>::show(); }
    MINITEN_HOSTDEVICE static inline void showCap()     { _ShowInt<T>::showCap(); }
    MINITEN_HOSTDEVICE static inline void show(T value) { printf("%lu", value); }
};

template <>
struct Show<long>
{
    using T = long;
    MINITEN_HOSTDEVICE static inline void show()        { _ShowInt<T>::show(); }
    MINITEN_HOSTDEVICE static inline void showCap()     { _ShowInt<T>::showCap(); }
    MINITEN_HOSTDEVICE static inline void show(T value) { printf("%ld", value); }
};

template <>
struct Show<unsigned long long>
{
    using T = unsigned long long;
    MINITEN_HOSTDEVICE static inline void show()        { _ShowInt<T>::show(); }
    MINITEN_HOSTDEVICE static inline void showCap()     { _ShowInt<T>::showCap(); }
    MINITEN_HOSTDEVICE static inline void show(T value) { printf("%llu", value); }
};

template <>
struct Show<long long>
{
    using T = long long;
    MINITEN_HOSTDEVICE static inline void show()        { _ShowInt<T>::show(); }
    MINITEN_HOSTDEVICE static inline void showCap()     { _ShowInt<T>::showCap(); }
    MINITEN_HOSTDEVICE static inline void show(T value) { printf("%lld", value); }
};

////////////////////////////////////////////////////////////////////////

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

} // namespace miniten

#endif // MINITEN_SHOW_TYPES_H
