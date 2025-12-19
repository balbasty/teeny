#ifndef MINITEN__DISP_TYPES
#define MINITEN__DISP_TYPES
#include <miniten/_core/defines.h>
#include <miniten/_core/types.h>
#include <miniten/_disp/impl.h>

NAMESPACE_BEGIN(miniten)

NAMESPACE_BEGIN(_)
using s = const char *;
NAMESPACE_END(_)

////////////////////////////////////////////////////////////////////////

template <class T, size_t L = sizeof(T), int64_t MIN = static_cast<int64_t>(TypeInfo<T>::Min)>
struct _DispInt {};

template <class T, int64_t MIN>
struct _DispInt<T, 1, MIN> {
    MINIDEF(H,D,S,I) void disp()    { miniten::disp("int8"); }
    MINIDEF(H,D,S,I) void dispCap() { miniten::disp("Int8"); }
};

template <class T>
struct _DispInt<T, 1, 0> {
    MINIDEF(H,D,S,I) void disp()    { miniten::disp("uint8"); }
    MINIDEF(H,D,S,I) void dispCap() { miniten::disp("UInt8"); }
};

template <class T, int64_t MIN>
struct _DispInt<T, 2, MIN> {
    MINIDEF(H,D,S,I) void disp()    { miniten::disp("int16"); }
    MINIDEF(H,D,S,I) void dispCap() { miniten::disp("Int16"); }
};

template <class T>
struct _DispInt<T, 2, 0> {
    MINIDEF(H,D,S,I) void disp()    { miniten::disp("uint16"); }
    MINIDEF(H,D,S,I) void dispCap() { miniten::disp("UInt16"); }
};

template <class T, int64_t MIN>
struct _DispInt<T, 4, MIN> {
    MINIDEF(H,D,S,I) void disp()    { miniten::disp("int32"); }
    MINIDEF(H,D,S,I) void dispCap() { miniten::disp("Int32"); }
};

template <class T>
struct _DispInt<T, 4, 0> {
    MINIDEF(H,D,S,I) void disp()    { miniten::disp("uint32"); }
    MINIDEF(H,D,S,I) void dispCap() { miniten::disp("UInt32"); }
};

template <class T, int64_t MIN>
struct _DispInt<T, 8, MIN> {
    MINIDEF(H,D,S,I) void disp()    { miniten::disp("int64"); }
    MINIDEF(H,D,S,I) void dispCap() { miniten::disp("Int64"); }
};

template <class T>
struct _DispInt<T, 8, 0> {
    MINIDEF(H,D,S,I) void disp()    { miniten::disp("uint64"); }
    MINIDEF(H,D,S,I) void dispCap() { miniten::disp("UInt64"); }
};

template <class T, int64_t MIN>
struct _DispInt<T, 16, MIN> {
    MINIDEF(H,D,S,I) void disp()    { miniten::disp("int128"); }
    MINIDEF(H,D,S,I) void dispCap() { miniten::disp("Int128"); }
};

template <class T>
struct _DispInt<T, 16, 0> {
    MINIDEF(H,D,S,I) void disp()    { miniten::disp("uint128"); }
    MINIDEF(H,D,S,I) void dispCap() { miniten::disp("UInt128"); }
};

////////////////////////////////////////////////////////////////////////

template <>
struct Display<bool> {
    MINIDEF(H,D,S,I,CX) _::s srepr()          { return "bool"; }
    MINIDEF(H,D,S,I)    void disp()           { miniten::disp("bool"); }
    MINIDEF(H,D,S,I)    void dispCap()        { miniten::disp("Bool"); }
    MINIDEF(H,D,S,I)    void disp(bool value) { miniten::disp(value ? "true" :"false"); }
};

////////////////////////////////////////////////////////////////////////

template <>
struct Display<uint8_t>
{
    using T = uint8_t;
    MINIDEF(H,D,S,I,CX) _::s srepr()       { return "uint8_t"; }
    MINIDEF(H,D,S,I)    void disp()        { _DispInt<T>::disp(); }
    MINIDEF(H,D,S,I)    void dispCap()     { _DispInt<T>::dispCap(); }
    MINIDEF(H,D,S,I)    void disp(T value) { printf("%u", value); }
};

template <>
struct Display<int8_t>
{
    using T = int8_t;
    MINIDEF(H,D,S,I,CX) _::s srepr()       { return "int8_t"; }
    MINIDEF(H,D,S,I)    void disp()        { _DispInt<T>::disp(); }
    MINIDEF(H,D,S,I)    void dispCap()     { _DispInt<T>::dispCap(); }
    MINIDEF(H,D,S,I)    void disp(T value) { printf("%d", value); }
};

template <>
struct Display<uint16_t>
{
    using T = uint16_t;
    MINIDEF(H,D,S,I,CX) _::s srepr()       { return "uint16_t"; }
    MINIDEF(H,D,S,I)    void disp()        { _DispInt<T>::disp(); }
    MINIDEF(H,D,S,I)    void dispCap()     { _DispInt<T>::dispCap(); }
    MINIDEF(H,D,S,I)    void disp(T value) { printf("%u", value); }
};

template <>
struct Display<int16_t>
{
    using T = int16_t;
    MINIDEF(H,D,S,I,CX) _::s srepr()       { return "int16_t"; }
    MINIDEF(H,D,S,I)    void disp()        { _DispInt<T>::disp(); }
    MINIDEF(H,D,S,I)    void dispCap()     { _DispInt<T>::dispCap(); }
    MINIDEF(H,D,S,I)    void disp(T value) { printf("%d", value); }
};

template <>
struct Display<uint32_t>
{
    using T = uint32_t;
    MINIDEF(H,D,S,I,CX) _::s srepr()       { return "uint32_t"; }
    MINIDEF(H,D,S,I)    void disp()        { _DispInt<T>::disp(); }
    MINIDEF(H,D,S,I)    void dispCap()     { _DispInt<T>::dispCap(); }
    MINIDEF(H,D,S,I)    void disp(T value) { printf("%u", value); }
};

template <>
struct Display<int32_t>
{
    using T = int32_t;
    MINIDEF(H,D,S,I,CX) _::s srepr()       { return "int32_t"; }
    MINIDEF(H,D,S,I)    void disp()        { _DispInt<T>::disp(); }
    MINIDEF(H,D,S,I)    void dispCap()     { _DispInt<T>::dispCap(); }
    MINIDEF(H,D,S,I)    void disp(T value) { printf("%d", value); }
};

template <>
struct Display<uint64_t>
{
    using T = uint64_t;
    MINIDEF(H,D,S,I,CX) _::s srepr()       { return "uint64_t"; }
    MINIDEF(H,D,S,I)    void disp()        { _DispInt<T>::disp(); }
    MINIDEF(H,D,S,I)    void dispCap()     { _DispInt<T>::dispCap(); }
    MINIDEF(H,D,S,I)    void disp(T value) { printf("%llu", value); }
};

template <>
struct Display<int64_t>
{
    using T = int64_t;
    MINIDEF(H,D,S,I,CX) _::s srepr()    { return "int64_t"; }
    MINIDEF(H,D,S,I) void disp()        { _DispInt<T>::disp(); }
    MINIDEF(H,D,S,I) void dispCap()     { _DispInt<T>::dispCap(); }
    MINIDEF(H,D,S,I) void disp(T value) { printf("%lld", value); }
};

////////////////////////////////////////////////////////////////////////

template <>
struct Display<float32_t>
{
    MINIDEF(H,D,S,I,CX) _::s srepr()               { return "float32"; }
    MINIDEF(H,D,S,I)    void disp()                { miniten::disp("float32"); }
    MINIDEF(H,D,S,I)    void dispCap()             { miniten::disp("Float32"); }
    MINIDEF(H,D,S,I)    void disp(float32_t value) { printf("%f", value); }
};

template <>
struct Display<float64_t>
{
    MINIDEF(H,D,S,I,CX) _::s srepr()               { return "float64"; }
    MINIDEF(H,D,S,I)    void disp()                { miniten::disp("float64"); }
    MINIDEF(H,D,S,I)    void dispCap()             { miniten::disp("Float64"); }
    MINIDEF(H,D,S,I)    void disp(float64_t value) { printf("%f", value); }
};

NAMESPACE_END(miniten)

#endif // MINITEN_SHOW_TYPES
