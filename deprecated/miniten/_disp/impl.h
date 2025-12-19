#ifndef MINITEN__DISP_IMPL
#define MINITEN__DISP_IMPL
#include <cstdio>
#include <miniten/_core/defines.h>
#include <miniten/_meta/traits.h>

NAMESPACE_BEGIN(miniten)

/// A structure that encapsulates static visualization tools. is
/// redeclared as a function than calls the method internally.
template <class Type>
struct Display {

    MINIDEF(H,D,S,I,CX) const char * srepr() {
        return "<THIS TYPE CANNOT BE REPRESENTED>";
    }

    MINIDEF(H,D,S,I,CX) const char * srepr(const Type & value)
    {
        return "<THIS VALUE CANNOT BE REPRESENTED>";
    }

    MINIDEF(H,D,S,I) void disp()
    {
        printf("<THIS VALUE CANNOT BE DISPLAYED>");
    }

    MINIDEF(H,D,S,I) void disp(const Type & value)
    {
        printf("<THIS VALUE CANNOT BE DISPLAYED>");
    }
};

/// Print a new line
MINIDEF(H,D,I) void newline()
{
    printf("\n");
}

/// Print the representation of a type
template <typename T>
MINIDEF(H,D,I) void disp()
{
    Display<T>::disp();
}

// /// Print the representation of an instance
// template <typename T>
// void disp(const T & value)
// {
//     Display<T>::disp(value);
// }

/// Print the representation of an instance
template <typename T>
MINIDEF(H,D,I) void disp(const T value)
{
    Display<T>::disp(value);
}

// /// Specific overload for string literals (seems to be needed)
// template <>
// void disp<const char *>(const char * value)
// {
//     Display<const char *>::disp(value);
// }

/// Return the representation of a type
template <typename T>
MINIDEF(H,D,I,CX) const char * srepr()
{
    return Display<T>::srepr();
}

/// Return the representation of a type
template <typename T>
MINIDEF(H,D,I,CX) const char * srepr(const T value)
{
    return Display<T>::srepr(value);
}

/// Specialization for string
template <>
struct Display<const char *> {

    MINIDEF(H,D,S,I,CX)
    const char * srepr() {
        return "String";
    }

    MINIDEF(H,D,S,I,CX) const char * srepr(const char * const value)
    {
        return "String(<value>)";
    }

    MINIDEF(H,D,S,I)
    void disp() {
        printf("String");
    }

    MINIDEF(H,D,S,I)
    void disp(const char * const value) {
        printf("%s", value);
    }
};

template <class T>
struct Display<T&> {
    using REF = T&;
    MINIDEF(H,D,S,I) void disp()                { Display<T>::disp(); miniten::disp("&"); }
    MINIDEF(H,D,S,I) void disp(REF & value)     { disp(reinterpret_cast<uintptr_t>(&value)); }
};

template <class T>
struct Display<T*> {
    MINIDEF(H,D,S,I) void disp()                { Display<T>::disp(); miniten::disp(" *"); }
    MINIDEF(H,D,S,I) void disp(T * value)       { miniten::disp(reinterpret_cast<uintptr_t>(value)); }
};

template <class T>
struct Display<const T> {
    MINIDEF(H,D,S,I) void disp()                {  miniten::disp("const "); Display<T>::disp(); }
    MINIDEF(H,D,S,I) void disp(const T value)   {  Display<T>::disp(value); }
};

template <class T>
struct Display<const T*> {
    MINIDEF(H,D,S,I) void disp()                { miniten::disp("const "); Display<T>::disp(); miniten::disp(" *"); }
    MINIDEF(H,D,S,I) void disp(const T * value) { miniten::disp(reinterpret_cast<uintptr_t>(value)); }
};

NAMESPACE_END(miniten)

#endif // MINITEN__DISP_IMPL
