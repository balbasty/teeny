#ifndef MINITEN__META__VECTOR_DISP
#define MINITEN__META__VECTOR_DISP
#include <miniten/_core/defines.h>
#include <miniten/disp.h>
#include <miniten/_meta/_vector/decl.h>

NAMESPACE_BEGIN(miniten)

template <typename T, T X0, T... X>
struct Display< meta::Vector<T, X0, X...> > {
    using Type = meta::Vector<T, X0, X...>;

    MINIDEF(H,D,S,I,CX)
    const char * srepr() {
        return "Pack<...>";
    }

    MINIDEF(H,D,S,I,CX) const char * srepr(const Type & value)
    {
        return srepr();
    }

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("Vector<");
        miniten::disp<T>();  miniten::disp(", ");
        showValues();        miniten::disp(">");
    }

    MINIDEF(H,D,S,I)
    void showValues() {
        miniten::disp(X0); miniten::disp(", ");
        Display<meta::Vector<T, X...>>::showValues();
    }

    MINIDEF(H,D,S,I)
    void disp(const Type & value)
    {
        disp();
    }
};


template <typename T, T X>
struct Display< meta::Vector<T, X> > {
    using Type = meta::Vector<T, X>;

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("Vector<");
        miniten::disp<T>();  miniten::disp(", ");
        miniten::disp(X);    miniten::disp(">");
    }

    MINIDEF(H,D,S,I)
    void showValues() {
        miniten::disp(X);
    }

    MINIDEF(H,D,S,I)
    void disp(const Type & value)
    {
        disp();
    }
};

template <typename T>
struct Display< meta::Vector<T> > {
    using Type = meta::Vector<T>;

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("Vector<");
        miniten::disp<T>();
        miniten::disp(">");
    }

    MINIDEF(H,D,S,I)
    void showValues() {}


    MINIDEF(H,D,S,I)
    void disp(const Type & value)
    {
        disp();
    }
};

#define DEFINE_SHOW_VECINT(T, NAME) \
    template <T X0, T... X> \
    struct Display< meta::Vector<T, X0, X...> > { \
        using Type = meta::Vector<T, X0, X...>; \
        MINIDEF(H,D,S,I) \
        void disp() { \
            miniten::disp(NAME); miniten::disp("<"); \
            showValues();        miniten::disp(">"); \
        } \
        MINIDEF(H,D,S,I) \
        void showValues() { \
            miniten::disp(X0); miniten::disp(", "); \
            Display<meta::Vector<T, X...>>::showValues(); \
        } \
        MINIDEF(H,D,S,I) \
        void disp(const Type & value) \
        { \
            disp(); \
        } \
    }; \
    template <T X> \
    struct Display< meta::Vector<T, X> > { \
        using Type = meta::Vector<T, X>; \
        MINIDEF(H,D,S,I) \
        void disp() { \
            miniten::disp(NAME); miniten::disp("<"); \
            miniten::disp(X);    miniten::disp(">"); \
        } \
        MINIDEF(H,D,S,I) \
        void showValues() { \
            miniten::disp(X); \
        } \
        MINIDEF(H,D,S,I) \
        void disp(const Type & value) \
        { \
            disp(); \
        } \
    }; \
    template <> \
    struct Display< meta::Vector<T> > { \
        using Type = meta::Vector<T>; \
        MINIDEF(H,D,S,I) \
        void disp() { \
            miniten::disp(NAME); miniten::disp("<>"); \
        } \
        MINIDEF(H,D,S,I) \
        void showValues() {} \
        MINIDEF(H,D,S,I) \
        void disp(const Type & value) \
        { \
            disp(); \
        } \
    }; \


DEFINE_SHOW_VECINT(bool,      "Bool")
DEFINE_SHOW_VECINT(uint8_t,   "UInt8")
DEFINE_SHOW_VECINT(uint16_t,  "UInt16")
DEFINE_SHOW_VECINT(uint32_t,  "UInt32")
DEFINE_SHOW_VECINT(uint64_t,  "UInt64")
DEFINE_SHOW_VECINT(int8_t,    "Int8")
DEFINE_SHOW_VECINT(int16_t,   "Int16")
DEFINE_SHOW_VECINT(int32_t,   "Int32")
DEFINE_SHOW_VECINT(int64_t,   "Int64")


template <>
struct Display< meta::True > {
    using Type = meta::True;

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("True");
    }

    MINIDEF(H,D,S,I)
    void showValues() {}

    MINIDEF(H,D,S,I)
    void disp(const Type & value)
    {
        disp();
    }
};

template <>
struct Display< meta::False > {
    using Type = meta::False;

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("False");
    }

    MINIDEF(H,D,S,I)
    void showValues() {}

    MINIDEF(H,D,S,I)
    void disp(const Type & value)
    {
        disp();
    }
};

template <>
struct Display< meta::None > {
    using Type = meta::None;

    MINIDEF(H,D,S,I)
    void disp() {
        miniten::disp("None");
    }

    MINIDEF(H,D,S,I)
    void showValues() {}

    MINIDEF(H,D,S,I)
    void disp(const Type & value)
    {
        disp();
    }
};

NAMESPACE_END(miniten)

#endif // MINITEN__META__VECTOR_DISP
