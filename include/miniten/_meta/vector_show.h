#ifndef MINITEN_META_VECTOR_SHOW_H
#define MINITEN_META_VECTOR_SHOW_H
#include "../_core/defines.h"
#include "../show.h"
#include "vector.h"

namespace miniten {

template <typename T, T X0, T... X>
struct Show< meta::Vector<T, X0, X...> > {
    using Type = meta::Vector<T, X0, X...>;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("Vector<");
        miniten::show<T>();  miniten::show(", ");
        showValues();        miniten::show(">");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues() {
        miniten::show(X0); miniten::show(", ");
        Show<meta::Vector<T, X...>>::showValues();
    }

    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show();
    }
};


template <typename T, T X>
struct Show< meta::Vector<T, X> > {
    using Type = meta::Vector<T, X>;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("Vector<");
        miniten::show<T>();  miniten::show(", ");
        miniten::show(X);    miniten::show(">");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues() {
        miniten::show(X);
    }

    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show();
    }
};

template <typename T>
struct Show< meta::Vector<T> > {
    using Type = meta::Vector<T>;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("Vector<");
        miniten::show<T>();
        miniten::show(">");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues() {}


    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show();
    }
};

#define DEFINE_SHOW_VECINT(T, NAME) \
    template <T X0, T... X> \
    struct Show< meta::Vector<T, X0, X...> > { \
        using Type = meta::Vector<T, X0, X...>; \
        MINITEN_HOSTDEVICE static inline \
        void show() { \
            miniten::show(NAME); miniten::show("<"); \
            showValues();        miniten::show(">"); \
        } \
        MINITEN_HOSTDEVICE static inline \
        void showValues() { \
            miniten::show(X0); miniten::show(", "); \
            Show<meta::Vector<T, X...>>::showValues(); \
        } \
        MINITEN_HOSTDEVICE static inline \
        void show(const Type & value) \
        { \
            show(); \
        } \
    }; \
    template <T X> \
    struct Show< meta::Vector<T, X> > { \
        using Type = meta::Vector<T, X>; \
        MINITEN_HOSTDEVICE static inline \
        void show() { \
            miniten::show(NAME); miniten::show("<"); \
            miniten::show(X);    miniten::show(">"); \
        } \
        MINITEN_HOSTDEVICE static inline \
        void showValues() { \
            miniten::show(X); \
        } \
        MINITEN_HOSTDEVICE static inline \
        void show(const Type & value) \
        { \
            show(); \
        } \
    }; \
    template <> \
    struct Show< meta::Vector<T> > { \
        using Type = meta::Vector<T>; \
        MINITEN_HOSTDEVICE static inline \
        void show() { \
            miniten::show(NAME); miniten::show("<>"); \
        } \
        MINITEN_HOSTDEVICE static inline \
        void showValues() {} \
        MINITEN_HOSTDEVICE static inline \
        void show(const Type & value) \
        { \
            show(); \
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
struct Show< meta::True > {
    using Type = meta::True;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("True");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues() {}

    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show();
    }
};

template <>
struct Show< meta::False > {
    using Type = meta::False;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("False");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues() {}

    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show();
    }
};

template <>
struct Show< meta::None > {
    using Type = meta::None;

    MINITEN_HOSTDEVICE static inline
    void show() {
        miniten::show("None");
    }

    MINITEN_HOSTDEVICE static inline
    void showValues() {}

    MINITEN_HOSTDEVICE static inline
    void show(const Type & value)
    {
        show();
    }
};

} // namespace miniten

#endif // MINITEN_META_VECTOR_SHOW_H
