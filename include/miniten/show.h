#ifndef MINITEN_SHOW_H
#define MINITEN_SHOW_H
#include <cstdio>
#include "defines.h"
#include "meta/base.h"

namespace miniten {

/// Traits that find static methods "Show()" and "Show(value)" in a class

typedef void (*_ShowTypeFn)();

template <class T>
struct _ShowValueFn {
    typedef void (*Type)(const T &);
};

template <class T, class = void>
struct HasShowType {
    using Type = meta::False;
};

template <class T>
struct HasShowType<T, typename meta::VoidFromValue<_ShowTypeFn, T::Show>::Type > {
    using Type = meta::True;
};

template <class T, class = void>
struct HasShowValue {
    using Type = meta::False;
};

template <class T>
struct HasShowValue<T, typename meta::VoidFromValue<typename _ShowValueFn<T>::Type, T::Show>::Type > {
    using Type = meta::True;
};

/// A structure that encapsulates static visualization tools.
///
/// This is usualy not called directly. Instead, each static method is
/// redeclared as a function than calls the method internally.
template <class Type,
          class _HasShowType  = typename HasShowType<Type>::Type,
          class _HasShowValue = typename HasShowValue<Type>::Type>
struct Show {};


template <class Type>
struct Show<Type, meta::False, meta::False> {

    /// Print the representation of the `Type`
    MINITEN_HOSTDEVICE static inline
    void show() {
        return Type::Show();
    }

    /// Print the representation of the `value`
    MINITEN_HOSTDEVICE static inline
    void show(const Type & value) {
        return Type::Show(value);
    }
};

template <class Type>
struct Show<Type, meta::True, meta::True> {

    /// Print the representation of the `Type`
    MINITEN_HOSTDEVICE static inline
    void show() {
        return Type::Show();
    }

    /// Print the representation of the `value`
    MINITEN_HOSTDEVICE static inline
    void show(const Type & value) {
        return Type::Show(value);
    }
};

template <class Type>
struct Show<Type, meta::True, meta::False> {

    /// Print the representation of the `Type`
    MINITEN_HOSTDEVICE static inline
    void show() {
        return Type::Show();
    }

    /// Print the representation of the `value`
    MINITEN_HOSTDEVICE static inline
    void show(const Type & value) {
        printf("{InvisibleValue}");
    }
};

template <class Type>
struct Show<Type, meta::False, meta::True> {

    /// Print the representation of the `Type`
    MINITEN_HOSTDEVICE static inline
    void show() {
        printf("{InvisibleType}");
    }

    /// Print the representation of the `value`
    MINITEN_HOSTDEVICE static inline
    void show(const Type & value) {
        return Type::Show(value);
    }
};


/// Specialization for string
template <>
struct Show<const char *> {
    MINITEN_HOSTDEVICE static inline
    void show() {
        printf("String");
    }

    MINITEN_HOSTDEVICE static inline
    void show(const char * const value) {
        printf("%s", value);
    }
};

/// Print a new line
void newline()
{
    printf("\n");
}

/// Print the representation of a type
template <typename T>
void show()
{
    Show<T>::show();
}

// /// Print the representation of an instance
// template <typename T>
// void show(const T & value)
// {
//     Show<T>::show(value);
// }

/// Print the representation of an instance
template <typename T>
void show(const T value)
{
    Show<T>::show(value);
}

// /// Specific overload for string literals (seems to be needed)
// template <>
// void show<const char *>(const char * value)
// {
//     Show<const char *>::show(value);
// }

} // namespace miniten

#endif // MINITEN_SHOW_H
