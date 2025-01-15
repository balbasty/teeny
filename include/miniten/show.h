#ifndef MINITEN_SHOW_H
#define MINITEN_SHOW_H
#include <cstdio>
#include "_core/defines.h"
#include "_meta/traits.h"

namespace miniten {

/// A structure that encapsulates static visualization tools. is
/// redeclared as a function than calls the method internally.
template <class Type>
struct Show {
    MINITEN_HOSTDEVICE static inline
    void show() {}

    MINITEN_HOSTDEVICE static inline
    void show(const Type & value) {}
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

#include "show_types.h"

#endif // MINITEN_SHOW_H
