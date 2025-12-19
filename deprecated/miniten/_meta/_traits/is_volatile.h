#ifndef MINITEN__META__TRAITS_IS_VOLATILE
#define MINITEN__META__TRAITS_IS_VOLATILE
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/is_same.h>
#include <miniten/_meta/_traits/add_volatile.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class T> using IsVolatile   = IsSame<T, AddVolatile<T>>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_IS_VOLATILE
