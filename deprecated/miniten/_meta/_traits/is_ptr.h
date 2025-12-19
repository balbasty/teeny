#ifndef MINITEN__META__TRAITS_IS_PTR
#define MINITEN__META__TRAITS_IS_PTR
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/is_same.h>
#include <miniten/_meta/_traits/add_ptr.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class T> using IsPtr = IsSame<T, AddPtr<T>>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_IS_PTR
