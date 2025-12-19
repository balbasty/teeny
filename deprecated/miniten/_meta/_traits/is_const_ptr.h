#ifndef MINITEN__META__TRAITS_IS_CONST_PTR
#define MINITEN__META__TRAITS_IS_CONST_PTR
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/is_same.h>
#include <miniten/_meta/_traits/add_const_ptr.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class T> using IsConstPtr = IsSame<T, AddConstPtr<T>>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_IS_CONST_PTR
