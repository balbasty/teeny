#ifndef MINITEN__META__TRAITS_ADD_CONST_PTR
#define MINITEN__META__TRAITS_ADD_CONST_PTR
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/add_ptr.h>
#include <miniten/_meta/_traits/add_const.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class T> using AddConstPtr = AddConst<AddPtr<T>>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_ADD_CONST_PTR
