#ifndef MINITEN__META__TRAITS_ADD_CONST_REF
#define MINITEN__META__TRAITS_ADD_CONST_REF
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/add_ref.h>
#include <miniten/_meta/_traits/add_const.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class T> using AddConstRef = AddConst<AddRef<T>>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_ADD_CONST_REF
