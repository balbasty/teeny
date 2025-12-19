#ifndef MINITEN__META__TRAITS_IS_RVALUE_REF
#define MINITEN__META__TRAITS_IS_RVALUE_REF
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/is_same.h>
#include <miniten/_meta/_traits/add_rvalue_ref.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class T> using IsRValueRef = IsSame<T, AddRValueRef<T>>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_IS_RVALUE_REF
