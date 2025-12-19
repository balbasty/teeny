#ifndef MINITEN__META__TRAITS_IS_CONST
#define MINITEN__META__TRAITS_IS_CONST
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/is_same.h>
#include <miniten/_meta/_traits/add_const.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class T> using IsConst = IsSame<T, AddConst<T>>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_IS_CONST
