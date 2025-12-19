#ifndef MINITEN__META__TRAITS_IS_VOID
#define MINITEN__META__TRAITS_IS_VOID
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/is_same.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class T> using IsVoid = IsSame<T, void>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_IS_VOID
