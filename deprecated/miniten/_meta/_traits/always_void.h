#ifndef MINITEN__META__TRAITS_ALWAYS_VOID
#define MINITEN__META__TRAITS_ALWAYS_VOID
#include <miniten/_core/defines.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class... T> using AlwaysVoid = void;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_ALWAYS_VOID
