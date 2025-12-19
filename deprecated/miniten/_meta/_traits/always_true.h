#ifndef MINITEN__META__TRAITS_ALWAYS_TRUE
#define MINITEN__META__TRAITS_ALWAYS_TRUE
#include <miniten/_core/defines.h>
#include <miniten/_meta/_vector/decl.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class... T> using AlwaysTrue = True;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_ALWAYS_TRUE
