#ifndef MINITEN__META__TRAITS_ADD_VOLATILE
#define MINITEN__META__TRAITS_ADD_VOLATILE
#include <miniten/_core/defines.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class T> using AddVolatile  = volatile T;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_ADD_VOLATILE
