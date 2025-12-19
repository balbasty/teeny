#ifndef MINITEN__META__TRAITS_ADD_CV
#define MINITEN__META__TRAITS_ADD_CV
#include <miniten/_core/defines.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class T> using AddCV = const volatile T;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_ADD_CV
