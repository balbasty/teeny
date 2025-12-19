#ifndef MINITEN__META__TRAITS_IS_DIFFERENT
#define MINITEN__META__TRAITS_IS_DIFFERENT
#include <miniten/_core/defines.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class, class> struct _IsDifferent       { using Type = True; };
template <class A>      struct _IsDifferent<A, A> { using Type = False;  };

/**
 * @brief Check that two types are different
 */
template <class A, class B>
using IsDifferent = typename _IsDifferent<A,B>::Type;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_IS_DIFFERENT
