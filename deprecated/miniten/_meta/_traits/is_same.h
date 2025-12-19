#ifndef MINITEN__META__TRAITS_IS_SAME
#define MINITEN__META__TRAITS_IS_SAME
#include <miniten/_core/defines.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class, class> struct _IsSame       { using Type = False; };
template <class A>      struct _IsSame<A, A> { using Type = True;  };

/**
 * @brief Check that two types are identical
 * (!! and have the same const/ref !!)
 */
template <class A, class B> using IsSame = typename _IsSame<A,B>::Type;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_IS_SAME
