#ifndef MINITEN__META__TRAITS_ADD_REF
#define MINITEN__META__TRAITS_ADD_REF
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/is_referenceable.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class T, class CanAddRef = IsReferenceable<T>> struct _AddRef           { using Type = T&; };
template <class T>                                       struct _AddRef<T, False> { using Type = T;  };

/**
 * @brief Add reference to a type if possible
 *
 * AddRef<T> = (IsReferenceable<T> ? T& : T)
 */
template <class T> using AddRef = typename _AddRef<T>::Type;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_ADD_REF
