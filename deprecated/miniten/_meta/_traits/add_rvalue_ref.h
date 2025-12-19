#ifndef MINITEN__META__TRAITS_ADD_RVALUE_REF
#define MINITEN__META__TRAITS_ADD_RVALUE_REF
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/is_referenceable.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class T, class CanAddRef = IsReferenceable<T>> struct _AddRValueRef           { using Type = T&&; };
template <class T>                                       struct _AddRValueRef<T, False> { using Type = T;  };

/**
 * @brief Add right-value reference to a type if possible
 *
 * AddRValueRef<T> = (IsReferenceable<T> ? T&& : T)
 */
template <class T> using AddRValueRef = typename _AddRValueRef<T>::Type;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_ADD_RVALUE_REF
