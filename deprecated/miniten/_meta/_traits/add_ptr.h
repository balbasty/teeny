#ifndef MINITEN__META__TRAITS_ADD_PTR
#define MINITEN__META__TRAITS_ADD_PTR
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/is_referenceable.h>
#include <miniten/_meta/_traits/is_void.h>
#include <miniten/_meta/_traits/remove_ref.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class T, bool CanAddPtr = IsReferenceable<T>::value || IsVoid<T>::value>
struct _AddPtr           { using Type = RemoveRef<T>*; };
template <class T>
struct _AddPtr<T, False> { using Type = T;  };

/**
 * @brief Add pointer to a type if possible
 *
 * AddPtr<T> = (IsReferenceable<T> || IsVoid<T> ? RemoveRef<T>* : T)
 */
template <class T> using AddPtr = typename _AddPtr<T>::Type;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_ADD_PTR
