#ifndef MINITEN__META__TRAITS_IF_ELSE
#define MINITEN__META__TRAITS_IF_ELSE
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/conditional.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class... T>                   struct _IfElse           {};
template <class T>                      struct _IfElse<T>        { using Type = T; };
template <class C, class A, class... T> struct _IfElse<C,A,T...> { using Type = Conditional<C::Value, A, IfElse<T...>>; };

/**
 * @brief Compile-time conditional type selection
 *
 * IfElse<C, A, E> = (C ? A : E)
 * IfElse<C1, A1, C2, A2, ..., CN, AN, E> = (C1 ? A1 : C2 ? A2 : ... CN ? AN : E)
 *
 * @tparam Ci [class] i-th condition. Must have a static boolean member `Value`.
 * @tparam Ai [class] Type if the first (i-1) conditions are false and the i-th condition is true
 * @tparam E  [class] Type if all conditions are false
 */
template <class... T>  using IfElse = typename _IfElse<T...>::Type;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_IF_ELSE
