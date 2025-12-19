#ifndef MINITEN__META__TRAITS_CONDITIONAL
#define MINITEN__META__TRAITS_CONDITIONAL
#include <miniten/_core/defines.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <bool C, class A, class B> struct _Conditional              { using Type = A; };
template <        class A, class B> struct _Conditional<false, A, B> { using Type = B; };

/**
 * @brief Compile-time conditional type selection
 *
 * Conditional<C, A, B> = (C ? A : B)
 *
 * @tparam C [bool]  Condition
 * @tparam A [class] Type if C is true
 * @tparam B [class] Type if C is false
 */
template <bool C, class A, class B> using Conditional = typename _Conditional<C, A, B>::Type;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_CONDITIONAL
