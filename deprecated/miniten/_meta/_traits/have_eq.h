#ifndef MINITEN__META__TRAITS_HAVE_EQ
#define MINITEN__META__TRAITS_HAVE_EQ
#include <miniten/_core/defines.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class Left, class Right, class = void>
struct _HaveEq {
    static constexpr bool Value = false;
};

template <class Left, class Right>
struct _HaveEq<
    Left, Right,
    Void<decltype(std::declval<Left>() == std::declval<Right>())>
> {
    static constexpr bool Value = true;
};

template <class Left, class Right>
using HaveEq = IfElse<_HaveEq<Left, Right>, True, False>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_HAVE_EQ
