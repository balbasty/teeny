#ifndef MINITEN__META__TRAITS_IS_FUNCTION
#define MINITEN__META__TRAITS_IS_FUNCTION
#include <miniten/_core/defines.h>
#include <miniten/_meta/_math/decl.h>
#include <miniten/_meta/_traits/is_ref.h>
#include <miniten/_meta/_traits/is_const.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class A> using IsFunction = Or<IsRef<A>, IsConst<const A>>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_IS_FUNCTION
