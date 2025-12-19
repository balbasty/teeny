#ifndef MINITEN__META__TRAITS_IS_REFERENCEABLE
#define MINITEN__META__TRAITS_IS_REFERENCEABLE
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/is_different.h>
#include <miniten/_meta/_vector/decl.h>  // False

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

struct _IsReferenceable
{
    template <class T> static T&    test(int);
    template <class T> static False test(...);
};

template <class T>
using IsReferenceable = IsDifferent<False, decltype(_IsReferenceable::test<T>(0))>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_IS_REFERENCEABLE
