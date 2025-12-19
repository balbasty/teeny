#ifndef MINITEN__META__TRAITS_DECAY
#define MINITEN__META__TRAITS_DECAY
#include <miniten/_core/defines.h>
#include <miniten/_meta/_traits/if_else.h>
#include <miniten/_meta/_traits/is_function.h>
#include <miniten/_meta/_traits/add_ptr.h>
#include <miniten/_meta/_traits/remove_cv.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

// Most types:     Remove constness and reference annotation
// Function types: Add pointer ot make it a function pointer type
// (std::decay further transforms array<T> into T* but we don't care about that)
template<class T>
struct _Decay
{
private:
    using U = RemoveRef<T>;
public:
    using Type = IfElse<IsFunction<U>, AddPtr<U>, RemoveCV<U>>;
};
template<class T> using Decay = typename _Decay<T>::Type;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_DECAY
