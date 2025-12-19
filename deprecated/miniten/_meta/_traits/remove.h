#ifndef MINITEN__META__TRAITS_REMOVE
#define MINITEN__META__TRAITS_REMOVE
#include <miniten/_core/defines.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template<class T> struct _RemoveRef                      { using Type = T; };
template<class T> struct _RemoveRef<T&>                  { using Type = T; };
template<class T> struct _RemoveRef<T&&>                 { using Type = T; };
template<class T> struct _RemovePtr                      { using Type = T; };
template<class T> struct _RemovePtr<T*>                  { using Type = T; };
template<class T> struct _RemoveCV                       { using Type = T; };
template<class T> struct _RemoveCV<const T>              { using Type = T; };
template<class T> struct _RemoveCV<volatile T>           { using Type = T; };
template<class T> struct _RemoveCV<volatile const T>     { using Type = T; };
template<class T> struct _RemoveConst                    { using Type = T; };
template<class T> struct _RemoveConst<const T>           { using Type = T; };
template<class T> struct _RemoveVolatile                 { using Type = T; };
template<class T> struct _RemoveVolatile<volatile T>     { using Type = T; };

template<class T> using RemoveRef      = typename _RemoveRef<T>::Type;
template<class T> using RemovePtr      = typename _RemovePtr<T>::Type;
template<class T> using RemoveCV       = typename _RemoveCV<T>::Type;
template<class T> using RemoveConst    = typename _RemoveConst<T>::Type;
template<class T> using RemoveVolatile = typename _RemoveVolatile<T>::Type;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_REMOVE
