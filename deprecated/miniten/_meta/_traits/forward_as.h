#ifndef MINITEN__META__TRAITS_FORWARD_AS
#define MINITEN__META__TRAITS_FORWARD_AS
#include <miniten/_core/defines.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

template <class A, class B> struct _ForwardAs               { using Type = B&&; };
template <class A, class B> struct _ForwardAs<A&, B>        { using Type = B&; };
template <class A, class B> struct _ForwardAs<A&&, B>       { using Type = B&&; };
template <class A, class B> struct _ForwardAs<A const&, B>  { using Type = B const&; };
template <class A, class B> struct _ForwardAs<A const&&, B> { using Type = B const&&; };

/**
 * @brief Propagate "reference" annotation from one type to another.
 *
 * If left type is not a reference, propagate a right value reference (&&)
 */
template <class A, class B> using ForwardAs = typename _ForwardAs<A, B>::Type;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__TRAITS_FORWARD_AS
