/***********************************************************************
 * This file implements various metaprogramming utilities
 *
 * Identity<T>              = T
 * Void<T...>               = void
 *
 * ForwardAs<A, B>          = Apply A's ref/const/volatile to B
 *
 * RemoveRef<T>             = T&               -> T
 * RemoveCV<T>              = const volatile T -> T
 * RemoveConst<T>           = const T          -> T
 * RemoveVolatile<T>        = volatile T       -> T
 *
 * AddConst<T>              = const T;
 * AddRValueRef<T>          = T&&;
 * AddRef<T>                = T&;
 * AddConstRef<T>           = const T &;
 * AddPtr<T>                = RemoveRef<T>*;
 * AddConstPtr<T>           = const RemoveRef<T> *;
 *
 * IsSame<A,B>              = check same type (True | False)
 * ConditionalB<C,A,B>      = C ? A : B
 * Conditional<C,A,B>       = C ? A : B
 * SwitchCase<C,A,...>      = C ? A : SwitchCase<...>
 * IsFunction<A>            = check is function (True | False)
 * Decay<T>                 = is_function ? make ptr : remove cv
 * HaveEq<A,B>              = defined(A==B) ? True : False
 * HaveLess<A,B>            = defined(A<B)  ? True : False
 * IsComparable<A,B>        = HaveEq && HaveLess
 * EnableIf<C,T>            = C ? T : SFINAE
 * EnableIfB<C,T>           = C ? T : SFINAE
 * OtherThan<A,B>           =
 ***********************************************************************/
#ifndef MINITEN_META_META_H
#define MINITEN_META_META_H
#include <type_traits>
#include "vector.h"

namespace miniten {
namespace  meta {

template <class T>     using Identity = T;
template <class... T>  using Void     = void;

// Propagate "reference" annotation from one type to another.
// If left type is not a reference, propagate a right value reference (&&)
template <class A, class B> struct _ForwardAs                { using Type = B&&; };
template <class A, class B> struct _ForwardAs<A&, B>         { using Type = B&; };
template <class A, class B> struct _ForwardAs<A&&, B>        { using Type = B&&; };
template <class A, class B> struct _ForwardAs<A const&, B>   { using Type = B const&; };
template <class A, class B> struct _ForwardAs<A const&&, B>  { using Type = B const&&; };
template <class A, class B> using ForwardAs = typename _ForwardAs<A, B>::Type;

// Remove type annotations

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

// Add type annotations
template <class T> using AddConst     = const T;
template <class T> using AddRValueRef = T&&;
template <class T> using AddRef       = T&;
template <class T> using AddConstRef  = const T &;
template <class T> using AddPtr       = RemoveRef<T>*;
template <class T> using AddConstPtr  = const RemoveRef<T> *;

// template<typename T>
// AddRValueRef<T> declval() noexcept
// {
//     static_assert(false, "declval not allowed in an evaluated context");
// }

// Check that two types are identical (!! and have the same const/ref !!)
template <class, class>     struct _IsSame       { using Type = False; };
template <class A>          struct _IsSame<A, A> { using Type = True;  };
template <class A, class B> using IsSame = typename _IsSame<A,B>::Type;

// Conditional type
template <bool C, class A, class B>  struct _ConditionalB              { using Type = A; };
template <class A, class B>          struct _ConditionalB<false, A, B> { using Type = B; };
template <bool C, class A, class B>  using ConditionalB = typename _ConditionalB<C, A, B>::Type;
template <class C, class A, class B> using Conditional  = typename _ConditionalB<C::Value, A, B>::Type;

template <class... T>                   struct _SwitchCase {};
template <class... T>                   using   SwitchCase = typename _SwitchCase<T...>::Type;
template <class T>                      struct _SwitchCase<T> { using Type = T; };
template <class C, class A, class... T> struct _SwitchCase<C,A,T...> { using Type = Conditional<C, A, SwitchCase<T...>>; };

// Is function
template <class A> using IsFunction = ConditionalB<std::is_function<A>::value, True, False>;

// Most types:     Remove constness and reference annotation
// Function types: Add pointer ot make it a function pointer type
// (std::decay further transforms array<T> into T* but we don't care about that)
template<class T>
struct _Decay
{
private:
    using U = RemoveRef<T>;
public:
    using Type = Conditional<IsFunction<U>, AddPtr<U>, RemoveCV<U>>;
};
template<class T> using Decay = typename _Decay<T>::Type;


// Check that values from two types are comparable

// ==
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
using HaveEq = Conditional<_HaveEq<Left, Right>, True, False>;


// <=
template <class Left, class Right, class = void>
struct _HaveLess {
    static constexpr bool Value = false;
};

template <class Left, class Right>
struct _HaveLess<
    Left, Right,
    Void<decltype(std::declval<Left>() < std::declval<Right>())>
> {
    static constexpr bool Value = true;
};

template <class Left, class Right>
using HaveLess = Conditional<_HaveLess<Left, Right>, True, False>;

template <class Left, class Right>
using IsComparable = ConditionalB<
    HaveEq<Left, Right>::Value &&
    HaveLess<Left, Right>::Value,
    True,
    False
>;

template <bool C, class T = void>
using EnableIfB = typename std::enable_if<C, T>::type;

template <class C, class T = void>
using EnableIf = EnableIfB<C::Value, T>;

// SFINAE-way of imposing that a type is not another type
template <class A, class B>
struct _OtherThan {
    static constexpr bool Value = !IsSame<Decay<A>,  Decay<B>>::Value;
};
template <class A, class B>
using OtherThan = ConditionalB<_OtherThan<A,B>::Value, True, False>;

template <class T> using IsTrue  = IsSame<T, True>;
template <class T> using IsFalse = IsSame<T, False>;
template <class T> using IsError = IsSame<T, Error>;
template <class T> using IsNone  = IsSame<T, None>;

} // namespace meta
} // namespace miniten

#endif // MINITEN_META_META_H
