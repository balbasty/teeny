/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file implements various metaprogramming utilities                  **
 **                                                                         **
 ** Identity<T>              = T                                            **
 ** Void<T...>               = void **                                      **
 **                                                                         **
 ** ForwardAs<A, B>          = Apply A's ref/const/volatile to B            **
 **                                                                         **
 ** RemoveRef<T>             = T&               -> T                        **
 ** RemoveCV<T>              = const volatile T -> T                        **
 ** RemoveConst<T>           = const T          -> T                        **
 ** RemoveVolatile<T>        = volatile T       -> T                        **
 **                                                                         **
 ** AddConst<T>              = const T                                      **
 ** AddRValueRef<T>          = T&&                                          **
 ** AddRef<T>                = T&                                           **
 ** AddConstRef<T>           = const T &                                    **
 ** AddPtr<T>                = RemoveRef<T> *                               **
 ** AddConstPtr<T>           = const RemoveRef<T> *                         **
 **                                                                         **
 ** IsSame<A,B>              = check same type (True | False)               **
 ** Conditional<C,A,B>       = C ? A : B                                    **
 ** IfElse<C,A,...>          = C ? A : IfElse<...>                          **
 ** IsFunction<A>            = check is function (True | False)             **
 ** Decay<T>                 = is_function ? make ptr : remove cv           **
 ** HaveEq<A,B>              = defined(A==B) ? True : False                 **
 ** HaveLess<A,B>            = defined(A<B)  ? True : False                 **
 ** IsComparable<A,B>        = HaveEq && HaveLess                           **
 ** EnableIf<C,T>            = C ? T : SFINAE                               **
 ** EnableIfB<C,T>           = C ? T : SFINAE                               **
 ** OtherThan<A,B>           = A is B ? Bool : SFINAE                       **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef MINITEN__META_TRAITS
#define MINITEN__META_TRAITS
#include <type_traits>
#include <miniten/_core/defines.h>
#include <miniten/_meta/_vector/decl.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)


// Check that values from two types are comparable

// ==


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
using HaveLess = IfElse<_HaveLess<Left, Right>, True, False>;

template <class Left, class Right>
using IsComparable = Conditional<
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
using OtherThan = Conditional<_OtherThan<A,B>::Value, True, False>;

template <class T> using IsTrue  = IsSame<T, True>;
template <class T> using IsFalse = IsSame<T, False>;
template <class T> using IsError = IsSame<T, Error>;
template <class T> using IsNone  = IsSame<T, None>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META_TRAITS
