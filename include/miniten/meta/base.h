/***********************************************************************
 * This file implements various metaprogramming utilities
 *
 * SameType<A, B>
 * CondType<Cond, A, B>
 * None
 * IsNone<T>
 * Error
 * IsError<T>
 ***********************************************************************/
#ifndef MINITEN_META_META_H
#define MINITEN_META_META_H

namespace miniten {
namespace  meta {

template <class T>      struct TypeIdentity  { using Type = T; };

// Something that always returns the void type, whatver the template type
template <class T>      struct VoidFromType  { using Type = void; };
template <class T, T V> struct VoidFromValue { using Type = void; };

// Propagate "reference" annotatiob from one type to another.
// If left type is not a reference, propagate a right value reference (&&)
template <class A, class B> struct ForwardAs                { using Type = B&&; };
template <class A, class B> struct ForwardAs<A&, B>         { using Type = B&; };
template <class A, class B> struct ForwardAs<A&&, B>        { using Type = B&&; };
template <class A, class B> struct ForwardAs<A const&, B>   { using Type = B const&; };
template <class A, class B> struct ForwardAs<A const&&, B>  { using Type = B const&&; };

// type Remove annotations

template<class T> struct RemoveRef      { typedef T type; };
template<class T> struct RemoveRef<T&>  { typedef T type; };
template<class T> struct RemoveRef<T&&> { typedef T type; };

template<class T> struct RemoveCV                       { typedef T type; };
template<class T> struct RemoveCV<const T>              { typedef T type; };
template<class T> struct RemoveCV<volatile T>           { typedef T type; };
template<class T> struct RemoveCV<volatile const T>     { typedef T type; };
template<class T> struct RemoveConst                    { typedef T type; };
template<class T> struct RemoveConst<const T>           { typedef T type; };
template<class T> struct RemoveVolatile                 { typedef T type; };
template<class T> struct RemoveVolatile<volatile T>     { typedef T type; };

// Add type annotations
template <class T> struct AddConst      { using Type = const T; };
template <class T> struct AddRValueRef  { using Type = T&&; };
template <class T> struct AddRef        { using Type = T&; };
template <class T> struct AddConstRef   { using Type = const T &; };
template <class T> struct AddPtr        { using Type = typename RemoveRef<T>::Type*; };
template <class T> struct AddConstPtr   { using Type = const typename RemoveRef<T>::Type *; };

template<typename T>
typename AddRValueRef<T>::Type declval() noexcept
{
    static_assert(false, "declval not allowed in an evaluated context");
}

// Check that two types are identical (!! and have the same const/ref !!)
template <class A, class B>
struct SameType {
    static constexpr bool Value = false;
    constexpr operator bool () const noexcept { return false; }

};

template <class A>
struct SameType<A, A> {
    static constexpr bool Value = true;
    constexpr operator bool () const noexcept { return false; }
};

// Conditional type
template <bool Cond, class A, class B>
struct CondType {};

template <class A, class B>
struct CondType<true, A, B> {
    using Type = A;
};

template <class A, class B>
struct CondType<false, A, B> {
    using Type = B;
};

// Most types:     Remove constness and reference annotation
// Function types: Add pointer ot make it a function pointer type
// (std::decay further transforms array<T> into T* but we don't care about that)
template<class T>
struct Decay
{
private:
    using U = typename RemoveRef<T>::Type;
public:
    using Type = typename CondType<
            std::is_function<U>::value,
            typename AddPtr<U>::Type,
            typename RemoveCV<U>::Type
        >::Type
};

// Check that values from two types are comparable
template <class Left, class Right, class = void>
struct TypesHaveEq {
    static constexpr bool Value = false;
};

template <class Left, class Right>
struct TypesHaveEq<
    Left, Right,
    typename VoidFromType<decltype(declval<Left>() == declval<Right>())>::Type
> {
    static constexpr bool Value = true;
};

template <class Left, class Right, class = void>
struct TypesHaveLess {
    static constexpr bool Value = false;
};

template <class Left, class Right>
struct TypesHaveLess<
    Left, Right,
    typename VoidFromType<decltype(declval<Left>() < declval<Right>())>::Type
> {
    static constexpr bool Value = true;
};

template <class Left, class Right>
struct TypesComparable {
    static constexpr bool Value = TypesHaveEq<Left, Right>::Value &&
                                  TypesHaveLess<Left, Right>::Value;
};

// SFINAE-way of imposing that a type is not another type
template <class A, class B>
    using other_than = std::enable_if_t<
        !std::is_same_v<std::decay_t<A>, std::decay_t<B>>>;

/// ---------------------------------------------------------------- ///
///     Special values                                               ///
/// ---------------------------------------------------------------- ///

struct True {};

template <typename T>
using IsTrue = SameType<T, True>;

struct False {};

template <typename T>
using IsFalse = SameType<T, False>;

/// The metatemplating equivalent of python's `None`
struct None {};

template <typename T>
using IsNone = SameType<T, None>;

/// Metatemplating error
struct Error {};

template <typename T>
using IsError = SameType<T, Error>;


} // namespace meta
} // namespace miniten

#endif // MINITEN_META_META_H
