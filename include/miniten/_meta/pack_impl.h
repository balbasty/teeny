/***********************************************************************
 * This file implements a compile-time "pack of types"
 *
 * This type **cannot** be used to store data. For a "tuple of types"
 * that can store objects of each of these types, use `Tuple`.
 *
 * Pack<T...>
 * NPack<N, T> = Pack<T... (N times)>
 ***********************************************************************/
#ifndef MINITEN_META_PACK_IMPL_H
#define MINITEN_META_PACK_IMPL_H
#include "../_core/defines.h"
#include "../_core/types.h"
#include "../show.h"
#include "traits.h"
#include "math.h"
#include "pack.h"
#include "vector.h"
#include "packapi.h"

namespace miniten {
namespace meta {

/// ---------------------------------------------------------------- ///
///     Pack API                                                     ///
/// ---------------------------------------------------------------- ///

/// Cat

template <class RIGHT>
struct _Cat2<Pack<>, RIGHT> {
    static_assert(IsPack<RIGHT>::Value, "L/R must be packs");
    using Type = RIGHT;
};

template <class X0, class... X>
struct _Cat2<Pack<X0>, Pack<X...>> {
    using Type = Pack<X0,X...>;
};

template <class RIGHT, class X0, class... X>
struct _Cat2<Pack<X0,X...>, RIGHT> {
    static_assert(IsPack<RIGHT>::Value, "L/R must be packs");
    using Type = Cat<Pack<X0>, Cat<Pack<X...>, RIGHT>>;
};

namespace _pack {
    template <class PACK>                 struct _Reversed;
    template <class PACK>                 using   Reversed  = typename _Reversed<PACK>::Type;
    template <class PACK, typename U>     struct _AsVector;
    template <class PACK, typename U>     using   AsVector  = typename _AsVector<PACK,U>::Type;
    template <long>                       struct  ApplyFn;
    template <class PACK, class APPLY>    struct _Apply;
    template <class PACK, class APPLY>    using   Apply     = typename _Apply<PACK, APPLY>::Type;
}

template <class... X>               struct _Length<Pack<X...>>             { using Type = CountTypes<X...>; };
template <class... X>               struct _EmptyLike<Pack<X...>>          { using Type = Pack<>; };
template <class... X>               struct _IsPack<Pack<X...>>             { using Type = True; };
template <class... X>               struct _Reversed<Pack<X...>>           { using Type = _pack::Reversed<Pack<X...>>; };
template <class U, class... X>      struct _AsVector<Pack<X...>, U>        { using Type = _pack::AsVector<Pack<X...>, U>; };
template <class... X>               struct _AsTuple<Pack<X...>>            { using Type = Tuple<X...>; };
template <class... X>               struct _AsPack<Pack<X...>>             { using Type = Pack<X...>; };
template <class M, class... X>      struct _LikeFrom<Pack<X...>, M>        { using Type = AsPack<M>; };
template <class X0, class... X>     struct _GetFirst<Pack<X0, X...>>       { using Type = Pack<X0>; };
template <class X0, class... X>     struct _DelFirst<Pack<X0, X...>>       { using Type = Pack<X...>; };
template <class X0, class... X>     struct _GetFirstValue<Pack<X0, X...>>  { using Type = X0; };
template <class... X>               struct _ApplyAddConst<Pack<X...>>      { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<0>>; };
template <class... X>               struct _ApplyAddConstPtr<Pack<X...>>   { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<1>>; };
template <class... X>               struct _ApplyAddConstRef<Pack<X...>>   { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<2>>; };
template <class... X>               struct _ApplyAddPtr<Pack<X...>>        { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<3>>; };
template <class... X>               struct _ApplyAddRef<Pack<X...>>        { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<4>>; };
template <class... X>               struct _ApplyAddRValueRef<Pack<X...>>  { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<5>>; };
template <class... X>               struct _ApplyRemoveConst<Pack<X...>>   { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<6>>; };
template <class... X>               struct _ApplyRemovePtr<Pack<X...>>     { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<7>>; };
template <class... X>               struct _ApplyRemoveRef<Pack<X...>>     { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<8>>; };
template <class... X>               struct _ApplyRemoveCV<Pack<X...>>      { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<9>>; };
template <class... X>               struct _ApplyDecay<Pack<X...>>         { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<10>>; };

/// ---------------------------------------------------------------- ///
///     Pack API Implementation                                      ///
/// ---------------------------------------------------------------- ///

namespace _pack {

/// Revert /////////////////////////////////////////////////////////////

template <class PACK>
struct _Reversed {};

template <class X0, class... X>
struct _Reversed<Pack<X0, X...>> {
    using Type = Cat< _Reversed<Pack<X...>>, Pack<X0> >;
};

template <class X0>
struct _Reversed<Pack<X0>> {
    using Type = Pack<X0>;
};

template <>
struct _Reversed<Pack<>> {
    using Type = Pack<>;
};

/// AsVector ///////////////////////////////////////////////////////////

template <class PACK, typename U>
struct _AsVector {};

template <typename U, class X0, class... X>
struct _AsVector<Pack<X0, X...>, U>
{
    using Type = Cat<
        Vector<U, static_cast<U>(X0())>,
        AsVector<Pack<X...>, U>
    >;
};

template <typename U, class X0>
struct _AsVector<Pack<X0>, U>
{
    using Type = Vector<U, static_cast<U>(X0())>;
};

template <typename U>
struct _AsVector<Pack<>, U>
{
    using Type = Vector<U>;
};

/// Apply //////////////////////////////////////////////////////////////

template <long>  struct ApplyFn {};
template <>      struct ApplyFn<0>  { template <class T> using Type = AddConst<T>; };
template <>      struct ApplyFn<1>  { template <class T> using Type = AddConstPtr<T>; };
template <>      struct ApplyFn<2>  { template <class T> using Type = AddConstRef<T>; };
template <>      struct ApplyFn<3>  { template <class T> using Type = AddPtr<T>; };
template <>      struct ApplyFn<4>  { template <class T> using Type = AddRef<T>; };
template <>      struct ApplyFn<5>  { template <class T> using Type = AddRValueRef<T>; };
template <>      struct ApplyFn<6>  { template <class T> using Type = RemoveConst<T>; };
template <>      struct ApplyFn<7>  { template <class T> using Type = RemovePtr<T>; };
template <>      struct ApplyFn<8>  { template <class T> using Type = RemoveRef<T>; };
template <>      struct ApplyFn<9>  { template <class T> using Type = RemoveCV<T>; };
template <>      struct ApplyFn<10> { template <class T> using Type = Decay<T>; };

template <class PACK, class APPLY>
struct _Apply {};

template <class APPLY, class X0, class... X>
struct _Apply<Pack<X0, X...>, APPLY>
{
    using Type = Cat<
        Pack<typename APPLY::template Type<X0>>,
        Apply<Pack<X...>, APPLY>
    >;
};

template <class APPLY, class X0>
struct _Apply<Pack<X0>, APPLY>
{
    using Type = Pack<typename APPLY::template Type<X0>>;
};

template <class APPLY>
struct _Apply<Pack<>, APPLY>
{
    using Type = Pack<>;
};

} // namespace _pack


/// ---------------------------------------------------------------- ///
///     Pack Specialization                                         ///
/// ---------------------------------------------------------------- ///

struct PackBase {};

/// A compile-time pack of types
template <class... X>
struct Pack: public PackBase {
    using ThisType = Pack<X...>;
    static constexpr size_t Length = CountTypes<X...>::Value;
};

/// ---------------------------------------------------------------- ///
///     Repeat                                                       ///
/// ---------------------------------------------------------------- ///

template <long N, class T>
struct _NPack {
    using Type = Append<NPack<N-1, T>, T>;
};

template <class T>
struct _NPack<0, T> {
    using Type = Pack<>;
};

} /// namespace meta
} /// namespace miniten

#endif /// MINITEN_META_PACK_IMPL_H
