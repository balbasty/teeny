/***********************************************************************
 * This file implements a compile-time "tuple of types"
 *
 * It is used by many tuple-like classes under the hood
 * (e.g., miniten::meta::Tuple, which is a compile-time tuple of types,
 * and miniten::Tuple, which is a runtime tuple of values, each with its
 * own type)
 *
 * TypePack<T...>
 *  A tuple of types
 *
 * TYPEPACK_STATIC_INHERIT(ClassName)
 *  A macro to "inherit" TypePack's static aliases and constexpr
 ***********************************************************************/
#ifndef MINITEN_META_TYPEPACK_H
#define MINITEN_META_TYPEPACK_H
#include "../defines.h"
#include "../types.h"
#include "../show.h"

namespace miniten {
namespace meta {

/// ---------------------------------------------------------------- ///
///     Forward declarations                                         ///
/// ---------------------------------------------------------------- ///

template <class T, T... N>                      struct Vector;
template <class Start, class Stop, class Step>  struct SmartSlice;
template <long  Start, long  Stop, long  Step>  struct Slice;

/// ---------------------------------------------------------------- ///
///     TypePack                                                     ///
/// ---------------------------------------------------------------- ///

/// A compile-time tuple of typenames
///
/// TypePack<T...>
///
///  ::Length                   Number of types in the tuple
///  ::Empty                    Return the empty tuple:     TupleType<>
///  ::Reversed                 Return reversed tuple
///
///  ::GetItem<i>               Return the indexed type:    T[0]
///  ::GetItems<Index>          Return the indexed types:   TupleType<T[i], ...>
///  ::GetFirstItem             Alias for GetItem<0>
///  ::GetLastItem              Alias for GetItem<-1>
///  ::GetFirstItems<n>         Alias for GetItems< Slice<0, n> >
///  ::GetLastItems<n>          Alias for GetItems< Slice<-n, Length> >
///  ::Get<i...>                Alias for GetItems<Vector<long, i...> >
///  ::Slice<i,j,k>             Alias for GetItems<Slice<i, j, k> >
///  ::SmartSlice<i,j,k>        Alias for GetItems<SmartSlice<i, j, k> >
///
///  ::DelItem<i>               Return a tuple with deleted item
///  ::DelItems<Index>          Return a tuple with deleted items (alias)
///  ::DelFirstItem             Alias for DelItem<0>
///  ::DelLastItem              Alias for DelItem<-1>
///  ::DelFirstItems<n>         Alias for DelItems< Slice<0, n> >
///  ::DelLastItems<n>          Alias for DelItems< Slice<-n, n> >
///  ::Del<i...>                Alias for DelItems<Vector<long, i...> >
///
///  ::SetItem<i, T>            Return a tuple with type T at index i
///  ::SetItems<Index, T...>    Return a tuple with assigned types
///  ::SetFirstItem<T>          Alias for SetItem<0, T>
///  ::SetLastItem<T>           Alias for SetItem<-1, T>
///  ::SetFirstItems<T...>      Alias for SetItems<Slice<0, n>, T...>
///  ::SetLastItems<T...>       Alias for SetItems<Slice<-n, Length>, T...>
///
///  ::Insert<i, T...>          Insert types in position i
///  ::Prepend<T...>            Alias for Insert<0, T...>
///  ::Append<T...>             Alias for Insert<Length, T...>
///  ::Extend<TypePack...>      Alias for Append<T...>
///  ::InsertPack<i, TypePack>  Alias for Insert<i, T...>
template <class... T>
struct TypePack {};

/// ---------------------------------------------------------------- ///
///     Implementation                                               ///
/// ---------------------------------------------------------------- ///

/// NOTE: all _pack helpers assume that Index is already wrapped
namespace _pack {

/// Forward declaration
template <class TUPLE, long  Index>                    struct GetItem;
template <class TUPLE, class Index>                    struct GetItems;
template <class TUPLE, long  Index>                    struct DelItem;
template <class TUPLE, class Index>                    struct DelItems;
template <class TUPLE, long  Index, class Value>       struct SetItem;
template <class TUPLE, class Index, class Values>      struct SetItems;
template <class TUPLE, long  Index, class... Values>   struct InsertTuple;

/// Concatenation helper
template <class LEFT, class RIGHT>
struct Cat {
    using Type = typename Cat<typename LEFT::ToPack, typename RIGHT::ToPack>::Type;
};


template <class... L, class... R>
struct Cat< TypePack<L...>, TypePack<R...> > {
    using Type = TypePack<L..., R...>;
};

// template <class RIGHT, class N0, class... N>
// struct Cat< TypePack<N0, N...>, RIGHT > {
// private:
//     using LeftTuple = TypePack<N0>;
//     using RightTuple = typename Cat<TypePack<N...>, RIGHT>::Type;
// public:
//     using Type = typename Cat<LeftTuple, RightTuple>::Type;
// };

// template <class N0, class... N>
// struct Cat< TypePack<N0>, TypePack<N...> > {
//     using Type = TypePack<N0, N...>;
// };

// template <class RIGHT>
// struct Cat<TypePack<>, RIGHT> {
//     using Type = typename RIGHT::ToPack;
// };


/// GetItem ////////////////////////////////////////////////////////////

/// Implementation of Vector::GetItem<Index>
template <class TUPLE, long Index>
struct GetItem {};

/// 1+ elements -> recursive call
template <long Index, class  N0, class... N>
struct GetItem<TypePack<N0, N...>, Index>
{
    using Type = typename GetItem<TypePack<N...>, Index-1>::Type;
};

/// Index first element
template <class N0, class... N>
struct GetItem<TypePack<N0, N...>, 0L>
{
    using Type = N0;
};

/// GetItems ///////////////////////////////////////////////////////////

/// Implementation of Vector::GetItems<Index>
template <class TUPLE, class Index>
struct GetItems {
    using Type = TUPLE;
};

/// Vector-Index: Take first indexed element, then recurse
template <class TUPLE, class T, T Index0, T... Index>
struct GetItems<TUPLE, Vector<T, Index0, Index...> >
{
private:
    using LeftTuple  = typename GetItems<TUPLE, Vector<T, Index0> >::Type;
    using RightTuple = typename GetItems<TUPLE, Vector<T, Index...> >::Type;
public:
    using Type = typename Cat<LeftTuple, RightTuple>::Type;
};

/// Vector-Index: Single index
template <class TUPLE, class T, T Index0>
struct GetItems<TUPLE, Vector<T, Index0> >
{
    using Type = TypePack<typename GetItem<TUPLE, Index0>::Type>;
};

/// Vector-Index: EmptyLike index list
template <class TUPLE, class T>
struct GetItems<TUPLE, Vector<T> >
{
    using Type = TypePack<>;
};

/// DelItem ////////////////////////////////////////////////////////////

/// Implementation of Vector::DelItem<Index>
template <class TUPLE, long Index>
struct DelItem {};

/// 1+ elements -> recursive call
template <long Index, class N0, class... N>
struct DelItem<TypePack<N0, N...>, Index>
{
private:
    using LeftTuple  = TypePack<N0>;
    using RightTuple = typename DelItem<TypePack<N...>, Index-1>::Type;
public:
    using Type = typename Cat<LeftTuple, RightTuple>::Type;
};

/// Delete first element
template <class N0, class... N>
struct DelItem<TypePack<N0, N...>, 0>
{
    using Type = TypePack<N...>;
};

/// DelItems ///////////////////////////////////////////////////////////

/// Implementation of Vector::GetItems<Index>
template <class TUPLE, class Index>
struct DelItems {};

/// Vector: Delete first indexed element, then recurse
template <class TUPLE, class T, T Index0, T... Index>
struct DelItems<TUPLE, Vector<T, Index0, Index...> >
{
private:
    using DelFirst = typename DelItem<TUPLE, Index0>::Type;
    using DelOther = typename DelItems<DelFirst, Vector<T, Index...> >::Type;
public:
    using Type = DelOther;
};

/// Vector: Single index
template <class TUPLE, class T, T Index0>
struct DelItems<TUPLE, Vector<T, Index0> >
{
    using Type = typename DelItem<TUPLE, Index0>::Type;
};

/// Vector: EmptyLike index list
template <class TUPLE, class T>
struct DelItems<TUPLE, Vector<T> >
{
    using Type = TUPLE;
};

/// SetItem ////////////////////////////////////////////////////////////

/// Implementation of Vector::SetItem<Index, T>
template <class TUPLE, long Index, class NewItem>
struct SetItem {};

/// 1+ elements -> recursive call
template <long Index, class NewItem, class N0, class... N>
struct SetItem<TypePack<N0, N...>, Index, NewItem>
{
private:
    using LeftTuple  = TypePack<N0>;
    using RightTuple = typename SetItem<TypePack<N...>, Index-1, NewItem>::Type;
public:
    using Type = typename Cat<LeftTuple, RightTuple>::Type;
};

/// Set first element
template <class NewItem, class N0, class... N>
struct SetItem<TypePack<N0, N...>, 0L, NewItem>
{
    using Type = TypePack<NewItem, N...>;
};

/// SetItems ///////////////////////////////////////////////////////////

/// Implementation of TypePack::SetItems<Index, Values>
template <class TUPLE, class Index, class Values>
struct SetItems {};

/// Vector: Delete first indexed element, then recurse
template <class TUPLE, class Values, class T, T Index0, T... Index>
struct SetItems<TUPLE, Vector<T, Index0, Index...>, Values>
{
    static_assert(Values::Length == 1 + Count<T, Index...>::Value,
                  "Index and Values length mismatch");
private:
    using OtherValues = typename Values::DelFirstItem;
    using SetFirst    = typename SetItem<TUPLE, Index0, typename Values::GetFirstItem>::Type;
    using SetOther    = typename SetItems<SetFirst, Vector<T, Index...>, OtherValues>::Type;
public:
    using Type = SetOther;
};

/// Vector: Single index
template <class TUPLE, class Values, class T, T Index0>
struct SetItems<TUPLE, Vector<T, Index0>, Values>
{
    static_assert(Values::Length == 1, "Index and Values length mismatch");
    using Type = typename SetItem<TUPLE, Index0, typename Values::GetFirstItem>::Type;
};

/// Vector: EmptyLike index list
template <class TUPLE, class Values, class T>
struct SetItems<TUPLE, Vector<T>, Values>
{
    static_assert(Values::Length == 0, "Index and Values length mismatch");
    using Type = TUPLE;
};

/// InsertPack /////////////////////////////////////////////////////////

template <class TUPLE, long  Index, class... Values>
struct InsertPack {};

template <class TUPLE, long  Index, class First, class... Values>
struct InsertPack<TUPLE, Index, First, Values...>
{
private:
    using Start   = typename TUPLE::template GetFirstItems<Index>;
    using End     = typename TUPLE::template GetLastItems<TUPLE::Length-Index>;
    using Middle  = typename Cat<Start, First>::Type;
    using FullMid = InsertPack<Middle, Middle::Length, Values...>;
public:
    using Type = typename Cat<FullMid, End>::Type;
};

template <class TUPLE, long  Index, class First>
struct InsertPack<TUPLE, Index, First>
{
private:
    using Start   = typename TUPLE::template GetFirstItems<Index>;
    using End     = typename TUPLE::template GetLastItems<TUPLE::Length-Index>;
    using Middle  = typename Cat<Start, First>::Type;
public:
    using Type = typename Cat<Middle, End>::Type;
};

template <class TUPLE, long  Index>
struct InsertPack<TUPLE, Index>
{
    using Type = TUPLE;
};

/// Revert /////////////////////////////////////////////////////////////

template <class TUPLE>
struct Reverse {};

template <class N0, class... N>
struct Reverse<TypePack<N0, N...> > {
    using Type = typename Cat< typename Reverse< TypePack<N...> >::Type, TypePack<N0> >::Type;
};

template <class N0>
struct Reverse<TypePack<N0> > {
    using Type = TypePack<N0>;
};

template <>
struct Reverse<TypePack<> > {
    using Type = TypePack<>;
};

} // namespace _pack

/// ---------------------------------------------------------------- ///
///     TypePack Specialization                                      ///
/// ---------------------------------------------------------------- ///

/// Specialization for empty tuples ////////////////////////////////////
template <class T0, class... T>
struct TypePack<T0, T...> {
private:
public:
    /// To/From Pack ///////////////////////////////////////////////////

    template <class Pack>
    struct _FromPack {};

    template <class... U>
    struct _FromPack< TypePack<U...> > {
        using Type = TypePack<U...>;
    };

    template <class Pack>
    using FromPack = typename _FromPack<Pack>::Type;

    using ToPack = TypePack<T0, T...>;

    /// Static types and values ////////////////////////////////////////

    using RefPack = TypePack<T0&, T&...>;

    static constexpr               long  Length        = CountTypes<T0, T...>::Value;
                                   using ThisType      = TypePack<T0, T...>;
                                   using Empty         = TypePack<>;
    template <class... M>          using PackLike      = TypePack<M...>;
                                   using Reversed      = typename _pack::Reverse<ThisType>::Type;

    template <class I>             using GetItems      = typename _pack::GetItems<ThisType, typename WrapIndexVector<Length, I>::Type>::Type;
    template <long... I>           using Get           = GetItems< Long<I...> >;
    template <long Nb>             using GetFirstItems = GetItems< Slice<0, Nb> >;
    template <long Nb>             using GetLastItems  = GetItems< Slice<Length-Nb, Length> >;
    template <long... I>           using Slice         = GetItems< Slice<I...> >;
    template <class... I>          using SmartSlice    = GetItems< SmartSlice<I...> >;
    template <long I>              using GetItem       = typename GetItems< Long<I> >::GetFirstItem;
                                   using GetFirstItem  = T0;
                                   using GetLastItem   = GetItem<Length-1L>;

    template <class I>             using RefItems      = typename GetItems<I>::RefPack;
    template <long... I>           using Ref           = typename Get<I...>::RefPack;
    template <long Nb>             using RefFirstItems = typename GetFirstItems<Nb>::RefPack;
    template <long Nb>             using RefLastItems  = typename GetLastItems<Nb>::RefPack;
    template <long... I>           using RefSlice      = typename Slice<I...>::RefPack;
    template <class... I>          using RefSmartSlice = typename SmartSlice<I...>::RefPack;
    template <long I>              using RefItem       = GetItem<I>   &;
                                   using RefFirstItem  = GetFirstItem &;
                                   using RefLastItem   = GetLastItem  &;

    template <class I>             using DelItems      = typename _pack::DelItems<ThisType, typename WrapIndexVector<Length, I>::Type>::Type;
    template <long... I>           using Del           = DelItems< Long<I...> >;
    template <long Nb>             using DelFirstItems = DelItems< Slice<0, Nb> >;
    template <long Nb>             using DelLastItems  = DelItems< Slice<Length-Nb, Length> >;
    template <long I>              using DelItem       = DelItems< Long<I> >;
                                   using DelFirstItem  = TypePack<T...>;
                                   using DelLastItem   = DelItem<Length-1L>;

    template <class I, class... M> using SetItems      = typename _pack::SetItems<ThisType, typename WrapIndexVector<Length, I>::Type, TypePack<M...> >::Type;
    template <class... M>          using SetFirstItems = SetItems< Slice<0, CountTypes<M...>::Value>, M... >;
    template <class... M>          using SetLastItems  = SetItems< Slice<Length-CountTypes<M...>::Value, Length>, M... >;
    template <long I, class M>     using SetItem       = SetItems< Long<I>, M>;
    template <class M>             using SetFirstItem  = TypePack<M, T...>;
    template <class M>             using SetLastItem   = SetItem<Length-1L, M>;

    template <long I, class... O>  using InsertPack    = typename _pack::InsertPack<ThisType, WrapIndex<Length, I>::Value, O...>::Type;
    template <long I, class... M>  using Insert        = InsertPack<I, TypePack<M...> >;
    template <class... M>          using Prepend       = Insert<0, M...>;
    template <class... M>          using Append        = Insert<Length, M...>;
    template <class... O>          using Extend        = InsertPack<Length, O...>;

    /// Static methods /////////////////////////////////////////////////

    MINITEN_HOSTDEVICE static inline
    void ShowValues()
    {
        show<GetFirstItem>();
        if (Length > 1)
            show(", ");
        DelFirstItem::ShowValues();
    }

    MINITEN_HOSTDEVICE static inline
    void Show()
    {
        show("meta::TypePack<");
        ShowValues();
        show(">");
    }
};

/// Specialization for singletons //////////////////////////////////////
template <class T0>
struct TypePack<T0> {
    /// To/From Pack ///////////////////////////////////////////////////

    template <class Pack>
    struct _FromPack {};

    template <class... U>
    struct _FromPack< TypePack<U...> > {
        using Type = TypePack<U...>;
    };

    template <class Pack>
    using FromPack = typename _FromPack<Pack>::Type;

    using ToPack = TypePack<T0>;

    using RefPack = TypePack<T0&>;

    /// Static types and values ////////////////////////////////////////

    static constexpr               long  Length        = 1L;
                                   using ThisType      = TypePack<T0>;
                                   using Empty         = TypePack<>;
    template <class... M>          using PackLike      = TypePack<M...>;
                                   using Reversed      = TypePack<T0>;

    template <class I>             using GetItems      = typename _pack::GetItems<ThisType, typename WrapIndexVector<Length, I>::Type>::Type;
    template <long... I>           using Get           = GetItems< Long<I...> >;
    template <long Nb>             using GetFirstItems = GetItems< Slice<0, Nb> >;
    template <long Nb>             using GetLastItems  = GetItems< Slice<Length-Nb, Length> >;
    template <long... I>           using Slice         = GetItems< Slice<I...> >;
    template <class... I>          using SmartSlice    = GetItems< SmartSlice<I...> >;
    template <long I>              using GetItem       = typename GetItems< Long<I> >::GetFirstItem;
                                   using GetFirstItem  = T0;
                                   using GetLastItem   = T0;

    template <class I>             using RefItems      = typename GetItems<I>::RefPack;
    template <long... I>           using Ref           = typename Get<I...>::RefPack;
    template <long Nb>             using RefFirstItems = typename GetFirstItems<Nb>::RefPack;
    template <long Nb>             using RefLastItems  = typename GetLastItems<Nb>::RefPack;
    template <long... I>           using RefSlice      = typename Slice<I...>::RefPack;
    template <class... I>          using RefSmartSlice = typename SmartSlice<I...>::RefPack;
    template <long I>              using RefItem       = GetItem<I>   &;
                                   using RefFirstItem  = GetFirstItem &;
                                   using RefLastItem   = GetLastItem  &;

    template <class I>             using DelItems      = typename _pack::DelItems<ThisType, typename WrapIndexVector<Length, I>::Type>::Type;
    template <long... I>           using Del           = DelItems< Long<I...> >;
    template <long Nb>             using DelFirstItems = DelItems< Slice<0, Nb> >;
    template <long Nb>             using DelLastItems  = DelItems< Slice<Length-Nb, Length> >;
    template <long I>              using DelItem       = DelItems< Long<I> >;
                                   using DelFirstItem  = TypePack<>;
                                   using DelLastItem   = TypePack<>;

    template <class I, class... M> using SetItems      = typename _pack::SetItems<ThisType, typename WrapIndexVector<Length, I>::Type, TypePack<M...> >::Type;
    template <class... M>          using SetFirstItems = SetItems< Slice<0, CountTypes<M...>::Value>, M... >;
    template <class... M>          using SetLastItems  = SetItems< Slice<Length-CountTypes<M...>::Value, Length>, M... >;
    template <long I, class M>     using SetItem       = SetItems< Long<I>, M>;
    template <class M>             using SetFirstItem  = TypePack<M>;
    template <class M>             using SetLastItem   = TypePack<M>;

    template <long I, class... O>  using InsertPack    = typename _pack::InsertPack<ThisType, WrapIndex<Length, I>::Value, O...>::Type;
    template <long I, class... M>  using Insert        = InsertPack<I, TypePack<M...> >;
    template <class... M>          using Prepend       = Insert<0, M...>;
    template <class... M>          using Append        = Insert<Length, M...>;
    template <class... O>          using Extend        = InsertPack<Length, O...>;

    /// Static methods /////////////////////////////////////////////////

    MINITEN_HOSTDEVICE static inline
    void ShowValues()
    {
        show<GetFirstItem>();
    }

    MINITEN_HOSTDEVICE static inline
    void Show()
    {
        show("meta::TypePack<");
        ShowValues();
        show(">");
    }
};

/// Specialization for empty tuples ////////////////////////////////////
template <>
struct TypePack<> {
    /// To/From Pack ///////////////////////////////////////////////////

    template <class Pack>
    struct _FromPack {};

    template <class... U>
    struct _FromPack< TypePack<U...> > {
        using Type = TypePack<U...>;
    };

    template <class Pack>
    using FromPack = typename _FromPack<Pack>::Type;

    using ToPack = TypePack<>;

    using RefPack = TypePack<>;

    /// Static types and values ////////////////////////////////////////

    static constexpr               long  Length        = 0L;
                                   using ThisType      = TypePack<>;
                                   using Empty         = TypePack<>;
    template <class... M>          using PackLike      = TypePack<M...>;
                                   using Reversed      = TypePack<>;

    template <class I>             using GetItems      = typename _pack::GetItems<ThisType, typename WrapIndexVector<Length, I>::Type>::Type;
    template <long... I>           using Get           = GetItems< Long<I...> >;
    template <long Nb>             using GetFirstItems = GetItems< Slice<0, Nb> >;
    template <long Nb>             using GetLastItems  = GetItems< Slice<Length-Nb, Length> >;
    template <long... I>           using Slice         = GetItems< Slice<I...> >;
    template <class... I>          using SmartSlice    = GetItems< SmartSlice<I...> >;
    template <long I>              using GetItem       = Error;
                                   using GetFirstItem  = Error;
                                   using GetLastItem   = Error;

    template <class I>             using RefItems      = typename GetItems<I>::RefPack;
    template <long... I>           using Ref           = typename Get<I...>::RefPack;
    template <long Nb>             using RefFirstItems = typename GetFirstItems<Nb>::RefPack;
    template <long Nb>             using RefLastItems  = typename GetLastItems<Nb>::RefPack;
    template <long... I>           using RefSlice      = typename Slice<I...>::RefPack;
    template <class... I>          using RefSmartSlice = typename SmartSlice<I...>::RefPack;
    template <long I>              using RefItem       = Error;
                                   using RefFirstItem  = Error;
                                   using RefLastItem   = Error;

    template <class I>             using DelItems      = typename _pack::DelItems<ThisType, typename WrapIndexVector<Length, I>::Type>::Type;
    template <long... I>           using Del           = DelItems< Long<I...> >;
    template <long Nb>             using DelFirstItems = DelItems< Slice<0, Nb> >;
    template <long Nb>             using DelLastItems  = DelItems< Slice<Length-Nb, Length> >;
    template <long I>              using DelItem       = Error;
                                   using DelFirstItem  = Error;
                                   using DelLastItem   = Error;

    template <class I, class... M> using SetItems      = typename _pack::SetItems<ThisType, typename WrapIndexVector<Length, I>::Type, TypePack<M...> >::Type;
    template <class... M>          using SetFirstItems = SetItems< Slice<0, CountTypes<M...>::Value>, M... >;
    template <class... M>          using SetLastItems  = SetItems< Slice<Length-CountTypes<M...>::Value, Length>, M... >;
    template <long I, class M>     using SetItem       = Error;
    template <class M>             using SetFirstItem  = Error;
    template <class M>             using SetLastItem   = Error;

    template <long I, class... O>  using InsertPack    = typename _pack::InsertPack<ThisType, WrapIndex<Length, I>::Value, O...>::Type;
    template <long I, class... M>  using Insert        = InsertPack<I, TypePack<M...> >;
    template <class... M>          using Prepend       = Insert<0, M...>;
    template <class... M>          using Append        = Insert<Length, M...>;
    template <class... O>          using Extend        = InsertPack<Length, O...>;

    /// Static methods /////////////////////////////////////////////////

    MINITEN_HOSTDEVICE static inline
    void ShowValues()
    {}

    MINITEN_HOSTDEVICE static inline
    void Show()
    {
        show("meta::TypePack<");
        ShowValues();
        show(">");
    }

};

} /// namespace meta
} /// namespace miniten

#define TYPEPACK_STATIC_TOPACK(ClassName)                                                                          \
    template <class Pack>                                                                                           \
    struct _FromPack {};                                                                                            \
                                                                                                                    \
    template <class... U>                                                                                           \
    struct _FromPack< TypePack<U...> > {                                                                            \
        using Type = ClassName<U...>;                                                                               \
    };                                                                                                              \
                                                                                                                    \
    template <>                                                                                                     \
    struct _FromPack<Error> {                                                                                       \
        using Type = Error;                                                                                         \
    };                                                                                                              \
                                                                                                                    \
    template <class Pack>                                                                                           \
    using FromPack = typename _FromPack<Pack>::Type;                                                                \
                                                                                                                    \
    using ToPack = TypePack<T...>;                                                                                  \


/// This macro should be used like this:
////
/// template <class... T>
/// struct MyStruct {
///     TYPEPACK_STATIC_TOPACK(MyStruct)
///     TYPEPACK_STATIC_INHERIT
/// };
#define TYPEPACK_STATIC_INHERIT                                                                                     \
    static constexpr               long  Length        = ToPack::Length;                                            \
                                                                                                                    \
                                   using ThisType      = FromPack<ToPack>;                                          \
                                   using Empty         = FromPack<::miniten::meta::TypePack<> >;                    \
    template <class... M>          using TupleLike     = FromPack<::miniten::meta::TypePack<M...> >;                \
                                   using Reversed      = FromPack<typename ToPack::Reversed>;                       \
                                                                                                                    \
    template <class I>             using GetItems      = FromPack<typename ToPack::template GetItems<I> >;          \
    template <long... I>           using Get           = FromPack<typename ToPack::template Get<I...> >;            \
    template <long Nb>             using GetFirstItems = FromPack<typename ToPack::template GetFirstItems<Nb> >;    \
    template <long Nb>             using GetLastItems  = FromPack<typename ToPack::template GetLastItems<Nb> >;     \
    template <long... I>           using Slice         = FromPack<typename ToPack::template Slice<I...> >;          \
    template <class... I>          using SmartSlice    = FromPack<typename ToPack::template SmartSlice<I...> >;     \
    template <long I>              using GetItem       = typename ToPack::template GetItem<I>;                      \
                                   using GetFirstItem  = typename ToPack::GetFirstItem;                             \
                                   using GetLastItem   = typename ToPack::GetLastItem;                              \
                                                                                                                    \
    template <class I>             using RefItems      = FromPack<typename ToPack::template RefItems<I> >;          \
    template <long... I>           using Ref           = FromPack<typename ToPack::template Ref<I...> >;            \
    template <long Nb>             using RefFirstItems = FromPack<typename ToPack::template RefFirstItems<Nb> >;    \
    template <long Nb>             using RefLastItems  = FromPack<typename ToPack::template RefLastItems<Nb> >;     \
    template <long... I>           using RefSlice      = FromPack<typename ToPack::template RefSlice<I...> >;       \
    template <class... I>          using RefSmartSlice = FromPack<typename ToPack::template RefSmartSlice<I...> >;  \
    template <long I>              using RefItem       = typename ToPack::template RefItem<I>;                      \
                                   using RefFirstItem  = typename ToPack::RefFirstItem;                             \
                                   using RefLastItem   = typename ToPack::RefLastItem;                              \
                                                                                                                    \
    template <class I>             using DelItems      = FromPack<typename ToPack::template DelItems<I> >;          \
    template <long... I>           using Del           = FromPack<typename ToPack::template Del<I...> >;            \
    template <long Nb>             using DelFirstItems = FromPack<typename ToPack::template DelFirstItems<Nb> >;    \
    template <long Nb>             using DelLastItems  = FromPack<typename ToPack::template DelLastItems<Nb> >;     \
    template <long I>              using DelItem       = FromPack<typename ToPack::template DelItem<I> >;           \
                                   using DelFirstItem  = FromPack<typename ToPack::DelFirstItem>;                   \
                                   using DelLastItem   = FromPack<typename ToPack::DelLastItem>;                    \
                                                                                                                    \
    template <class I, class... M> using SetItems      = FromPack<typename ToPack::template SetItems<I, M...> >;    \
    template <class... M>          using SetFirstItems = FromPack<typename ToPack::template SetFirstItems<M...> >;  \
    template <class... M>          using SetLastItems  = FromPack<typename ToPack::template SetLastItems<M...> >;   \
    template <long I, class M>     using SetItem       = FromPack<typename ToPack::template SetItem<I, M> >;        \
    template <class M>             using SetFirstItem  = FromPack<typename ToPack::template SetFirstItem<M> >;      \
    template <class M>             using SetLastItem   = FromPack<typename ToPack::template SetLastItem<M> >;       \
                                                                                                                    \
    template <long I, class... O>  using InsertTuple   = FromPack<typename ToPack::template InsertPack<I, O...> >;  \
    template <long I, class... M>  using Insert        = FromPack<typename ToPack::template Insert<I, M...> >;      \
    template <class... M>          using Prepend       = FromPack<typename ToPack::template Prepend<M...> >;        \
    template <class... M>          using Append        = FromPack<typename ToPack::template Append<M...> >;         \
    template <class... O>          using Extend        = FromPack<typename ToPack::template Extend<O...> >;         \
                                                                                                                    \
    MINITEN_HOSTDEVICE static inline void ShowValues() { return ToPack::ShowValues(); }                             \



#endif /// MINITEN_META_TYPEPACK_H
