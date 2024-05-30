/***********************************************************************
 * This file implements a compile-time "vector of values"
 *
 * Vector<T, N...>
 ***********************************************************************/
#ifndef MINITEN_META_VECTOR_H
#define MINITEN_META_VECTOR_H
#include "../defines.h"
#include "../types.h"
#include "../show.h"

namespace miniten {
namespace meta {

/// ---------------------------------------------------------------- ///
///     Forward declarations                                         ///
/// ---------------------------------------------------------------- ///

struct None;
struct Error;

/// Containers
template <class... T>      struct Tuple;
template <class T, T... N> struct Vector;

/// Slices
template <class Start, class Stop, class Step>  struct SmartSlice;
template <long  Start, long  Stop, long  Step>  struct Slice;

/// ---------------------------------------------------------------- ///
///     Aliases                                                      ///
/// ---------------------------------------------------------------- ///

/// A compile-time value
template <typename T, T N>
using Scalar = Vector<T, N>;

/// A compile-time signed char
template <signed char... N>
using SignedChar = Vector<signed char, N...>;

/// A compile-time short integer
template <short... N>
using Short = Vector<short, N...>;

/// A compile-time integer
template <int... N>
using Int = Vector<int, N...>;

/// A compile-time long integer
template <long... N>
using Long = Vector<long, N...>;

/// A compile-time long long integer
template <long long... N>
using LongLong = Vector<long long, N...>;

/// A compile-time unsigned char
template <unsigned char... N>
using UnsignedChar = Vector<unsigned char, N...>;

/// A compile-time unsigned short integer
template <unsigned short... N>
using UnsignedShort = Vector<unsigned short, N...>;

/// A compile-time unsigned integer
template <unsigned int... N>
using UnsignedInt = Vector<unsigned int, N...>;

/// A compile-time unsigned long integer
template <unsigned long... N>
using UnsignedLong = Vector<unsigned long, N...>;

/// A compile-time unsigned long long integer
template <unsigned long long... N>
using UnsignedLongLong = Vector<unsigned long long, N...>;

/// A compile-time signed char
template <int8_t... N>
using Int8 = Vector<int8_t, N...>;

/// A compile-time short integer
template <int16_t... N>
using Int16 = Vector<int16_t, N...>;

/// A compile-time integer
template <int32_t... N>
using Int32 = Vector<int32_t, N...>;

/// A compile-time long integer
template <int64_t... N>
using Int64 = Vector<int64_t, N...>;

/// A compile-time long long integer
template <int128_t... N>
using Int128 = Vector<int128_t, N...>;

/// A compile-time unsigned char
template <uint8_t... N>
using UInt8 = Vector<uint8_t, N...>;

/// A compile-time unsigned short integer
template <uint16_t... N>
using UInt16 = Vector<uint16_t, N...>;

/// A compile-time unsigned integer
template <uint32_t... N>
using UInt32 = Vector<uint32_t, N...>;

/// A compile-time unsigned long integer
template <uint64_t... N>
using UInt64 = Vector<uint64_t, N...>;

/// A compile-time unsigned long long integer
template <uint128_t... N>
using UInt128 = Vector<uint128_t, N...>;

/// ---------------------------------------------------------------- ///
///     Vector                                                       ///
/// ---------------------------------------------------------------- ///

/// A compile-time tuple of values with the same type
///
/// Vector<T, N...>
///
///  ::ItemType                 T
///  ::Length                   Number of items in the vector
///  ::EmptyLike                Return the empty vector:     Vector<T>
///  ::Reversed                 Return reversed vector
///
///  ::GetItem<i>               Return the indexed item:    N[0]
///  ::GetItems<Index>          Return the indexed items:   Vector<T, N[i], ...>
///  ::GetFirstItem             Alias for GetItem<0>
///  ::GetLastItem              Alias for GetItem<-1>
///  ::GetFirstItems<n>         Alias for GetItems< Slice<0, n> >
///  ::GetLastItems<n>          Alias for GetItems< Slice<-n, Length> >
///  ::Get<i...>                Alias for GetItems<Vector<long, i...> >
///  ::Slice<i,j,k>             Alias for GetItems<Slice<i, j, k> >
///  ::SmartSlice<i,j,k>        Alias for GetItems<SmartSlice<i, j, k> >
///
///  ::DelItem<i>               Return a vector with deleted item
///  ::DelItems<Index>          Return a vector with deleted items (alias)
///  ::DelFirstItem             Alias for DelItem<0>
///  ::DelLastItem              Alias for DelItem<-1>
///  ::DelFirstItems<n>         Alias for DelItems< Slice<0, n> >
///  ::DelLastItems<n>          Alias for DelItems< Slice<-n, n> >
///  ::Del<i...>                Alias for DelItems<Vector<long, i...> >
///
///  ::SetItem<i, N>            Return a vector with val N at index i
///  ::SetItems<Index, N...>    Return a vector with assigned values
///  ::SetFirstItem<N>          Alias for SetItem<0, N>
///  ::SetLastItem<N>           Alias for SetItem<-1, N>
///  ::SetFirstItems<M, N...>   Alias for SetItems<Slice<0, M>, N...>
///  ::SetLastItems<M, N...>    Alias for SetItems<Slice<-M, Length>, N...>
///
///  ::Insert<i, T...>          Insert items in position i
///  ::Prepend<T...>            Alias for Insert<0, T...>
///  ::Append<T...>             Alias for Insert<Length, T...>
///  ::Extend<Vector...>        Alias for Append<T...>
///  ::InsertVector<i, Vector>  Alias for Insert<i, T...>
///
///  ::AsTuple                  Convert to Tuple<Scalar<T, N>, ...>
template <typename T, T... N>
struct Vector {};

/// ---------------------------------------------------------------- ///
///     Vector Implementation                                        ///
/// ---------------------------------------------------------------- ///

/// NOTE: all _vector helpers assume that Index is already wrapped
namespace _vector {

/// Forward declaration
template <class VECTOR, long  Index>                    struct GetItem;
template <class VECTOR, class Index>                    struct GetItems;
template <class VECTOR, long  Index>                    struct DelItem;
template <class VECTOR, class Index>                    struct DelItems;
template <class VECTOR, long  Index, class T, T Value>  struct SetItem;
template <class VECTOR, class Index, class Values>      struct SetItems;
template <class VECTOR, long  Index, class... Values>   struct InsertVector;

/// Concatenation helper
template <class LEFT, class RIGHT>
struct Cat {};

template <class RIGHT, class T, T N0, T... N>
struct Cat< Vector<T, N0, N...>, RIGHT > {
    using Type = typename Cat<Vector<T, N0>, typename Cat<Vector<T, N...>, RIGHT>::Type>::Type;
};

template <class T, T N0, T... N>
struct Cat< Vector<T, N0>, Vector<T, N...> > {
    using Type = Vector<T, N0, N...>;
};

template <class RIGHT, class T>
struct Cat<Vector<T>, RIGHT> {
    using Type = RIGHT;
};


/// GetItem ////////////////////////////////////////////////////////////

/// Implementation of Vector::GetItem<Index>
template <class VECTOR, long Index>
struct GetItem {};

/// 1+ elements -> recursive call
template <long Index, class T, T N0, T... N>
struct GetItem<Vector<T, N0, N...>, Index>
{
    static constexpr T Value = GetItem<Vector<T, N...>, Index-1>::Value;
};

/// Index first element
template <class T, T N0, T... N>
struct GetItem<Vector<T, N0, N...>, 0L>
{
    static constexpr T Value = N0;
};

/// GetItems ///////////////////////////////////////////////////////////

/// Implementation of Vector::GetItems<Index>
template <class VECTOR, class Index>
struct GetItems {
    using Type = VECTOR;
};

/// Vector-Index: Take first indexed element, then recurse
template <class VECTOR, class T, T Index0, T... Index>
struct GetItems<VECTOR, Vector<T, Index0, Index...> >
{
private:
    using LeftVector  = typename GetItems<VECTOR, Vector<T, Index0> >::Type;
    using RightVector = typename GetItems<VECTOR, Vector<T, Index...> >::Type;
public:
    using Type = typename Cat<LeftVector, RightVector>::Type;
};

/// Vector-Index: Single index
template <class VECTOR, class T, T Index0>
struct GetItems<VECTOR, Vector<T, Index0> >
{
    using Type = typename VECTOR::template VectorLike<GetItem<VECTOR, Index0>::Value>;
};

/// Vector-Index: EmptyLike index list
template <class VECTOR, class T>
struct GetItems<VECTOR, Vector<T> >
{
    using Type = typename VECTOR::EmptyLike;
};

/// DelItem ////////////////////////////////////////////////////////////

/// Implementation of Vector::DelItem<Index>
template <class VECTOR, long Index>
struct DelItem {};

/// 1+ elements -> recursive call
template <long Index, class T, T N0, T... N>
struct DelItem<Vector<T, N0, N...>, Index>
{
private:
    using LeftVector  = Vector<T, N0>;
    using RightVector = typename DelItem<Vector<T, N...>, Index-1>::Type;
public:
    using Type = typename Cat<LeftVector, RightVector>::Type;
};

/// Delete first element
template <class T, T N0, T... N>
struct DelItem<Vector<T, N0, N...>, 0>
{
    using Type = Vector<T, N...>;
};

/// DelItems ///////////////////////////////////////////////////////////

/// Implementation of Vector::GetItems<Index>
template <class VECTOR, class Index>
struct DelItems {};

/// Vector: Delete first indexed element, then recurse
template <class VECTOR, class T, T Index0, T... Index>
struct DelItems<VECTOR, Vector<T, Index0, Index...> >
{
private:
    using DelFirst = typename DelItem<VECTOR, Index0>::Type;
    using DelOther = typename DelItems<DelFirst, Vector<T, Index...> >::Type;
public:
    using Type = DelOther;
};

/// Vector: Single index
template <class VECTOR, class T, T Index0>
struct DelItems<VECTOR, Vector<T, Index0> >
{
    using Type = typename DelItem<VECTOR, Index0>::Type;
};

/// Vector: EmptyLike index list
template <class VECTOR, class T>
struct DelItems<VECTOR, Vector<T> >
{
    using Type = VECTOR;
};

/// SetItem ////////////////////////////////////////////////////////////

/// Implementation of Vector::SetItem<Index, T>
template <class VECTOR, long Index, typename T, T NewItem>
struct SetItem {};

/// 1+ elements -> recursive call
template <long Index, class T, T NewItem, T N0, T... N>
struct SetItem<Vector<T, N0, N...>, Index, T, NewItem>
{
private:
    using LeftVector  = Vector<T, N0>;
    using RightVector = typename SetItem<Vector<T, N...>, Index-1, T, NewItem>::Type;
public:
    using Type = typename Cat<LeftVector, RightVector>::Type;
};

/// Set first element
template <class T, T NewItem, T N0, T... N>
struct SetItem<Vector<T, N0, N...>, 0L, T, NewItem>
{
    using Type = Vector<T, NewItem, N...>;
};

/// SetItems ///////////////////////////////////////////////////////////

/// Implementation of Tuple::SetItems<Index, Values>
template <class VECTOR, class Index, class Values>
struct SetItems {};

/// Vector: Delete first indexed element, then recurse
template <class VECTOR, class Values, class T, T Index0, T... Index>
struct SetItems<VECTOR, Vector<T, Index0, Index...>, Values>
{
private:
    using OtherValues = typename Values::DelFirstItem;
    using SetFirst    = typename SetItem<VECTOR, Index0, T, Values::GetFirstItem>::Type;
    using SetOther    = typename SetItems<SetFirst, Vector<T, Index...>, OtherValues>::Type;
public:
    using Type = SetOther;
};

/// Vector: Single index
template <class VECTOR, class Value, class T, T Index0>
struct SetItems<VECTOR, Vector<T, Index0>, Value>
{
    using Type = typename SetItem<VECTOR, Index0, T, Value::GetFirstItem>::Type;
};

/// Vector: EmptyLike index list
template <class VECTOR, class Value, class T>
struct SetItems<VECTOR, Vector<T>, Value>
{
    using Type = VECTOR;
};


/// InsertVector ///////////////////////////////////////////////////////

template <class VECTOR, long  Index, class... Values>
struct InsertVector {};

template <class VECTOR, long  Index, class First, class... Values>
struct InsertVector<VECTOR, Index, First, Values...>
{
private:
    using Start   = typename VECTOR::template GetFirstItems<Index>;
    using End     = typename VECTOR::template GetLastItems<VECTOR::Length-Index>;
    using Middle  = typename Cat<Start, First>::Type;
    using FullMid = InsertVector<Middle, Middle::Length, Values...>;
public:
    using Type = typename Cat<FullMid, End>::Type;
};

template <class VECTOR, long  Index, class First>
struct InsertVector<VECTOR, Index, First>
{
private:
    using Start   = typename VECTOR::template GetFirstItems<Index>;
    using End     = typename VECTOR::template GetLastItems<VECTOR::Length-Index>;
    using Middle  = typename Cat<Start, First>::Type;
public:
    using Type = typename Cat<Middle, End>::Type;
};

template <class VECTOR, long  Index>
struct InsertVector<VECTOR, Index>
{
    using Type = VECTOR;
};

/// Revert /////////////////////////////////////////////////////////////

template <class VECTOR>
struct Reverse {};

template <typename T, T N0, T... N>
struct Reverse<Vector<T, N0, N...> > {
    using Type = typename Cat< typename Reverse< Vector<T, N...> >::Type, Vector<T, N0> >::Type;
};

template <typename T, T N0>
struct Reverse<Vector<T, N0> > {
    using Type = Vector<T, N0>;
};

template <typename T>
struct Reverse<Vector<T> > {
    using Type = Vector<T>;
};

} // namespace _vector


/// ---------------------------------------------------------------- ///
///     Vector Specialization                                        ///
/// ---------------------------------------------------------------- ///

/// Empty Vector
template <typename T>
struct Vector<T> {

    /// Static types and values ////////////////////////////////////////

    static constexpr           long  Length        = 0L;
    static constexpr           T     Prod          = 1L;
    static constexpr           T     Sum           = 0L;

                               using ThisType      = Vector<T>;
                               using ItemType      = T;
                               using EmptyLike     = Vector<T>;
                               using Reversed      = Vector<T>;
    template <T... M>          using VectorLike    = Vector<T, M...>;

    template <class I>         using GetItems      = typename _vector::GetItems<ThisType, typename WrapIndexVector<Length, I>::Type>::Type;
    template <long... I>       using Get           = GetItems< Long<I...> >;
    template <long Nb>         using GetFirstItems = GetItems< Slice<0, Nb> >;
    template <long Nb>         using GetLastItems  = GetItems< Slice<Length-Nb, Length> >;
    template <long... I>       using Slice         = GetItems< Slice<I...> >;
    template <class... I>      using SmartSlice    = GetItems< SmartSlice<I...> >;

    template <class I>         using DelItems      = typename _vector::DelItems<ThisType, typename WrapIndexVector<Length, I>::Type>::Type;
    template <long... I>       using Del           = DelItems< Long<I...> >;
    template <long Nb>         using DelFirstItems = DelItems< Slice<0, Nb> >;
    template <long Nb>         using DelLastItems  = DelItems< Slice<Length-Nb, Length> >;

    template <class I, T... M> using SetItems      = typename _vector::SetItems<ThisType, typename WrapIndexVector<Length, I>::Type, Vector<T, M...> >::Type;
    template <long Nb, T... M> using SetFirstItems = SetItems< Slice<0, Nb>, M... >;
    template <long Nb, T... M> using SetLastItems  = SetItems< Slice<Length-Nb, Length>, M... >;

    template <long I, class... Vec>
                               using InsertVector  = typename _vector::InsertVector<ThisType, WrapIndex<Length, I>::Value, Vec...>::Type;
    template <long I, T... M>  using Insert        = InsertVector<I, Vector<T, M...> >;
    template <T... M>          using Prepend       = Vector<T, M...>;
    template <T... M>          using Append        = Vector<T, M...>;
    template <class... Vec>    using Extend        = InsertVector<0, Vec...>;

    /// Static methods /////////////////////////////////////////////////

    MINITEN_HOSTDEVICE static inline
    void ShowValues()
    {}

    MINITEN_HOSTDEVICE static inline
    void Show()
    {
        show("Vector<");
        show<T>();
        show(">");
    }
};

/// Single-element Vector
template <typename T, T N>
struct Vector<T, N> {

    /// Static types and values ////////////////////////////////////////

    static constexpr           long  Length        = 1L;
    static constexpr           T     Prod          = N;
    static constexpr           T     Sum           = N;
    static constexpr           T     Min           = N;
    static constexpr           T     Max           = N;

                               using ThisType      = Vector<T, N>;
                               using ItemType      = T;
                               using EmptyLike     = Vector<T>;
                               using Reversed      = Vector<T, N>;
    template <T... M>          using VectorLike    = Vector<T, M...>;

    template <class I>         using GetItems      = typename _vector::GetItems<ThisType, typename WrapIndexVector<Length, I>::Type>::Type;
    template <long I>          using GetItem       = GetItems< Long<I> >;
    template <long... I>       using Get           = GetItems< Long<I...> >;
    template <long Nb>         using GetFirstItems = GetItems< Slice<0, Nb> >;
    template <long Nb>         using GetLastItems  = GetItems< Slice<Length-Nb, Length> >;
    template <long... I>       using Slice         = GetItems< Slice<I...> >;
    template <class... I>      using SmartSlice    = GetItems< SmartSlice<I...> >;
    static constexpr           T     Value         = N;
    static constexpr           T     GetFirstItem  = N;
    static constexpr           T     GetLastItem   = N;

    template <class I>         using DelItems      = typename _vector::DelItems<ThisType, typename WrapIndexVector<Length, I>::Type>::Type;
    template <long I>          using DelItem       = DelItems< Long<I> >;
    template <long... I>       using Del           = DelItems< Long<I...> >;
    template <long Nb>         using DelFirstItems = DelItems< Slice<0, Nb> >;
    template <long Nb>         using DelLastItems  = DelItems< Slice<Length-Nb, Length> >;
                               using DelFirstItem  = DelItem<0L>;
                               using DelLastItem   = DelItem<0L>;

    template <class I, T... M> using SetItems      = typename _vector::SetItems<ThisType, typename WrapIndexVector<Length, I>::Type, Vector<T, M...> >::Type;
    template <long I, T M>     using SetItem       = SetItems< Long<I>, M>;
    template <long Nb, T... M> using SetFirstItems = SetItems< Slice<0, Nb>, M... >;
    template <long Nb, T... M> using SetLastItems  = SetItems< Slice<Length-Nb, Length>, M... >;
    template <T M>             using SetFirstItem  = SetItem<0L, M>;
    template <T M>             using SetLastItem   = SetItem<0L, M>;

    template <long I, class... Vec>
                               using InsertVector  = typename _vector::InsertVector<ThisType, WrapIndex<Length, I>::Value, Vec...>::Type;
    template <long I, T... M>  using Insert        = InsertVector<I, Vector<T, M...> >;
    template <T... M>          using Prepend       = Insert<0, M...>;
    template <T... M>          using Append        = Insert<Length, M...>;
    template <class... Vec>    using Extend        = InsertVector<Length, Vec...>;

    /// Static methods /////////////////////////////////////////////////

    MINITEN_HOSTDEVICE static inline
    void ShowValues()
    {
        show(GetFirstItem);
    }

    MINITEN_HOSTDEVICE static inline
    void Show()
    {
        show("Vector<");
        show<T>();
        show(", ");
        ShowValues();
        show(">");
    }
};

/// 2+ elements vector
template <typename T, T N0, T... N>
struct Vector<T, N0, N...> {

    /// Static types and values ////////////////////////////////////////

    static constexpr           long  Length        = Count<T, N0, N...>::Value;
    static constexpr           T     Prod          = meta::Prod<T, N0, N...>::Value;
    static constexpr           T     Sum           = meta::Sum<T, N0, N...>::Value;
    static constexpr           T     Min           = meta::Min<T, N0, N...>::Value;
    static constexpr           T     Max           = meta::Max<T, N0, N...>::Value;

                               using ThisType      = Vector<T, N0, N...>;
                               using ItemType      = T;
                               using EmptyLike     = Vector<T>;
                               using Reversed      = typename _vector::Reverse<ThisType>::Type;
    template <T... M>          using VectorLike    = Vector<T, M...>;

    template <class I>         using GetItems      = typename _vector::GetItems<ThisType, typename WrapIndexVector<Length, I>::Type>::Type;
    template <long I>          using GetItem       = GetItems< Long<I> >;
    template <long... I>       using Get           = GetItems< Long<I...> >;
    template <long Nb>         using GetFirstItems = GetItems< Slice<0, Nb> >;
    template <long Nb>         using GetLastItems  = GetItems< Slice<Length-Nb, Length> >;
    template <long... I>       using Slice         = GetItems< Slice<I...> >;
    template <class... I>      using SmartSlice    = GetItems< SmartSlice<I...> >;
    static constexpr           T     GetFirstItem  = N0;
    static constexpr           T     GetLastItem   = GetItem<Length-1L>::Value;

    template <class I>         using DelItems      = typename _vector::DelItems<ThisType, typename WrapIndexVector<Length, I>::Type>::Type;
    template <long I>          using DelItem       = DelItems< Long<I> >;
    template <long... I>       using Del           = DelItems< Long<I...> >;
    template <long Nb>         using DelFirstItems = DelItems< Slice<0, Nb> >;
    template <long Nb>         using DelLastItems  = DelItems< Slice<Length-Nb, Length> >;
                               using DelFirstItem  = Vector<T, N...>;
                               using DelLastItem   = DelItem<Length-1L>;

    template <class I, T... M> using SetItems      = typename _vector::SetItems<ThisType, typename WrapIndexVector<Length, I>::Type, Vector<T, M...> >::Type;
    template <long I, T M>     using SetItem       = SetItems< Long<I>, M>;
    template <long Nb, T... M> using SetFirstItems = SetItems< Slice<0, Nb>, M... >;
    template <long Nb, T... M> using SetLastItems  = SetItems< Slice<Length-Nb, Length>, M... >;
    template <T M>             using SetFirstItem  = Vector<T, M, N...>;
    template <T M>             using SetLastItem   = SetItem<Length-1L, M>;

    template <long I, class... Vec>
                               using InsertVector  = typename _vector::InsertVector<ThisType, WrapIndex<Length, I>::Value, Vec...>::Type;
    template <long I, T... M>  using Insert        = InsertVector<I, Vector<T, M...> >;
    template <T... M>          using Prepend       = Insert<0, M...>;
    template <T... M>          using Append        = Insert<Length, M...>;
    template <class... Vec>    using Extend        = InsertVector<Length, Vec...>;

    /// Static methods /////////////////////////////////////////////////

    MINITEN_HOSTDEVICE static inline
    void ShowValues()
    {
        show(GetFirstItem);
        show(", ");
        DelFirstItem::ShowValues();
    }

    MINITEN_HOSTDEVICE static inline
    void Show()
    {
        show("Vector<");
        show<T>();
        show(", ");
        ShowValues();
        show(">");
    }
};

/// ---------------------------------------------------------------- ///
///     Repeat                                                       ///
/// ---------------------------------------------------------------- ///

/// Create a Vector of N times the same value
///
/// NVector<N, T, V>
///   Type = Vector<T, V...>
template <long N, typename T, T Val>
struct NVector {
    using Type = typename NVector<N-1, T, Val>::Type::template Prepend<Val>;
};

template <typename T, T Val>
struct NVector<0, T, Val> {
    using Type = Vector<T>;
};

/// ---------------------------------------------------------------- ///
///     Convert from Tuple< Scalar<T> > to Vector<T>                 ///
/// ---------------------------------------------------------------- ///

/// Convert from Tuple< Scalar<T> > to Vector<T>
template <typename VECTOR>
struct TupleToVector {};

template <typename FirstType, FirstType FirstValue, typename... OtherTypes>
struct TupleToVector< Tuple<Scalar<FirstType, FirstValue>, OtherTypes...> > {
private:
    using _LeftVector  = Scalar<FirstType, FirstValue>;
    using _RightVector = typename TupleToVector<Tuple<OtherTypes...> >::Type;
public:
    using Type = typename _LeftVector::template AppendVector<_RightVector>;
};

template <typename FirstType, FirstType FirstValue>
struct TupleToVector< Tuple< Scalar<FirstType, FirstValue> > > {
    using Type = Scalar<FirstType, FirstValue>;
};

template <>
struct TupleToVector< Tuple<> > {
    using Type = Vector<None>;
};

/// Convert from Vector<T> to Tuple< Scalar<T> >
template <typename VECTOR>
struct VectorToTuple {};

template <typename T, T FirstValue, T... OtherValues>
struct VectorToTuple< Vector<T, FirstValue, OtherValues...> > {
private:
    using _LeftTuple  = Tuple< Scalar<T, FirstValue> >;
    using _RightTuple = typename VectorToTuple< Vector<T, OtherValues...> >::Type;
public:
    using Type = typename _LeftTuple::template AppendTuple<_RightTuple>;
};

template <typename T, T FirstValue>
struct VectorToTuple< Vector<T, FirstValue> > {
    using Type = Tuple< Scalar<T, FirstValue> >;
};

template <typename T>
struct VectorToTuple< Vector<T> > {
    using Type = Tuple<>;
};


} /// namespace meta
} /// namespace miniten

#endif /// MINITEN_META_VECTOR_H
