#ifndef MINITEN_TUPLE_H
#define MINITEN_TUPLE_H
#include "defines.h"
#include "types.h"
#include "show.h"
#include "meta.h"

namespace miniten {

/// ---------------------------------------------------------------- ///
///     Tuple                                                        ///
/// ---------------------------------------------------------------- ///

/// A runtime tuple of values
///
/// Tuple<T...>
///
///  # --- MetaProgramming ---------------------------------------------
///
///  ::Length                   Number of types in the tuple
///  ::Empty                    Return the empty tuple:     Tuple<>
///  ::Reversed                 Return reversed tuple
///
///  ::GetItem<i>               Return the indexed type:    T[0]
///  ::GetItems<Index>          Return the indexed types:   Tuple<T[i], ...>
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
///  ::Extend<Tuple...>         Alias for Append<T...>
///  ::InsertTuple<i, Tuple>    Alias for Insert<i, T...>
///
///  # --- Dynamic Programming -----------------------------------------
///
///  tuple(T0 x0, T1 x1, ...) -> Tuple<T0, T1, ...>;
///
///  ::getItem<i>()        -> GetItem<i>
///  ::getItems<Index>()   -> GetItems<Index>
///  ::getFirstItem()      -> GetFirstItem
///  ::getLastItem()       -> GetLastItem
///  ::getFirstItems<n>()  -> GetFirstItems<n>
///  ::getLastItems<n>()   -> GetLastItems<n>
///  ::slice<i,j,k>()      -> Slice<i,j,k>
///  ::smartSlice<i,j,k>() -> SmartSlice<i,j,k>
template <class... T>
struct Tuple {};

template <class... T>
Tuple<T...> tuple(T... values)
{
    return Tuple<T...>(values...);
}


/// ---------------------------------------------------------------- ///
///     Aliases                                                      ///
/// ---------------------------------------------------------------- ///

/// A compile-time single element
template <typename T>
using Element = Tuple<T>;

/// A compile-time pair of elements
template <typename T, typename U = T>
using Pair = Tuple<T, U>;

/// ---------------------------------------------------------------- ///
///     Implementation                                               ///
/// ---------------------------------------------------------------- ///

/// NOTE: all _pack helpers assume that Index is already wrapped
namespace _tuple {

#if 0

/// Forward declaration
template <class TupleType, TupleType TUPLE, long  Index>                    struct GetItem;
template <class TupleType, TupleType TUPLE, class Index>                    struct GetItems;
template <class TupleType, TupleType TUPLE, long  Index>                    struct DelItem;
template <class TupleType, TupleType TUPLE, class Index>                    struct DelItems;
template <class TupleType, TupleType TUPLE, long  Index, class Value>       struct SetItem;
template <class TupleType, TupleType TUPLE, class Index, class Values>      struct SetItems;
template <class TupleType, TupleType TUPLE, long  Index, class... Values>   struct InsertTuple;

/// Concatenation helper
template <class LeftType, class RightType>
struct Cat {};

template <class RightType, class T0, class... T>
struct Cat<Tuple<T0, T...>, RightType> {
private:
    using LeftType   = Tuple<T0, T...>;
    using ReturnType = typename LeftType::template Extend<RightType>;
public:
    MINITEN_HOSTDEVICE static inline
    ReturnType do(const LeftType & left, const RightType & right)
    {
        return cat(left.getFirstItems<1>(), cat(left.getLastItems<LeftType::Length-1>(), right));
    }
};

template <class T0, class... T>
struct Cat< Tuple<T0>, Tuple<T...> > {
private:
    using LeftType   = Tuple<T0>;
    using RightType  = Tuple<T...>;
    using ReturnType = typename LeftType::template Extend<RightType>;
public:
    MINITEN_HOSTDEVICE static inline
    ReturnType do(const LeftType & left, const RightType & right)
    {
        return ReturnType(
            left.getFirstItem(),
            cat(left.getLastItems<LeftType::Length-1>(), right)
        );
    }
};

template <class RightType>
struct Cat<Tuple<>, RightType> {
private:
    using LeftType   = Tuple<>;
    using ReturnType = RightType;
public:
    MINITEN_HOSTDEVICE static inline
    ReturnType do(const LeftType & left, const RightType & right)
    {
        return right;
    }
};

template <class LeftType, class RightType>
MINITEN_HOSTDEVICE inline
typename LeftType::template Extend<RightType>
cat(const LeftType & left, const RightType & right)
{
    return Cat<LeftType, RightType>::do(left, right);
}

/// GetItem ////////////////////////////////////////////////////////////

/// Implementation of Vector::GetItem<Index>
template <class TupleType, long Index>
struct GetItem {};

/// 1+ elements -> recursive call
template <long Index, class  T0, class... T>
struct GetItem<Tuple<T0, T...>, Index>
{
private:
    using InputType  = Tuple<T0, T...>;
    using ReturnType = typename InputType::GetItem<Index>;
public:
    MINITEN_HOSTDEVICE static inline
    const ReturnType & do(const InputType & x)
    {
        return dynamic_cast<typename InputType::BaseType *>(&x)->getItem<Index-1>();
    }

    MINITEN_HOSTDEVICE static inline
    ReturnType & do(InputType & x)
    {
        return dynamic_cast<typename InputType::BaseType *>(&x)->getItem<Index-1>();
    }
};

/// Index first element
template <class T0, class... T>
struct GetItem<Tuple<T0, T...>, 0L>
{
private:
    using InputType  = Tuple<T0, T...>;
    using ReturnType = T0;
public:
    MINITEN_HOSTDEVICE static inline
    const ReturnType & do(const InputType & x)
    {
        return x.value;
    }

    MINITEN_HOSTDEVICE static inline
    ReturnType & do(InputType & x)
    {
        return x.value;
    }
};

template <class TupleType, long Index>
MINITEN_HOSTDEVICE inline
const typename TupleType::GetItem<Index> &
getItem(const TupleType & x)
{
    return GetItem<TupleType, Index>::do(x);
}

template <class TupleType, long Index>
MINITEN_HOSTDEVICE inline
typename TupleType::GetItem<Index> &
getItem(TupleType & x)
{
    return GetItem<TupleType, Index>::do(x);
}


/// GetItems ///////////////////////////////////////////////////////////

/// Implementation of Vector::GetItems<Index>
template <class TupleType, class Index>
struct GetItems {
    using Type = TupleType;
};

/// Vector-Index: Take first indexed element, then recurse
template <class TupleType, class T, T Index0, T... Index>
struct GetItems<TupleType, Vector<T, Index0, Index...> >
{
private:
    using InputType  = TupleType;
    using IndexType  = Vector<T, Index0, Index...>;
    using ReturnType = typename InputType::template GetItems<IndexType>;
    using RefType    = typename InputType::template RefItems<IndexType>;
public:
    MINITEN_HOSTDEVICE static inline
    ReturnType do(const InputType & x)
    {
        return cat(x.getFirstItems<1>(), x.getLastItems<InputType::Length-1>());
    }

    MINITEN_HOSTDEVICE static inline
    RefType do(InputType & x)
    {
        return cat(x.refFirstItems<1>(), x.refLastItems<InputType::Length-1>());
    }
};

/// Vector-Index: Single index
template <class TUPLE, class T, T Index0>
struct GetItems<TUPLE, Vector<T, Index0> >
{
private:
    using InputType  = TupleType;
    using IndexType  = Vector<T, Index0>;
    using ReturnType = typename InputType::template GetItems<IndexType>;
    using RefType    = typename InputType::template RefItems<IndexType>;
public:
    MINITEN_HOSTDEVICE static inline
    ReturnType do(const InputType & x)
    {
        return x.getFirstItems<1>();
    }

    MINITEN_HOSTDEVICE static inline
    RefType do(InputType & x)
    {
        return x.refFirstItems<1>();
    }
};

/// Vector-Index: EmptyLike index list
template <class TUPLE, class T>
struct GetItems<TUPLE, Vector<T> >
{
    MINITEN_HOSTDEVICE static inline
    typename TUPLE::Empty do(const TUPLE & x)
    {
        return TUPLE::Empty();
    }

    MINITEN_HOSTDEVICE static inline
    typename TUPLE::Empty do(TUPLE & x)
    {
        return TUPLE::Empty();
    }
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

#endif

} // namespace _pack

/// ---------------------------------------------------------------- ///
///     Tuple Specialization                                         ///
/// ---------------------------------------------------------------- ///

/// Specialization for empty tuples ////////////////////////////////////
template <>
struct Tuple<> {

    /// Conversion To/From TypePack ////////////////////////////////////

    template <class Pack>
    struct _FromPack {};

    template <class... U>
    struct _FromPack< meta::TypePack<U...> > {
        using Type = Tuple<U...>;
    };

    template <>
    struct _FromPack<meta::Error> {
        using Type = meta::Error;
    };

    template <class Pack>
    using FromPack = typename _FromPack<Pack>::Type;

    using ToPack = meta::TypePack<>;

    /// Metaprogramming ////////////////////////////////////////////////

    TYPEPACK_STATIC_INHERIT

    using MetaType = meta::Tuple<>;

    /// Static methods /////////////////////////////////////////////////

    /// Pretty-print compile-time values (i.e., types)
    MINITEN_HOSTDEVICE static inline
    void Show()
    {
        miniten::show("Tuple<>");
    }

    /// Print runtime values
    MINITEN_HOSTDEVICE static inline
    void ShowValues(const ThisType & value)
    {}

    /// Pretty-print runtime values
    MINITEN_HOSTDEVICE static inline
    void Show(const ThisType & value)
    {
        miniten::show("Tuple()");
    }

    /// Runtime methods ////////////////////////////////////////////////

    MINITEN_HOSTDEVICE inline
    void show()
    {
        Show(*this);
    }

    /// constructor
    constexpr Tuple() {}

    /// copy operator
    Tuple(const Tuple &) = default;
    Tuple(Tuple &&) = default;

    /// copy assignment
    constexpr const Tuple & operator= (const Tuple&) const { return *this; }
    constexpr const Tuple & operator= (Tuple &&)     const { return *this; }

    /// destructor
    virtual ~Tuple() {}
};

/// Specialization for single-element tuples ///////////////////////////
template <typename T>
struct Tuple<T>: Tuple<> {

    using BaseType = Tuple<>;

    /// Conversion To/From TypePack ////////////////////////////////////

    template <class Pack>
    struct _FromPack {};

    template <class... U>
    struct _FromPack< meta::TypePack<U...> > {
        using Type = Tuple<U...>;
    };

    template <>
    struct _FromPack<meta::Error> {
        using Type = meta::Error;
    };

    template <class Pack>
    using FromPack = typename _FromPack<Pack>::Type;

    using ToPack = meta::TypePack<T>;

    /// Metaprogramming ////////////////////////////////////////////////

    TYPEPACK_STATIC_INHERIT

    using MetaType = meta::Tuple<T>;

    /// Static methods /////////////////////////////////////////////////

    MINITEN_HOSTDEVICE static inline
    void Show()
    {
        miniten::show("Tuple<");
        ShowValues();
        miniten::show(">");
    }

    MINITEN_HOSTDEVICE static inline
    void ShowValues(const ThisType & tuple)
    {
        miniten::show(tuple.value);
    }

    MINITEN_HOSTDEVICE static inline
    void Show(const ThisType & tuple)
    {
        miniten::show("Tuple(");
        ShowValues(tuple);
        miniten::show(")");
    }

    /// Runtime methods ////////////////////////////////////////////////

    MINITEN_HOSTDEVICE inline
    void show()
    {
        Show(*this);
    }

    /// constructor
    constexpr Tuple() {}

    Tuple(const T & value): value(value) {}

    Tuple(const T & value, const Tuple<> & values): value(value) {}

    /// destructor
    virtual ~Tuple() {}

    /// copy constructor
    Tuple(const Tuple &) = default;
    Tuple(Tuple &&) = default;

    template <class U>
    Tuple(const Tuple<U> & tuple): value(static_cast<T>(tuple.value)) {}

    template <class U>
    Tuple(Tuple<U> && tuple): value(static_cast<T>(tuple.value)) {}

    /// copy operator
    Tuple & operator= (const Tuple & tuple)
    {
        this->value = tuple.value;
        return *this;
    }

    Tuple & operator= (Tuple && tuple) noexcept
    {
        this->value = tuple.value;
        return *this;
    }

    /// conversion from scalar
    Tuple & operator= (const T & value)
    {
        this->value = value;
        return *this;
    }

    /// conversion to scalar
    operator T() const
    {
        return value;
    }

    /// accessors
    template <int N> const T & getItem() const
    {
        static_assert(meta::WrapIndex<Length, N>::Value == 0, "Index must be 0");
        return value;
    }
    template <int N>       T & getItem()
    {
        static_assert(meta::WrapIndex<Length, N>::Value == 0, "Index must be 0");
        return value;
    }

    /// Members ////////////////////////////////////////////////////////

    T value;
};

/// Specialization for multi-element tuples ////////////////////////////
template <typename T0, typename... T>
struct Tuple<T0, T...>: Tuple<T...> {

    using BaseType = Tuple<T...>;

    /// Conversion To/From TypePack ////////////////////////////////////

    template <class Pack>
    struct _FromPack {};

    template <class... U>
    struct _FromPack< meta::TypePack<U...> > {
        using Type = Tuple<U...>;
    };

    template <>
    struct _FromPack<meta::Error> {
        using Type = meta::Error;
    };

    template <class Pack>
    using FromPack = typename _FromPack<Pack>::Type;

    using ToPack = meta::TypePack<T0, T...>;

    /// Metaprogramming ////////////////////////////////////////////////

    TYPEPACK_STATIC_INHERIT

    using MetaType = meta::Tuple<T0, T...>;

    /// Static methods /////////////////////////////////////////////////

    MINITEN_HOSTDEVICE static inline
    void Show()
    {
        miniten::show("Tuple<");
        ShowValues();
        miniten::show(">");
    }

    MINITEN_HOSTDEVICE static inline
    void ShowValues(const ThisType & tuple)
    {
        miniten::show(tuple.value);
        miniten::show(", ");
        DelFirstItem::ShowValues(tuple);
    }

    MINITEN_HOSTDEVICE static inline
    void Show(const ThisType & tuple)
    {
        miniten::show("Tuple(");
        ShowValues(tuple);
        miniten::show(")");
    }

    /// Runtime methods ////////////////////////////////////////////////

    MINITEN_HOSTDEVICE inline
    void show()
    {
        Show(*this);
    }

    /// constructor
    constexpr Tuple() {}

    Tuple(const T0 & value1, const T&... values):
        value(value1), DelFirstItem(values...)
    {}

    Tuple(const T0 & value1, const DelFirstItem & values):
        value(value1), DelFirstItem(values)
    {}

    /// destructor
    virtual ~Tuple() {}

    /// copy-constructor
    Tuple(const Tuple &) = default;
    Tuple(Tuple &&) = default;

    /// copy-assignment
    #if 0
    Tuple & operator= (const Tuple & tuple)
    {
        this->value = tuple.value;
        DelFirstItem::operator=(tuple.delFirstItem());
        return *this;
    }

    Tuple & operator= (Tuple && tuple) noexcept
    {
        this->value = tuple.value;
        DelFirstItem::operator=(tuple.delFirstItem());
        return *this;
    }
    #endif


    /// accessors
private:
    template <int n> const GetItem<n> & getItemWrapped()    const
    {
        static_assert(n >= 0 && n < Length, "Index must be in [0, Length-1]");
        return DelFirstItem::template get<n-1>();
    }
    template <int n>       GetItem<n> & getItemWrapped()
    {
        static_assert(n >= 0 && n < Length, "Index must be in [0, Length-1]");
        return DelFirstItem::template get<n-1>();
    }
    template <>      const T0         & getItemWrapped<0>() const
    {
        return value;
    }
    template <>            T0         & getItemWrapped<0>()
    {
        return value;
    }
public:
    template <int n> const GetItem<n> & getItem()    const
    {
        return getItemWrapped<meta::WrapIndex<Length, n>::Value>();
    }
    template <int n>       GetItem<n> & getItem()
    {
        return getItemWrapped<meta::WrapIndex<Length, n>::Value>();
    }
    // template <class index> const GetItems<index> & getItems()    const
    // {
    //     return getItemsWrapped<meta::WrapIndexVector<Length, index>::Type>();
    // }
    // template <class index>       GetItems<index> & getItems()
    // {
    //     return getItemsWrapped<meta::WrapIndexVector<Length, index>::Type>();
    // }

    /// members
    T0 value;
};

/// ---------------------------------------------------------------- ///
///     Repeat                                                       ///
/// ---------------------------------------------------------------- ///

/// Create a Tuple of N times the same type
///
/// NTuple<N, T>
///   Type = Tuple<T...>
template <long N, class T>
struct NTuple {
    using Type = typename NTuple<N-1, T>::Type::template Append<T>;
};

template <class T>
struct NTuple<0, T> {
    using Type = Tuple<>;
};
} /// namespace miniten

#endif /// MINITEN_META_TUPLE_H
