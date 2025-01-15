/***********************************************************************
 * This file implements a compile-time "tuple of types"
 *
 * Tuple<T...>
 * Element<T>   = Tuple<T>
 * Pair<T, U>   = Tuple<T, U>
 * NTuple<N, T> = Tuple<T... (N times)>
 ***********************************************************************/
#ifndef MINITEN_META_TUPLE_IMPL_H
#define MINITEN_META_TUPLE_IMPL_H
#include "../_core/defines.h"
#include "../_core/types.h"
#include "../show.h"
#include "traits.h"
#include "math.h"
#include "tuple.h"
#include "vector.h"
#include "packapi.h"

namespace miniten {
namespace meta {

/// ---------------------------------------------------------------- ///
///     Pack API                                                     ///
/// ---------------------------------------------------------------- ///

/// Cat

template <class RIGHT>
struct _Cat2<Tuple<>, RIGHT> {
    static_assert(IsTuple<RIGHT>::Value, "L/R must be tuples");
    using Type = RIGHT;
};

template <class X0, class... X>
struct _Cat2<Tuple<X0>, Tuple<X...>> {
    using Type = Tuple<X0,X...>;
};

template <class RIGHT, class X0, class... X>
struct _Cat2<Tuple<X0,X...>, RIGHT> {
    static_assert(IsTuple<RIGHT>::Value, "L/R must be tuples");
    using Type = Cat<Tuple<X0>, Cat<Tuple<X...>, RIGHT>>;
};

namespace _tuple {
    template <class TUPLE>                 struct _Reversed;
    template <class TUPLE>                 using   Reversed = typename _Reversed<TUPLE>::Type;
    template <class TUPLE, typename U>     struct _AsVector;
    template <class TUPLE, typename U>     using   AsVector = typename _AsVector<TUPLE,U>::Type;
    template <long>                         struct  ApplyFn;
    template <class TUPLE, class APPLY>    struct _Apply;
    template <class TUPLE, class APPLY>    using   Apply = typename _Apply<TUPLE, APPLY>::Type;
}

template <class... X>               struct _Length<Tuple<X...>>             { using Type = CountTypes<X...>; };
template <class... X>               struct _EmptyLike<Tuple<X...>>          { using Type = Tuple<>; };
template <class... X>               struct _IsTuple<Tuple<X...>>            { using Type = True; };
template <class... X>               struct _Reversed<Tuple<X...>>           { using Type = _tuple::Reversed<Tuple<X...>>; };
template <class U, class... X>      struct _AsVector<Tuple<X...>, U>        { using Type = _tuple::AsVector<Tuple<X...>, U>; };
template <class... X>               struct _AsTuple<Tuple<X...>>            { using Type = Tuple<X...>; };
template <class... X>               struct _AsPack<Tuple<X...>>             { using Type = Pack<X...>; };
template <class M, class... X>      struct _LikeFrom<Tuple<X...>, M>        { using Type = AsTuple<M>; };
template <class X0, class... X>     struct _GetFirst<Tuple<X0, X...>>       { using Type = Tuple<X0>; };
template <class X0, class... X>     struct _DelFirst<Tuple<X0, X...>>       { using Type = Tuple<X...>; };
template <class X0, class... X>     struct _GetFirstValue<Tuple<X0, X...>>  { using Type = X0; };
template <class... X>               struct _ApplyAddConst<Tuple<X...>>      { using Type = _tuple::Apply<Tuple<X...>, _tuple::ApplyFn<0>>; };
template <class... X>               struct _ApplyAddConstPtr<Tuple<X...>>   { using Type = _tuple::Apply<Tuple<X...>, _tuple::ApplyFn<1>>; };
template <class... X>               struct _ApplyAddConstRef<Tuple<X...>>   { using Type = _tuple::Apply<Tuple<X...>, _tuple::ApplyFn<2>>; };
template <class... X>               struct _ApplyAddPtr<Tuple<X...>>        { using Type = _tuple::Apply<Tuple<X...>, _tuple::ApplyFn<3>>; };
template <class... X>               struct _ApplyAddRef<Tuple<X...>>        { using Type = _tuple::Apply<Tuple<X...>, _tuple::ApplyFn<4>>; };
template <class... X>               struct _ApplyAddRValueRef<Tuple<X...>>  { using Type = _tuple::Apply<Tuple<X...>, _tuple::ApplyFn<5>>; };
template <class... X>               struct _ApplyRemoveConst<Tuple<X...>>   { using Type = _tuple::Apply<Tuple<X...>, _tuple::ApplyFn<6>>; };
template <class... X>               struct _ApplyRemovePtr<Tuple<X...>>     { using Type = _tuple::Apply<Tuple<X...>, _tuple::ApplyFn<7>>; };
template <class... X>               struct _ApplyRemoveRef<Tuple<X...>>     { using Type = _tuple::Apply<Tuple<X...>, _tuple::ApplyFn<8>>; };
template <class... X>               struct _ApplyRemoveCV<Tuple<X...>>      { using Type = _tuple::Apply<Tuple<X...>, _tuple::ApplyFn<9>>; };
template <class... X>               struct _ApplyDecay<Tuple<X...>>         { using Type = _tuple::Apply<Tuple<X...>, _tuple::ApplyFn<10>>; };

/// ---------------------------------------------------------------- ///
///     Pack API Implementation                                      ///
/// ---------------------------------------------------------------- ///

namespace _tuple {

/// Revert /////////////////////////////////////////////////////////////

template <class TUPLE>
struct _Reversed {};

template <class X0, class... X>
struct _Reversed<Tuple<X0, X...>> {
    using Type = Cat< _Reversed<Tuple<X...>>, Tuple<X0> >;
};

template <class X0>
struct _Reversed<Tuple<X0>> {
    using Type = Tuple<X0>;
};

template <>
struct _Reversed<Tuple<>> {
    using Type = Tuple<>;
};

/// AsVector ///////////////////////////////////////////////////////////

template <class TUPLE, typename U>
struct _AsVector {};

template <typename U, class X0, class... X>
struct _AsVector<Tuple<X0, X...>, U>
{
    using Type = Cat<
        Vector<U, static_cast<U>(X0())>,
        AsVector<Tuple<X...>, U>
    >;
};

template <typename U, class X0>
struct _AsVector<Tuple<X0>, U>
{
    using Type = Vector<U, static_cast<U>(X0())>;
};

template <typename U>
struct _AsVector<Tuple<>, U>
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

template <class TUPLE, class APPLY>
struct _Apply {};

template <class APPLY, class X0, class... X>
struct _Apply<Tuple<X0, X...>, APPLY>
{
    using Type = Cat<
        Tuple<typename APPLY::template Type<X0>>,
        Apply<Tuple<X...>, APPLY>
    >;
};

template <class APPLY, class X0>
struct _Apply<Tuple<X0>, APPLY>
{
    using Type = Tuple<typename APPLY::template Type<X0>>;
};

template <class APPLY>
struct _Apply<Tuple<>, APPLY>
{
    using Type = Tuple<>;
};

} // namespace _tuple


/// ---------------------------------------------------------------- ///
///     Tuple Specialization                                         ///
/// ---------------------------------------------------------------- ///

template <class... X> MINITEN_HOSTDEVICE inline
Tuple<X...> tuple(const X &... x) {
    return Tuple<X...>(x...);
}

template <class... X> MINITEN_HOSTDEVICE inline
GetFirstValue<Tuple<X...>> getFirstValue(const X &... x) {
    return tuple(x...).GetFirstValue();
}

template <class... X> MINITEN_HOSTDEVICE inline
GetLastValue<Tuple<X...>> getLastValue(const X &... x) {
    return tuple(x...).getLastValue();
}

template <class N ,class... X> MINITEN_HOSTDEVICE inline
GetFirst<Tuple<X...>, N::Value> getFirst(const X &... x, const N & n = UZ1()) {
    return tuple(x...).getFirst(n);
}

template <class N ,class... X> MINITEN_HOSTDEVICE inline
GetLast<Tuple<X...>, N::Value> getLast(const X &... x, const N & n = UZ1()) {
    return tuple(x...).getLast(n);
}

template <class N ,class... X> MINITEN_HOSTDEVICE inline
DelFirst<Tuple<X...>, N::Value> delFirst(const X &... x, const N & n = UZ1()) {
    return tuple(x...).delFirst(n);
}

template <class N ,class... X> MINITEN_HOSTDEVICE inline
DelLast<Tuple<X...>, N::Value> delLast(const X &... x, const N & n = UZ1()) {
    return tuple(x...).delLast(n);
}

struct TupleBase {};

/// A compile-time tuple of types
/// (default implementation is only used by empty tuples)
template <class... X>
struct Tuple: public TupleBase {};

// Empty tuple
template <>
struct Tuple<>: TupleBase
{
    /// Static types and values ////////////////////////////////////////

    using ThisType = Tuple<>;
    static constexpr size_t Length = 0;

    ////////////////////////////////////////////////////////////////////

    using AsConstRef = ApplyAddConstRef<ThisType>;
    using AsRef      = ApplyAddRValueRef<ThisType>;

    MINITEN_HOSTDEVICE inline AsConstRef asConstRef() const
    {
        return AsConstRef(*this);
    }

    /// Constexpr methods ///////////////////////////////////////////////

    MINITEN_HOSTDEVICE constexpr size_t     length()    const { return Length; }

    template <class I, class... M>  MINITEN_HOSTDEVICE constexpr InsertFrom<ThisType,I,M...>    insertFrom(I, M...) const { return InsertFrom<ThisType,I,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr PrependFrom<ThisType,M...>     prependFrom(M...)   const { return PrependFrom<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr AppendFrom<ThisType,M...>      appendFrom(M...)    const { return AppendFrom<ThisType,M...>(); }

    template <class I, class... M>  MINITEN_HOSTDEVICE constexpr Insert<ThisType,I,M...>        insert(I, M...)     const { return Insert<ThisType,I,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr Prepend<ThisType,M...>         prepend(M...)       const { return Prepend<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr Append<ThisType,M...>          append(M...)        const { return Append<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr Extend<ThisType,M...>          extend(M...)        const { return Extend<ThisType,M...>(); }

    /// Constructors ///////////////////////////////////////////////////

    constexpr Tuple() {};
    constexpr Tuple(const ThisType &...) {};
};

/// Non-empty tuple
template <class X0, class... X>
struct Tuple<X0, X...>: public DelLast<Tuple<X0, X...>> {
    /// Static types and values ////////////////////////////////////////

    using ParentType    = DelLast<Tuple<X0, X...>>;
    using ThisType      = Tuple<X0, X...>;
    using NextType      = Tuple<X...>;
    using FirstItem     = GetFirstValue<ThisType>;
    using LastItem      = GetLastValue<ThisType>;
    static constexpr size_t Length = CountTypes<X0, X...>::Value;

    /// Constexpr methods ///////////////////////////////////////////////

    MINITEN_HOSTDEVICE constexpr size_t     length()    const { return Length; }

    /// Other methods //////////////////////////////////////////////////

    template <class I>              MINITEN_HOSTDEVICE inline GetValue<ThisType,I>           getValue(I i)       const { return get(i).getLastValue(); }
                                    MINITEN_HOSTDEVICE inline GetFirstValue<ThisType>        getFirstValue()     const { return getValue(Z0()); }
                                    MINITEN_HOSTDEVICE inline GetLastValue<ThisType>         getLastValue()      const { return Value; }

    template <class I>              MINITEN_HOSTDEVICE inline Get<ThisType,I>                get(I)              const { return Get<ThisType,I>(); }
    template <class N>              MINITEN_HOSTDEVICE inline GetFirst<ThisType,N::Value>    getFirst(N)         const { return GetFirst<ThisType,N::Value>(); }
    template <class N>              MINITEN_HOSTDEVICE inline GetLast<ThisType,N::Value>     getLast(N)          const { return GetLast<ThisType,N::Value>(); }
    template <class N>              MINITEN_HOSTDEVICE inline GetFirst<ThisType>             getFirst()          const { return GetFirst<ThisType>(); }
    template <class N>              MINITEN_HOSTDEVICE inline GetLast<ThisType>              getLast()           const { return GetLast<ThisType>(); }

    // template <class I>              MINITEN_HOSTDEVICE inline AddConstRef<ThisType,I>        ref(I)              const { return Get<ThisType,I>(); }
    // template <class N>              MINITEN_HOSTDEVICE inline AddConstRef<GetFirst<ThisType,N::Value>>    refFirst(N = UZ1()) const { return GetFirst<ThisType,N::Value>(); }
    // template <class N>              MINITEN_HOSTDEVICE inline AddConstRef<GetLast<ThisType,N::Value>>     refLast(N = UZ1())  const { return GetLast<ThisType,N::Value>(); }

    template <class I>              MINITEN_HOSTDEVICE inline Del<ThisType,I>                del(I)              const { return Del<ThisType,I>(); }
    template <class N>              MINITEN_HOSTDEVICE inline DelFirst<ThisType,N::Value>    delFirst(N) const { return DelFirst<ThisType,N::Value>(); }
    template <class N>              MINITEN_HOSTDEVICE inline DelLast<ThisType,N::Value>     delLast(N)  const { return DelLast<ThisType,N::Value>(); }
    template <class N>              MINITEN_HOSTDEVICE inline DelFirst<ThisType>             delFirst()  const { return DelFirst<ThisType>(); }
    template <class N>              MINITEN_HOSTDEVICE inline DelLast<ThisType>              delLast()   const { return DelLast<ThisType>(); }

    // template <class I, class... M>  MINITEN_HOSTDEVICE inline SetFrom<ThisType,I,M...>       setFrom(I, M...)    const { return SetFrom<ThisType,I,M...>(); }
    // template <class... M>           MINITEN_HOSTDEVICE inline SetFirstFrom<ThisType,M...>    setFirstFrom(M...)  const { return SetFirstFrom<ThisType,M...>(); }
    // template <class... M>           MINITEN_HOSTDEVICE inline SetLastFrom<ThisType,M...>     setLastFrom(M...)   const { return SetLastFrom<ThisType,M...>(); }

    // template <class I, class... M>  MINITEN_HOSTDEVICE inline Set<ThisType,I,M...>           set(I, M...)        const { return Set<ThisType,I,M...>(); }
    // template <class... M>           MINITEN_HOSTDEVICE inline SetFirst<ThisType,M...>        setFirst(M...)      const { return SetFirst<ThisType,M...>(); }
    // template <class... M>           MINITEN_HOSTDEVICE inline SetLast<ThisType,M...>         setLast(M...)       const { return SetLast<ThisType,M...>(); }

    // template <class I, class... M>  MINITEN_HOSTDEVICE inline InsertFrom<ThisType,I,M...>    insertFrom(I, M...) const { return InsertFrom<ThisType,I,M...>(); }
    // template <class... M>           MINITEN_HOSTDEVICE inline PrependFrom<ThisType,M...>     prependFrom(M...)   const { return PrependFrom<ThisType,M...>(); }
    // template <class... M>           MINITEN_HOSTDEVICE inline AppendFrom<ThisType,M...>      appendFrom(M...)    const { return AppendFrom<ThisType,M...>(); }

    // template <class I, class... M>  MINITEN_HOSTDEVICE inline Insert<ThisType,I,M...>        insert(I, M...)     const { return Insert<ThisType,I,M...>(); }
    // template <class... M>           MINITEN_HOSTDEVICE inline Prepend<ThisType,M...>         prepend(M...)       const { return Prepend<ThisType,M...>(); }
    // template <class... M>           MINITEN_HOSTDEVICE inline Append<ThisType,M...>          append(M...)        const { return Append<ThisType,M...>(); }
    // template <class... M>           MINITEN_HOSTDEVICE inline Extend<ThisType,M...>          extend(M...)        const { return Extend<ThisType,M...>(); }

    /// Constructors ///////////////////////////////////////////////////

protected:
    MINITEN_HOSTDEVICE static inline
    const ParentType & _as_parent(const ThisType & x)
    {
        return *reinterpret_cast<const ParentType *>(&x);
    }

    MINITEN_HOSTDEVICE static inline
    ParentType & _as_parent(ThisType & x)
    {
        return *reinterpret_cast<ParentType *>(&x);
    }

    MINITEN_HOSTDEVICE inline const ParentType & asParentRef() const
    {
        return _as_parent(*this);
    }

    MINITEN_HOSTDEVICE inline ParentType & asParentRef()
    {
        return _as_parent(*this);
    }

    MINITEN_HOSTDEVICE inline ParentType asParent() const
    {
        return ParentType(_as_parent(*this));
    }

public:
    constexpr   Tuple():                                          ParentType(),                 Value()     {}
                Tuple(const X0 & x0, const Tuple<X...>     & x):  ParentType(x0, x.delLast()),  Value(x.getLastValue())   {}
                Tuple(const X0 & x0, const X &...            x):  Tuple(x0, tuple(x...))                            {}
                Tuple(const ThisType                       & x):  ParentType(x.asParent()),     Value(x.getLastValue()) {}
                // Tuple(const AsConstRef<ThisType>           & x):  ParentType(x.asParent()), Value(x.getLastValue()) {}
                // Tuple(const AsRef<ThisType>                & x):  ParentType(x.asParent()), Value(x.getLastValue()) {}
                Tuple(const ParentType & x, const LastItem & x1): ParentType(x),                Value(x1)               {}

protected:
    LastItem Value;
};

/// ---------------------------------------------------------------- ///
///     Repeat                                                       ///
/// ---------------------------------------------------------------- ///

template <long N, class T>
struct _NTuple {
    using Type = Append<NTuple<N-1, T>, T>;
};

template <class T>
struct _NTuple<0, T> {
    using Type = Tuple<>;
};

} /// namespace meta
} /// namespace miniten

#endif /// MINITEN_META_TUPLE_IMPL_H
