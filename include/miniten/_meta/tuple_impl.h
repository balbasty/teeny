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
#include <utility>
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
    template <class TUPLE>                 using   Reversed         = typename _Reversed<TUPLE>::Type;
    template <class TUPLE, typename U>     struct _AsVector;
    template <class TUPLE, typename U>     using   AsVector         = typename _AsVector<TUPLE,U>::Type;
    template <long>                        struct  ApplyFn;
    template <class TUPLE, class APPLY>    struct _Apply;
    template <class TUPLE, class APPLY>    using   Apply            = typename _Apply<TUPLE, APPLY>::Type;
    template <class TUPLE>                 struct _ApplySizeOf;
    template <class TUPLE>                 using   ApplySizeOf      = typename _ApplySizeOf<TUPLE>::Type;
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
template <class... X>               struct _ApplySizeOf<Tuple<X...>>        { using Type = _tuple::ApplySizeOf<Tuple<X...>>; };

/// ---------------------------------------------------------------- ///
///     Pack API Implementation                                      ///
/// ---------------------------------------------------------------- ///

namespace _tuple {

    /// Revert /////////////////////////////////////////////////////////

    template <class TUPLE>
    struct _Reversed {};

    template <class TUPLE>
    using Reversed = typename _Reversed<TUPLE>::Type;

    template <class X0, class... X>
    struct _Reversed<Tuple<X0, X...>> {
        using Type = Cat< Reversed<Tuple<X...>>, Tuple<X0> >;
    };

    template <class X0>
    struct _Reversed<Tuple<X0>> {
        using Type = Tuple<X0>;
    };

    template <>
    struct _Reversed<Tuple<>> {
        using Type = Tuple<>;
    };

    /// AsVector ///////////////////////////////////////////////////////

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

    /// Apply //////////////////////////////////////////////////////////

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

    template <class TUPLE>
    struct _ApplySizeOf
    {};

    template <class X0, class... X>
    struct _ApplySizeOf<Tuple<X0, X...>>
    {
        using Type = Cat<
            ApplySizeOf<Tuple<X0>>,
            ApplySizeOf<Tuple<X...>>
        >;
    };

    template <class X0>
    struct _ApplySizeOf<Tuple<X0>>
    {
        using Type = SizeT<sizeof(X0)>;
    };

    template <>
    struct _ApplySizeOf<Tuple<>>
    {
        using Type = SizeT<>;
    };

} // namespace _tuple


// These are needed in the body of Tuple - def/spec later on
namespace _tuple {
    // Small tuple-like structure that holds a tuple's values
    template <class... X> struct Values {};

    template <>
    struct Values<> {
        static constexpr size_t Length = 0;
        using AsPack  = Pack<>;
        using AsTuple = Tuple<>;
        template <class I> using At = Values<>;
    };

    // Helper for smart indexing
    template <class TUPLE, class Index> struct HelperGet;

    // Helper for smart indexing
    template <class... X> struct HelperCat;
}

/// ---------------------------------------------------------------- ///
///     Tuple Specialization                                         ///
/// ---------------------------------------------------------------- ///

struct TupleBase {};

/// Non-empty tuple
template <class... X>
struct Tuple: TupleBase
{
    /// Static types and values ////////////////////////////////////////

    static constexpr size_t Length = CountTypes<X...>::Value;

    using ParentType    = TupleBase;
    using ThisType      = Tuple<X...>;
    using ValueType     = _tuple::Values<X...>;
    using NextType      = DelFirst<ThisType>;
    using FirstItem     = GetFirstValue<ThisType>;
    using LastItem      = GetLastValue<ThisType>;

    ////////////////////////////////////////////////////////////////////

    using AsNoRef       = ApplyRemoveRef<ThisType>;
    using AsNoConst     = ApplyRemoveConst<ThisType>;
    using AsNoConstRef  = ApplyRemoveConst<AsNoRef>;
    using AsRef         = ApplyAddRef<AsNoRef>;
    using AsConstRef    = ApplyAddConstRef<AsNoRef>;
    using AsRValueRef   = ApplyAddRValueRef<AsNoRef>;

protected:
    template <class I>      using HelperGet     = _tuple::HelperGet<ThisType,I>;

                            using GetFirstValue = RemoveRef<GetFirstValue<ThisType>>;
                            using GetLastValue  = RemoveRef<GetLastValue<ThisType>>;
    template <class I>      using GetValue      = meta::GetValue<AsNoRef,I>;
    template <class I>      using Get           = typename HelperGet<I>::ReturnType;
    template <class I>      using GetConst      = typename HelperGet<I>::ReturnConstType;
    template <class N=UZ1>  using GetFirst      = meta::GetFirst<ThisType,N::Value>;
    template <class N=UZ1>  using GetLast       = meta::GetLast<ThisType,N::Value>;

    template <class N=UZ1>  using DelFirst      = meta::DelFirst<ThisType,N::Value>;
    template <class N=UZ1>  using DelLast       = meta::DelLast<ThisType,N::Value>;

public:
    ////////////////////////////////////////////////////////////////////

    _MHD_ inline AsRef asRef()
    {
        using N = SizeT<Length-1>;
        return AsRef(head(), tail(N()).asRef()._asValue());
    }

    _MHD_ inline AsConstRef asConstRef() const
    {
        using N = SizeT<Length-1>;
        return AsConstRef(head(), tail(N()).asConstRef()._asValue());
    }

    /// Constexpr methods //////////////////////////////////////////////

    _MHD_ constexpr size_t length() const { return Length; }

    /// Other methods //////////////////////////////////////////////////

public:

    template <class I>              _MHD_ inline const GetValue<I>      &  getValue(I i)       const { return head(Add<I,PtrDiff<1>>()).tail(); }
                                    _MHD_ inline const GetFirstValue    &  getFirstValue()     const { return head(); }
                                    _MHD_ inline const GetLastValue     &  getLastValue()      const { return tail(); }
    template <class I>              _MHD_ inline       GetValue<I>      &  getValue(I i)             { return head(Add<I,PtrDiff<1>>()).tail(); }
                                    _MHD_ inline       GetFirstValue    &  getFirstValue()           { return head(); }
                                    _MHD_ inline       GetLastValue     &  getLastValue()            { return tail(); }

    template <class I>              _MHD_ inline       GetConst<I>         get(I)              const { return HelperGet<I>::get(*this); }
    template <class I>              _MHD_ inline       Get<I>              get(I)                    { return HelperGet<I>::get(*this); }

    template <class N>              _MHD_ inline const GetFirst<N>      &  getFirst(N)         const { return head(N()); }
    template <class N>              _MHD_ inline const GetLast<N>       &  getLast(N)          const { return tail(N()); }
                                    _MHD_ inline const GetFirst<>       &  getFirst()          const { return head(UZ1()); }
                                    _MHD_ inline const GetLast<>        &  getLast()           const { return tail(UZ1()); }

    template <class N>              _MHD_ inline       GetFirst<N>      &  getFirst(N)               { return head(N()); }
    template <class N>              _MHD_ inline       GetLast<N>       &  getLast(N)                { return tail(N()); }
                                    _MHD_ inline       GetFirst<>       &  getFirst()                { return head(UZ1()); }
                                    _MHD_ inline       GetLast<>        &  getLast()                 { return tail(UZ1()); }

    // template <class I>              _MHD_ inline Del<ThisType,I>                del(I)      const { return _tuple_obj::Del<ThisType,I>::del(); }
    // template <class N>              _MHD_ inline DelFirst<ThisType,N::Value>    delFirst(N) const { return del(SimpleSlice<0,N::Value>()); }
    // template <class N>              _MHD_ inline DelLast<ThisType,N::Value>     delLast(N)  const { return del(SimpleSlice<Length-N::Value,Length>()); }
    // template <class N>              _MHD_ inline DelFirst<ThisType>             delFirst()  const { return delFirst(SizeT<1>()); }
    // template <class N>              _MHD_ inline DelLast<ThisType>              delLast()   const { return delLast(SizeT<1>()); }

    // template <class I, class... M>  _MHD_ inline SetFrom<ThisType,I,M...>       setFrom(I, M...)    const { return SetFrom<ThisType,I,M...>(); }
    // template <class... M>           _MHD_ inline SetFirstFrom<ThisType,M...>    setFirstFrom(M...)  const { return SetFirstFrom<ThisType,M...>(); }
    // template <class... M>           _MHD_ inline SetLastFrom<ThisType,M...>     setLastFrom(M...)   const { return SetLastFrom<ThisType,M...>(); }

    // template <class I, class... M>  _MHD_ inline Set<ThisType,I,M...>           set(I, M...)        const { return Set<ThisType,I,M...>(); }
    // template <class... M>           _MHD_ inline SetFirst<ThisType,M...>        setFirst(M...)      const { return SetFirst<ThisType,M...>(); }
    // template <class... M>           _MHD_ inline SetLast<ThisType,M...>         setLast(M...)       const { return SetLast<ThisType,M...>(); }

    // template <class I, class... M>  _MHD_ inline InsertFrom<ThisType,I,M...>    insertFrom(I, M...) const { return InsertFrom<ThisType,I,M...>(); }
    // template <class... M>           _MHD_ inline PrependFrom<ThisType,M...>     prependFrom(M...)   const { return PrependFrom<ThisType,M...>(); }
    // template <class... M>           _MHD_ inline AppendFrom<ThisType,M...>      appendFrom(M...)    const { return AppendFrom<ThisType,M...>(); }

    // template <class I, class... M>  _MHD_ inline Insert<ThisType,I,M...>        insert(I, M...)     const { return Insert<ThisType,I,M...>(); }
    // template <class... M>           _MHD_ inline Prepend<ThisType,M...>         prepend(M...)       const { return Prepend<ThisType,M...>(); }
    // template <class... M>           _MHD_ inline Append<ThisType,M...>          append(M...)        const { return Append<ThisType,M...>(); }
    // template <class... M>           _MHD_ inline Extend<ThisType,M...>          extend(M...)        const { return Extend<ThisType,M...>(); }

    /// Self reference /////////////////////////////////////////////////
public:

    template <class N>
    _MHD_ inline
    const GetFirst<N> & head(N) const
    {
        static_assert(N::Value <= Length, "Too many elements requested");
        return *reinterpret_cast<const GetFirst<N> *>(this);
    }

    template <class N>
    _MHD_ inline
    GetFirst<N> & head(N)
    {
        static_assert(N::Value <= Length, "Too many elements requested");
        return *reinterpret_cast<GetFirst<N> *>(this);
    }

    _MHD_ inline
    const GetFirstValue & head() const
    {
        return _values.FirstValue;
    }

    _MHD_ inline
    GetFirstValue & head()
    {
        return _values.FirstValue;
    }

    template <class N>
    _MHD_ inline
    const GetLast<N> & tail(N) const
    {
        static_assert(N::Value <= Length, "Too many elements requested");
        static constexpr size_t Skip = Length - N::Value;
        return *reinterpret_cast<const GetLast<N> *>(
            &(_values.at(PtrDiff<Skip>()))
        );
    }

    template <class N>
    _MHD_ inline
    GetLast<N> & tail(N)
    {
        static_assert(N::Value <= Length, "Too many elements requested");
        static constexpr size_t Skip = Length - N::Value;
        return *reinterpret_cast<GetLast<N> *>(
            &(_values.at(PtrDiff<Skip>()))
        );
    }

    _MHD_ inline
    const GetLastValue & tail() const
    {
        return tail(UZ1()).head();
    }

    _MHD_ inline
    GetLastValue & tail()
    {
        return tail(UZ1()).head();
    }

    /// Constructors ///////////////////////////////////////////////////

// protected:
//     // Helper constructor to generate asRef/asConstRef
//     template <class Y0>
//     Tuple(Y0&& y0, NextType&& y): _values() {}

//     template <class Y0>
//     Tuple(Y0&& y0, const NextType&& y): _values(std::forward<Y0>(y0), y._values) {}

public:
    // Default
    constexpr Tuple(): _values() {}

    // From values
    template <class... Y>
    Tuple(Y&&... y): _values(std::forward<Y>(y)...) {}

    // Copy constructors
    Tuple(      ThisType &   x): _values(x._values)            {}
    Tuple(const ThisType &   x): _values(x._values)            {}
    Tuple(const ThisType &&  x): _values(x._values)            {}
    Tuple(      ThisType &&  x): _values(std::move(x)._values) {}

    // Copy operator
    _MHD_ inline ThisType & operator=(const ThisType & x)
    {
        _values = std::move(x._values);
        return *this;
    }

    // Move copy operator
    _MHD_ inline ThisType & operator=(ThisType && x)
    {
        _values = std::move(x._values);
        return *this;
    }

protected:
    // Conversion
    _MHD_ inline       ValueType & _asValue ()       { return _values; }
    _MHD_ inline const ValueType & _asValue () const { return _values; }

protected:
    ValueType _values;

    template <class... Y>  friend struct Tuple;
    template <class... Y> friend struct _tuple::HelperCat;
};


// Empty tuple
template <>
struct Tuple<>: TupleBase
{
    /// Static types and values ////////////////////////////////////////

    using ParentType = TupleBase;
    using ThisType   = Tuple<>;
    using ValueType  = _tuple::Values<>;
    static constexpr size_t Length = 0;

    ////////////////////////////////////////////////////////////////////

    using AsNoRef       = ThisType;
    using AsNoConst     = ThisType;
    using AsNoConstRef  = ThisType;
    using AsRef         = ThisType;
    using AsConstRef    = ThisType;
    using AsRValueRef   = ThisType;

    _MHD_ inline       ThisType & asRef()            { return *this; }
    _MHD_ inline const ThisType & asConstRef() const { return *this; }

    /// Constexpr methods ///////////////////////////////////////////////

    _MHD_ constexpr size_t     length()    const { return Length; }

    // template <class I, class... M>  _MHD_ constexpr InsertFrom<ThisType,I,M...>    insertFrom(I, M...) const { return InsertFrom<ThisType,I,M...>(); }
    // template <class... M>           _MHD_ constexpr PrependFrom<ThisType,M...>     prependFrom(M...)   const { return PrependFrom<ThisType,M...>(); }
    // template <class... M>           _MHD_ constexpr AppendFrom<ThisType,M...>      appendFrom(M...)    const { return AppendFrom<ThisType,M...>(); }

    // template <class I, class... M>  _MHD_ constexpr Insert<ThisType,I,M...>        insert(I, M...)     const { return Insert<ThisType,I,M...>(); }
    // template <class... M>           _MHD_ constexpr Prepend<ThisType,M...>         prepend(M...)       const { return Prepend<ThisType,M...>(); }
    // template <class... M>           _MHD_ constexpr Append<ThisType,M...>          append(M...)        const { return Append<ThisType,M...>(); }
    // template <class... M>           _MHD_ constexpr Extend<ThisType,M...>          extend(M...)        const { return Extend<ThisType,M...>(); }

    /// Constructors ///////////////////////////////////////////////////

    constexpr Tuple():                  ParentType() {}
    constexpr Tuple(const ThisType &):  ParentType() {}

protected:
    // Copy to value
    _MHD_ constexpr ValueType _asValue () const { return ValueType(); }

    /// Friends ////////////////////////////////////////////////////////
    template <class... Y> friend struct Tuple;
    template <class... Y> friend struct _tuple::HelperCat;
};


/// ---------------------------------------------------------------- ///
///     Tuple helpers                                                ///
/// ---------------------------------------------------------------- ///

template <class... X>
_MHD_ inline
Tuple<X...> tuple(const X &... x) {
    return Tuple<X...>(x...);
}

_MHD_ constexpr
Tuple<> tuple()  {
    return Tuple<>();
}

namespace tuple {

template <class... X>
_MHD_ inline
Cat<X...> cat(X&&... x) {
    return _tuple::HelperCat<X...>::cat(std::forward<X>(x)...);
}

template <class... X>
_MHD_ inline
typename Cat<X...>::AsRef view(X&... x) {
    return _tuple::HelperCat<X...>::view(x...);
}

template <class... X>
_MHD_ inline
typename Cat<X...>::AsRefConst view(const X&... x) {
    return _tuple::HelperCat<X...>::view(x...);
}

} // namespace tuple


/// ---------------------------------------------------------------- ///
///     Helper for recurive tuple construction (impl)                ///
/// ---------------------------------------------------------------- ///

namespace _tuple {

    /// Cat ////////////////////////////////////////////////////////////

    template <class... X>
    struct HelperCat {};

    template <>
    struct HelperCat<> {
        using ReturnType        = Tuple<>;
        using ReturnView        = Tuple<>;
        using ReturnViewConst   = Tuple<>;
        _MHD_ static constexpr Tuple<> cat()     { return Tuple<>(); }
        _MHD_ static constexpr Tuple<> catview() { return Tuple<>(); }
    };

    template <class X>
    struct HelperCat<X> {
        using ReturnType      = X;
        using ReturnView      = X&;
        using ReturnViewConst = const X&;

        _MHD_ static inline       X &  view(      X&  x) { return x; }
        _MHD_ static inline const X &  view(const X&  x) { return x; }
        _MHD_ static inline       X    cat (const X&  x) { return x; }
        _MHD_ static inline       X && cat (      X&& x) { return std::move(x); }
    };

    template <class X, class Y>
    struct HelperCat<X, Y> {
        using ReturnType      = Cat<X, Y>;
        using ReturnView      = typename ReturnType::AsView;
        using ReturnViewConst = typename ReturnType::AsViewConst;

        template <class X0, class Y0>
        _MHD_ static inline ReturnType cat(X0&& x, Y0&& y)
        {
            return tuple::cat(
                std::forward<X0>(x).getFirst(),
                tuple::cat(
                    std::forward<X0>(x).getLast(SizeT<X::Length-1>()),
                    std::forward<Y0>(y)
                )
            );
        }

        _MHD_ static inline ReturnView view(X& x, Y& y)
        {
            return tuple::view(
                x.getFirst(),
                tuple::view(x.getLast(SizeT<X::Length-1>()), y)
            );
        }

        _MHD_ static inline ReturnViewConst view(const X& x, const Y& y)
        {
            return tuple::view(
                x.getFirst(),
                tuple::view(x.getLast(SizeT<X::Length-1>()), y)
            );
        }
    };

    template <class X, class Y>
    struct HelperCat<Tuple<X>, Y> {
        using ReturnType      = Cat<Tuple<X>, Y>;
        using ReturnView      = typename ReturnType::AsView;
        using ReturnViewConst = typename ReturnType::AsViewConst;

        template <class X0, class Y0>
        _MHD_ static inline ReturnType cat(Tuple<X0>&& x, Y0&& y)
        {
            return ReturnType(
                std::forward<X0>(x).getFirstValue(),
                std::forward<Y0>(y)._asValue()
            );
        }

        _MHD_ static inline ReturnView view(Tuple<X>& x, Y& y)
        {
            return ReturnView(
                x.getFirstValue(),
                y.asRef()._asValue()
            );
        }

        _MHD_ static inline ReturnViewConst view(const Tuple<X>& x, const Y& y)
        {
            return ReturnView(
                x.getFirstValue(),
                y.asConstRef()._asValue()
            );
        }
    };


    template <class Y>
    struct HelperCat<Tuple<>, Y> {
        using ReturnType      = Y;
        using ReturnView      = Y &;
        using ReturnViewConst = const Y &;

        template <class X0, class Y0>
        _MHD_ static inline ReturnType cat(Tuple<X0>&& x, Y0&& y)
        {
            return y;
        }

        _MHD_ static inline ReturnView view(Tuple<> x, Y& y)
        {
            return y;
        }

        _MHD_ static inline ReturnViewConst view(Tuple<> x, const Y& y)
        {
            return y;
        }
    };

    template <class X, class Y, class... Z>
    struct HelperCat<X, Y, Z...> {
        using ReturnType      = Cat<X, Y, Z...>;
        using ReturnView      = typename ReturnType::AsView;
        using ReturnViewConst = typename ReturnType::AsViewConst;

        template <class X0, class Y0, class... Z0>
        _MHD_ static inline ReturnType cat(X0&& x, Y0&& y, Z0&&... z)
        {
            return tuple::cat(
                tuple::cat(std::forward<X0>(x), std::forward<Y0>(y)),
                std::forward<Z0>(z)...
            );
        }

        _MHD_ static inline ReturnView view(X& x, Y& y, Z&... z)
        { return tuple::view(tuple::view(x, y), z...); }

        _MHD_ static inline ReturnViewConst view(const X& x, const Y& y, const Z&... z)
        { return tuple::view(tuple::view(x, y), z...); }
    };

    /// At /////////////////////////////////////////////////////////////

    template <class VALUES, class I,
              ptrdiff_t Idx = WrapIndex<SizeT<VALUES::Length>, I>::Value>
    struct HelperAt {
        using Index     = WrapIndex<SizeT<VALUES::Length>, I>;
        using NextType  = LikeFrom<Values<>, DelFirst<typename VALUES::AsPack>>;
        using ValueType = typename VALUES::template At<Index>;

        _MHD_ static inline ValueType & at(VALUES & t)
        {
            using  NextIndex  = PtrDiff<Index::Value-1>;
            using  NextHelper = HelperAt<NextType, NextIndex>;
            return NextHelper::at(t.NextValues);
        }

        _MHD_ static inline const ValueType & at(const VALUES & t)
        {
            using  NextIndex  = PtrDiff<Index::Value-1>;
            using  NextHelper = HelperAt<NextType, NextIndex>;
            return NextHelper::at(t.NextValues);
        }
    };

    template <class VALUES, class I>
    struct HelperAt<VALUES,I,0> {
        using NextType  = LikeFrom<Values<>, DelFirst<typename VALUES::AsPack>>;
        using ValueType = typename VALUES::template At<I>;

        _MHD_ static inline ValueType & at(VALUES & t)
        {
            return *reinterpret_cast<ValueType *>(&t);
        }

        _MHD_ static inline const ValueType & at(const VALUES & t)
        {
            return *reinterpret_cast<const ValueType *>(&t);
        }
    };

    /// Values /////////////////////////////////////////////////////////

} // namespace _tuple

template <class Y, class... X>
struct _LikeFrom<_tuple::Values<X...>, Y>
{
    using Type = LikeFrom<_tuple::Values<>, Y>;
};

template <class... X>
struct _LikeFrom<_tuple::Values<>, Pack<X...>>
{
    using Type = _tuple::Values<X...>;
};

namespace _tuple {

    template <class X>
    struct Values<X> {
        using ThisType = Values<X>;
        using NextType = Values<>;
        using AsPack   = Pack<X>;
        using AsTuple  = Tuple<X>;
        static constexpr size_t Length = 1;

        template <class I>
        using At = LikeFrom<Values<>, Get<Pack<X>, I>>;

        // From values
        template <class Y0, class...Y>
        Values(Y0&& y0, Y&&... y):   FirstValue(std::forward<Y0>(y0))    {}

        // Copy
        Values(const ThisType &  x): FirstValue(x.FirstValue)            {}
        Values(      ThisType &  x): FirstValue(x.FirstValue)            {}
        Values(const ThisType && x): FirstValue(x.FirstValue)            {}
        Values(      ThisType && x): FirstValue(std::move(x).FirstValue) {}

        _MHD_ inline ThisType & operator=(const ThisType & x)
        {
            FirstValue = x.FirstValue;
            return *this;
        }

        _MHD_ inline ThisType & operator=(ThisType && x)
        {
            FirstValue = std::move(x.FirstValue);
            return *this;
        }

        template <class I>
        _MHD_ At<I> & at(I) {
            // NOTE that this function is sometimes called with I == 1
            // (in which case, At<I> == Values<>). When this happens,
            // we return a reference to ourselves, but disguise it
            // as an empty Values<>. This is definitely not good practice
            // but a hack that I keep for now to get things working.
            //
            // FIXME Tuple.tail(N) should use a helper so that
            // a constexpr Tuple<> is returns when N == 0, and a reference
            // is returned otherwise. In this case, only the case N > 1
            // will call Values.at(I), and we won't have this issue anymore.
            return *reinterpret_cast<At<I>*>(this);
        }

        template <class I>
        _MHD_ const At<I> & at(I) const {
            // Same comment as before.
            return *reinterpret_cast<const At<I>*>(this);
        }

        X FirstValue;
    };

    template <class X0, class X1, class... X>
    struct Values<X0, X1, X...> {
        using ThisType = Values<X0, X1, X...>;
        using NextType = Values<X1, X...>;
        using AsPack   = Pack<X0, X1, X...>;
        using AsTuple  = Tuple<X0, X1, X...>;
        static constexpr size_t Length = Pack<X0, X1, X...>::Length;

        template <class I>
        using At = LikeFrom<Values<>, Get<Pack<X0, X1, X...>, I>>;

        // From values
        template <class Y0, class...Y>
        Values(Y0&& y0, Y&&... y):   FirstValue(std::forward<Y0>(y0)),    NextValues(std::forward<Y>(y)...)   {}

        // Copy
        Values(const ThisType &  x): FirstValue(x.FirstValue),            NextValues(x.NextValues)            {}
        Values(      ThisType &  x): FirstValue(x.FirstValue),            NextValues(x.NextValues)            {}
        Values(const ThisType && x): FirstValue(x.FirstValue),            NextValues(x.NextValues)            {}
        Values(      ThisType && x): FirstValue(std::move(x).FirstValue), NextValues(std::move(x).NextValues) {}

        _MHD_ inline ThisType & operator=(const ThisType & x)
        {
            FirstValue = x.FirstValue;
            NextValues = x.NextValues;
            return *this;
        }

        _MHD_ inline ThisType & operator=(ThisType && x)
        {
            FirstValue = std::move(x.FirstValue);
            NextValues = std::move(x.NextValues);
            return *this;
        }

        template <class I>
        _MHD_ At<I> & at(I) {
            static_assert(WrapIndex<SizeT<Length>,I>::Value < Length, "");
            using  Index = WrapIndex<SizeT<Length>,I>;
            return HelperAt<ThisType,Index>::at(*this);
        }

        template <class I>
        _MHD_ const At<I> & at(I) const {
            static_assert(WrapIndex<SizeT<Length>,I>::Value < Length, "");
            using  Index = WrapIndex<SizeT<Length>,I>;
            return HelperAt<ThisType,Index>::at(*this);
        }

        X0       FirstValue;
        NextType NextValues;
    };

    /// Get ////////////////////////////////////////////////////////////

    // _HelperGet only accepts wrapped indices
    template <class TUPLE, class InputIndex>
    struct _HelperGet {};

    // Many indices -> build tuple of references
    template <class TUPLE, ptrdiff_t I0, ptrdiff_t I1, ptrdiff_t... I>
    struct _HelperGet<TUPLE, PtrDiff<I0, I1, I...>>
    {
        using Index            = PtrDiff<I0, I1, I...>;
        using ReturnType       = typename meta::Get<TUPLE,Index>::AsRef;
        using ReturnTypeConst  = typename meta::Get<TUPLE,Index>::AsConstRef;

        _MHD_ static inline ReturnType get(TUPLE & t) {
            auto first = &t.getValue(PtrDiff<I0>());
            auto other = Get<TUPLE,PtrDiff<I1, I...>>::getValueAsRef(t);
            return ReturnType(*first, other);
        }

        _MHD_ static inline ReturnTypeConst get(const TUPLE & t) {
            auto first = &t.getValue(PtrDiff<I0>());
            auto other = Get<TUPLE,PtrDiff<I1, I...>>::getValueAsRef(t);
            return ReturnTypeConst(*first, other);
        }

        // Aliases for recursive calls

        _MHD_ static inline ReturnType getValueAsRef(TUPLE & t) {
            return get(t);
        }

        _MHD_ static inline ReturnTypeConst getValueAsRef(const TUPLE & t) {
            return get(t);
        }
    };

    // Single index -> can point to the original tuple
    template <class TUPLE, ptrdiff_t I>
    struct _HelperGet<TUPLE, PtrDiff<I>>
    {
        using Index             = PtrDiff<I>;
        using ReturnItem        = Get<TUPLE, Index>;
        using ReturnType        = ReturnItem &;
        using ReturnTypeConst   = const ReturnItem &;

        _MHD_ static inline ReturnType get(TUPLE & t)
        {
            auto v = &t.getValue(Index());
            return *reinterpret_cast<ReturnItem *>(v);
        }

        _MHD_ static inline ReturnTypeConst get(const TUPLE & t)
        {
            auto v = &t.getValue(Index());
            return *reinterpret_cast<const ReturnItem *>(v);
        }

        // Versions that return a tuple of pointers,
        // used when recursively constucting a tuple of pointers.

        using ReturnItemAsRef       = typename ReturnItem::AsRef;
        using ReturnTypeAsRef       = ReturnItemAsRef &;
        using ReturnTypeConstAsRef  = const ReturnItemAsRef &;

        _MHD_ static inline ReturnTypeAsRef getAsRef(TUPLE & t) {
            return ReturnTypeAsRef(t.getValue(PtrDiff<I>()));
        }

        _MHD_ static inline ReturnTypeConstAsRef getasRef(const TUPLE & t) {
            return ReturnTypeConstAsRef(t.getValue(PtrDiff<I>()));
        }

    };

    // General case -> wrap indices and delegate to _HelperGet
    template <class TUPLE, class I>
    struct HelperGet
    {
        using Index           = I;
        using WrappedIndex    = WrapIndex<Length<TUPLE>, Index>;
        using Helper          = _HelperGet<TUPLE,WrappedIndex>;
        using ReturnType      = typename Helper::ReturnType;
        using ReturnTypeConst = typename Helper::ReturnTypeConst;

        _MHD_ static inline ReturnType get(TUPLE & t)
        {
            return Helper::get(t);
        }

        _MHD_ static inline ReturnTypeConst get(const TUPLE & t)
        {
            return Helper::get(t);
        }
    };

    // Slice -> can point to the original tuple
    template <class TUPLE, ptrdiff_t Start, ptrdiff_t Stop>
    struct HelperGet<TUPLE, SimpleSlice<Start, Stop>>
    {
        using Index             = SimpleSlice<Start, Stop>;
        using ReturnItem        = Get<TUPLE, Index>;
        using ReturnType        = ReturnItem &;
        using ReturnTypeConst   = const ReturnItem &;
        using FirstIndex        = GetFirst<WrapIndex<Length<TUPLE>, Index>>;

        _MHD_ static inline ReturnType get(TUPLE & t)
        {
            auto v = &t.getValue(FirstIndex());
            return *reinterpret_cast<ReturnItem *>(v);
        }

        _MHD_ static inline ReturnTypeConst get(const TUPLE & t)
        {
            auto v = &t.getValue(FirstIndex());
            return *reinterpret_cast<const ReturnItem *>(v);
        }
    };

    // Slice -> can point to the original tuple
    template <class TUPLE, class Start, class Stop>
    struct HelperGet<TUPLE, Slice<Start, Stop>>
    {
        using Index             = Slice<Start, Stop>;
        using ReturnItem        = Get<TUPLE, Index>;
        using ReturnType        = ReturnItem &;
        using ReturnTypeConst   = const ReturnItem &;
        using FirstIndex        = GetFirst<WrapIndex<Length<TUPLE>, Index>>;

        _MHD_ static inline ReturnType get(TUPLE & t)
        {
            auto v = &t.getValue(FirstIndex());
            return *reinterpret_cast<ReturnItem *>(v);
        }

        _MHD_ static inline ReturnTypeConst get(const TUPLE & t)
        {
            auto v = &t.getValue(FirstIndex());
            return *reinterpret_cast<const ReturnItem *>(v);
        }
    };

} // namespace _tuple

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
