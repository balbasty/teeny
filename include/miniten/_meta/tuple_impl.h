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
    template <class TUPLE>                 using   Reversed = typename _Reversed<TUPLE>::Type;
    template <class TUPLE, typename U>     struct _AsVector;
    template <class TUPLE, typename U>     using   AsVector = typename _AsVector<TUPLE,U>::Type;
    template <long>                         struct  ApplyFn;
    template <class TUPLE, class APPLY>    struct _Apply;
    template <class TUPLE, class APPLY>    using   Apply = typename _Apply<TUPLE, APPLY>::Type;
    template <class PACK>                 struct _ApplySizeOf;
    template <class PACK>                 using   ApplySizeOf   = typename _ApplySizeOf<PACK>::Type;
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

    /// Revert /////////////////////////////////////////////////////////////

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


/// ---------------------------------------------------------------- ///
///     Tuple object implementation                                  ///
/// ---------------------------------------------------------------- ///

namespace _tuple_obj {

    /// Get ////////////////////////////////////////////////////////////

    // __Get only accepts wrapped indices
    template <class InputTuple, class InputIndex>
    struct __Get {};

    template <class InputTuple, ptrdiff_t I>
    struct __Get<InputTuple, PtrDiff<I>>
    {
        using InputIndex   = PtrDiff<I>;
        // using ReturnType   = meta::GetValue<InputTuple,InputIndex>;
        using FirstTuple   = meta::GetFirst<InputTuple,I+1>;
        using ReturnTuple  = meta::GetLast<FirstTuple>;
        using ReturnType   = meta::GetLastValue<FirstTuple>;

        _MHD_
        static inline ReturnType getValue(const InputTuple & t) {
            const FirstTuple * f = reinterpret_cast<const FirstTuple *>(&t);
            return f->getLastValue();
        }

        _MHD_
        static inline ReturnTuple get(const InputTuple & t) {
            return tuple(getValue(t));
        }
    };

    template <class InputTuple, ptrdiff_t I0, ptrdiff_t... I>
    struct __Get<InputTuple, PtrDiff<I0, I...>>
    {
        using InputIndex   = PtrDiff<I0, I...>;
        using ReturnTuple  = meta::Get<InputTuple,InputIndex>;
        using WrappedIndex = WrapIndex<Length<InputTuple>, InputIndex>;

        _MHD_
        static inline ReturnTuple get(const InputTuple & t) {
            auto first = Get<InputTuple,PtrDiff<I0>>::getValue(t);
            auto other = Get<InputTuple,PtrDiff<I...>>::get(t);
            return ReturnTuple(first, other);
        }
    };

    // _Get Wraps indices and calls __Get
    template <class InputTuple, class InputIndex>
    struct _Get {
        using WrappedIndex = WrapIndex<Length<InputTuple>, InputIndex>;
        using WrappedGet   = Get<InputTuple,WrappedIndex>;
        using Type         = __Get<InputTuple, WrappedIndex>;
    };

    // Get is a pretty alias for _Get::Type == __Get
    template <class InputTuple, class InputIndex>
    using Get = typename _Get<InputTuple, InputIndex>::Type;

}

/// ---------------------------------------------------------------- ///
///     Tuple helpers                                                ///
/// ---------------------------------------------------------------- ///

template <class... X>
_MHD_ inline
Tuple<X...> tuple(const X &... x) {
    return Tuple<X...>(x...);
}

namespace _tuple {

    // Specialized later on
    template <class TUPLE> struct Helper {};

    template <class T>
    _MHD_ inline
    const typename T::ParentType & asParentRef(const T & t)
    {
        return t.head(SizeT<T::Length-1>());
    }

    template <class T>
    _MHD_ inline
    typename T::ParentType & asParentRef(T & t)
    {
        return t.head(SizeT<T::Length-1>());
    }

    template <class T>
    _MHD_ inline
    typename T::ParentType asParent(const T & t)
    {
        return ParentType(t.asParentRef());
    }
}

/// ---------------------------------------------------------------- ///
///     Tuple Specialization                                         ///
/// ---------------------------------------------------------------- ///

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

    using ParentType = TupleBase;
    using ThisType   = Tuple<>;
    static constexpr size_t Length = 0;

    ////////////////////////////////////////////////////////////////////

    using AsNoRef       = ThisType;
    using AsNoConst     = ThisType;
    using AsNoConstRef  = ThisType;
    using AsRef         = ThisType;
    using AsConstRef    = ThisType;
    using AsRValueRef   = ThisType;

    _MHD_ inline       ThisType & asRef()            { return *this; }
    _MHD_ inline       ThisType & asRValueRef()      { return *this; }
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

    template <class... Y> friend struct Tuple;
};

/// Non-empty tuple
template <class X0, class... X>
struct Tuple<X0, X...>: public DelLast<Tuple<X0, X...>>
{
    /// Static types and values ////////////////////////////////////////

    using ParentType    = DelLast<Tuple<X0, X...>>;
    using ThisType      = Tuple<X0, X...>;
    using NextType      = Tuple<X...>;
    using FirstItem     = RemoveRef<GetFirstValue<ThisType>>;
    using LastItem      = RemoveRef<GetLastValue<ThisType>>;
    using Helper        = _tuple::Helper<ThisType>;

    static constexpr size_t Length = CountTypes<X0, X...>::Value;

    ////////////////////////////////////////////////////////////////////

    using AsNoRef       = ApplyRemoveRef<ThisType>;
    using AsNoConst     = ApplyRemoveConst<ThisType>;
    using AsNoConstRef  = ApplyRemoveConst<AsNoRef>;
    using AsRef         = ApplyAddRef<AsNoRef>;
    using AsConstRef    = ApplyAddConstRef<AsNoRef>;
    using AsRValueRef   = ApplyAddRValueRef<AsNoRef>;

    _MHD_ inline AsRef asRef()
    {
        using N = SizeT<Length-1>;
        return AsRef(head(), tail(N()).asRef());
    }

    _MHD_ inline AsConstRef asConstRef() const
    {
        using N = SizeT<Length-1>;
        return AsConstRef(head(), tail(N()).asConstRef());
    }

    _MHD_ inline AsRValueRef asRValueRef() const
    {
        using N = SizeT<Length-1>;
        return AsRValueRef(head(), tail(N()).asRValueRef());
    }

    /// Constexpr methods ///////////////////////////////////////////////

    _MHD_ constexpr size_t     length()    const { return Length; }

    /// Other methods //////////////////////////////////////////////////

    template <class I>              _MHD_ inline const GetValue<AsNoRef,I>        &  getValue(I i)       const { return head(Add<I,PtrDiff<1>>()).tail(); }
                                    _MHD_ inline const GetFirstValue<AsNoRef>     &  getFirstValue()     const { return head(); }
                                    _MHD_ inline const GetLastValue<AsNoRef>      &  getLastValue()      const { return tail(); }
    template <class I>              _MHD_ inline       GetValue<AsNoRef,I>        &  getValue(I i)             { return head(Add<I,PtrDiff<1>>()).tail(); }
                                    _MHD_ inline       GetFirstValue<AsNoRef>     &  getFirstValue()           { return head(); }
                                    _MHD_ inline       GetLastValue<AsNoRef>      &  getLastValue()            { return tail(); }

    template <class I>              _MHD_ inline       Get<ThisType,I>                get(I)              const { return _tuple_obj::Get<ThisType,I>::get(*this); }
    template <class N>              _MHD_ inline const GetFirst<ThisType,N::Value> &  getFirst(N)         const { return head(N()); }
    template <class N>              _MHD_ inline const GetLast<ThisType,N::Value>  &  getLast(N)          const { return tail(N()); }
                                    _MHD_ inline const GetFirst<ThisType>          &  getFirst()          const { return head(UZ1()); }
                                    _MHD_ inline const GetLast<ThisType>           &  getLast()           const { return tail(UZ1()); }
    template <class N>              _MHD_ inline       GetFirst<ThisType,N::Value> &  getFirst(N)               { return head(N()); }
    template <class N>              _MHD_ inline       GetLast<ThisType,N::Value>  &  getLast(N)                { return tail(N()); }
                                    _MHD_ inline       GetFirst<ThisType>          &  getFirst()                { return head(UZ1()); }
                                    _MHD_ inline       GetLast<ThisType>           &  getLast()                 { return tail(UZ1()); }

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

    template <class N>
    _MHD_ inline
    const GetFirst<ThisType, N::Value> & head(N) const
    {
        static_assert(N::Value <= Length, "Too many elements requested");
        return *reinterpret_cast<const GetFirst<ThisType, N::Value> *>(this);
    }

    template <class N>
    _MHD_ inline
    GetFirst<ThisType, N::Value> & head(N)
    {
        static_assert(N::Value <= Length, "Too many elements requested");
        return *reinterpret_cast<GetFirst<ThisType, N::Value> *>(this);
    }

    _MHD_ inline
    const FirstItem & head() const
    {
        return *reinterpret_cast<const FirstItem *>(this);
    }

    _MHD_ inline
    FirstItem & head()
    {
        return *reinterpret_cast<FirstItem *>(this);
    }

    template <class N>
    _MHD_ inline
    const GetLast<ThisType, N::Value> & tail(N) const
    {
        static_assert(N::Value <= Length, "Too many elements requested");
        using ReturnType = GetLast<ThisType, N::Value>;
        static constexpr size_t HeadLength = (
            Length+1-N::Value > Length ? Length : Length+1-N::Value
        );
        return *reinterpret_cast<const ReturnType *>(
            &(head(PtrDiff<HeadLength>()).tail())
        );
    }

    template <class N>
    _MHD_ inline
    GetLast<ThisType, N::Value> & tail(N)
    {
        static_assert(N::Value <= Length, "Too many elements requested");
        using ReturnType = GetLast<ThisType, N::Value>;
        static constexpr size_t HeadLength = (
            Length+1-N::Value > Length ? Length : Length+1-N::Value
        );
        return *reinterpret_cast<ReturnType *>(
            &(head(PtrDiff<HeadLength>()).tail())
        );
    }

    _MHD_ inline
    const LastItem & tail() const
    {
        return Value;
    }

    _MHD_ inline
    LastItem & tail()
    {
        return Value;
    }

    /// Constructors ///////////////////////////////////////////////////

    // Default
    constexpr Tuple(): ParentType(), Value() {}

    // Series of values || first value + tuple (for recursive construction)
    template <class Y0, class... Y>
    Tuple(Y0 y0, Y... y):       ParentType(Helper::getFirstValues(y0, y...)), Value(Helper::getLastValue(y0, y...)) {}

    // Copy constructors
    Tuple(      ThisType &   x): ParentType(_tuple::asParentRef(x)), Value(x.getLastValue()) {}
    Tuple(      ThisType &&  x): ParentType(_tuple::asParentRef(x)), Value(x.getLastValue()) {}
    Tuple(const ThisType &   x): ParentType(_tuple::asParentRef(x)), Value(x.getLastValue()) {}
    Tuple(const ThisType &&  x): ParentType(_tuple::asParentRef(x)), Value(x.getLastValue()) {}

    // Copy operator
    _MHD_ inline ThisType & operator=(const ThisType & x)
    {
        Value = x.getLastValue();
        ParentType::operator=(_tuple::asParentRef(x));
        return *this;
    }

    // Move copy operator
    _MHD_ inline ThisType & operator=(ThisType && x)
    {
        Value = std::move(x.getLastValue());
        ParentType::operator=(std::move(_tuple::asParentRef(x)));
        return *this;
    }

protected:
    LastItem Value;

    template <class... Y>  friend struct Tuple;
    template <class TUPLE> friend struct _tuple::Helper;
};

/// ---------------------------------------------------------------- ///
///     Helper for recurive tuple construction (impl)                ///
/// ---------------------------------------------------------------- ///

namespace _tuple {

    //// Tuple of Values ///////////////////////////////////////////////

    template <class X0, class Y0, class... Y>
    struct Helper<Tuple<X0, Y0, Y...>>
    {
        using ThisType   = Tuple<X0, Y0, Y...>;
        using ParentType = DelLast<ThisType>;
        using FirstItem  = GetFirstValue<ThisType>;
        using LastItem   = GetLastValue <ThisType>;

        _MHD_ static inline
        const LastItem & getLastValue(const X0 & x0, const Y0 & y0, const Y &... y)
        {
            using NextHelper = Helper<DelFirst<ThisType>>;
            return NextHelper::getLastValue(y0, y...);
        }

        _MHD_ static inline
        LastItem && getLastValue(const X0 && x0, Y0 && y0, Y &&... y)
        {
            using NextHelper = Helper<DelFirst<ThisType>>;
            LastItem && last = NextHelper::getLastValue(std::move(y0), std::move(y)...);
            return std::move(last);
        }

        _MHD_ static inline
        const LastItem & getLastValue(const X0 & x0, const Tuple<Y0, Y...> & y)
        {
            return y.getLastValue();
        }

        _MHD_ static inline
        LastItem && getLastValue(const X0 && x0, Tuple<Y0, Y...> && y)
        {
            return std::move(y.getLastValue());
        }

        _MHD_ static inline
        ParentType getFirstValues(const X0 & x0, const Y0 & y0, const Y &... y)
        {
            using NextType = DelFirst<ThisType>;
            return ParentType(x0, _tuple::asParentRef(NextType(y0, y...)));
        }

        _MHD_ static inline
        ParentType getFirstValues(X0 && x0, Y0 && y0, Y &&... y)
        {
            using NextType = DelFirst<ThisType>;
            return ParentType(std::move(x0), std::move(_tuple::asParentRef(NextType(y0, y...))));
        }

        _MHD_ static inline
        ParentType getFirstValues(const X0 & x0, const Tuple<Y0, Y...> & y)
        {
            return ParentType(x0, _tuple::asParentRef(y));
        }

        _MHD_ static inline
        ParentType getFirstValues(X0 && x0, Tuple<Y0, Y...> && y)
        {
            return ParentType(std::move(x0), std::move(_tuple::asParentRef(y)));
        }
    };

    template <class X0>
    struct Helper<Tuple<X0>>
    {
        using ThisType   = Tuple<X0>;
        using ParentType = Tuple<>;
        using FirstItem  = X0;
        using LastItem   = X0;

        _MHD_ static inline
        const LastItem & getLastValue(const X0 & x0)
        {
            return x0;
        }

        _MHD_ static inline
        LastItem && getLastValue(X0 && x0)
        {
            return std::move(x0);
        }

        _MHD_ static inline
        const LastItem & getLastValue(const X0 & x0, const Tuple<> & y)
        {
            return x0;
        }

        _MHD_ static constexpr
        ParentType getFirstValues(const X0 & x0)
        {
            return ParentType();
        }

        _MHD_ static constexpr
        ParentType getFirstValues(const X0 && x0)
        {
            return ParentType();
        }

        _MHD_ static inline
        LastItem && getLastValue(X0 && x0, const Tuple<> && y)
        {
            return std::move(x0);
        }

        _MHD_ static constexpr
        ParentType getFirstValues(const X0 & x0, const Tuple<> & y)
        {
            return ParentType();
        }

        _MHD_ static constexpr
        ParentType getFirstValues(const X0 && x0, const Tuple<> && y)
        {
            return ParentType();
        }
    };

    //// Tuple of References ///////////////////////////////////////////

    template <class X0, class Y0, class... Y>
    struct Helper<Tuple<X0&, Y0&, Y&...>>
    {
        using ThisType   = Tuple<X0&, Y0&, Y&...>;
        using ParentType = DelLast<ThisType>;
        using FirstItem  = RemovePtr<GetFirstValue<ThisType>>;
        using LastItem   = RemovePtr<GetLastValue <ThisType>>;

        _MHD_ static inline
        const LastItem & getLastValue(X0 & x0, Y0 & y0, Y &... y)
        {
            using NextHelper = Helper<DelFirst<ThisType>>;
            return NextHelper::getLastValue(y0, y...);
        }

        _MHD_ static inline
        LastItem && getLastValue(const X0 && x0, Y0 && y0, Y &&... y)
        {
            using NextHelper = Helper<DelFirst<ThisType>>;
            LastItem && last = NextHelper::getLastValue(std::move(y0), std::move(y)...);
            return std::move(last);
        }

        _MHD_ static inline
        LastItem & getLastValue(const X0 & x0, Tuple<Y0&, Y&...> & y)
        {
            return y.getLastValue();
        }

        _MHD_ static inline
        const LastItem & getLastValue(const X0 & x0, const Tuple<Y0&, Y&...> & y)
        {
            return y.getLastValue();
        }

        _MHD_ static inline
        LastItem && getLastValue(const X0 && x0, Tuple<Y0&, Y&...> && y)
        {
            return std::move(y.getLastValue());
        }

        _MHD_ static inline
        ParentType getFirstValues(X0 & x0, Y0 & y0, Y &... y)
        {
            using NextType = DelFirst<ThisType>;
            return ParentType(x0, _tuple::asParentRef(NextType(y0, y...)));
        }

        _MHD_ static inline
        ParentType getFirstValues(X0 && x0, Y0 && y0, Y &&... y)
        {
            using NextType = DelFirst<ThisType>;
            return ParentType(std::move(x0), std::move(_tuple::asParentRef(NextType(y0, y...))));
        }

        _MHD_ static inline
        ParentType getFirstValues(X0 & x0, Tuple<Y0&, Y&...> & y)
        {
            return ParentType(x0, _tuple::asParentRef(y));
        }

        _MHD_ static inline
        ParentType getFirstValues(X0 & x0, const Tuple<Y0&, Y&...> & y)
        {
            return ParentType(x0, _tuple::asParentRef(y));
        }

        _MHD_ static inline
        ParentType getFirstValues(X0 && x0, Tuple<Y0&, Y&...> && y)
        {
            return ParentType(std::move(x0), std::move(_tuple::asParentRef(y)));
        }
    };

    template <class X0>
    struct Helper<Tuple<X0&>>
    {
        using ThisType   = Tuple<X0&>;
        using ParentType = Tuple<>;
        using FirstItem  = X0;
        using LastItem   = X0;

        _MHD_ static inline
        LastItem & getLastValue(X0 & x0)
        {
            return x0;
        }

        _MHD_ static inline
        LastItem && getLastValue(X0 && x0)
        {
            return std::move(x0);
        }

        _MHD_ static inline
        LastItem & getLastValue(X0 & x0, const Tuple<> & y)
        {
            return x0;
        }

        _MHD_ static inline
        LastItem && getLastValue(X0 && x0, const Tuple<> && y)
        {
            return std::move(x0);
        }

        _MHD_ static constexpr
        ParentType getFirstValues(X0 & x0)
        {
            return ParentType();
        }

        _MHD_ static constexpr
        ParentType getFirstValues(X0 && x0)
        {
            return ParentType();
        }

        _MHD_ static constexpr
        ParentType getFirstValues(X0 & x0, const Tuple<> & y)
        {
            return ParentType();
        }

        _MHD_ static constexpr
        ParentType _getFirstValues(const X0 && x0, const Tuple<> && y)
        {
            return ParentType();
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
