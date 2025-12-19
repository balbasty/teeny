/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file implements the "pack API" for the type `Tuple`                **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef MINITEN__TUPLE_IMPL
#define MINITEN__TUPLE_IMPL
#include <utility>
#include <miniten/core.h>
#include <miniten/disp.h>
#include <miniten/meta.h>

NAMESPACE_BEGIN(miniten)

/* ------------------------------------------------------------------ *
 *     Helpers                                                        *
 * ------------------------------------------------------------------ */
// These are needed in the body of Tuple - def/spec later on

NAMESPACE_BEGIN(_tuple)

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

// Helper for concat
template <class... X> struct HelperCat;

// Helper for access to first/last element sequences
template <class TUPLE, class N, size_t NValue = N::Value>
struct HelperHeadTail;

template <class A>
struct CastAs;

NAMESPACE_END(_tuple)

/* ------------------------------------------------------------------ *
 *     Tuple Specialization                                           *
 * ------------------------------------------------------------------ */

struct TupleBase {};

/// Non-empty tuple
template <class... X>
struct Tuple: TupleBase
{
    /// Static types and values ////////////////////////////////////////

    using                   LengthT = CountTypes<X...>;
    MINIDEF(S,CX) size_t    Length  = LengthT::Value;

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
    template <class N=UZ1>  using HelperHeadTail= _tuple::HelperHeadTail<ThisType,N>;

                            using GetFirstValue = RemoveRef<meta::GetFirstValue<ThisType>>;
                            using GetLastValue  = RemoveRef<meta::GetLastValue<ThisType>>;
    template <class I>      using GetValue      = meta::GetValue<AsNoRef,I>;

    template <class I>      using Get           = typename HelperGet<I>::ReturnType;
    template <class I>      using GetConst      = typename HelperGet<I>::ReturnConstType;

    template <class N=UZ1>  using GetFirst      = typename HelperHeadTail<N>::HeadType;
    template <class N=UZ1>  using GetLast       = typename HelperHeadTail<N>::TailType;
    template <class N=UZ1>  using GetFirstConst = typename HelperHeadTail<N>::HeadConstType;
    template <class N=UZ1>  using GetLastConst  = typename HelperHeadTail<N>::TailConstType;

    template <class I>      using Del           = Get       <IndexComplement<LengthT,I>>;
    template <class I>      using DelConst      = GetConst  <IndexComplement<LengthT,I>>;

    template <class N=UZ1>  using DelFirst      = GetLast       <SizeT<Length-N::Value>>;
    template <class N=UZ1>  using DelLast       = GetFirst      <SizeT<Length-N::Value>>;
    template <class N=UZ1>  using DelFirstConst = GetLastConst  <SizeT<Length-N::Value>>;
    template <class N=UZ1>  using DelLastConst  = GetFirstConst <SizeT<Length-N::Value>>;

public:
    ////////////////////////////////////////////////////////////////////

    MINIDEF(H,D,I) AsRef asRef()
    {
        using N = SizeT<Length-1>;
        return AsRef(getFirstValue(), getLast(N()).asRef()._asValue());
    }

    MINIDEF(H,D,I) AsConstRef asConstRef() const
    {
        using N = SizeT<Length-1>;
        return AsConstRef(getFirstValue(), getLast(N()).asConstRef()._asValue());
    }

    /// Constexpr methods //////////////////////////////////////////////

    _MHD_ constexpr size_t length() const { return Length; }

    /// Other methods //////////////////////////////////////////////////

    template <class I>              MINIDEF(H,D,I) const GetValue<I>      &  getValue(I i)       const { return getFirst(Add<I,Z1>()).getLastValue(); }
                                    MINIDEF(H,D,I) const GetFirstValue    &  getFirstValue()     const { return _values.FirstValue; }
                                    MINIDEF(H,D,I) const GetLastValue     &  getLastValue()      const { return getLast(UZ1()).getFirstValue(); }
    template <class I>              MINIDEF(H,D,I)       GetValue<I>      &  getValue(I i)             { return getFirst(Add<I,Z1>()).getLastValue(); }
                                    MINIDEF(H,D,I)       GetFirstValue    &  getFirstValue()           { return _values.FirstValue; }
                                    MINIDEF(H,D,I)       GetLastValue     &  getLastValue()            { return getLast(UZ1()).getFirstValue(); }

    template <class I>              MINIDEF(H,D,I)       GetConst<I>         get(I)              const { return HelperGet<I>::get(*this); }
    template <class I>              MINIDEF(H,D,I)       Get<I>              get(I)                    { return HelperGet<I>::get(*this); }

    template <class N>              MINIDEF(H,D,I)       GetFirstConst<N>    getFirst(N)         const { return HelperHeadTail<N>::head(*this); }
    template <class N>              MINIDEF(H,D,I)       GetLastConst<N>     getLast(N)          const { return HelperHeadTail<N>::tail(*this); }
                                    MINIDEF(H,D,I)       GetFirstConst<>     getFirst()          const { return HelperHeadTail< >::head(*this); }
                                    MINIDEF(H,D,I)       GetLastConst<>      getLast()           const { return HelperHeadTail< >::tail(*this); }

    template <class N>              MINIDEF(H,D,I)       GetFirst<N>         getFirst(N)               { return HelperHeadTail<N>::head(*this); }
    template <class N>              MINIDEF(H,D,I)       GetLast<N>          getLast(N)                { return HelperHeadTail<N>::tail(*this); }
                                    MINIDEF(H,D,I)       GetFirst<>          getFirst()                { return HelperHeadTail< >::head(*this); }
                                    MINIDEF(H,D,I)       GetLast<>           getLast()                 { return HelperHeadTail< >::tail(*this); }

    template <class I>              MINIDEF(H,D,I)       DelConst<I>         del(I)              const { return get(IndexComplement<LengthT,I>()); }
    template <class I>              MINIDEF(H,D,I)       Del<I>              del(I)                    { return get(IndexComplement<LengthT,I>()); }

    template <class N>              MINIDEF(H,D,I)       DelFirstConst<N>    delFirst(N)          const { return getLast (UZ<Length-N::Value>()); }
    template <class N>              MINIDEF(H,D,I)       DelLastConst<N>     delLast(N)           const { return getFirst(UZ<Length-N::Value>()); }
                                    MINIDEF(H,D,I)       DelFirstConst<>     delFirst()           const { return getLast (UZ<Length-1>()); }
                                    MINIDEF(H,D,I)       DelLastConst<>      delLast()            const { return getFirst(UZ<Length-1>()); }

    template <class N>              MINIDEF(H,D,I)       DelFirst<N>         delFirst(N)                { return getLast (UZ<Length-N::Value>()); }
    template <class N>              MINIDEF(H,D,I)       DelLast<N>          delLast(N)                 { return getFirst(UZ<Length-N::Value>()); }
                                    MINIDEF(H,D,I)       DelFirst<>          delFirst()                 { return getLast (UZ<Length-1>()); }
                                    MINIDEF(H,D,I)       DelLast<>           delLast()                  { return getFirst(UZ<Length-1>()); }

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

    /// Constructors ///////////////////////////////////////////////////

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
    template <class Y>
    MINIDEF(H,D,I) ThisType & operator=(Y && y)
    {
        static_assert(IsTuple<Y>(), "Only tuples can be assigned into tuples");
        _values = std::forward<Y>(y).as(Type<ThisType>())._values;
        return *this;
    }

    // Conversion
    template <class T> using As      = typename _tuple::template CastAs<T>::template Type<      ThisType>;
    template <class T> using AsConst = typename _tuple::template CastAs<T>::template Type<const ThisType>;

    template <class T> MINIDEF(H,D,I) As<T>      as(meta::Type<T>)       { return _tuple::CastAs<T>::as(*this); }
    template <class T> MINIDEF(H,D,I) AsConst<T> as(meta::Type<T>) const { return _tuple::CastAs<T>::as(*this); }

    // Conversion
    MINIDEF(H,D,I)       ValueType & _asValue ()       { return _values; }
    MINIDEF(H,D,I) const ValueType & _asValue () const { return _values; }

// protected:
    ValueType _values;

    template <class... Y>                   friend struct Tuple;
    template <class... Y>                   friend struct _tuple::HelperCat;
    template <class T, class N, size_t V>   friend struct _tuple::HelperHeadTail;
};


// Empty tuple
template <>
struct Tuple<>: TupleBase
{
    /// Static types and values ////////////////////////////////////////

    using ParentType = TupleBase;
    using ThisType   = Tuple<>;
    using ValueType  = _tuple::Values<>;
    MINIDEF(S,CX) size_t Length = 0;

    ////////////////////////////////////////////////////////////////////

    using AsNoRef       = ThisType;
    using AsNoConst     = ThisType;
    using AsNoConstRef  = ThisType;
    using AsRef         = ThisType;
    using AsConstRef    = ThisType;
    using AsRValueRef   = ThisType;

    MINIDEF(H,D,I)       ThisType & asRef()            { return *this; }
    MINIDEF(H,D,I) const ThisType & asConstRef() const { return *this; }

    /// Constexpr methods ///////////////////////////////////////////////

    MINIDEF(H,D,CX) size_t     length()    const { return Length; }

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

    // Copy to value
    MINIDEF(H,D,CX) ValueType _asValue () const { return ValueType(); }

    /// Friends ////////////////////////////////////////////////////////
    template <class... Y> friend struct Tuple;
    template <class... Y> friend struct _tuple::HelperCat;
};

/// ---------------------------------------------------------------- ///
///     Tuple helpers                                                ///
/// ---------------------------------------------------------------- ///

template <class... X>
MINIDEF(H,D,I)
Tuple<X...> tuple(const X &... x) {
    return Tuple<X...>(x...);
}

MINIDEF(H,D,CX)
Tuple<> tuple()  {
    return Tuple<>();
}

NAMESPACE_BEGIN(tuple)

template <class... X>
MINIDEF(H,D,I)
Cat<X...> cat(X&&... x) {
    return _tuple::HelperCat<X...>::cat(std::forward<X>(x)...);
}

template <class... X>
MINIDEF(H,D,I)
typename Cat<X...>::AsRef view(X&... x) {
    return _tuple::HelperCat<X...>::view(x...);
}

template <class... X>
MINIDEF(H,D,I)
typename Cat<X...>::AsRefConst view(const X&... x) {
    return _tuple::HelperCat<X...>::view(x...);
}

NAMESPACE_END(tuple)


/// ---------------------------------------------------------------- ///
///     Helper for recurive tuple construction (impl)                ///
/// ---------------------------------------------------------------- ///

NAMESPACE_BEGIN(_tuple)

    /// Head/Tail //////////////////////////////////////////////////////

    template <class TUPLE, class N, size_t NValue>
    struct HelperHeadTail {
        using _HeadType     = GetFirst<TUPLE, N::Value>;
        using HeadType      =       _HeadType &;
        using HeadConstType = const _HeadType &;
        using _TailType     = GetLast<TUPLE, N::Value>;
        using TailType      =       _TailType &;
        using TailConstType = const _TailType &;
        static constexpr size_t Length = TUPLE::Length;

        MINIDEF(H,D,S,I)
        HeadConstType head(const TUPLE & t)
        {
            static_assert(N::Value <= Length, "Too many elements requested");
            return reinterpret_cast<HeadConstType>(t);
        }

        MINIDEF(H,D,S,I)
        HeadType head(TUPLE & t)
        {
            static_assert(N::Value <= Length, "Too many elements requested");
            return reinterpret_cast<HeadType>(t);
        }

        MINIDEF(H,D,S,I)
        TailConstType tail(const TUPLE & t)
        {
            static_assert(N::Value <= Length, "Too many elements requested");
            static constexpr size_t Skip = Length - N::Value;
            return reinterpret_cast<TailConstType>(t._values.at(PtrDiff<Skip>()));
        }

        MINIDEF(H,D,S,I)
        TailType tail(TUPLE & t)
        {
            static_assert(N::Value <= Length, "Too many elements requested");
            static constexpr size_t Skip = Length - N::Value;
            return reinterpret_cast<TailType>(t._values.at(PtrDiff<Skip>()));
        }
    };

    template <class TUPLE, class N>
    struct HelperHeadTail<TUPLE, N, 0> {
        using HeadType      = Tuple<>;
        using HeadConstType = Tuple<>;
        using TailType      = Tuple<>;
        using TailConstType = Tuple<>;
        MINIDEF(S,CX) size_t Length = TUPLE::Length;

        MINIDEF(H,D,S,CX)
        HeadType head(const TUPLE &)
        {
            return HeadType();
        }

        MINIDEF(H,D,S,CX)
        TailType tail(const TUPLE &)
        {
            return TailType();
        }
    };

    /// Cat ////////////////////////////////////////////////////////////

    template <class... X>
    struct HelperCat {};

    template <>
    struct HelperCat<> {
        using ReturnType        = Tuple<>;
        using ReturnView        = Tuple<>;
        using ReturnViewConst   = Tuple<>;
        MINIDEF(H,D,S,CX) Tuple<> cat()     { return Tuple<>(); }
        MINIDEF(H,D,S,CX) Tuple<> catview() { return Tuple<>(); }
    };

    template <class X>
    struct HelperCat<X> {
        using ReturnType      = X;
        using ReturnView      = X&;
        using ReturnViewConst = const X&;

        MINIDEF(H,D,S,I)       X &  view(      X&  x) { return x; }
        MINIDEF(H,D,S,I) const X &  view(const X&  x) { return x; }
        MINIDEF(H,D,S,I)       X    cat (const X&  x) { return x; }
        MINIDEF(H,D,S,I)       X && cat (      X&& x) { return std::move(x); }
    };

    template <class X, class Y>
    struct HelperCat<X, Y> {
        using ReturnType      = Cat<X, Y>;
        using ReturnView      = typename ReturnType::AsView;
        using ReturnViewConst = typename ReturnType::AsViewConst;

        template <class X0, class Y0>
        MINIDEF(H,D,S,I) ReturnType cat(X0&& x, Y0&& y)
        {
            return tuple::cat(
                std::forward<X0>(x).getFirst(),
                tuple::cat(
                    std::forward<X0>(x).getLast(SizeT<X::Length-1>()),
                    std::forward<Y0>(y)
                )
            );
        }

        MINIDEF(H,D,S,I) ReturnView view(X& x, Y& y)
        {
            return tuple::view(
                x.getFirst(),
                tuple::view(x.getLast(SizeT<X::Length-1>()), y)
            );
        }

        MINIDEF(H,D,S,I) ReturnViewConst view(const X& x, const Y& y)
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
        MINIDEF(H,D,S,I) ReturnType cat(Tuple<X0>&& x, Y0&& y)
        {
            return ReturnType(
                std::forward<X0>(x).getFirstValue(),
                std::forward<Y0>(y)._asValue()
            );
        }

        MINIDEF(H,D,S,I) ReturnView view(Tuple<X>& x, Y& y)
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
            static_assert(Idx < VALUES::Length, "");
            using  NextIndex  = PtrDiff<Index::Value-1>;
            using  NextHelper = HelperAt<NextType, NextIndex>;
            return NextHelper::at(t.NextValues);
        }

        _MHD_ static inline const ValueType & at(const VALUES & t)
        {
            static_assert(Idx < VALUES::Length, "");
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

    /// CastAs /////////////////////////////////////////////////////////

    template <class A, class B> struct _CastAsType {};
    template <class A, class B> using   CastAsType =
        typename _CastAsType<
            ApplyRemoveRef<RemoveRef<A>>,
            ApplyRemoveRef<RemoveRef<B>>
        >::Type;

    // const Tuple -> Tuple<const>
    template <class X, class Y>
    struct _CastAsType<X, const Y>
    { using Type = CastAsType<X, ApplyAddConst<Y>>; };

    // empty -> empty
    template <class X>
    struct _CastAsType<Tuple<>, X>
    { using Type = Tuple<>; };

    // Single element
    template <class X, class Y>
    struct _CastAsType<Tuple<X>, Tuple<Y>>
    {
        using SameBaseType = IsSame<RemoveConst<RemoveRef<X>>, RemoveConst<RemoveRef<Y>>>;
        using Type = Tuple<IfElse<
            Not<SameBaseType>,
                RemoveConst<RemoveRef<X>>,
            IsConst<Y>,
                AddConstRef<RemoveRef<X>>,
            // Else
                AddRef<X>
        >>;
    };

    // several elements -> unwrap
    template <class T, class X0, class X1, class... X>
    struct _CastAsType<Tuple<X0, X1, X...>, T>
    {
        using Type = Cat<
            CastAsType<Tuple<X0>,       GetFirst<T>>,
            CastAsType<Tuple<X1, X...>, DelFirst<T>>
        >;
    };

    // single element converted to several elements
    template <class Y, class X0, class X1, class... X>
    struct _CastAsType<Tuple<X0, X1, X...>, Tuple<Y>>
    {
        using Type = Cat<
            CastAsType<Tuple<X0>,       Tuple<Y>>,
            CastAsType<Tuple<X1, X...>, Tuple<Y>>
        >;
    };

    // needed to remove ambiguity
    template <class T, class X0, class X1, class... X>
    struct _CastAsType<Tuple<X0, X1, X...>, const T>
    {
        using Type = Cat<
            CastAsType<Tuple<X0>,       AddConst<GetFirst<T>>>,
            CastAsType<Tuple<X1, X...>, AddConst<DelFirst<T>>>
        >;
    };

    template <class A, class B> struct _CastAs {};

    template <class A>
    struct CastAs {
        template <typename B>
        using Type = CastAsType<A,B>;

        template <typename B>
        _MHD_ static inline Type<B> as(B&& b) {
            using  As =  _CastAs<Type<B>,RemoveConst<RemoveRef<B>>>;
            return As::as(std::forward<B>(b));
        }
    };

    template <class X0, class... X>
    struct _CastAs<Tuple<X0, X...>,Tuple<X0, X...>> {
        using A = Tuple<X0, X...>;
        _MHD_ static inline       A&& as(      A&& a) { return std::move(a); }
        _MHD_ static inline       A&  as(      A&  a) { return a; }
        _MHD_ static inline const A&  as(const A&  a) { return a; }
    };

    template <class B>
    struct _CastAs<Tuple<>, B> {
        _MHD_ static constexpr const Tuple<>  as(const B&  b) { return Tuple<>(); }
    };

    template <class A, class X0, class... X>
    struct _CastAs<A, Tuple<X0, X...>> {
        using First = GetFirstValue<A>;
        using Next  = DelFirst<A>;

        template <typename B>
        _MHD_ static inline A as(B&&  b) {
            return A(
                static_cast<First>(std::forward<B>(b).getFirstValue()),
                CastAs<Next>::as(std::forward<B>(b).delFirst())._asValue()
            );
        }
    };

NAMESPACE_END(_tuple)

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

template <class... X>
struct _LikeFrom<_tuple::Values<>, Tuple<X...>>
{
    using Type = _tuple::Values<X...>;
};

namespace _tuple {

    template <class T>      struct _IsValues               { using Type = False; };
    template <class... X>   struct _IsValues<Values<X...>> { using Type = True;  };
    template <class T>       using  IsValues = typename _IsValues<T>::Type;

    /// Values Helper //////////////////////////////////////////////////

    template <class VALUES, bool IsValue, class ARG1, class... ARGS>
    struct _BuildValues {};

    template <class VALUES, class ARG1, class... ARGS>
    using BuildValues = _BuildValues<VALUES, IsValues<RemoveRef<ARG1>>::Value, RemoveRef<ARG1>, RemoveRef<ARGS>...>;

    // Values(x...)
    template <class VALUES, class Y0, class... Y>
    struct _BuildValues<VALUES, false, Y0, Y...> {
            using FirstType = typename VALUES::FirstType;
            using NextType  = typename VALUES::NextType;

        template <class Z0, class...Z>
        _MHD_ static inline FirstType firstValue(Z0&& z0, Z&&... z) { return FirstType(std::forward<Z0>(z0)); }

        template <class Z0, class...Z>
        _MHD_ static inline NextType  nextValues(Z0&& z0, Z&&... z) { return NextType(std::forward<Z>(z)...); }
    };

    // Values(x0, Values(x...)) [used by asRef/asConstRef]
    template <class VALUES, class Y0, class... Y>
    struct _BuildValues<VALUES, false, Y0, Values<Y...>> {
            using FirstType = typename VALUES::FirstType;
            using NextType  = typename VALUES::NextType;

        template <class Z0, class Z>
        _MHD_ static inline FirstType firstValue(Z0&& z0, Z&& z) { return FirstType(std::forward<Z0>(z0)); }

        template <class Z0, class Z>
        _MHD_ static inline NextType  nextValues(Z0&& z0, Z&& z) { return NextType(std::forward<Z>(z)); }
    };

    // Values<X>(Y)
    template <class X, class Y>
    struct _BuildValues<Values<X>, false, Y> {
            using ThisType  = Values<X>;
            using FirstType = Y;
            using NextType  = Values<>;

        _MHD_ static inline       FirstType && firstValue(      Y && y) { return std::move(y); }
        _MHD_ static inline       FirstType &  firstValue(      Y &  y) { return y;  }
        _MHD_ static inline const FirstType &  firstValue(const Y &  y) { return y; }
        _MHD_ static constexpr    NextType     nextValues(const Y &  y) { return NextType(); }
    };

    // Values<X...>(Values<Y...>)
    template <class VALUES, class... Y>
    struct _BuildValues<VALUES, true, Values<Y...>> {
            using ThisType       = VALUES;
            using OtherType      = Values<Y...>;
            using FirstType      = typename OtherType::FirstType;
            using NextType       = typename OtherType::NextType;

        _MHD_ static inline       FirstType && firstValue(      OtherType && y) { return std::move(y.firstValue()); }
        _MHD_ static inline       FirstType &  firstValue(      OtherType &  y) { return y.firstValue(); }
        _MHD_ static inline const FirstType &  firstValue(const OtherType &  y) { return y.firstValue(); }
        _MHD_ static inline       NextType  && nextValues(      OtherType && y) { return std::move(y.nextValues()); }
        _MHD_ static inline       NextType  &  nextValues(      OtherType &  y) { return y.nextValues(); }
        _MHD_ static inline const NextType  &  nextValues(const OtherType &  y) { return y.nextValues(); }
    };

    // Values<X...>(Tuple<Y...>)
    template <class VALUES, class... Y>
    struct _BuildValues<VALUES, false, Tuple<Y...>> {
            using ThisType       = VALUES;
            using OtherType      = Tuple<Y...>;
            using ValueType      = typename OtherType::ValueType;
            using FirstType      = typename ValueType::FirstType;
            using NextType       = typename ValueType::NextType;

        _MHD_ static inline       FirstType && firstValue(      OtherType && y) { return std::move(y._values.firstValue()); }
        _MHD_ static inline       FirstType &  firstValue(      OtherType &  y) { return y._values.firstValue(); }
        _MHD_ static inline const FirstType &  firstValue(const OtherType &  y) { return y._values.firstValue(); }
        _MHD_ static inline       NextType  && nextValues(      OtherType && y) { return std::move(y._values.nextValues()); }
        _MHD_ static inline       NextType  &  nextValues(      OtherType &  y) { return y._values.nextValues(); }
        _MHD_ static inline const NextType  &  nextValues(const OtherType &  y) { return y._values.nextValues(); }
    };

    /// Values /////////////////////////////////////////////////////////

    template <class X>
    struct Values<X> {
        using FirstType = X;
        using ThisType  = Values<X>;
        using NextType  = Values<>;
        using AsPack    = Pack<X>;
        using AsTuple   = Tuple<X>;
        static constexpr size_t Length = 1;

        template <class Y0, class... Y>
        using Build = BuildValues<ThisType,Y0,Y...>;

        template <class I>
        using At = LikeFrom<Values<>, Get<Pack<X>, I>>;

        // Default
        Values(): FirstValue() {}

        // From values
        template <class Y0, class... Y>
        Values(Y0&& y0, Y&&... y):
            FirstValue(Build<Y0, Y...>::firstValue(std::forward<Y0>(y0), std::forward<Y>(y)...))
            {}

        // // From Values
        // template <class Y> Values(const Values<Y&&> &  y): FirstValue(std::forward<Y>(y.FirstValue)) {}
        // template <class Y> Values(      Values<Y&&> &  y): FirstValue(std::forward<Y>(y.FirstValue)) {}
        // template <class Y> Values(const Values<Y&&> && y): FirstValue(std::forward<Y>(y.FirstValue)) {}
        // template <class Y> Values(      Values<Y&&> && y): FirstValue(std::forward<Y>(std::move(y).FirstValue)) {}

        // // Copy
        // Values(const ThisType &  x): FirstValue(x.FirstValue)            {}
        // Values(      ThisType &  x): FirstValue(x.FirstValue)            {}
        // Values(const ThisType && x): FirstValue(x.FirstValue)            {}
        // Values(      ThisType && x): FirstValue(std::move(x).FirstValue) {}

        template <class Y>
        _MHD_ inline ThisType & operator=(const Values<Y> & x)
        {
            FirstValue = x.FirstValue;
            return *this;
        }

        template <class Y>
        _MHD_ inline ThisType & operator=(Values<Y> & x)
        {
            FirstValue = x.FirstValue;
            return *this;
        }

        template <class Y>
        _MHD_ inline ThisType & operator=(Values<Y> && x)
        {
            FirstValue = std::move(x.FirstValue);
            return *this;
        }

        template <class I>
        _MHD_ At<I> & at(I) {
            return reinterpret_cast<At<I> &>(*this);
        }

        template <class I>
        _MHD_ const At<I> & at(I) const {
            // Same comment as before.
            return reinterpret_cast<const At<I> &>(*this);
        }

        _MHD_ inline     const X & firstValue() const { return FirstValue; }
        _MHD_ inline           X & firstValue()       { return FirstValue; }
        _MHD_ constexpr NextType   nextValues() const { return NextType(); }

        _MHD_ inline     const ThisType & nextValuesOrPad() const { return *this; }
        _MHD_ inline           ThisType & nextValuesOrPad()       { return *this; }

        X FirstValue;
    };

    template <class X0, class X1, class... X>
    struct Values<X0, X1, X...> {
        using FirstType = X0;
        using ThisType  = Values<X0, X1, X...>;
        using NextType  = Values<X1, X...>;
        using AsPack    = Pack<X0, X1, X...>;
        using AsTuple   = Tuple<X0, X1, X...>;
        static constexpr size_t Length = Pack<X0, X1, X...>::Length;

        template <class I>
        using At = LikeFrom<Values<>, Get<Pack<X0, X1, X...>, I>>;

        template <class Y0, class... Y>
        using Build = BuildValues<ThisType,Y0,Y...>;

        // Default
        Values(): FirstValue(), NextValues() {}

        // From values or `Values`
        template <class Y0, class...Y>
        Values(Y0&& y0, Y&&... y):
            FirstValue(Build<Y0,Y...>::firstValue(std::forward<Y0>(y0), std::forward<Y>(y)...)),
            NextValues(Build<Y0,Y...>::nextValues(std::forward<Y0>(y0), std::forward<Y>(y)...))
            {}

        // // From values
        // template <class Y0, class...Y>
        // Values(Y0&& y0, Y&&... y):   FirstValue(std::forward<Y0>(y0)),    NextValues(std::forward<Y>(y)...)   {}

        // // Copy
        // Values(const ThisType &  x): FirstValue(x.FirstValue),            NextValues(x.NextValues)            {}
        // Values(      ThisType &  x): FirstValue(x.FirstValue),            NextValues(x.NextValues)            {}
        // Values(const ThisType && x): FirstValue(x.FirstValue),            NextValues(x.NextValues)            {}
        // Values(      ThisType && x): FirstValue(std::move(x).FirstValue), NextValues(std::move(x).NextValues) {}

        template <class... Y>
        _MHD_ inline ThisType & operator=(const Values<Y...> & x)
        {
            FirstValue = x.firstValue();
            NextValues = x.nextValuesOrPad();
            return *this;
        }

        template <class... Y>
        _MHD_ inline ThisType & operator=(Values<Y...> & x)
        {
            FirstValue = x.firstValue();
            NextValues = x.nextValuesOrPad();
            return *this;
        }

        template <class... Y>
        _MHD_ inline ThisType & operator=(Values<Y...> && x)
        {
            FirstValue = std::move(x.firstValue());
            NextValues = std::move(x.nextValuesOrPad());
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

        _MHD_ inline const       X0 & firstValue() const { return FirstValue; }
        _MHD_ inline             X0 & firstValue()       { return FirstValue; }
        _MHD_ inline const NextType & nextValues() const { return NextValues; }
        _MHD_ inline       NextType & nextValues()       { return NextValues; }
        _MHD_ inline const NextType & nextValuesOrPad() const { return nextValues(); }
        _MHD_ inline       NextType & nextValuesOrPad()       { return nextValues(); }

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
///     Operators                                                    ///
/// ---------------------------------------------------------------- ///


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

NAMESPACE_BEGIN(miniten)

#endif /// MINITEN__TUPLE_IMPL
