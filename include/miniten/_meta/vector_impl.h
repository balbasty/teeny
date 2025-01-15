/***********************************************************************
 * This file implements a compile-time "vector of values"
 *
 * Vector<T, N...>
 ***********************************************************************/
#ifndef MINITEN_META_VECTOR_IMPL_H
#define MINITEN_META_VECTOR_IMPL_H
#include "../_core/defines.h"
#include "../_core/types.h"
#include "../show.h"
#include "traits.h"
#include "vector.h"
#include "tuple.h"
#include "pack.h"
#include "packapi.h"
#include "math.h"

namespace miniten {
namespace meta {

/// ---------------------------------------------------------------- ///
///     Pack API                                                     ///
/// ---------------------------------------------------------------- ///

/// Cat

template <class RIGHT, class T>
struct _Cat2<Vector<T>, RIGHT> {
    static_assert(IsVector<RIGHT>::Value, "L/R should be vectors");
    static_assert(IsSame<ItemType<RIGHT>,T>::Value, "L/R should have same item type");
    using Type = RIGHT;
};

template <class T, T X0, T... X>
struct _Cat2<Vector<T,X0>, Vector<T, X...>> {
    using Type = Vector<T,X0,X...>;
};

template <class RIGHT, class T, T X0, T... X>
struct _Cat2<Vector<T,X0,X...>, RIGHT> {
    using Type = Cat<Vector<T,X0>, Cat<Vector<T,X...>, RIGHT>>;
};


namespace _vector {
    template <class VECTOR>          struct _Reversed;
    template <class VECTOR>          using   Reversed = typename _Reversed<VECTOR>::Type;
    template <class VECTOR, class T> struct _Cast;
    template <class VECTOR, class T> using   Cast = typename _Cast<VECTOR, T>::Type;
    template <class VECTOR>          struct _AsTuple;
    template <class VECTOR>          using   AsTuple = typename _AsTuple<VECTOR>::Type;
    template <class VECTOR>          struct _AsPack;
    template <class VECTOR>          using   AsPack = typename _AsPack<VECTOR>::Type;
}

template <class T, T... X>          struct _ItemType<Vector<T, X...>>           { using Type = T; };
template <class T, T... X>          struct _Length<Vector<T, X...>>             { using Type = CountValues<T, X...>; };
template <class T, T... X>          struct _EmptyLike<Vector<T, X...>>          { using Type = Vector<T>; };
template <class T, T... X>          struct _IsVector<Vector<T, X...>>           { using Type = True; };
template <class T, T... X>          struct _Reversed<Vector<T, X...>>           { using Type = _vector::Reversed<Vector<T, X...>>; };
template <class T, class U, T... X> struct _AsVector<Vector<T, X...>, U>        { using Type = _vector::Cast<Vector<T, X...>, U>; };
template <class T, T... X>          struct _AsTuple<Vector<T, X...>>            { using Type = _vector::AsTuple<Vector<T, X...>>; };
template <class T, T... X>          struct _AsPack<Vector<T, X...>>             { using Type = _vector::AsPack<Vector<T, X...>>; };
template <class T, T... X, class M> struct _LikeFrom<Vector<T, X...>, M>        { using Type = AsVector<M, T>; };
template <class T, T X0, T... X>    struct _GetFirst<Vector<T, X0, X...>>       { using Type = Vector<T, X0>; };
template <class T, T X0, T... X>    struct _DelFirst<Vector<T, X0, X...>>       { using Type = Vector<T, X...>; };
template <class T, T X0, T... X>    struct _GetFirstValue<Vector<T, X0, X...>>  { using Type = Vector<T, X0>; };

/// ---------------------------------------------------------------- ///
///     Pack API Implementation                                      ///
/// ---------------------------------------------------------------- ///

namespace _vector {

/// Revert /////////////////////////////////////////////////////////////

template <class VECTOR>
struct _Reversed {};

template <class VECTOR>
using Reversed = typename _Reversed<VECTOR>::Type;

template <typename T, T N0, T... N>
struct _Reversed<Vector<T, N0, N...> > {
    using Type = Cat< _Reversed<Vector<T, N...>>, Vector<T, N0> >;
};

template <typename T, T N0>
struct _Reversed<Vector<T, N0> > {
    using Type = Vector<T, N0>;
};

template <typename T>
struct _Reversed<Vector<T> > {
    using Type = Vector<T>;
};

/// Cast ///////////////////////////////////////////////////////////////

template <class VECTOR, typename U>
struct _Cast {};

template <class VECTOR, typename U>
using Cast = typename _Cast<VECTOR, U>::Type;

template <typename T, typename U, T X0, T... X>
struct _Cast<Vector<T, X0, X...>, U>
{
    using Type = Cat<
        Vector<U, static_cast<U>(X0)>,
        Cast<Vector<T, X...>, U>
    >;
};

template <typename T, typename U, T X0>
struct _Cast<Vector<T, X0>, U>
{
    using Type = Vector<U, static_cast<U>(X0)>;
};

template <typename T, typename U>
struct _Cast<Vector<T>, U>
{
    using Type = Vector<U>;
};

/// AsTuple ////////////////////////////////////////////////////////////

template <class VECTOR>
struct _AsTuple {};

template <class VECTOR>
using AsTuple = typename _AsTuple<VECTOR>::Type;

template <class T, T X0, T... X>
struct _AsTuple<Vector<T, X0, X...>>
{
    using Type = Cat<
        Tuple<Vector<T, X0>>,
        AsTuple<Vector<T, X...>>
    >;
};

template <class T, T X0>
struct _AsTuple<Vector<T, X0>>
{
    using Type = Tuple<Vector<T, X0>>;
};

template <class T>
struct _AsTuple<Vector<T>>
{
    using Type = Tuple<>;
};

/// AsPack /////////////////////////////////////////////////////////////

template <class VECTOR>
struct _AsPack {};

template <class VECTOR>
using AsPack = typename _AsPack<VECTOR>::Type;

template <class T, T X0, T... X>
struct _AsPack<Vector<T, X0, X...>>
{
    using Type = Cat<
        Pack<Vector<T, X0>>,
        AsPack<Vector<T, X...>>
    >;
};

template <class T, T X0>
struct _AsPack<Vector<T, X0>>
{
    using Type = Pack<Vector<T, X0>>;
};

template <class T>
struct _AsPack<Vector<T>>
{
    using Type = Pack<>;
};

} // namespace _vector

/// ---------------------------------------------------------------- ///
///     Vector Specialization                                        ///
/// ---------------------------------------------------------------- ///

struct AnyVectorBase {};

template <typename T>
struct VectorBase: public AnyVectorBase {
    using ItemType = T;
};

/// A compile-time tuple of values with the same type
template <typename T, T... X>
struct Vector: public VectorBase<T> {};

/// 1+ elements
template <typename T, T X0, T... X>
struct Vector<T, X0, X...>: public VectorBase<T> {

    /// Static types and values ////////////////////////////////////////

    using ThisType = Vector<T, X0, X...>;
    using NextType = Vector<T, X...>;
    using ItemType = T;
    static constexpr size_t Length = CountValues<T, X0, X...>::Value;

    /// Constexpr methods ///////////////////////////////////////////////

    MINITEN_HOSTDEVICE constexpr size_t     length()    const { return Length;}

    template <class I>              MINITEN_HOSTDEVICE constexpr T                              getValue(I)         const { return GetValue<ThisType,I>::Value; }
                                    MINITEN_HOSTDEVICE constexpr T                              getFirstValue()     const { return GetFirstValue<ThisType>::Value; }
                                    MINITEN_HOSTDEVICE constexpr T                              getLastValue()      const { return GetLastValue<ThisType>::Value; }

    template <class I>              MINITEN_HOSTDEVICE constexpr Get<ThisType,I>                get(I)              const { return Get<ThisType,I>(); }
    template <class N>              MINITEN_HOSTDEVICE constexpr GetFirst<ThisType,N::Value>    getFirst(N)         const { return GetFirst<ThisType,N::Value>(); }
                                    MINITEN_HOSTDEVICE constexpr GetFirst<ThisType>             getFirst()          const { return GetFirst<ThisType>(); }
    template <class N>              MINITEN_HOSTDEVICE constexpr GetLast<ThisType,N::Value>     getLast(N)          const { return GetLast<ThisType,N::Value>(); }
                                    MINITEN_HOSTDEVICE constexpr GetLast<ThisType>              getLast()           const { return GetLast<ThisType>(); }

    template <class I>              MINITEN_HOSTDEVICE constexpr Del<ThisType,I>                del(I)              const { return Del<ThisType,I>(); }
    template <class N>              MINITEN_HOSTDEVICE constexpr DelFirst<ThisType,N::Value>    delFirst(N)         const { return DelFirst<ThisType,N::Value>(); }
    template <class N>              MINITEN_HOSTDEVICE constexpr DelLast<ThisType,N::Value>     delLast(N)          const { return DelLast<ThisType,N::Value>(); }

    template <class I, class... M>  MINITEN_HOSTDEVICE constexpr SetFrom<ThisType,I,M...>       setFrom(I, M...)    const { return SetFrom<ThisType,I,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr SetFirstFrom<ThisType,M...>    setFirstFrom(M...)  const { return SetFirstFrom<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr SetLastFrom<ThisType,M...>     setLastFrom(M...)   const { return SetLastFrom<ThisType,M...>(); }

    template <class I, class... M>  MINITEN_HOSTDEVICE constexpr Set<ThisType,I,M...>           set(I, M...)        const { return Set<ThisType,I,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr SetFirst<ThisType,M...>        setFirst(M...)      const { return SetFirst<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr SetLast<ThisType,M...>         setLast(M...)       const { return SetLast<ThisType,M...>(); }

    template <class I, class... M>  MINITEN_HOSTDEVICE constexpr InsertFrom<ThisType,I,M...>    insertFrom(I, M...) const { return InsertFrom<ThisType,I,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr PrependFrom<ThisType,M...>     prependFrom(M...)   const { return PrependFrom<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr AppendFrom<ThisType,M...>      appendFrom(M...)    const { return AppendFrom<ThisType,M...>(); }

    template <class I, class... M>  MINITEN_HOSTDEVICE constexpr Insert<ThisType,I,M...>        insert(I, M...)     const { return Insert<ThisType,I,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr Prepend<ThisType,M...>         prepend(M...)       const { return Prepend<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr Append<ThisType,M...>          append(M...)        const { return Append<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr Extend<ThisType,M...>          extend(M...)        const { return Extend<ThisType,M...>(); }
};

/// Single element
template <typename T, T X>
struct Vector<T, X>: public VectorBase<T> {

    /// Static types and values ////////////////////////////////////////

    using ThisType = Vector<T, X>;
    using NextType = Vector<T>;
    using ItemType = T;
    static constexpr size_t Length = 1;
    static constexpr T Value = X;

    /// Constexpr methods ///////////////////////////////////////////////

    MINITEN_HOSTDEVICE constexpr size_t     length()    const { return Length; }
    MINITEN_HOSTDEVICE constexpr operator   T()         const { return Value; }

    template <class I>              MINITEN_HOSTDEVICE constexpr T                              getValue(I)         const { return GetValue<ThisType,I>::Value; }
                                    MINITEN_HOSTDEVICE constexpr T                              getFirstValue()     const { return GetFirstValue<ThisType>::Value; }
                                    MINITEN_HOSTDEVICE constexpr T                              getLastValue()      const { return GetLastValue<ThisType>::Value; }

    template <class I>              MINITEN_HOSTDEVICE constexpr Get<ThisType,I>                get(I)              const { return Get<ThisType,I>(); }
    template <class N>              MINITEN_HOSTDEVICE constexpr GetFirst<ThisType,N::Value>    getFirst(N)         const { return GetFirst<ThisType,N::Value>(); }
                                    MINITEN_HOSTDEVICE constexpr ThisType                       getFirst()          const { return ThisType(); }
    template <class N>              MINITEN_HOSTDEVICE constexpr GetLast<ThisType,N::Value>     getLast(N)          const { return GetLast<ThisType,N::Value>(); }
                                    MINITEN_HOSTDEVICE constexpr ThisType                       getLast()           const { return ThisType(); }

    template <class I>              MINITEN_HOSTDEVICE constexpr Del<ThisType,I>                del(I)              const { return Del<ThisType,I>(); }
    template <class N>              MINITEN_HOSTDEVICE constexpr DelFirst<ThisType,N::Value>    delFirst(N)         const { return DelFirst<ThisType,N::Value>(); }
    template <class N>              MINITEN_HOSTDEVICE constexpr DelLast<ThisType,N::Value>     delLast(N)          const { return DelLast<ThisType,N::Value>(); }

    template <class I, class... M>  MINITEN_HOSTDEVICE constexpr SetFrom<ThisType,I,M...>       setFrom(I, M...)    const { return SetFrom<ThisType,I,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr SetFirstFrom<ThisType,M...>    setFirstFrom(M...)  const { return SetFirstFrom<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr SetLastFrom<ThisType,M...>     setLastFrom(M...)   const { return SetLastFrom<ThisType,M...>(); }

    template <class I, class... M>  MINITEN_HOSTDEVICE constexpr Set<ThisType,I,M...>           set(I, M...)        const { return Set<ThisType,I,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr SetFirst<ThisType,M...>        setFirst(M...)      const { return SetFirst<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr SetLast<ThisType,M...>         setLast(M...)       const { return SetLast<ThisType,M...>(); }

    template <class I, class... M>  MINITEN_HOSTDEVICE constexpr InsertFrom<ThisType,I,M...>    insertFrom(I, M...) const { return InsertFrom<ThisType,I,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr PrependFrom<ThisType,M...>     prependFrom(M...)   const { return PrependFrom<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr AppendFrom<ThisType,M...>      appendFrom(M...)    const { return AppendFrom<ThisType,M...>(); }

    template <class I, class... M>  MINITEN_HOSTDEVICE constexpr Insert<ThisType,I,M...>        insert(I, M...)     const { return Insert<ThisType,I,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr Prepend<ThisType,M...>         prepend(M...)       const { return Prepend<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr Append<ThisType,M...>          append(M...)        const { return Append<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr Extend<ThisType,M...>          extend(M...)        const { return Extend<ThisType,M...>(); }
};

/// Empty Vector
template <typename T>
struct Vector<T>: public VectorBase<T> {

    /// Static types and values ////////////////////////////////////////

    using ThisType = Vector<T>;
    using ItemType = T;
    static constexpr size_t Length = 0;

    /// Constexpr methods ///////////////////////////////////////////////

    MINITEN_HOSTDEVICE constexpr size_t     length()    const { return Length; }

    template <class I, class... M>  MINITEN_HOSTDEVICE constexpr InsertFrom<ThisType,I,M...>    insertFrom(I, M...) const { return InsertFrom<ThisType,I,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr PrependFrom<ThisType,M...>     prependFrom(M...)   const { return PrependFrom<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr AppendFrom<ThisType,M...>      appendFrom(M...)    const { return AppendFrom<ThisType,M...>(); }

    template <class I, class... M>  MINITEN_HOSTDEVICE constexpr Insert<ThisType,I,M...>        insert(I, M...)     const { return Insert<ThisType,I,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr Prepend<ThisType,M...>         prepend(M...)       const { return Prepend<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr Append<ThisType,M...>          append(M...)        const { return Append<ThisType,M...>(); }
    template <class... M>           MINITEN_HOSTDEVICE constexpr Extend<ThisType,M...>          extend(M...)        const { return Extend<ThisType,M...>(); }
};

/// ---------------------------------------------------------------- ///
///     Repeat                                                       ///
/// ---------------------------------------------------------------- ///

template <long N, typename T, T Val>
struct _NVector {
    using Type = Cat<NVector<N-1, T, Val>, Vector<T, Val>>;
};

template <typename T, T Val>
struct _NVector<0, T, Val> {
    using Type = Vector<T>;
};


} /// namespace meta
} /// namespace miniten

#endif /// MINITEN_META_VECTOR_IMPL_H
