/***********************************************************************
 * This file implements a compile-time "tuple of types"
 *
 * Tuple<T...>
 ***********************************************************************/
#ifndef MINITEN_META_TUPLE2_H
#define MINITEN_META_TUPLE2_H
#include "../defines.h"
#include "../types.h"
#include "../show.h"
#include "typepack.h"

namespace miniten {
namespace meta {

/// ---------------------------------------------------------------- ///
///     Tuple                                                        ///
/// ---------------------------------------------------------------- ///

/// A compile-time tuple of typenames
///
/// Tuple<T...>
///
///  ::Length                   Number of types in the tuple
///  ::Empty                    Return the empty tuple:     Tuple<>
///  ::Reversed                 Return reversed tuple
///  ::TupleLike<T...>          A tuple with the same general type
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
template <class... T>
struct Tuple {

    TYPEPACK_STATIC_TOPACK(Tuple)
    TYPEPACK_STATIC_INHERIT

    MINITEN_HOSTDEVICE static inline
    void Show()
    {
        show("meta::Tuple<");
        ShowValues();
        show(">");
    }
};

/// ---------------------------------------------------------------- ///
///     Aliases                                                      ///
/// ---------------------------------------------------------------- ///

/// A compile-time single element
template <class T>
using Element = Tuple<T>;

/// A compile-time pair of elements
template <class T, class U = T>
using Pair = Tuple<T, U>;


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

} /// namespace meta
} /// namespace miniten

#endif /// MINITEN_META_TUPLE_H
