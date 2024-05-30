#ifndef MINITEN_META_NDINDEX_H
#define MINITEN_META_NDINDEX_H
#include "meta/vector.h"

namespace miniten {
namespace meta {

/// ---------------------------------------------------------------- ///
///     Shape specialization                                         ///
/// ---------------------------------------------------------------- ///

template <long... N> using Shape = Long<N...>;

template <class... N>
struct SmartShape {};

template <>
struct SmartShape {

};

template <class N0>
struct SmartShape {

};

template <class N0, class... N>
struct SmartShape {

};

/// ---------------------------------------------------------------- ///
///     Indices                                                      ///
/// ---------------------------------------------------------------- ///

/// A tuple of indices
template <typename... T>
using Slicer = Tuple<T...>;

struct Ellipsis {};

template <typename T>
struct IsEllipsis { static constexpr bool Value = false; };
template <>
struct IsEllipsis<Ellipsis> { static constexpr bool Value = true; };

template <long Begin, long End, long Step = 1>
struct Slice {};

using FullSlice = Slice<0, -1>;

template <long Length>
struct SizedRange {};

/// Check if Slicer contains an ellipsis
template <typename Tuple>
struct HasEllipsis {
    static constexpr bool Value = IsEllipsis<typename Tuple::First>::Value ||
                                  HasEllipsis<typename Tuple::PopFirst>::Value;
};
template <>
struct HasEllipsis< Tuple<> > {
    static constexpr bool Value = false;
};

/// Find the index of the ellipsis
///
/// FindEllipsis< Tuple<
template <typename Tuple, typename First = typename Tuple::First, long N=0>
struct FindEllipsis {
    static constexpr long Index = FindEllipsis<typename Tuple::PopFirst, typename Tuple::PopFirst::First, N+1>::Index;
};
template <typename Tuple, long N>
struct FindEllipsis<Tuple, Ellipsis, N> {
    static constexpr long Index = N;
};

/// replace ellipses

template <long NbDim, typename SlicerLeft, typename SlicerRight = Slicer<> >
struct FillWithFullSlices {};


} /// namespace meta
} /// namespace

#endif // MINITEN_META_NDINDEX_H
