/***********************************************************************
 * This file implements indexing utility for template parameter packs
 *
 * WrapIndexValue   : Makes python-like negative indices positive
 * WrapIndex        : WrapIndex applied to all elements in a container
 * Slice            : The metatemplating equivalent of python's `slice`
 * SimpleSlice      : A simpler slice that only accepts concrete arguments
 * AsIndexVector    : Convert slice to wrapped PtrDiff indices
 ***********************************************************************/
#ifndef MINITEN_META_INDEX_H
#define MINITEN_META_INDEX_H
#include "../_core/defines.h"
#include "../_core/types.h"
#include "vector.h"

namespace miniten {
namespace meta {

/// ---------------------------------------------------------------- ///
///     Python-like slice                                            ///
/// ---------------------------------------------------------------- ///

template <ptrdiff_t Start, ptrdiff_t Stop, ptrdiff_t Step = 1>
struct SimpleSlice;

template <class Start, class Stop, class Step = PtrDiff<1>>
struct Slice;

template <class SLICE, size_t LENGTH>
struct _AsIndexVector;

template <class SLICE, size_t LENGTH>
using   AsIndexVector = typename _AsIndexVector<SLICE,LENGTH>::Type;

/// ---------------------------------------------------------------- ///
///     Python-like negative indexing                                ///
/// ---------------------------------------------------------------- ///

template <size_t Length, ptrdiff_t... Index>
struct _WrapIndexValue; // Type = PtrDiff<Index...>

/// Transform negative indices into positive indices (Python convention).
///
/// WrapIndexValue<Length, Index...> = PtrDiff<Index...>
///
/// @tparam Length  Number of elements in the tuple/vector.
/// @tparam Index   Index to (maybe) wrap. If negative, counts from the end.
template <size_t Length, ptrdiff_t... Index>
using WrapIndexValue = typename _WrapIndexValue<Length, Index...>::Type;

template <class Length, class Index>
struct _WrapIndex;

/// Transform negative indices into positive indices (Python convention).
///
/// WrapIndex<Length, Index> = Index
///
/// @tparam Length  Number of elements in the tuple/vector.
/// @tparam Index   Index to (maybe) wrap. If negative, counts from the end.
template <class Length, class Index>
using WrapIndex = typename _WrapIndex<Length, Index>::Type;


} // namespace meta
} // namespace miniten

#endif // MINITEN_META_INDEX_H
