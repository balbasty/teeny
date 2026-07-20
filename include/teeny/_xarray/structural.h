#ifndef TNY__XARRAY_STRUCTURAL
#define TNY__XARRAY_STRUCTURAL
#include <cuda/std/utility>       // index_sequence, make_index_sequence
#include <teeny/core.h>
#include <teeny/statix.h>
#include <teeny/_xarray/decl.h>
#include <teeny/_xarray/impl.h>

_TNY_NAMESPACE_BEGIN(tny)

/* ================================================================== *
 *  Structural xarray ops.                                            *
 *                                                                    *
 *  Each rebuilds a new xarray whose `values` descriptor is computed  *
 *  at the type level (via `statix` pack algebra) and whose dynamic    *
 *  leaves are copied from the source by logical index. Static slots   *
 *  carry no storage, so only dynamic outputs are assigned.           *
 * ================================================================== */

namespace _xarray {

// Copy output slot J from source logical index `Src` iff that slot is dynamic.
template <class Out, class Src, size_t J, ptrdiff_t SrcIdx>
_TNYDEF(H,D,I,CX) void _copy_slot(Out& out, const Src& src) {
    if constexpr (statix::is_cnone<
                      statix::at<typename Out::values_type, statix::csize<J> > >::value)
        out.at(statix::csize<J>()) = src.at(statix::cptrdiff<SrcIdx>());
    else { (void)out; (void)src; }
}

// select: output slot J takes source index Idx_J.
template <class Out, class T, class V, ptrdiff_t... Idx, size_t... J>
_TNYDEF(H,D,I,CX) void _select_fill(Out& out, const xarray<T,V>& a,
                                    cuda::std::index_sequence<J...>) {
    constexpr ptrdiff_t idx[] = { Idx..., 0 /* guard for the empty pack */ };
    (_copy_slot<Out, xarray<T,V>, J, idx[J]>(out, a), ...);
}

// erase: output slot J takes source slot (J < Dw ? J : J+1).
template <ptrdiff_t Dw, class Out, class T, class V, size_t... J>
_TNYDEF(H,D,I,CX) void _erase_fill(Out& out, const xarray<T,V>& a,
                                   cuda::std::index_sequence<J...>) {
    (_copy_slot<Out, xarray<T,V>, J, (ptrdiff_t)(J < (size_t)Dw ? J : J + 1)>(out, a), ...);
}

// reversed: output slot J takes source slot (N-1-J).
template <size_t N, class Out, class T, class V, size_t... J>
_TNYDEF(H,D,I,CX) void _reverse_fill(Out& out, const xarray<T,V>& a,
                                     cuda::std::index_sequence<J...>) {
    (_copy_slot<Out, xarray<T,V>, J, (ptrdiff_t)(N - 1 - J)>(out, a), ...);
}

} // namespace _xarray

/**
 * @brief Select / permute / gather elements by static index list.
 *
 * `select<i0, i1, ...>(a)` returns an xarray of the chosen slots in order;
 * indices may repeat and may be negative. A permutation is the special case
 * where the indices are a rearrangement of `0..N-1`.
 */
template <ptrdiff_t... Idx, class T, class V>
_TNYDEF(H,D,I,CX) auto select(const xarray<T,V>& a)
    -> xarray<T, statix::get_index<statix::as_tuple<V>, Idx...> >
{
    using Out = xarray<T, statix::get_index<statix::as_tuple<V>, Idx...> >;
    Out out{};
    _xarray::_select_fill<Out, T, V, Idx...>(
        out, a, cuda::std::make_index_sequence<sizeof...(Idx)>{});
    return out;
}

/** @brief Alias for `select`, read as a permutation. */
template <ptrdiff_t... Idx, class T, class V>
_TNYDEF(H,D,I,CX) auto permute(const xarray<T,V>& a)
    -> xarray<T, statix::get_index<statix::as_tuple<V>, Idx...> >
{ return select<Idx...>(a); }

/**
 * @brief Drop the element at (possibly negative) index `D` -- i.e. squeeze.
 */
template <ptrdiff_t D, class T, class V>
_TNYDEF(H,D,I,CX) auto erase(const xarray<T,V>& a)
    -> xarray<T, statix::erase_index<statix::as_tuple<V>, D> >
{
    using Out = xarray<T, statix::erase_index<statix::as_tuple<V>, D> >;
    constexpr ptrdiff_t Dw = statix::wrap_index<
        statix::size<statix::as_tuple<V> >, statix::cptrdiff<D> >::value;
    Out out{};
    _xarray::_erase_fill<Dw, Out, T, V>(
        out, a, cuda::std::make_index_sequence<
                    statix::size<statix::as_tuple<V> >::value - 1>{});
    return out;
}

/** @brief Reverse the element order. */
template <class T, class V>
_TNYDEF(H,D,I,CX) auto reversed(const xarray<T,V>& a)
    -> xarray<T, statix::reversed<statix::as_tuple<V> > >
{
    using Out = xarray<T, statix::reversed<statix::as_tuple<V> > >;
    constexpr size_t N = statix::size<statix::as_tuple<V> >::value;
    Out out{};
    _xarray::_reverse_fill<N, Out, T, V>(
        out, a, cuda::std::make_index_sequence<N>{});
    return out;
}

_TNY_NAMESPACE_END(tny)

#endif // TNY__XARRAY_STRUCTURAL
