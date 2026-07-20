#ifndef TNY_MD_HELPERS
#define TNY_MD_HELPERS
#include <cuda/std/mdspan>
#include <cuda/std/utility>
#include <teeny/_core/defines.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/**
 * @brief Fortran-order linear index over the leading (rank-1) dims -> offset.
 *
 * The `index2offset` operation over the batch dimensions (all but the last):
 * decodes `lin` with dim 0 varying fastest, weighting by each dim's stride.
 * Fully unrolled, so static extents/strides fold.
 */
template <class MD, cs::size_t... D>
_TNY_API auto batch_offset(const MD & a, typename MD::index_type lin, cs::index_sequence<D...>) {
    using O = typename MD::index_type;
    O off = 0, cur = 1;
    ( ( off += ((lin % (cur * a.extent(D))) / cur) * a.stride(D), cur *= a.extent(D) ), ... );
    return off;
}
template <class MD>
_TNY_API auto batch_offset(const MD & a, typename MD::index_type lin) {
    return batch_offset(a, lin, cs::make_index_sequence<MD::rank() - 1>{});
}

/**
 * @brief Peel axis 0 (e.g. the channel axis) at index `c` -> a spatial view.
 *
 * Rank-generic wrapper over `submdspan`. Works on any mdspan.
 */
template <class MD, cs::size_t... I>
_TNY_API auto channel(const MD & a, typename MD::index_type c, cs::index_sequence<I...>) {
    return cs::submdspan(a, c, ((void)I, cs::full_extent)...);
}
template <class MD>
_TNY_API auto channel(const MD & a, typename MD::index_type c) {
    return channel(a, c, cs::make_index_sequence<MD::rank() - 1>{});
}

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_HELPERS
