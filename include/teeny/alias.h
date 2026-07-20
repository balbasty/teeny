#ifndef TNY_ALIAS
#define TNY_ALIAS
// Convenience: pull the mdspan vocabulary teeny builds on into `tny`, so user
// code needs only `using namespace tny;`. These are the standard cuda::std
// names -- none collide with teeny's own, so exposing them is safe (not
// breaking): teeny defines `tensor`, `view`, `layout_static_stride`, ... which
// are all distinct from `extents`, `layout_right`, `array`, etc.
#include <cuda/std/mdspan>
#include <cuda/std/array>
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <teeny/_core/defines.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

// shapes / layouts / views
using cs::extents;
using cs::dextents;
using cs::dynamic_extent;
using cs::full_extent;
using cs::full_extent_t;
using cs::layout_right;
using cs::layout_left;
using cs::layout_stride;
using cs::mdspan;
using cs::submdspan;

// containers / utilities
using cs::array;
using cs::size_t;
using cs::ptrdiff_t;
using cs::index_sequence;
using cs::make_index_sequence;
using cs::integral_constant;

/** @brief A compile-time index value, e.g. `t(ic<1>, j, ic<3>)`. Converts
 *         implicitly to a runtime integral, and carries `::value`. */
template <long V>
using ic = cs::integral_constant<long, V>;

/** @brief Keep-this-axis marker for slicing (an alias of `full_extent`). */
constexpr cs::full_extent_t all{};

_TNY_NAMESPACE_END(tny)

#endif // TNY_ALIAS
