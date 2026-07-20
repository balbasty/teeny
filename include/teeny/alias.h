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
#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
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

/* ------------------------------------------------------------------ *
 *     Static integral values (compile-time indices / extents)        *
 *                                                                    *
 *  cuda::std / std do not ship short names for integral_constant, so  *
 *  teeny provides them. Each converts implicitly to a runtime         *
 *  integral and carries `::value`, so it works as both a compile-time *
 *  and a runtime value. Pass one where a tensor wants a static index   *
 *  (e.g. `t(Int<1>(), j)`, `t.extent(Int<0>())`).                     *
 * ------------------------------------------------------------------ */
template <int            V> using Int    = cs::integral_constant<int, V>;
template <long           V> using Long   = cs::integral_constant<long, V>;
template <cs::size_t     V> using Size   = cs::integral_constant<cs::size_t, V>;
template <unsigned       V> using Uint   = cs::integral_constant<unsigned, V>;
template <cs::int32_t    V> using Int32  = cs::integral_constant<cs::int32_t, V>;
template <cs::int64_t    V> using Int64  = cs::integral_constant<cs::int64_t, V>;
template <cs::ptrdiff_t  V> using Diff   = cs::integral_constant<cs::ptrdiff_t, V>;
template <bool           V> using Bool   = cs::integral_constant<bool, V>;

/** @brief Alias of `Long`; a compile-time index value. */
template <long V> using ic = cs::integral_constant<long, V>;

/** @brief Keep-this-axis marker for slicing (an alias of `full_extent`). */
constexpr cs::full_extent_t all{};

_TNY_NAMESPACE_END(tny)

#endif // TNY_ALIAS
