#ifndef TNY__XARRAY_ACCESS
#define TNY__XARRAY_ACCESS
#include <cuda/std/tuple>
#include <teeny/core.h>
#include <teeny/statix.h>
#include <teeny/_xarray/decl.h>

_TNY_NAMESPACE_BEGIN(tny)

/**
 * @brief Access policy for a single, statically-indexed xarray element.
 *
 * Dispatches on whether the *slot* at `Index` is dynamic or static
 * (decided on the logical `values` descriptor, not the storage layout):
 *   - dynamic slot -> returns a reference into storage (readable & writable),
 *   - static  slot -> returns the compile-time value by prvalue.
 *
 * `Index` is a static index (e.g. `statix::csize<0>` or `statix::cptrdiff<-1>`).
 *
 * @tparam XArray     An `xarray<T, values>` type.
 * @tparam Index      A static index type.
 * @tparam IsDynamic  Deduced: whether the slot at `Index` is dynamic.
 */
template <class XArray, class Index,
          bool IsDynamic = statix::is_cnone<
              statix::at<typename XArray::values_type, Index>
          >::value>
struct xarray_access;

/* --- dynamic slot: reference into the dynamic-only storage tuple --- */
template <class XArray, class Index>
struct xarray_access<XArray, Index, true> {
    using values_type = typename XArray::values_type;
    using value_type  = typename XArray::value_type;

    // Non-negative logical index, then its ordinal among the stored dynamics.
    static constexpr size_t logical  = static_cast<size_t>(
        statix::wrap_index<statix::size<values_type>, Index>::value);
    static constexpr size_t position = dynamic_ordinal<values_type, logical>::value;

    using type       = value_type &;
    using const_type = const value_type &;

    _TNYDEF(H,D,I,CX) static type at(XArray & self, Index) noexcept
    { return cuda::std::get<position>(self); }

    _TNYDEF(H,D,I,CX) static const_type at(const XArray & self, Index) noexcept
    { return cuda::std::get<position>(self); }
};

/* --- static slot: compile-time value by prvalue ------------------- */
template <class XArray, class Index>
struct xarray_access<XArray, Index, false> {
    using values_type = typename XArray::values_type;
    using value_type  = typename XArray::value_type;
    using element     = statix::at<values_type, Index>;  // carray<value_type, V>

    using type       = value_type;
    using const_type = value_type;

    _TNYDEF(H,D,I,CX) static value_type at(const XArray &, Index) noexcept
    { return static_cast<value_type>(element::value); }
};

/* --- convenience aliases ------------------------------------------ */

template <class XArray, class Index>
using xarray_access_type = typename xarray_access<XArray, Index>::type;

template <class XArray, class Index>
using xarray_access_const_type = typename xarray_access<XArray, Index>::const_type;

_TNY_NAMESPACE_END(tny)

#endif // TNY__XARRAY_ACCESS
