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
 * Dispatches on whether the element stored at `Index` is a compile-time
 * (static) value or a runtime (dynamic) value:
 *   - dynamic element -> returns a reference into storage (readable & writable),
 *   - static  element -> returns the compile-time value by prvalue.
 *
 * `Index` is a static index (e.g. `statix::csize<0>` or `statix::cptrdiff<-1>`).
 *
 * @tparam XArray    An `xarray<T, values>` type.
 * @tparam Index     A static index type.
 * @tparam IsStatic  Deduced: whether the element at `Index` is static.
 */
template <class XArray, class Index,
          bool IsStatic = statix::is_carray<
              statix::at<typename XArray::tuple_type, Index>
          >::value>
struct xarray_access;

/* --- dynamic element: reference into the underlying tuple ---------- */
template <class XArray, class Index>
struct xarray_access<XArray, Index, false> {
    using tuple_type = typename XArray::tuple_type;
    using value_type = typename XArray::value_type;

    // Position of the element among *all* tuple leaves (static leaves
    // included). The storage tuple has one leaf per element, so the wrapped
    // index directly names the leaf.
    static constexpr size_t position = static_cast<size_t>(
        statix::wrap_index<statix::size<tuple_type>, Index>::value);

    using type       = value_type &;
    using const_type = const value_type &;

    _TNYDEF(H,D,I) static type at(XArray & self, Index) noexcept
    { return cuda::std::get<position>(self); }

    _TNYDEF(H,D,I) static const_type at(const XArray & self, Index) noexcept
    { return cuda::std::get<position>(self); }
};

/* --- static element: compile-time value by prvalue ---------------- */
template <class XArray, class Index>
struct xarray_access<XArray, Index, true> {
    using tuple_type = typename XArray::tuple_type;
    using value_type = typename XArray::value_type;
    using element    = statix::at<tuple_type, Index>;  // carray<value_type, V>

    using type       = value_type;
    using const_type = value_type;

    _TNYDEF(H,D,I) static value_type at(const XArray &, Index) noexcept
    { return static_cast<value_type>(element::value); }
};

/* --- convenience aliases ------------------------------------------ */

template <class XArray, class Index>
using xarray_access_type = typename xarray_access<XArray, Index>::type;

template <class XArray, class Index>
using xarray_access_const_type = typename xarray_access<XArray, Index>::const_type;

_TNY_NAMESPACE_END(tny)

#endif // TNY__XARRAY_ACCESS
