#ifndef TNY__XARRAY_DECL
#define TNY__XARRAY_DECL
#include <teeny/core.h>

_TNY_NAMESPACE_BEGIN(tny)

/**
 * @brief Hybrid one-dimensional array.
 *
 * Each element in the array can be either static (compile-time known value)
 * or dynamic (run-time value).
 *
 * `values` is a pack-like type containing either `cvalue<T, X>` for static
 * values or `cnone` for dynamic values. E.g.: `tuple<cvalue<T,X0>, cnone, cvalue<T,X2>, ...>`
 *
 * The size of the array in memory is equal to the number of dynamic elements
 * times the size of the element type T (static elements are stored as empty
 * base subobjects and take no space).
 *
 * xarray<T, values>
 *
 * @tparam T        Element Type.
 * @tparam values   Compile-time values.
 */
template <class T, class values>
struct xarray;

/* ------------------------------------------------------------------ *
 *     xarray_tuple                                                   *
 * ------------------------------------------------------------------ */

template <class T, class values>
struct _xarray_tuple;

/**
 * @brief The storage tuple-type equivalent of an xarray type.
 *
 * Static elements become empty `carray` leaves (zero storage under EBO),
 * dynamic elements become plain `T` leaves.
 */
template <class T, class values>
using xarray_tuple = typename _xarray_tuple<T, values>::type;

/* ------------------------------------------------------------------ *
 *     xarray_num_dynamic                                             *
 * ------------------------------------------------------------------ */

template <class values>
struct _xarray_num_dynamic;

/**
 * @brief Number of dynamic (runtime-stored) elements in a `values` pack.
 *
 * Result is a `statix::csize<D>`.
 */
template <class values>
using xarray_num_dynamic = typename _xarray_num_dynamic<values>::type;

_TNY_NAMESPACE_END(tny)

#endif // TNY__XARRAY_DECL
