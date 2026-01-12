#ifndef TNY__XARRAY_DECL
#define TNY__XARRAY_DECL
#include <cuda/std/tuple>
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
 * times the size of the element type T.
 *
 * xarray<T, values>
 *
 * @tparam T        Element Type.
 * @tparam values   Compile-time values.
 */
template <class T, class values>
struct xarray;

/**
 * @brief Number of dynamic elements in an xarray type.
 */
template <size_t>
struct xarray_numel;

template <class T, class values>
struct _xarray_tuple;

/**
 * @brief The tuple-type equivalent of an xarray type.
 */
template <class T, class values>
using xarray_tuple = typename _xarray_tuple<T, values>::type;

_TNY_NAMESPACE_END(tny)

#endif // TNY__XARRAY_DECL
