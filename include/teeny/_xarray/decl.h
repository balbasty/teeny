#ifndef TNY__XARRAY_DECL
#define TNY__XARRAY_DECL
#include <teeny/core.h>

_TNY_NAMESPACE_BEGIN(tny)

/**
 * @brief Hybrid one-dimensional array.
 *
 * Each element in the array is either static (a compile-time known value,
 * spelled `cvalue<T, X>`) or dynamic (a run-time value, spelled `cnone`).
 *
 * `values` is a pack-like type listing the slots, e.g.
 * `tuple<cvalue<T,X0>, cnone, cvalue<T,X2>, ...>`.
 *
 * Only the *dynamic* elements are stored: the storage is exactly
 * `num_dynamic` values of type `T`, so
 * `sizeof(xarray) == max(1, num_dynamic * sizeof(T))` -- static elements
 * cost no memory (they live only in the type). This makes an `xarray`
 * cheap to pass into a `__global__` kernel by value.
 *
 * xarray<T, values>
 *
 * @tparam T        Element Type.
 * @tparam values   Compile-time slot descriptor (pack of cvalue / cnone).
 */
template <class T, class values>
struct xarray;

/* ------------------------------------------------------------------ *
 *     xarray_tuple                                                   *
 * ------------------------------------------------------------------ */

template <class T, class values>
struct _xarray_tuple;

/**
 * @brief The storage type of an xarray: a plain tuple of exactly the
 *        dynamic elements (`cuda::std::tuple<T, ..., T>`, one `T` per
 *        dynamic slot). Static slots are absent from storage.
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
 *        Result is a `statix::csize<D>`.
 */
template <class values>
using xarray_num_dynamic = typename _xarray_num_dynamic<values>::type;

/* ------------------------------------------------------------------ *
 *     dynamic_ordinal                                                *
 * ------------------------------------------------------------------ */

template <class values, size_t I>
struct _dynamic_ordinal;

/**
 * @brief Storage position of the dynamic element at logical index `I`.
 *
 * Equals the number of dynamic (`cnone`) slots strictly before `I`, i.e.
 * the ordinal of that dynamic element within the dynamic-only storage
 * tuple. Result is a `statix::csize<P>`. (Only meaningful when the slot
 * at `I` is itself dynamic.)
 */
template <class values, size_t I>
using dynamic_ordinal = typename _dynamic_ordinal<values, I>::type;

_TNY_NAMESPACE_END(tny)

#endif // TNY__XARRAY_DECL
