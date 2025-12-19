/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file implements indexing utility for template parameter packs      **
 **                                                                         **
 ** wrap_index_value : Makes python-like negative indices positive          **
 ** wrap_index       : WrapIndex applied to all elements in a container     **
 ** slice            : The metatemplating equivalent of python's `slice`    **
 ** simple_slice     : A simpler slice that only accepts concrete arguments **
 ** as_index_carray  : Convert slice to wrapped PtrDiff indices             **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef TNY__STATIX__INDEX_DECL
#define TNY__STATIX__INDEX_DECL
#include <teeny/core.h>
#include <teeny/_statix/_carray/decl.h>     // cptrdiff

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                           *
 *                            Python-like slice                              *
 *                                                                           *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**
 * @brief A simple slice with concrete values.
 *
 * @tparam start  Start index (inclusive)
 * @tparam stop   Stop index  (exclusive)
 * @tparam step   Step size   (default: 1)
 *
 * @example simple_slice<1,4>
 */
template <ptrdiff_t start, ptrdiff_t stop, ptrdiff_t step = 1>
struct simple_slice;

/**
 * @brief A general slice with possibly non-concrete values.
 *
 * Start/Stop/Step must be typename if we want to support the value `cnone`.
 * For values that are not `cnone`, users should wrap them in a `cscalar`
 * to transform them into types.
 *
 * @tparam start  Start index (cscalar or cnone, inclusive)
 * @tparam stop   Stop index  (cscalar or cnone, exclusive)
 * @tparam step   Step size   (cscalar or cnone, default: cptrdiff<1>)
 *
 * @example slice<clong<1>, cnone, clong<2>>
 */
template <class start, class stop, class step = cptrdiff<1>>
struct slice;

/* ================================================================== *
 *      Slice to range                                                *
 * ================================================================== */

template <class slice, size_t length>
struct _as_index_carray;

/**
 * @brief Convert a slice to a carray of wrapped indices.
 *
 * as_index_carray<slice, length> = cptrdiff<I...>
 *
 * @tparam slice   Input slice
 * @tparam length  Number of elements in the tuple/carray being sliced
 *
 * @example as_index_carray<simple_slice<1,4>, 5>      = cptrdiff<1,2,3>
 * @example as_index_carray<slice<clong<1>, cnone>, 5> = cptrdiff<1,2,3,4>
 */
template <class slice, size_t length>
using as_index_carray = typename _as_index_carray<slice,length>::type;

/* ================================================================== *
 *      Index complement                                              *
 * ================================================================== */

template <class length, class index>
struct _index_complement;

/**
 * @brief Get the complement of indices I in [0, N)
 *
 * index_complement<N, I> = cptrdiff<...>
 *
 * @tparam length   Number of elements in the tuple/vector.
 * @tparam index    Indices to exclude.
 *
 * @example index_complement<csize<5>, cptrdiff<1, 3>> = cptrdiff<0, 2, 4>
 */
template <class length, class index>
using index_complement = typename _index_complement<length, index>::type;

/* ================================================================== *
 *      Python-like negative indexing                                 *
 * ================================================================== */

template <size_t length, ptrdiff_t... index>
struct _wrap_index_value; // type = cptrdiff<index...>

/**
 * @brief Transform negative indices into positive indices (Python convention).
 *
 * wrap_index_value<Length, Index...> = cptrdiff<Index...>
 *
 * @tparam length  Number of elements in the tuple/vector.
 * @tparam index   Index to (maybe) wrap. If negative, counts from the end.
 *
 * @example wrap_index_value<3, -1>     = cptrdiff<2>
 * @example wrap_index_value<5, -4, -1> = cptrdiff<1, 4>
 */
template <size_t length, ptrdiff_t... index>
using wrap_index_value = typename _wrap_index_value<length, index...>::type;

template <class length, class index>
struct _wrap_index;

/**
 * @brief Transform negative indices into positive indices (Python convention).
 *
 * If the index is a slice, it is further converted to a cptrdiff carray.
 *
 * wrap_index<length, index> = index
 *
 * @tparam length  Number of elements in the tuple/carray.
 * @tparam index   Index to (maybe) wrap. If negative, counts from the end.
 *
 * @example wrap_index<csize<3>, cptrdiff<-1>>            = cptrdiff<2>
 * @example wrap_index<csize<5>, simple_slice<-4, -1, 2>> = cptrdiff<1, 3>
 */
template <class length, class index>
using wrap_index = typename _wrap_index<length, index>::type;

/* ================================================================== *
 *      Convert Scalar (or None) to integer                           *
 * ================================================================== */

template <class output_type, class scalar_type, output_type default_value>
struct _set_default;

/**
 * @brief Convert a type that can be `cscalar<output_type, value>` or `cnone`
 *        into `value` or `default_value`.
 *
 * @tparam output_type    The scalar data type (e.g., `long`)
 * @tparam scalar_type    The meta scalar type (e.g., `clong<1>`)
 * @tparam default_value  Default value to use instead of `cnone`
 */
template <class output_type, class scalar_type,
          output_type default_value = static_cast<output_type>(0)>
using set_default = typename _set_default<output_type, scalar_type, default_value>::type;

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                           *
 *                     Register compile-time indices                         *
 *                                                                           *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

template <class T> struct _is_static_index { using type = cfalse; };
template <class T> using   is_static_index = typename _is_static_index<T>::type;

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif // TNY__STATIX__INDEX_DECL
