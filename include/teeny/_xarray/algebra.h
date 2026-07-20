#ifndef TNY__XARRAY_ALGEBRA
#define TNY__XARRAY_ALGEBRA
#include <cuda/std/tuple>
#include <cuda/std/utility>       // index_sequence, make_index_sequence
#include <cuda/std/type_traits>   // common_type_t
#include <teeny/core.h>
#include <teeny/statix.h>
#include <teeny/_xarray/decl.h>
#include <teeny/_xarray/impl.h>

_TNY_NAMESPACE_BEGIN(tny)

/* ================================================================== *
 *  Hybrid xarray algebra.                                            *
 *                                                                    *
 *  Folds constant-fold their fully-static inputs and compute the     *
 *  rest at run time. When *every* participating slot is static the    *
 *  result is a compile-time `cvalue<T, R>` (usable in `static_assert` *
 *  and implicitly convertible to `T`); otherwise it is a plain `T`.  *
 *                                                                    *
 *  All internals are lambda-free (index-sequence folds) so they are  *
 *  safe to instantiate on device without `--extended-lambda`.        *
 * ================================================================== */

namespace _xarray {

template <class V>
using _len = statix::size<statix::as_tuple<V> >;

template <class T, class V>
using _static_prod = statix::prod<statix::as_carray<statix::as_tuple<V>, T> >;
template <class T, class V>
using _static_sum  = statix::sum<statix::as_carray<statix::as_tuple<V>, T> >;
template <class T, class V>
using _static_max  = statix::max<statix::as_carray<statix::as_tuple<V>, T> >;

template <class T, class V, size_t... I>
_TNYDEF(H,D,I,CX) T _prod_dyn(const xarray<T,V>& a, cuda::std::index_sequence<I...>) noexcept {
    T acc = T(1);
    ((acc *= static_cast<T>(a[statix::csize<I>()])), ...);
    return acc;
}

template <class T, class V, size_t... I>
_TNYDEF(H,D,I,CX) T _sum_dyn(const xarray<T,V>& a, cuda::std::index_sequence<I...>) noexcept {
    T acc = T(0);
    ((acc += static_cast<T>(a[statix::csize<I>()])), ...);
    return acc;
}

template <class T, class V, size_t... I>
_TNYDEF(H,D,I,CX) T _max_dyn(const xarray<T,V>& a, cuda::std::index_sequence<I...>) noexcept {
    T acc = static_cast<T>(a[statix::csize<0>()]);
    ((acc = (static_cast<T>(a[statix::csize<I>()]) > acc)
                ? static_cast<T>(a[statix::csize<I>()]) : acc), ...);
    return acc;
}

} // namespace _xarray

/* ------------------------------------------------------------------ *
 *     Folds: prod / sum / max                                        *
 * ------------------------------------------------------------------ */

/** @brief Product of all elements (== numel of a shape). Empty -> 1. */
template <class T, class V>
_TNYDEF(H,D,I,CX) auto prod(const xarray<T,V>& a) noexcept {
    if constexpr (xarray_num_dynamic<V>::value == 0) {
        (void)a;  return _xarray::_static_prod<T,V>{};
    } else {
        return _xarray::_prod_dyn(a, cuda::std::make_index_sequence<_xarray::_len<V>::value>{});
    }
}

/** @brief Sum of all elements. Empty -> 0. */
template <class T, class V>
_TNYDEF(H,D,I,CX) auto sum(const xarray<T,V>& a) noexcept {
    if constexpr (xarray_num_dynamic<V>::value == 0) {
        (void)a;  return _xarray::_static_sum<T,V>{};
    } else {
        return _xarray::_sum_dyn(a, cuda::std::make_index_sequence<_xarray::_len<V>::value>{});
    }
}

/** @brief Maximum element. Requires a non-empty array. */
template <class T, class V>
_TNYDEF(H,D,I,CX) auto max(const xarray<T,V>& a) noexcept {
    static_assert(_xarray::_len<V>::value > 0, "max: empty xarray");
    if constexpr (xarray_num_dynamic<V>::value == 0) {
        (void)a;  return _xarray::_static_max<T,V>{};
    } else {
        return _xarray::_max_dyn(a, cuda::std::make_index_sequence<_xarray::_len<V>::value>{});
    }
}

/* ------------------------------------------------------------------ *
 *     dot  (linear offset from index . stride)                       *
 * ------------------------------------------------------------------ */

namespace _xarray {
template <class TA, class VA, class TB, class VB, size_t... I>
_TNYDEF(H,D,I,CX)
cuda::std::common_type_t<TA,TB>
_dot(const xarray<TA,VA>& a, const xarray<TB,VB>& b, cuda::std::index_sequence<I...>) noexcept {
    using R = cuda::std::common_type_t<TA,TB>;
    R acc = R(0);
    ((acc += static_cast<R>(a[statix::csize<I>()]) * static_cast<R>(b[statix::csize<I>()])), ...);
    return acc;
}
} // namespace _xarray

/**
 * @brief Inner product of two equal-length xarrays.
 *
 * The workhorse for turning a multi-index into a memory offset: any term
 * whose stride slot is static folds its multiply to an immediate.
 */
template <class TA, class VA, class TB, class VB>
_TNYDEF(H,D,I,CX) auto dot(const xarray<TA,VA>& a, const xarray<TB,VB>& b) noexcept {
    static_assert(_xarray::_len<VA>::value == _xarray::_len<VB>::value, "dot: size mismatch");
    return _xarray::_dot(a, b, cuda::std::make_index_sequence<_xarray::_len<VA>::value>{});
}

/* ------------------------------------------------------------------ *
 *     for_each  (unrolled compile-time loop; replaces iterators)     *
 * ------------------------------------------------------------------ */

namespace _xarray {
template <class T, class V, class F, size_t... I>
_TNYDEF(H,D,I) void _for_each(xarray<T,V>& a, F& f, cuda::std::index_sequence<I...>) {
    ((f(statix::csize<I>(), a[statix::csize<I>()])), ...);
}
template <class T, class V, class F, size_t... I>
_TNYDEF(H,D,I) void _for_each(const xarray<T,V>& a, F& f, cuda::std::index_sequence<I...>) {
    ((f(statix::csize<I>(), a[statix::csize<I>()])), ...);
}
} // namespace _xarray

/**
 * @brief Apply `f(index, element)` to every slot, fully unrolled.
 *
 * `index` is a `statix::csize<I>` compile-time tag (usable for per-dim
 * dispatch); `element` is a mutable reference for dynamic slots and a
 * prvalue for static ones. On device, pass a `__host__ __device__` lambda
 * (compile with `--extended-lambda`).
 */
template <class T, class V, class F>
_TNYDEF(H,D,I) void for_each(xarray<T,V>& a, F&& f) {
    _xarray::_for_each(a, f, cuda::std::make_index_sequence<_xarray::_len<V>::value>{});
}
template <class T, class V, class F>
_TNYDEF(H,D,I) void for_each(const xarray<T,V>& a, F&& f) {
    _xarray::_for_each(a, f, cuda::std::make_index_sequence<_xarray::_len<V>::value>{});
}

/* ------------------------------------------------------------------ *
 *     from_pointer  (kernel-boundary construction)                   *
 * ------------------------------------------------------------------ */

namespace _xarray {
// Assign the K-th logical slot from p[K] iff it is dynamic (static slots
// keep their compile-time value; the read is discarded).
template <size_t K, class T, class V>
_TNYDEF(H,D,I) void _fp_one(xarray<T,V>& a, const T* p) {
    if constexpr (statix::is_cnone<statix::at<statix::as_tuple<V>, statix::csize<K> > >::value)
        a.at(statix::csize<K>()) = p[K];
    else { (void)a; (void)p; }
}
template <class V, class T, size_t... I>
_TNYDEF(H,D,I) xarray<T,V> _from_pointer(const T* p, cuda::std::index_sequence<I...>) {
    xarray<T,V> a{};
    (_fp_one<I>(a, p), ...);
    return a;
}
} // namespace _xarray

/**
 * @brief Build an `xarray<T, V>` from a raw pointer of `size()` logical
 *        values (as jitfields passes `size[]` / `stride[]`).
 *
 * Dynamic slots take `p[i]`; static slots keep their compile-time value
 * (their `p[i]` is ignored).
 */
template <class V, class T>
_TNYDEF(H,D,I) xarray<T,V> from_pointer(const T* p) {
    return _xarray::_from_pointer<V>(
        p, cuda::std::make_index_sequence<_xarray::_len<V>::value>{});
}

_TNY_NAMESPACE_END(tny)

#endif // TNY__XARRAY_ALGEBRA
