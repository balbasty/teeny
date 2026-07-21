#ifndef TNY_MD_DYNAMIC
#define TNY_MD_DYNAMIC
#include <cuda/std/mdspan>
#include <cuda/std/array>
#include <teeny/defines.h>
#include <teeny/tensor.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/** @brief A fixed-rank, fully-dynamic, arbitrarily-strided tensor view. */
template <class T, class offset_t, cs::size_t R>
using dyn_tensor = tensor<T, cs::dextents<offset_t, R>, cs::layout_stride, own::view>;

/**
 * @brief A rank-erased tensor for the host dispatch boundary.
 *
 * Carries a pointer plus bounded shape/stride arrays and a runtime `ndim`
 * (numpy/torch/cupy tensors have a small maximum rank; `MaxRank` defaults to 8).
 * You never compute on it directly -- you `dispatch_rank` it into a fixed-rank
 * `md::tensor`, so the actual kernel stays fully static. Trivially copyable.
 *
 * This is the ndindex/rank plumbing jitfields does by hand at the host boundary.
 */
template <class T, class offset_t, cs::size_t MaxRank = 8>
struct any_tensor {
    T *    data = nullptr;
    offset_t shape [MaxRank] = {};
    offset_t stride[MaxRank] = {};
    int    ndim = 0;

    static constexpr cs::size_t max_rank = MaxRank;

    /** @brief View this tensor as a fixed rank `R` (requires `ndim == R`). */
    template <cs::size_t R>
    _TNY_HOST dyn_tensor<T, offset_t, R> fixed() const {
        using E = cs::dextents<offset_t, R>;
        cs::array<offset_t, R> ext{}, str{};
        for (cs::size_t i = 0; i < R; ++i) { ext[i] = shape[i]; str[i] = stride[i]; }
        cs::layout_stride::mapping<E> m(E(ext), str);
        return dyn_tensor<T, offset_t, R>(data, m);
    }
};

/** @brief Build an `any_tensor` from raw data + shape/stride + runtime rank. */
template <cs::size_t MaxRank = 8, class T, class offset_t>
_TNY_HOST any_tensor<T, offset_t, MaxRank>
any(T * data, const offset_t * shape, const offset_t * stride, int ndim) {
    any_tensor<T, offset_t, MaxRank> t;
    t.data = data; t.ndim = ndim;
    for (int i = 0; i < ndim; ++i) { t.shape[i] = shape[i]; t.stride[i] = stride[i]; }
    return t;
}

namespace _detail {
template <cs::size_t R, class T, class offset_t, cs::size_t MaxRank, class F>
_TNY_HOST bool dispatch_from(const any_tensor<T, offset_t, MaxRank> & t, F & f) {
    if constexpr (R <= MaxRank) {
        if (t.ndim == static_cast<int>(R)) { f(t.template fixed<R>()); return true; }
        return dispatch_from<R + 1>(t, f);
    } else {
        (void)t; (void)f; return false;   // ndim > MaxRank
    }
}
} // namespace _detail

/**
 * @brief Call `f` with a fixed-rank view of `t` chosen by its runtime `ndim`.
 *
 * `f` is a generic callable (e.g. `[](auto view){ ... }`) instantiated once per
 * possible rank; the device kernel it launches is fully static. Returns false
 * if `ndim` exceeds `MaxRank`.
 *
 *     dispatch_rank(any(data, size, stride, ndim), [&](auto v){ my_kernel(v); });
 */
template <class T, class offset_t, cs::size_t MaxRank, class F>
_TNY_HOST bool dispatch_rank(const any_tensor<T, offset_t, MaxRank> & t, F && f) {
    return _detail::dispatch_from<1>(t, f);
}

/**
 * @brief Turn a runtime value into a compile-time one from a candidate list.
 *
 * `dispatch_value<1,2,3>(D, f)` calls `f(Int<k>{})` for the matching candidate
 * `k == D` (so `f` receives a static `integral_constant` it can use as a
 * template argument), and returns whether any matched. This is how a kernel
 * turns a runtime spatial rank / interpolation order / boundary mode into a
 * template parameter *early*, then dispatches to fully-static code — the
 * `(*batch, *spatial, C)` pattern where spatial rank D ∈ {1,2,3} is specialised.
 *
 *     dispatch_value<1,2,3>(ndim_spatial, [&](auto d){ kernel<d.value>(view); });
 */
template <int... Vs, class F>
_TNY_HOST bool dispatch_value(int v, F && f) {
    bool matched = false;
    ( (v == Vs ? (f(cs::integral_constant<int, Vs>{}), matched = true) : false), ... );
    return matched;
}

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_DYNAMIC
