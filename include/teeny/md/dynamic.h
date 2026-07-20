#ifndef TNY_MD_DYNAMIC
#define TNY_MD_DYNAMIC
#include <cuda/std/mdspan>
#include <cuda/std/array>
#include <teeny/_core/defines.h>
#include <teeny/md/tensor.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(md)

namespace cs = cuda::std;

/** @brief A fixed-rank, fully-dynamic, arbitrarily-strided tensor view. */
template <class T, class Offset, cs::size_t R>
using dyn_tensor = tensor<T, cs::dextents<Offset, R>, cs::layout_stride, own::view>;

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
template <class T, class Offset, cs::size_t MaxRank = 8>
struct any_tensor {
    T *    data = nullptr;
    Offset shape [MaxRank] = {};
    Offset stride[MaxRank] = {};
    int    ndim = 0;

    static constexpr cs::size_t max_rank = MaxRank;

    /** @brief View this tensor as a fixed rank `R` (requires `ndim == R`). */
    template <cs::size_t R>
    _TNY_HOST dyn_tensor<T, Offset, R> fixed() const {
        using E = cs::dextents<Offset, R>;
        cs::array<Offset, R> ext{}, str{};
        for (cs::size_t i = 0; i < R; ++i) { ext[i] = shape[i]; str[i] = stride[i]; }
        cs::layout_stride::mapping<E> m(E(ext), str);
        return dyn_tensor<T, Offset, R>(data, m);
    }
};

/** @brief Build an `any_tensor` from raw data + shape/stride + runtime rank. */
template <cs::size_t MaxRank = 8, class T, class Offset>
_TNY_HOST any_tensor<T, Offset, MaxRank>
any(T * data, const Offset * shape, const Offset * stride, int ndim) {
    any_tensor<T, Offset, MaxRank> t;
    t.data = data; t.ndim = ndim;
    for (int i = 0; i < ndim; ++i) { t.shape[i] = shape[i]; t.stride[i] = stride[i]; }
    return t;
}

namespace _detail {
template <cs::size_t R, class T, class Offset, cs::size_t MaxRank, class F>
_TNY_HOST bool dispatch_from(const any_tensor<T, Offset, MaxRank> & t, F & f) {
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
template <class T, class Offset, cs::size_t MaxRank, class F>
_TNY_HOST bool dispatch_rank(const any_tensor<T, Offset, MaxRank> & t, F && f) {
    return _detail::dispatch_from<1>(t, f);
}

_TNY_NAMESPACE_END(md)
_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_DYNAMIC
