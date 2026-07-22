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

template <class T, class offset_t, cs::size_t MaxRank, cs::size_t Sr> struct anyrank_front;  // fwd

/**
 * @brief A rank-erased tensor for the host/ndarray dispatch boundary.
 *
 * Carries a pointer plus bounded shape/stride arrays and a runtime `ndim`
 * (numpy/torch/cupy tensors have a small maximum rank; `MaxRank` defaults to 8).
 * Trivially copyable, so it passes into a CUDA kernel by value.
 *
 * You do NOT compute on it — it is a *doorway*, not a room. Turn it into a
 * statically-typed view at the boundary and compute on that:
 *   - `fixed<R>()`            — force a known total rank R.
 *   - `dispatch_rank(...)`    — pick R from the runtime `ndim`.
 *   - `peel_front<Sr>()`      — the batch idiom: peel the runtime number of
 *                               leading batch dims, keep the trailing `Sr`
 *                               "interesting" dims STATIC. One kernel per Sr.
 *
 * Deliberately no `add_`/`mul_`/etc. here. A runtime-rank arithmetic path would
 * either loop over `ndim` (killing the compile-time folding teeny exists for) or
 * dispatch internally to every rank (the binary bloat `peel_front<Sr>` avoids) —
 * so host-side setup math should go through `fixed<R>()`/`peel_front<Sr>()` onto
 * a static view, not onto the carrier. (See the dynamic-rank design notes.)
 */
template <class T, class offset_t, cs::size_t MaxRank = 8>
struct anyrank {
    T *      data = nullptr;
    offset_t shape [MaxRank] = {};
    offset_t stride[MaxRank] = {};
    int      ndim = 0;

    static constexpr cs::size_t max_rank = MaxRank;

    /** @brief View this tensor as a fixed rank `R` (requires `ndim == R`). */
    template <cs::size_t R>
    _TNY_API dyn_tensor<T, offset_t, R> fixed() const {
        using E = cs::dextents<offset_t, R>;
        cs::array<offset_t, R> ext{}, str{};
        for (cs::size_t i = 0; i < R; ++i) { ext[i] = shape[i]; str[i] = stride[i]; }
        cs::layout_stride::mapping<E> m(E(ext), str);
        return dyn_tensor<T, offset_t, R>(data, m);
    }

    /** @brief The `lin`-th sub-view obtained by peeling the leading `ndim - Sr`
     *         BATCH axes (runtime count) -> a fixed-rank-`Sr` view over the
     *         trailing "interesting" axes. Grid-stride friendly (device-safe).
     *         Follow with `recast<shape<-1,...>>()` to fold known inner dims. */
    template <cs::size_t Sr>
    _TNY_API dyn_tensor<T, offset_t, Sr> peel_front_at(offset_t lin) const {
        const int nb = ndim - static_cast<int>(Sr);          // # batch dims (runtime)
        _TNY_CHECK(nb >= 0, "peel_front: Sr exceeds ndim");
        offset_t off = 0, rem = lin;                          // decode lin over batch axes
        for (int d = nb - 1; d >= 0; --d) { offset_t k = rem % shape[d]; rem /= shape[d]; off += k * stride[d]; }
        using E = cs::dextents<offset_t, Sr>;
        cs::array<offset_t, Sr> ext{}, str{};
        for (cs::size_t i = 0; i < Sr; ++i) { ext[i] = shape[nb + i]; str[i] = stride[nb + i]; }
        cs::layout_stride::mapping<E> m(E(ext), str);
        return dyn_tensor<T, offset_t, Sr>(data + off, m);
    }

    /** @brief Peel the leading batch axes -> an iterable of fixed-rank-`Sr`
     *         sub-views (range-for, `size()`, `operator[]`). This is the
     *         `(*batch, *spatial, C)` boundary with `Sr = spatial + channels`:
     *         the kernel instantiates ONCE for `Sr`, not once per total rank. */
    template <cs::size_t Sr>
    _TNY_API anyrank_front<T, offset_t, MaxRank, Sr> peel_front() const { return { *this }; }
};

/** @brief A range of fixed-rank-`Sr` sub-views over an `anyrank`'s batch axes. */
template <class T, class offset_t, cs::size_t MaxRank, cs::size_t Sr>
struct anyrank_front {
    anyrank<T, offset_t, MaxRank> src;

    _TNY_API offset_t size() const noexcept {
        offset_t n = 1;
        for (int d = 0; d < src.ndim - static_cast<int>(Sr); ++d) n *= src.shape[d];
        return n;
    }
    _TNY_API auto operator[](offset_t i) const { return src.template peel_front_at<Sr>(i); }

    struct iterator {
        const anyrank_front * r; offset_t i;
        _TNY_API auto operator*() const { return (*r)[i]; }
        _TNY_API iterator & operator++() { ++i; return *this; }
        _TNY_API bool operator!=(const iterator & o) const { return i != o.i; }
    };
    _TNY_API iterator begin() const { return { this, 0 }; }
    _TNY_API iterator end()   const { return { this, size() }; }
};

/** @brief Build an `anyrank` from raw data + shape/stride + runtime rank.
 *         (DLPack strides are in ELEMENTS; numpy `__array_interface__` in BYTES.) */
template <cs::size_t MaxRank = 8, class T, class offset_t>
_TNY_HOST anyrank<T, offset_t, MaxRank>
any(T * data, const offset_t * shape, const offset_t * stride, int ndim) {
    anyrank<T, offset_t, MaxRank> t;
    t.data = data; t.ndim = ndim;
    for (int i = 0; i < ndim; ++i) { t.shape[i] = shape[i]; t.stride[i] = stride[i]; }
    return t;
}

namespace _detail {
template <cs::size_t R, class T, class offset_t, cs::size_t MaxRank, class F>
_TNY_HOST bool dispatch_from(const anyrank<T, offset_t, MaxRank> & t, F & f) {
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
 * `f` is a generic callable instantiated once per possible rank; the kernel it
 * launches is fully static. Returns false if `ndim` exceeds `MaxRank`. Prefer
 * `peel_front<Sr>` when only the trailing dims need to be static — it costs one
 * instantiation instead of one per total rank.
 *
 *     dispatch_rank(any(data, size, stride, ndim), [&](auto v){ my_kernel(v); });
 */
template <class T, class offset_t, cs::size_t MaxRank, class F>
_TNY_HOST bool dispatch_rank(const anyrank<T, offset_t, MaxRank> & t, F && f) {
    return _detail::dispatch_from<1>(t, f);
}

/**
 * @brief Turn a runtime value into a compile-time one from a candidate list.
 *
 * `dispatch_value<1,2,3>(D, f)` calls `f(Int<k>{})` for the matching candidate
 * `k == D` (so `f` receives a static `integral_constant` it can use as a
 * template argument), and returns whether any matched. This is how a kernel
 * turns a runtime spatial rank / interpolation order / boundary mode into a
 * template parameter *early*, then dispatches to fully-static code.
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
