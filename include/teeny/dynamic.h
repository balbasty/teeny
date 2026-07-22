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

// The shape/stride store of an `anyrank` is itself a 1-D teeny tensor:
//   - `_meta_store` : an INLINE stack tensor of `TNY_MAX_RANK` (default) — the
//     sizes travel WITH the carrier, so it stays trivially copyable and can be
//     passed into a CUDA kernel by value (peel on device).
//   - `_meta_view`  : a non-owning VIEW of external size/stride arrays (e.g. a
//     DLPack tensor's), so the carrier wraps them with NO copy. HOST use only —
//     those pointers are not valid inside a device kernel.
template <class offset_t, cs::size_t N>
using _meta_store = tensor<offset_t, cs::extents<offset_t, N>, cs::layout_right, own::stack>;
template <class offset_t>
using _meta_view = tensor<offset_t, cs::dextents<offset_t, 1>, cs::layout_right, own::view>;

template <class T, class offset_t, class Meta, cs::size_t Sr> struct anyrank_front;  // fwd

/**
 * @brief A rank-erased tensor for the host/ndarray dispatch boundary.
 *
 * Holds a data pointer, a runtime `ndim`, and 1-D `shape`/`stride` tensors
 * (`Meta`). By default those are INLINE (a `TNY_MAX_RANK` stack store), so the
 * carrier is trivially copyable and passes into a CUDA kernel by value; wrap a
 * DLPack tensor with NO copy via `as_anyrank_view` (a `_meta_view` store).
 *
 * You do NOT compute on it — it is a *doorway*, not a room. Turn it into a
 * statically-typed view at the boundary and compute on that:
 *   - `fixed<R>()`            — force a known total rank R.
 *   - `dispatch_rank(...)`    — pick R from the runtime `ndim`.
 *   - `peel_front<Sr>()`      — the batch idiom: peel the runtime number of
 *                               leading batch dims, keep the trailing `Sr`
 *                               "interesting" dims STATIC. One kernel per Sr.
 *
 * Deliberately no `add_`/`mul_`/etc.: a runtime-rank arithmetic path would loop
 * over `ndim` (killing folding) or dispatch to every rank (the bloat
 * `peel_front<Sr>` avoids). Do host-side math on a `fixed<R>()`/`peel_front<Sr>()`
 * view instead.
 */
template <class T, class offset_t = cs::int64_t, class Meta = _meta_store<offset_t, TNY_MAX_RANK>>
struct anyrank {
    T *  data = nullptr;
    Meta shape{};      // 1-D tensor of sizes   (inline, or a view of external memory)
    Meta stride{};     // 1-D tensor of strides
    int  ndim = 0;

    // Largest rank the store can hold: the inline store's static length, else
    // (a view store) the compile-time dispatch bound TNY_MAX_RANK.
    static constexpr cs::size_t max_rank =
        Meta::extents_type::static_extent(0) != cs::dynamic_extent
            ? Meta::extents_type::static_extent(0) : cs::size_t(TNY_MAX_RANK);

    _TNY_API offset_t size(int i)  const noexcept { return shape(i); }   // size of dim i
    _TNY_API offset_t step(int i)  const noexcept { return stride(i); }  // stride of dim i

    /** @brief View this tensor as a fixed rank `R` (requires `ndim == R`). */
    template <cs::size_t R>
    _TNY_API dyn_tensor<T, offset_t, R> fixed() const {
        using E = cs::dextents<offset_t, R>;
        cs::array<offset_t, R> ext{}, st{};
        for (cs::size_t i = 0; i < R; ++i) { ext[i] = shape(i); st[i] = stride(i); }
        cs::layout_stride::mapping<E> m(E(ext), st);
        return dyn_tensor<T, offset_t, R>(data, m);
    }

    /** @brief The `lin`-th sub-view obtained by peeling the leading `ndim - Sr`
     *         BATCH axes (runtime count) -> a fixed-rank-`Sr` view over the
     *         trailing "interesting" axes. Grid-stride friendly (device-safe
     *         with the inline store). Follow with `recast<shape<-1,...>>()`. */
    template <cs::size_t Sr>
    _TNY_API dyn_tensor<T, offset_t, Sr> peel_front_at(offset_t lin) const {
        const int nb = ndim - static_cast<int>(Sr);          // # batch dims (runtime)
        _TNY_CHECK(nb >= 0, "peel_front: Sr exceeds ndim");
        offset_t off = 0, rem = lin;                          // decode lin over batch axes
        for (int d = nb - 1; d >= 0; --d) { offset_t k = rem % shape(d); rem /= shape(d); off += k * stride(d); }
        using E = cs::dextents<offset_t, Sr>;
        cs::array<offset_t, Sr> ext{}, st{};
        for (cs::size_t i = 0; i < Sr; ++i) { ext[i] = shape(nb + i); st[i] = stride(nb + i); }
        cs::layout_stride::mapping<E> m(E(ext), st);
        return dyn_tensor<T, offset_t, Sr>(data + off, m);
    }

    /** @brief Peel the leading batch axes -> an iterable of fixed-rank-`Sr`
     *         sub-views (range-for, `size()`, `operator[]`). The
     *         `(*batch, *spatial, C)` boundary with `Sr = spatial + channels`:
     *         the kernel instantiates ONCE for `Sr`, not once per total rank. */
    template <cs::size_t Sr>
    _TNY_API anyrank_front<T, offset_t, Meta, Sr> peel_front() const { return { *this }; }
};

/** @brief A range of fixed-rank-`Sr` sub-views over an `anyrank`'s batch axes. */
template <class T, class offset_t, class Meta, cs::size_t Sr>
struct anyrank_front {
    anyrank<T, offset_t, Meta> src;

    _TNY_API offset_t size() const noexcept {
        offset_t n = 1;
        for (int d = 0; d < src.ndim - static_cast<int>(Sr); ++d) n *= src.shape(d);
        return n;
    }
    _TNY_API auto operator[](offset_t i) const { return src.template peel_front_at<Sr>(i); }

    struct iterator {
        anyrank_front r; offset_t i;   // by value (POD carrier) -> no dangle on a temporary range
        _TNY_API auto operator*() const { return r[i]; }
        _TNY_API iterator & operator++() { ++i; return *this; }
        _TNY_API bool operator!=(const iterator & o) const { return i != o.i; }
    };
    _TNY_API iterator begin() const { return { *this, 0 }; }
    _TNY_API iterator end()   const { return { *this, size() }; }
};

/** @brief Build an `anyrank` that COPIES shape/stride into its inline store
 *         (device-passable). `MaxRank` sets the inline capacity (default
 *         `TNY_MAX_RANK`). DLPack strides are in ELEMENTS; numpy
 *         `__array_interface__` in BYTES (divide by the itemsize first). */
template <cs::size_t MaxRank = TNY_MAX_RANK, class T, class offset_t>
_TNY_HOST anyrank<T, offset_t, _meta_store<offset_t, MaxRank>>
as_anyrank(T * data, const offset_t * shape, const offset_t * stride, int ndim) {
    anyrank<T, offset_t, _meta_store<offset_t, MaxRank>> t;
    t.data = data; t.ndim = ndim;
    // Never write past the inline store: ndim can come straight from a DLPack
    // caller (torch allows 64 dims). Copy at most MaxRank; an oversized ndim is
    // then simply never matched by dispatch_rank / fixed<R>.
    _TNY_CHECK(ndim <= static_cast<int>(MaxRank), "as_anyrank: ndim exceeds MaxRank (raise -DTNY_MAX_RANK)");
    const int n = ndim < static_cast<int>(MaxRank) ? ndim : static_cast<int>(MaxRank);
    for (int i = 0; i < n; ++i) { t.shape(i) = shape[i]; t.stride(i) = stride[i]; }
    return t;
}

/** @brief Build an `anyrank` that WRAPS external shape/stride arrays with no
 *         copy (a view store) — e.g. straight off a DLPack tensor. The arrays
 *         must outlive the carrier. HOST only: the pointers are not valid inside
 *         a device kernel, so peel/dispatch on the host and pass the resulting
 *         fixed-rank views to the device. */
template <class T, class offset_t>
_TNY_HOST anyrank<T, offset_t, _meta_view<offset_t>>
as_anyrank_view(T * data, offset_t * shape, offset_t * stride, int ndim) {
    anyrank<T, offset_t, _meta_view<offset_t>> t;
    t.data = data; t.ndim = ndim;
    cs::dextents<offset_t, 1> e{ static_cast<offset_t>(ndim) };
    t.shape  = _meta_view<offset_t>(shape,  e);
    t.stride = _meta_view<offset_t>(stride, e);
    return t;
}

namespace _detail {
template <cs::size_t R, class T, class offset_t, class Meta, class F>
_TNY_HOST bool dispatch_from(const anyrank<T, offset_t, Meta> & t, F & f) {
    if constexpr (R <= anyrank<T, offset_t, Meta>::max_rank) {
        if (t.ndim == static_cast<int>(R)) { f(t.template fixed<R>()); return true; }
        return dispatch_from<R + 1>(t, f);
    } else {
        (void)t; (void)f; return false;   // ndim > max_rank
    }
}
} // namespace _detail

/**
 * @brief Call `f` with a fixed-rank view of `t` chosen by its runtime `ndim`.
 *
 * `f` is a generic callable instantiated once per possible rank; the kernel it
 * launches is fully static. Returns false if `ndim` exceeds `max_rank`. Prefer
 * `peel_front<Sr>` when only the trailing dims need to be static — one
 * instantiation instead of one per total rank.
 *
 *     dispatch_rank(as_anyrank(data, size, stride, ndim), [&](auto v){ kernel(v); });
 */
template <class T, class offset_t, class Meta, class F>
_TNY_HOST bool dispatch_rank(const anyrank<T, offset_t, Meta> & t, F && f) {
    return _detail::dispatch_from<0>(t, f);   // R=0 handles a rank-0 (scalar) ndarray
}

/**
 * @brief Turn a runtime value into a compile-time one from a candidate list.
 *
 * `dispatch_value<1,2,3>(D, f)` calls `f(Int<k>{})` for the matching candidate
 * `k == D` (so `f` receives a static `integral_constant` it can use as a
 * template argument), and returns whether any matched.
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
