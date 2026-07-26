#ifndef TNY_MD_DYNAMIC
#define TNY_MD_DYNAMIC
#include <cuda/std/mdspan>
#include <cuda/std/array>
#include <teeny/defines.h>
#include <teeny/alias.h>
#include <teeny/tensor.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/** @brief A fixed-rank, fully-dynamic, arbitrarily-strided tensor view. `O` is the
 *         memory space of the view — `storage::view` (host) by default, `storage::gpu_view`
 *         when the pointer lives in device memory (see `anyrank`'s `Space`). */
template <class T, class offset_t, cs::size_t R, storage O = storage::view>
using dyn_tensor = tensor<T, cs::dextents<offset_t, R>, cs::layout_stride, O>;

// Is `E` a `cs::extents<...>` (a teeny `shape<...>`)? Gates the shape-form
// `peel_front[_at]<NewE>` overloads apart from the negative-`long` ones so a
// mis-typed `peel_front_at<Int<3>>` gives a clear message, not a deep recast error.
template <class> struct _is_extents : cs::false_type {};
template <class I, cs::size_t... E> struct _is_extents<cs::extents<I, E...>> : cs::true_type {};

// The shape/stride store of an `anyrank` is itself a 1-D teeny tensor:
//   - `_meta_store` : an INLINE stack tensor of `TNY_MAX_RANK` (default) — the
//     sizes travel WITH the carrier, so it stays trivially copyable and can be
//     passed into a CUDA kernel by value (peel on device).
//   - `_meta_view`  : a non-owning VIEW of external size/stride arrays (e.g. a
//     DLPack tensor's), so the carrier wraps them with NO copy. HOST use only —
//     those pointers are not valid inside a device kernel.
template <class offset_t, cs::size_t N>
using _meta_store = tensor<offset_t, cs::extents<offset_t, N>, ccontiguous, storage::stack>;
template <class offset_t>
using _meta_view = tensor<offset_t, cs::dextents<offset_t, 1>, ccontiguous, storage::view>;

template <class T, class offset_t, class Meta, storage Space, cs::size_t Sr> struct anyrank_front;  // fwd

/** @brief Tag for `as_anyrank(..., copy_meta)`: COPY shape/stride into an inline,
 *         device-passable store instead of wrapping the caller's arrays. Named
 *         `copy_meta`, not `copy`: a bare `copy` variable in `tny` would, under
 *         `using namespace tny`, shadow an unqualified `std::copy(...)` call
 *         (finding a variable suppresses ADL) — a nasty surprise. */
struct copy_meta_t {};
constexpr copy_meta_t copy_meta{};

/**
 * @brief A rank-erased tensor for the host/ndarray dispatch boundary.
 *
 * Holds a data pointer, a runtime `ndim`, and 1-D `shape`/`stride` tensors
 * (`Meta`). `as_anyrank(...)` **wraps** the caller's arrays with no copy (a
 * `_meta_view` store, HOST only) — the default; `as_anyrank(..., copy_meta)`
 * COPIES them into an INLINE `TNY_MAX_RANK` store, so the carrier is trivially
 * copyable and passes into a CUDA kernel by value (`device_passable == true`).
 *
 * You do NOT compute on it — it is a *doorway*, not a room. Turn it into a
 * statically-typed view at the boundary and compute on that:
 *   - `fixed<R>()`            — force a known total rank R.
 *   - `dispatch_rank(...)`    — pick R from the runtime `ndim`.
 *   - `peel_front<-Sr>()`     — the batch idiom: peel the runtime number of
 *                               leading batch dims, keep the trailing `Sr`
 *                               "interesting" dims STATIC. One kernel per Sr.
 *                               NB the template arg is NEGATIVE: pass `-Sr`
 *                               (`peel_front<-2>()` keeps the last two dims),
 *                               matching the tensor's `peel_front` sign rule —
 *                               a positive front-count would leave a runtime
 *                               rank, which can't be a static view (asserted).
 *
 * Deliberately no `add_`/`mul_`/etc.: a runtime-rank arithmetic path would loop
 * over `ndim` (killing folding) or dispatch to every rank (the bloat
 * `peel_front<-Sr>` avoids). Do host-side math on a `fixed<R>()`/`peel_front<-Sr>()`
 * view instead.
 */
template <class T, class offset_t = cs::int64_t, class Meta = _meta_store<offset_t, TNY_MAX_RANK>,
          storage Space = storage::view>
struct anyrank {
    T *  data = nullptr;
    Meta shape{};      // 1-D tensor of sizes   (inline, or a view of external memory)
    Meta stride{};     // 1-D tensor of strides
    int  ndim = 0;

    // The MEMORY SPACE the `data` pointer lives in (a compile-time tag, set at the
    // boundary — `from_dlpack` from the DLPack `device`, `as_anyrank<Space>` by
    // hand). `fixed()`/`peel_front` tag every view they hand out with the matching
    // view kind (`storage::view` for a host pointer, `storage::gpu_view` for device), so a
    // `kDLCUDA` capsule no longer erases into a host-tagged view over device memory.
    static constexpr storage  space     = Space;
    static constexpr bool is_device = storage_is_device(Space);
    // the view kind produced by fixed()/peel_front — preserves the carrier's space.
    static constexpr storage  view_space = storage_view_of(Space);

    // Largest rank the store can hold: the inline store's static length, else
    // (a view store) the compile-time dispatch bound TNY_MAX_RANK.
    static constexpr cs::size_t max_rank =
        Meta::extents_type::static_extent(0) != cs::dynamic_extent
            ? Meta::extents_type::static_extent(0) : cs::size_t(TNY_MAX_RANK);

    // True for an inline (copy) store; false for a view store that wraps external
    // host arrays. CONTRACT: only a `device_passable` carrier (built with the
    // `copy_meta` tag) may be passed into a kernel — a view carrier holds host
    // pointers, so using it on the device is UB. This is the caller's guarantee,
    // not a compile-time trip-wire: a former `static_assert(device_passable)` under
    // `#ifdef __CUDA_ARCH__` OVER-FIRED — a HOST-only `fixed()`/`peel_front` on a
    // view carrier is still instantiated in nvcc's device pass and tripped it,
    // breaking valid host code under nvcc (#59). It could not distinguish a
    // host-only instantiation from an actual device use, so it is gone; assert on
    // `device_passable` yourself at your kernel boundary if you want the check.
    static constexpr bool device_passable =
        (Meta::extents_type::static_extent(0) != cs::dynamic_extent);

    _TNY_API offset_t size(int i)  const noexcept { return shape(i); }   // size of dim i
    _TNY_API offset_t step(int i)  const noexcept { return stride(i); }  // stride of dim i

    /** @brief View this tensor as a fixed rank `R` (requires `ndim == R`). */
    template <cs::size_t R>
    _TNY_API dyn_tensor<T, offset_t, R, view_space> fixed() const {
        _TNY_CHECK(static_cast<cs::size_t>(ndim) == R, "fixed<R>(): R must equal ndim (else reads past the shape/stride arrays)");
        using E = cs::dextents<offset_t, R>;
        cs::array<offset_t, R> ext{}, st{};
        for (cs::size_t i = 0; i < R; ++i) { ext[i] = shape(i); st[i] = stride(i); }
        cs::layout_stride::mapping<E> m(E(ext), st);
        return dyn_tensor<T, offset_t, R, view_space>(data, m);
    }

    // internal: the lin-th sub-view keeping the last `Sr` axes static (peeling
    // the leading `ndim - Sr` runtime batch axes into the pointer offset).
    template <cs::size_t Sr>
    _TNY_API dyn_tensor<T, offset_t, Sr, view_space> _keep_last(offset_t lin) const {
        const int nb = ndim - static_cast<int>(Sr);          // # batch dims (runtime)
        _TNY_CHECK(nb >= 0, "peel_front: keep-count exceeds ndim");
        offset_t off = 0, rem = lin;                          // decode lin over batch axes
        for (int d = nb - 1; d >= 0; --d) { offset_t k = rem % shape(d); rem /= shape(d); off += k * stride(d); }
        using E = cs::dextents<offset_t, Sr>;
        cs::array<offset_t, Sr> ext{}, st{};
        for (cs::size_t i = 0; i < Sr; ++i) { ext[i] = shape(nb + i); st[i] = stride(nb + i); }
        cs::layout_stride::mapping<E> m(E(ext), st);
        return dyn_tensor<T, offset_t, Sr, view_space>(data + off, m);
    }

    /** @brief The `lin`-th sub-view keeping the last `|N|` axes static (grid-stride
     *         style). `N` is **negative** — matching the tensor's `peel_front`,
     *         negative means "keep the last |N| dims". (A positive front-count
     *         would leave a runtime rank, which can't be a static view — hence
     *         the assert.) Follow with `recast<shape<-1,...>>()`. */
    template <long N>
    _TNY_API auto peel_front_at(offset_t lin) const {
        static_assert(N < 0, "anyrank::peel_front_at needs a NEGATIVE index (keep the last |N| dims)");
        return _keep_last<static_cast<cs::size_t>(-N)>(lin);
    }

    /** @brief The `lin`-th cell peeled DIRECTLY to a target trailing shape — fuses
     *         `peel_front_at<-NewE::rank()>(lin).recast<NewE, NewL>()` into one call, so
     *         no separate `recast` in the caller. `NewE`'s rank = the number of KEPT
     *         trailing dims (the batch is the leading `ndim - rank` dims, decoded into
     *         the pointer); a static extent in `NewE` folds, a `-1` extent stays
     *         dynamic (read from the carrier). `(*batch, *spatial, C)` -> 2-D pull with
     *         C=3 is `peel_front_at<shape<-1,-1,3>>(i)`. Removes the hand-kept `Sr ==
     *         recast-shape rank` invariant. STRIDES: `NewL` defaults to `keep_strides`
     *         so the cell keeps the carrier's RUNTIME strides (`layout_stride`) — an
     *         anyrank has no compile-time stride info to fold. To fold the inner
     *         strides, either pass a layout (`peel_front_at<shape<-1,c,c>, ccontiguous>`
     *         — a debug-checked "I promise it's contiguous") or use the runtime-proven
     *         `dispatch_layout` on the result. UB if a baked static extent doesn't match
     *         the carrier (debug-checked in `recast`, same contract). */
    template <class NewE, class NewL = keep_strides, cs::enable_if_t<_is_extents<NewE>::value, int> = 0>
    _TNY_API auto peel_front_at(offset_t lin) const {
        return _keep_last<NewE::rank()>(lin).template recast<NewE, NewL>();
    }
    /** @brief Value-form twins (no `.template` on a dependent receiver): pass the target
     *         shape (and optional layout) as a tag — `at.peel_front_at(i, shape<-1,c,c>{})`
     *         / `at.peel_front_at(i, shape<-1,c,c>{}, ccontiguous{})`. */
    template <class NewE, cs::enable_if_t<_is_extents<NewE>::value, int> = 0>
    _TNY_API auto peel_front_at(offset_t lin, NewE) const { return peel_front_at<NewE>(lin); }
    template <class NewE, class NewL, cs::enable_if_t<_is_extents<NewE>::value, int> = 0>
    _TNY_API auto peel_front_at(offset_t lin, NewE, NewL) const { return peel_front_at<NewE, NewL>(lin); }

    /** @brief Peel the leading batch axes -> an iterable of fixed-rank-`|N|`
     *         sub-views (range-for, `size()`, `operator[]`). The
     *         `(*batch, *spatial, C)` boundary with `|N| = spatial + channels`:
     *         one kernel instantiation for `|N|`, not one per total rank.
     *         `N` is negative (keep the last |N| dims), as on the tensor. */
    template <long N>
    _TNY_API anyrank_front<T, offset_t, Meta, Space, static_cast<cs::size_t>(N < 0 ? -N : 0)> peel_front() const {
        static_assert(N < 0, "anyrank::peel_front needs a NEGATIVE index (keep the last |N| dims)");
        return { *this };
    }

    /** @brief The number of cells `peel_front<N>()` would yield — the product of
     *         the peeled leading (batch) extents — computed directly, without
     *         building the range. `N` is NEGATIVE (keep the last |N| dims), the
     *         same sign as `peel_front`; `size_front<-2>()` is the flattened
     *         batch count of a `(*batch, C, C)` carrier. */
    template <long N>
    _TNY_API offset_t size_front() const noexcept {
        static_assert(N < 0, "anyrank::size_front needs a NEGATIVE index (keep the last |N| dims)");
        offset_t n = 1;
        for (int d = 0; d < ndim - static_cast<int>(-N); ++d) n *= shape(d);
        return n;
    }
};

/** @brief A range of fixed-rank-`Sr` sub-views over an `anyrank`'s batch axes.
 *         Inherits the carrier's `Space`, so each cell is a host or `gpu_view`
 *         view accordingly. */
template <class T, class offset_t, class Meta, storage Space, cs::size_t Sr>
struct anyrank_front {
    anyrank<T, offset_t, Meta, Space> src;
    using Cell = dyn_tensor<T, offset_t, Sr, storage_view_of(Space)>;
    static constexpr cs::size_t MaxNb = anyrank<T, offset_t, Meta, Space>::max_rank;

    _TNY_API offset_t size() const noexcept { return src.template size_front<-static_cast<long>(Sr)>(); }
    // Random access (grid-stride `i += nthreads`): decode the batch index from scratch.
    _TNY_API auto operator[](offset_t i) const { return src.template _keep_last<Sr>(i); }

    // Every cell keeps the SAME trailing-Sr extents/strides (only the base offset
    // moves), so the iterator carries one template cell (invariant mapping) and an
    // INCREMENTAL odometer over the runtime batch axes (#110) — one stride-add per
    // step instead of an O(#batch) decode. Seedable at any index (single decode) so a
    // thread/block can start mid-range; a grid-stride loop keeps `operator[]`.
    struct iterator {
        Cell     tmpl;                        // cell at offset 0 -> invariant mapping
        T *      base;                        // tmpl.data() (== src.data)
        offset_t ctr[MaxNb ? MaxNb : 1];      // odometer over batch axes 0..nb-1
        offset_t ext[MaxNb ? MaxNb : 1];      // batch extents
        offset_t str[MaxNb ? MaxNb : 1];      // batch strides
        int      nb;                          // # batch axes = ndim - Sr (runtime)
        offset_t off, lin;
        _TNY_API Cell operator*() const { return Cell(base + off, tmpl.mapping()); }
        _TNY_API iterator & operator++() {
            ++lin;
            for (int d = nb - 1; d >= 0; --d) {
                if (ctr[d] + 1 < ext[d]) { ++ctr[d]; off += str[d]; return *this; }
                off -= ctr[d] * str[d]; ctr[d] = 0;   // wrap axis d, carry up
            }
            return *this;
        }
        _TNY_API bool operator!=(const iterator & o) const { return lin != o.lin; }
        _TNY_API bool operator==(const iterator & o) const { return lin == o.lin; }
    };
    _TNY_API iterator _iter_at(offset_t i) const {
        iterator it{};
        it.tmpl = src.template _keep_last<Sr>(0);      // template at offset 0
        it.base = it.tmpl.data();
        it.nb   = src.ndim - static_cast<int>(Sr);
        for (int d = 0; d < it.nb; ++d) { it.ext[d] = src.size(d); it.str[d] = src.step(d); }
        it.lin = i; it.off = 0; offset_t rem = i;      // seed the odometer at i (one decode)
        for (int d = it.nb - 1; d >= 0; --d) {
            const offset_t e = it.ext[d]; const offset_t k = e ? rem % e : offset_t(0); rem = e ? rem / e : rem;
            it.ctr[d] = k; it.off += k * it.str[d];
        }
        return it;
    }
    _TNY_API iterator begin() const { return _iter_at(0); }
    _TNY_API iterator end()   const { iterator it = _iter_at(0); it.lin = size(); return it; }

    /** @brief A `[lo, hi)` slice of the batch cells for chunked/threaded sweeps: seed
     *         the incremental cursor once at `lo`, then O(1) per step. Split
     *         `[0, size())` across threads/blocks; each sweeps its own chunk. */
    struct subrange_t {
        iterator b, e;
        _TNY_API iterator begin() const { return b; }
        _TNY_API iterator end()   const { return e; }
    };
    _TNY_API subrange_t subrange(offset_t lo, offset_t hi) const {
        iterator b = _iter_at(lo);
        iterator e = b; e.lin = hi;   // end sentinel: only `lin` is compared
        return { b, e };
    }
};

/** @brief Build an `anyrank` that **wraps** the caller's shape/stride arrays with
 *         **no copy** (the default) — e.g. straight off a DLPack tensor. The
 *         arrays must outlive the carrier. HOST only: the pointers are not valid
 *         inside a device kernel, so peel/dispatch on the host and pass the
 *         resulting fixed-rank views to the device. To instead copy into an
 *         inline, device-passable store, pass the `copy_meta` tag (overload
 *         below). DLPack strides are in ELEMENTS; numpy `__array_interface__` in
 *         BYTES (divide by the itemsize first).
 *
 *         `Space` is the memory space of `data` (default `storage::view` = host); pass
 *         `as_anyrank<storage::gpu_view>(...)` for a device pointer so the views peeled
 *         off it are `gpu_view`-tagged. (The shape/stride metadata arrays are host
 *         either way — `Space` labels the DATA, not the metadata store.) */
template <storage Space = storage::view, class T, class offset_t>
_TNY_HOST anyrank<T, offset_t, _meta_view<offset_t>, Space>
as_anyrank(T * data, offset_t * shape, offset_t * stride, int ndim) {
    static_assert(!cs::is_const<offset_t>::value,
        "as_anyrank: the default WRAPS mutable shape/stride arrays; for `const` arrays "
        "(or to build a device-passable carrier) pass the `copy_meta` tag");
    anyrank<T, offset_t, _meta_view<offset_t>, Space> t;
    t.data = data; t.ndim = ndim;
    cs::dextents<offset_t, 1> e{ static_cast<offset_t>(ndim) };
    t.shape  = _meta_view<offset_t>(shape,  e);
    t.stride = _meta_view<offset_t>(stride, e);
    return t;
}

/** @brief `as_anyrank(data, shape, stride, ndim, copy_meta)` — COPY shape/stride
 *         into an inline store, so the carrier is trivially copyable and can be
 *         passed into a CUDA kernel by value (peel on device). `MaxRank` sets the
 *         inline capacity (default `TNY_MAX_RANK`); pass it as
 *         `as_anyrank<64>(..., copy_meta)`. Accepts `const` arrays (it copies). */
template <cs::size_t MaxRank = TNY_MAX_RANK, storage Space = storage::view, class T, class offset_t>
_TNY_HOST anyrank<T, offset_t, _meta_store<offset_t, MaxRank>, Space>
as_anyrank(T * data, const offset_t * shape, const offset_t * stride, int ndim, copy_meta_t) {
    anyrank<T, offset_t, _meta_store<offset_t, MaxRank>, Space> t;
    t.data = data; t.ndim = ndim;
    // Never write past the inline store: ndim can come straight from a DLPack
    // caller (torch allows 64 dims). Copy at most MaxRank; an oversized ndim is
    // then simply never matched by dispatch_rank / fixed<R>.
    _TNY_CHECK(ndim <= static_cast<int>(MaxRank), "as_anyrank(copy_meta): ndim exceeds MaxRank (raise -DTNY_MAX_RANK)");
    const int n = ndim < static_cast<int>(MaxRank) ? ndim : static_cast<int>(MaxRank);
    for (int i = 0; i < n; ++i) { t.shape(i) = shape[i]; t.stride(i) = stride[i]; }
    return t;
}

/**
 * @brief Narrow a fixed-rank view's OFFSET INDEX WIDTH to `Idx2` (default `int32_t`)
 *        when its element span fits, then call `f` — else call `f` with the view as-is.
 *
 * The kernel-boundary primitive behind the int32 fast path (#115): it instantiates
 * `f` for BOTH widths and picks at run time via `index_fits`/`reindex`, so a genuinely
 * dynamic view runs its offset math in 32-bit (half the by-value footprint, fewer
 * device registers) exactly when that is lossless. `_TNY_HOST`; preserves the view's
 * mutability. Use it standalone on a known-rank view (or a `peel_front` batch cell), or
 * via `dispatch_rank<narrow_index>` to fuse it with the rank dispatch.
 *
 *     for (auto cell : at.peel_front<-Sr>()) dispatch_index(cell, [&](auto c){ kernel<Sr>(c); });
 */
template <class Idx2 = cs::int32_t, class V, class F>
_TNY_HOST void dispatch_index(V && v, F && f) {
    if (v.template index_fits<Idx2>()) f(v.template reindex<Idx2>());   // int32 arm
    else                               f(v);                            // wide (int64) arm
}

/**
 * @brief Runtime-classify a DYNAMIC-strided view's contiguity and hand `f` a view whose
 *        LAYOUT is baked into the type — `ccontiguous` (C-order) or `fcontiguous`
 *        (F-order) when the runtime strides match, else the original `dynamic_strides`.
 *
 * The layout counterpart of `dispatch_index`. An `anyrank` boundary erases the
 * producer's contiguity into `layout_stride`, so a later `recast<shape<…>>` can only
 * KEEP runtime strides. `dispatch_layout` cheaply checks (`is_dense<ccontiguous>()` /
 * `<fcontiguous>()` — a stride compare, no data touched) and, in the contiguous arms,
 * hands `f` a view whose strides are EXTENT-DERIVED — so `recast<shape<-1,c,c>>()` then
 * folds the inner strides to immediates SAFELY (no "I promise it's contiguous" — the
 * runtime check already proved it). `f` is instantiated up to 3× (only the matching arm
 * runs), so make it generic over the view type.
 *
 * OPT-IN per call site (like `dispatch_index`): do NOT wrap `from_dlpack` in it by
 * default — it triples instantiations and composes multiplicatively with the rank/width
 * dispatchers. Reach for it when the inner block's folded strides actually matter (a
 * small static-`C` kernel; see the efficient-kernels guide).
 *
 *     for (auto cell : at.peel_front<-Sr>())
 *         dispatch_layout(cell, [&](auto v){ kernel<Sr>(v.recast(shape<-1,c,c>{})); });
 */
template <class T, class E, storage O, class F>
_TNY_HOST void dispatch_layout(tensor<T, E, dynamic_strides, O> v, F && f) {
    static_assert(storage_is_view(O), "dispatch_layout: expects a view (an anyrank fixed()/peel cell)");
    if      (v.template is_dense<ccontiguous>()) f(tensor<T, E, ccontiguous, O>(v.data(), v.extents()));  // C-order strides fold
    else if (v.template is_dense<fcontiguous>()) f(tensor<T, E, fcontiguous, O>(v.data(), v.extents()));  // F-order strides fold
    else                                         f(v);                                                    // genuinely strided
}
/** @brief The spelling for `dispatch_rank`'s opt-in flag: `dispatch_rank<narrow_index>(at, f)`. */
inline constexpr bool narrow_index = true;

namespace _detail {
template <cs::size_t R, bool Narrow, class T, class offset_t, class Meta, storage Space, class F>
_TNY_HOST bool dispatch_from(const anyrank<T, offset_t, Meta, Space> & t, F & f) {
    if constexpr (R <= anyrank<T, offset_t, Meta, Space>::max_rank) {
        if (t.ndim == static_cast<int>(R)) {
            if constexpr (Narrow) dispatch_index(t.template fixed<R>(), f);   // width innermost
            else                  f(t.template fixed<R>());
            return true;
        }
        return dispatch_from<R + 1, Narrow>(t, f);
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
 * `peel_front<-Sr>` when only the trailing dims need to be static — one
 * instantiation instead of one per total rank.
 *
 *     dispatch_rank(as_anyrank(data, size, stride, ndim), [&](auto v){ kernel(v); });
 *
 * Opt into the int32 fast path with the compile-time `narrow_index` flag: each fixed
 * cell is then also `dispatch_index`-narrowed (rank OUTER, width INNER — only the leaf
 * doubles). `Narrow = false` (the default) is exactly the plain rank dispatch — no
 * extra instantiation.
 *
 *     dispatch_rank<narrow_index>(at, [&](auto v){ kernel(v); });   // int32 cells when they fit
 */
template <bool Narrow = false, class T, class offset_t, class Meta, storage Space, class F>
_TNY_HOST bool dispatch_rank(const anyrank<T, offset_t, Meta, Space> & t, F && f) {
    return _detail::dispatch_from<0, Narrow>(t, f);   // R=0 handles a rank-0 (scalar) ndarray
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
