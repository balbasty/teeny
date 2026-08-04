#ifndef TNY_DYNAMIC_H
#define TNY_DYNAMIC_H
#include <cuda/std/mdspan>
#include <cuda/std/array>
#include <cuda/std/limits>
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

// --- static trailing geometry (#208/#209/#210) -------------------------------
// An `anyrank` can carry a compile-time `Tail` (a `shape<...>`) of static TRAILING
// extents AND a `TailS` (a `strides<...>`) of static trailing STRIDES — the only
// dims with stable identity under rank erasure (anchored at `ndim`, like
// `peel_front<-Sr>` and numpy right-alignment). A kept-`Sr` window is filled
// RIGHT-ALIGNED from `Tail`/`TailS`: cell dim `j` maps to tail slot `j - (Sr - K)`
// (K = Tail::rank()); `< 0` is a leading batch-kept dim (dynamic extent + stride).
// Covers `Sr > K` (partial), `Sr == K` (fully folded), `Sr < K` (peel into the
// suffix, keeping its last `Sr` dims).
//
// `TailS` defaults to all-`dynamic_stride` (#209 behaviour: extents fold, strides
// stay runtime). When it marks strides static, the cell's LAYOUT folds to a
// `strides<...>` — and a KNOWN-CONTIGUOUS inner block's mapping is empty (EBO), so
// the cell loses its stride words (fewer registers/thread). Crucially, an all-dynamic
// `TailS` (which includes the empty-`Tail` default) yields EXACTLY `cs::layout_stride`,
// so today's `dyn_tensor` and every `dispatch_layout` match are byte-identical.
template <cs::size_t Sr, class Tail>
struct _tail_ext {
    static constexpr cs::size_t K = Tail::rank();
    static constexpr cs::size_t at(cs::size_t j) noexcept {
        const long tj = static_cast<long>(j) - (static_cast<long>(Sr) - static_cast<long>(K));
        return tj < 0 ? cs::dynamic_extent : Tail::static_extent(static_cast<cs::size_t>(tj));
    }
};
// The cell's EXTENTS type for a kept-`Sr` window over `Tail` (index type `offset_t`,
// static where `Tail` says so, dynamic elsewhere).
template <class offset_t, cs::size_t Sr, class Tail, class Seq = cs::make_index_sequence<Sr>>
struct _cell_ext;
template <class offset_t, cs::size_t Sr, class Tail, cs::size_t... J>
struct _cell_ext<offset_t, Sr, Tail, cs::index_sequence<J...>> {
    using type = cs::extents<offset_t, _tail_ext<Sr, Tail>::at(J)...>;
};

// Right-align `TailS` into a rank-`Sr` window (leading batch-kept dims -> dynamic
// stride). `all_dyn()` is the back-compat trip-wire: if nothing folds, the cell
// stays `cs::layout_stride` (== today), not a `strides<dynamic_stride×Sr>`.
template <cs::size_t Sr, class TailS>
struct _tail_str {
    static constexpr cs::size_t K = TailS::N;
    static constexpr cs::int64_t at(cs::size_t j) noexcept {
        const long tj = static_cast<long>(j) - (static_cast<long>(Sr) - static_cast<long>(K));
        return tj < 0 ? dynamic_stride : TailS::static_stride(static_cast<cs::size_t>(tj));
    }
    static constexpr bool all_dyn() noexcept {
        for (cs::size_t j = 0; j < Sr; ++j) if (at(j) != dynamic_stride) return false;
        return true;
    }
};
// The cell's LAYOUT: a folded `strides<...>` when some trailing stride is static,
// else `cs::layout_stride` (the mandatory back-compat degenerate case).
template <cs::size_t Sr, class TailS, bool AllDyn = _tail_str<Sr, TailS>::all_dyn(),
          class Seq = cs::make_index_sequence<Sr>>
struct _cell_layout;
template <cs::size_t Sr, class TailS, cs::size_t... J>
struct _cell_layout<Sr, TailS, /*AllDyn=*/true, cs::index_sequence<J...>> { using type = cs::layout_stride; };
template <cs::size_t Sr, class TailS, cs::size_t... J>
struct _cell_layout<Sr, TailS, /*AllDyn=*/false, cs::index_sequence<J...>> {
    using type = strides<_tail_str<Sr, TailS>::at(J)...>;
};

// The cell TYPE `fixed`/`_keep_last`/`peel_front` hand out (== `dyn_tensor` when
// `Tail`/`TailS` are empty/all-dynamic; folded extents+strides recover the static
// inner block otherwise).
template <class T, class offset_t, cs::size_t Sr, class Tail, class TailS, storage O>
using _tail_cell = tensor<T, typename _cell_ext<offset_t, Sr, Tail>::type,
                          typename _cell_layout<Sr, TailS>::type, O>;

// Derive the `TailS` a boundary layout tag imposes on `Tail`:
//   - `keep_strides` (default): all `dynamic_stride` (extents-only, #209 behaviour).
//   - `ccontiguous`/`fcontiguous`: the contiguous strides over `Tail` — folds the
//     trailing static run (`shape<-1,c,c>` C-order -> `strides<c*c, c, 1>`, the outer
//     one static because it is the product of the STATIC trailing extents).
//   - `strides<S...>`: those strides verbatim.
// Reuses `layout.h`'s `_src_sstride`, so it folds exactly where a real tensor would.
template <class Tail, class L, class Seq = cs::make_index_sequence<Tail::rank()>>
struct _tail_strides_of;
template <class Tail, class L, cs::size_t... D>
struct _tail_strides_of<Tail, L, cs::index_sequence<D...>> { using type = strides<_src_sstride<D, L, Tail>()...>; };
template <class Tail, cs::size_t... D>
struct _tail_strides_of<Tail, keep_strides, cs::index_sequence<D...>> { using type = strides<((void)D, dynamic_stride)...>; };

// --- both-ends geometry (#219): a static leading Head folds too, but ONLY in a
// FULL-RANK window (`fixed<R>` / `dispatch_rank`), where the window's left edge is the
// absolute dim 0 (compile-time). `peel_front<-Sr>` keeps a TRAILING window whose left
// edge is runtime (`ndim - Sr`), so a leading Head can't anchor there — the peel path
// keeps the `_cell_*`/`_tail_cell` (Tail-only) machinery above unchanged. The
// full-rank cell folds `Head` left-aligned at `[0, H)`, `Tail` right-aligned at
// `[R-K, R)`, dynamic middle. An empty Head reduces `_ends_*` to `_tail_*`, so a
// no-Head `fixed<R>` type is byte-identical to #209/#210.
template <cs::size_t R, class Head, class Tail>
struct _ends_ext {
    static constexpr cs::size_t H = Head::rank(), K = Tail::rank();
    static constexpr cs::size_t at(cs::size_t j) noexcept {
        if (j < H)      return Head::static_extent(j);
        if (j + K >= R) return Tail::static_extent(j - (R - K));     // j >= R-K (H<=R-K, no overlap)
        return cs::dynamic_extent;
    }
};
template <class offset_t, cs::size_t R, class Head, class Tail, class Seq = cs::make_index_sequence<R>>
struct _ends_cell_ext;
template <class offset_t, cs::size_t R, class Head, class Tail, cs::size_t... J>
struct _ends_cell_ext<offset_t, R, Head, Tail, cs::index_sequence<J...>> {
    using type = cs::extents<offset_t, _ends_ext<R, Head, Tail>::at(J)...>;
};
template <cs::size_t R, class HeadS, class TailS>
struct _ends_str {
    static constexpr cs::size_t H = HeadS::N, K = TailS::N;
    static constexpr cs::int64_t at(cs::size_t j) noexcept {
        if (j < H)      return HeadS::static_stride(j);
        if (j + K >= R) return TailS::static_stride(j - (R - K));
        return dynamic_stride;
    }
    static constexpr bool all_dyn() noexcept {
        for (cs::size_t j = 0; j < R; ++j) if (at(j) != dynamic_stride) return false;
        return true;
    }
};
template <cs::size_t R, class HeadS, class TailS, bool AllDyn = _ends_str<R, HeadS, TailS>::all_dyn(),
          class Seq = cs::make_index_sequence<R>>
struct _ends_layout;
template <cs::size_t R, class HeadS, class TailS, cs::size_t... J>
struct _ends_layout<R, HeadS, TailS, /*AllDyn=*/true, cs::index_sequence<J...>> { using type = cs::layout_stride; };
template <cs::size_t R, class HeadS, class TailS, cs::size_t... J>
struct _ends_layout<R, HeadS, TailS, /*AllDyn=*/false, cs::index_sequence<J...>> {
    using type = strides<_ends_str<R, HeadS, TailS>::at(J)...>;
};
template <class T, class offset_t, cs::size_t R, class Head, class HeadS, class Tail, class TailS, storage O>
using _ends_cell = tensor<T, typename _ends_cell_ext<offset_t, R, Head, Tail>::type,
                          typename _ends_layout<R, HeadS, TailS>::type, O>;

// Debug-check the runtime shape/stride ends against the static `Head`/`HeadS`
// (leading, dims `[0,H)`) and `Tail`/`TailS` (trailing, dims `[ndim-K, ndim)`) — run
// ONCE at the boundary (`as_anyrank`/`from_dlpack`), then trusted (same contract class
// as `recast`, hoisted host-side to the import). Reads the ORIGINAL arrays (so it
// validates even when a copy-meta store is clamped to TNY_MAX_RANK). A static stride
// whose dim has runtime extent <= 1 is unobservable, so it is exempt (recast's rule).
// `const` args accept DLPack's `const int64_t *`. Empty Head == the #209/#210 tail check.
template <class Head, class HeadS, class Tail, class TailS, class Idx>
_TNY_HOST void _check_ends(const Idx * shp, const Idx * strd, int ndim) {
    constexpr cs::size_t H = Head::rank(), K = Tail::rank();
    _TNY_CHECK(ndim >= static_cast<int>(H + K),
               "as_anyrank/from_dlpack: ndim < the static Head+Tail rank");
    for (cs::size_t j = 0; j < H; ++j) {                       // leading Head, anchored at 0
        const int d = static_cast<int>(j);
        _TNY_CHECK(Head::static_extent(j) == cs::dynamic_extent ||
                   static_cast<Idx>(Head::static_extent(j)) == shp[d],
                   "as_anyrank/from_dlpack: a static Head extent does not match the runtime shape");
        _TNY_CHECK(HeadS::static_stride(j) == dynamic_stride || shp[d] <= Idx(1) ||
                   static_cast<Idx>(HeadS::static_stride(j)) == strd[d],
                   "as_anyrank/from_dlpack: a static Head stride does not match the runtime strides");
    }
    for (cs::size_t j = 0; j < K; ++j) {                       // trailing Tail, anchored at ndim
        const int d = ndim - static_cast<int>(K) + static_cast<int>(j);
        _TNY_CHECK(Tail::static_extent(j) == cs::dynamic_extent ||
                   static_cast<Idx>(Tail::static_extent(j)) == shp[d],
                   "as_anyrank/from_dlpack: a static tail extent does not match the runtime shape");
        _TNY_CHECK(TailS::static_stride(j) == dynamic_stride || shp[d] <= Idx(1) ||
                   static_cast<Idx>(TailS::static_stride(j)) == strd[d],
                   "as_anyrank/from_dlpack: a static tail stride does not match the runtime strides "
                   "(the inner block is not laid out as the layout tag promised)");
    }
}

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

template <class T, class offset_t, class Meta, storage Space, class Tail, class TailS, cs::size_t Sr> struct anyrank_front;  // fwd

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
 * `peel_front` / `peel_front_at` / `size_front` each also take the keep-count as a
 * static integer VALUE — `at.peel_front(Int<-Sr>())`, `at.peel_front_at(lin, Int<-Sr>())`,
 * `at.size_front(Int<-Sr>())` — identical to the `<-Sr>` template spelling but deduced,
 * so a carrier whose type is a template parameter needs no `.template` disambiguator.
 *
 * Before any of those, the carrier itself can be narrowed to 32-bit offsets:
 * `index_fits<Idx2>()` asks whether every reachable offset fits, `reindex<Idx2>()`
 * returns the same carrier with an `Idx2` meta store — so a GPU boundary narrows
 * ONCE, host-side, and every cell it later hands out is `Idx2`-indexed.
 *
 * Deliberately no `add_`/`mul_`/etc.: a runtime-rank arithmetic path would loop
 * over `ndim` (killing folding) or dispatch to every rank (the bloat
 * `peel_front<-Sr>` avoids). Do host-side math on a `fixed<R>()`/`peel_front<-Sr>()`
 * view instead.
 */
template <class T, class offset_t = cs::int64_t, class Meta = _meta_store<offset_t, TNY_MAX_RANK>,
          storage Space = storage::view, class Tail = shape<>,
          class TailS = _runtime_strides_t<Tail::rank()>,
          class Head = shape<>, class HeadS = _runtime_strides_t<Head::rank()>>
struct anyrank {
    T *  data = nullptr;
    Meta shape{};      // 1-D tensor of sizes   (inline, or a view of external memory)
    Meta stride{};     // 1-D tensor of strides
    int  ndim = 0;

    // The static geometry: `Tail`/`TailS` are the static TRAILING extents/strides
    // (anchored at `ndim`), `Head`/`HeadS` the static LEADING ones (anchored at 0) —
    // all empty / all-dynamic by default == today's fully-dynamic carrier. `Tail`
    // folds into EVERY cell `fixed`/`peel_front` hands out; `Head` folds ONLY in
    // `fixed`/`dispatch_rank` (a full-rank window), since `peel_front` peels the
    // leading dims into the batch (the Head is normally there). So `(*batch,*spatial,C)`
    // needs only `Tail`, while `(C_in,*spatial,C_out)` uses both — recovered WITHOUT a
    // per-call `recast`. Set at the boundary via the `anyshape<...>` tag (+ optional
    // layout) on `as_anyrank`/`from_dlpack`, which debug-checks them against the runtime
    // shape/strides ONCE, then trusts them (same contract class as `recast`). Pure type
    // info: the members above are unchanged, so size/layout/trivial-copyability hold.
    using tail_type        = Tail;
    using tail_stride_type = TailS;
    using head_type        = Head;
    using head_stride_type = HeadS;
    static constexpr cs::size_t tail_rank = Tail::rank();
    static constexpr cs::size_t head_rank = Head::rank();
    static constexpr cs::size_t ends_rank = head_rank + tail_rank;   // where dispatch_rank starts
    static_assert(TailS::N == Tail::rank(), "anyrank: TailS must have one stride per Tail dim");
    static_assert(HeadS::N == Head::rank(), "anyrank: HeadS must have one stride per Head dim");

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

    // internal: build a cell of extents `E` / layout `CL` at pointer `p`, reading the
    // shape/strides of dims `[d0, d0+E::rank())`. A folded (`strides<...>`) layout only
    // reads the DYNAMIC strides from the runtime array (an all-static/contiguous block
    // reads none — EBO mapping); an all-dynamic layout keeps the `cs::layout_stride`
    // path (== #209 / today). Shared by the peel (Tail-only) and fixed (both-ends) cells.
    template <class E, class CL>
    _TNY_API tensor<T, E, CL, view_space> _build_cell(T * p, int d0) const {
        constexpr cs::size_t R = E::rank();
        using Cell = tensor<T, E, CL, view_space>;
        cs::array<offset_t, R> ext{};
        for (cs::size_t i = 0; i < R; ++i) ext[i] = shape(d0 + static_cast<int>(i));
        if constexpr (cs::is_same<CL, cs::layout_stride>::value) {
            cs::array<offset_t, R> st{};
            for (cs::size_t i = 0; i < R; ++i) st[i] = stride(d0 + static_cast<int>(i));
            return Cell(p, cs::layout_stride::mapping<E>(E(ext), st));
        } else if constexpr (CL::ndyn() == 0) {
            return Cell(p, typename CL::template mapping<E>(E(ext)));   // fully-static strides: EBO
        } else {
            cs::array<offset_t, CL::ndyn() ? CL::ndyn() : 1> dyn{};     // dynamic slots only, dim order
            for (cs::size_t i = 0, s = 0; i < R; ++i)
                if (CL::static_stride(i) == dynamic_stride) dyn[s++] = stride(d0 + static_cast<int>(i));
            return Cell(p, typename CL::template mapping<E>(E(ext), dyn));
        }
    }
    // the kept-`Sr` peel cell (Tail-only, right-aligned; Head is inert here — see below)
    template <cs::size_t Sr>
    _TNY_API _tail_cell<T, offset_t, Sr, Tail, TailS, view_space> _cell_at(T * p, int d0) const {
        return _build_cell<typename _cell_ext<offset_t, Sr, Tail>::type,
                           typename _cell_layout<Sr, TailS>::type>(p, d0);
    }

    /** @brief View this tensor as a fixed rank `R` (requires `ndim == R`). BOTH the
     *         static `Head` (first `head_rank` dims) and `Tail` (last `tail_rank`) fold —
     *         the full-rank window has a compile-time left edge, so the Head anchors. */
    template <cs::size_t R>
    _TNY_API _ends_cell<T, offset_t, R, Head, HeadS, Tail, TailS, view_space> fixed() const {
        static_assert(R >= ends_rank, "fixed<R>(): R must be >= the static Head+Tail rank");
        _TNY_CHECK(static_cast<cs::size_t>(ndim) == R, "fixed<R>(): R must equal ndim (else reads past the shape/stride arrays)");
        return _build_cell<typename _ends_cell_ext<offset_t, R, Head, Tail>::type,
                           typename _ends_layout<R, HeadS, TailS>::type>(data, 0);
    }

    // internal: the lin-th sub-view keeping the last `Sr` axes static (peeling
    // the leading `ndim - Sr` runtime batch axes into the pointer offset). Only the
    // static `Tail`/`TailS` fold (right-aligned) — the kept window's left edge is
    // runtime, so a leading `Head` can't anchor and is simply peeled into the batch.
    template <cs::size_t Sr>
    _TNY_API _tail_cell<T, offset_t, Sr, Tail, TailS, view_space> _keep_last(offset_t lin) const {
        const int nb = ndim - static_cast<int>(Sr);          // # batch dims (runtime)
        _TNY_CHECK(nb >= 0, "peel_front: keep-count exceeds ndim");
        offset_t off = 0, rem = lin;                          // decode lin over batch axes
        for (int d = nb - 1; d >= 0; --d) { offset_t k = rem % shape(d); rem /= shape(d); off += k * stride(d); }
        return _cell_at<Sr>(data + off, nb);
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
    /** @brief Value form: `at.peel_front_at(lin, Int<-Sr>())` == `at.peel_front_at<-Sr>(lin)`.
     *         The keep-count is the only compile-time selector here, so it takes the plain
     *         `Int<k>()` static integer (the single-selector spelling `t.squeeze(Int<1>())`
     *         uses) — deduced, so a type-dependent receiver needs no `.template`. The
     *         `lin` argument stays an ordinary runtime index; a static integer in the
     *         SECOND position is always the selector, and a `shape<...>` tag there is
     *         the fused-recast twin below — never confusable. */
    template <class I, cs::enable_if_t<_is_ic<I>::value, int> = 0>
    _TNY_API auto peel_front_at(offset_t lin, I) const { return peel_front_at<static_cast<long>(I::value)>(lin); }

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
    _TNY_API anyrank_front<T, offset_t, Meta, Space, Tail, TailS, static_cast<cs::size_t>(N < 0 ? -N : 0)> peel_front() const {
        static_assert(N < 0, "anyrank::peel_front needs a NEGATIVE index (keep the last |N| dims)");
        // peel folds Tail only (the leading Head is peeled into the batch), so hand the
        // range a HEAD-LESS carrier — same members, so the peel cell type is unchanged.
        return { anyrank<T, offset_t, Meta, Space, Tail, TailS>{ data, shape, stride, ndim } };
    }
    /** @brief Value form: `at.peel_front(Int<-Sr>())` == `at.peel_front<-Sr>()` — the
     *         keep-count as a static integer, deduced, so a type-dependent receiver
     *         needs no `.template`. */
    template <class I, cs::enable_if_t<_is_ic<I>::value, int> = 0>
    _TNY_API auto peel_front(I) const { return peel_front<static_cast<long>(I::value)>(); }

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
    /** @brief Value form: `at.size_front(Int<-Sr>())` == `at.size_front<-Sr>()` — the
     *         keep-count as a static integer, deduced, so a type-dependent receiver
     *         needs no `.template`. */
    template <class I, cs::enable_if_t<_is_ic<I>::value, int> = 0>
    _TNY_API offset_t size_front(I) const noexcept { return size_front<static_cast<long>(I::value)>(); }

    // --- whole-carrier offset-width narrowing (#467) -------------------------
    // The fixed-rank sibling of these two lives on `tensor` (`index_fits`/`reindex`);
    // this is the same pair on the CARRIER, so a boundary can narrow ONCE, host-side,
    // before a launch and keep the batch idiom (`peel_front<-Sr>`) — cells peeled from
    // an `Idx2` carrier are already `Idx2`-indexed, since `offset_t` is a parameter.
    // (`dispatch_index(v, f)` is `_TNY_HOST` and per-cell, so it cannot be the device
    // mechanism; `dispatch_rank` would collapse the rank the batch idiom keeps runtime.)

    /** @brief The carrier type `reindex<Idx2>()` produces: same `T`/`Space` and the same
     *         static `Head`/`Tail` geometry, with the offset width — and the meta store —
     *         narrowed to `Idx2`. Always an INLINE (`copy_meta`) store: narrowing has to
     *         copy, there is nothing to narrow a wrapped array into. */
    template <class Idx2, cs::size_t MaxRank = max_rank>
    using reindexed = anyrank<T, Idx2, _meta_store<Idx2, MaxRank>, Space,
                              _reindex_extents_t<Idx2, Tail>, TailS,
                              _reindex_extents_t<Idx2, Head>, HeadS>;

    /** @brief Does every element offset reachable through this carrier fit the index
     *         type `Idx2`? The whole-carrier twin of the view's `index_fits<Idx2>()`,
     *         with the same SIGNED reach contract (teeny has negative-stride views):
     *         `max = Σ_{s>0}(e−1)·s`, `min = Σ_{s<0}(e−1)·s`; fits ⟺ `min..max` ⊆ `Idx2`.
     *         Accumulates in a wide type, with the accumulation itself overflow-checked
     *         (#471 — safe even against adversarial/corrupted shape/stride, e.g. off a
     *         raw DLPack import); a broadcast (stride-0) axis adds 0. The precondition
     *         `reindex<Idx2>()` debug-checks. */
    template <class Idx2>
    _TNY_API bool index_fits() const noexcept {
        // `shape`/`stride` are themselves callable (1-D tensor members), so they
        // pass straight into the shared loop with no adapter (contrast tensor.h's
        // `_extent_method_fn`/`_stride_method_fn`, which wrap METHOD accessors).
        return _detail::_signed_reach_fits<Idx2>(ndim, shape, stride);
    }

    /** @brief Narrow the whole carrier's OFFSET INDEX WIDTH to `Idx2` — same data
     *         pointer, same memory `Space`, same static `Head`/`Tail` geometry, with
     *         `ndim` and the runtime shape/strides copied into an inline `Idx2` store.
     *         Every cell peeled off the result is then `Idx2`-indexed for free, so a
     *         GPU boundary narrows once, on the host, and still launches the batch
     *         idiom (`peel_front<-Sr>`) — 32-bit offset math, fewer registers, and
     *         half the meta store to pass by value into a `__global__`.
     *
     *         The static extents/strides baked into the type are PURE TYPE INFO and are
     *         untouched (only their extents' index type follows `Idx2`). `MaxRank` sets
     *         the inline capacity (default: this carrier's own `max_rank`).
     *
     *         Debug-checks `index_fits<Idx2>()`; UB if the caller lies — the same
     *         contract as the view's `reindex`:
     *
     *             if (at.index_fits<int32_t>()) launch(at.reindex<int32_t>());
     *             else                          launch(at);
     *
     *         which is exactly what `dispatch_index(at, f)` does for you. */
    template <class Idx2, cs::size_t MaxRank = max_rank>
    _TNY_API reindexed<Idx2, MaxRank> reindex() const {
        _TNY_CHECK(index_fits<Idx2>(),
                   "anyrank::reindex: element offsets don't fit the target index type (span exceeds its range)");
        _TNY_CHECK(ndim <= static_cast<int>(MaxRank),
                   "anyrank::reindex: ndim exceeds MaxRank (raise -DTNY_MAX_RANK)");
        reindexed<Idx2, MaxRank> t;
        t.data = data; t.ndim = ndim;
        const int n = ndim < static_cast<int>(MaxRank) ? ndim : static_cast<int>(MaxRank);
        for (int i = 0; i < n; ++i) {
            t.shape(i)  = static_cast<Idx2>(shape(i));
            t.stride(i) = static_cast<Idx2>(stride(i));
        }
        return t;
    }
};

/** @brief Free forms of `reindex`/`index_fits` for `anyrank` — the whole-carrier
 *         twins of `tensor`'s free forms (tensor.h): deduce the carrier, so a
 *         type-dependent receiver (e.g. inside a generic `dispatch_dlpack_dtype`-
 *         style functor) can write `reindex<Idx2>(at)` / `index_fits<Idx2>(at)`
 *         without the `.template` disambiguator. (`Idx2` is a TYPE, so there is
 *         no value form.) `anyrank::reindex` is const-only (unlike `tensor`'s
 *         mutable/const pair), so only one overload is needed here. No `MaxRank`
 *         parameter: it would sit 10th in the deduced list, so a caller could not
 *         practically override it without spelling out every preceding deduced
 *         parameter by hand; this free form always takes the member's own default
 *         (the carrier's `max_rank`) — call the member directly (`at.reindex<Idx2,
 *         8>()`) when a custom capacity is needed. */
template <class Idx2, class T, class offset_t, class Meta, storage Space, class Tail, class TailS, class Head, class HeadS>
_TNY_API auto reindex(const anyrank<T, offset_t, Meta, Space, Tail, TailS, Head, HeadS> & a) {
    return a.template reindex<Idx2>();
}
template <class Idx2, class T, class offset_t, class Meta, storage Space, class Tail, class TailS, class Head, class HeadS>
_TNY_API bool index_fits(const anyrank<T, offset_t, Meta, Space, Tail, TailS, Head, HeadS> & a) {
    return a.template index_fits<Idx2>();
}

/** @brief A range of fixed-rank-`Sr` sub-views over an `anyrank`'s batch axes.
 *         Inherits the carrier's `Space`, so each cell is a host or `gpu_view`
 *         view accordingly. */
template <class T, class offset_t, class Meta, storage Space, class Tail, class TailS, cs::size_t Sr>
struct anyrank_front {
    anyrank<T, offset_t, Meta, Space, Tail, TailS> src;
    using Cell = _tail_cell<T, offset_t, Sr, Tail, TailS, storage_view_of(Space)>;
    static constexpr cs::size_t MaxNb = anyrank<T, offset_t, Meta, Space, Tail, TailS>::max_rank;

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
        // The current cell's BATCH multi-index / flat index — the odometer the cursor
        // already maintains, surfaced for free (#217). `index(d)` = coord of batch axis
        // d (d in [0, nbatch())); `linear()` = the flat batch index. NB the batch rank
        // `nbatch()` is RUNTIME (ndim - Sr), so there is no fixed-size `index()` tuple —
        // use `index(d)` per axis, or `enumerate()`'s coord proxy.
        _TNY_API offset_t index(int d)  const noexcept { return ctr[d]; }
        _TNY_API int      nbatch()      const noexcept { return nb; }
        _TNY_API offset_t linear()      const noexcept { return lin; }
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

    // --- enumerate: batch-cell range-for that ALSO yields the batch multi-index (#217)
    // Mirrors the tensor-side peel `enumerate()` (#213). Opt-in: the bare
    // `for (auto cell : at.peel_front<-Sr>())` cell stays lean; enumerate pairs each
    // cell with the batch coordinates for a per-batch-axis table `param[d][m[d]]`:
    //   for (auto [m, cell] : at.peel_front<-Sr>().enumerate()) use(m[0], m[1], cell);
    // `m` is a lightweight VIEW of the live odometer (valid THIS iteration — don't store
    // it past the loop body); `m[d]` = coord of batch axis d, `m.rank()` = #batch axes,
    // `m.linear()` = the flat batch index. Composes with `subrange` for chunked sweeps.
    struct coord {
        const offset_t * ctr; int nb; offset_t lin;
        _TNY_API offset_t operator[](int d) const noexcept { return ctr[d]; }
        _TNY_API int      rank()    const noexcept { return nb; }
        _TNY_API offset_t linear()  const noexcept { return lin; }
    };
    struct item { coord index; Cell cell; };
    struct enum_iterator {
        iterator it;
        _TNY_API item operator*() const { return { coord{ it.ctr, it.nb, it.lin }, *it }; }
        _TNY_API enum_iterator & operator++() { ++it; return *this; }
        _TNY_API bool operator!=(const enum_iterator & o) const { return it != o.it; }
        _TNY_API bool operator==(const enum_iterator & o) const { return it == o.it; }
    };
    struct enum_range {
        anyrank_front r;   // by VALUE -> safe on a `peel_front<-Sr>().enumerate()` temporary
        _TNY_API enum_iterator begin() const { return { r.begin() }; }
        _TNY_API enum_iterator end()   const { return { r.end() }; }
        struct enum_subrange {
            enum_iterator b, e;
            _TNY_API enum_iterator begin() const { return b; }
            _TNY_API enum_iterator end()   const { return e; }
        };
        _TNY_API enum_subrange subrange(offset_t lo, offset_t hi) const {
            subrange_t s = r.subrange(lo, hi);
            return { { s.b }, { s.e } };
        }
    };
    _TNY_API enum_range enumerate() const { return { *this }; }
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

/** @brief `as_anyrank(..., anyshape<etc,c,c>{}[, layout])` — carry a STATIC TRAILING
 *         shape (and, with a layout tag, static trailing STRIDES) in the carrier's type,
 *         so `fixed`/`peel_front` hand out cells with those inner extents/strides already
 *         folded (no per-call `recast`). The runtime shape/strides' trailing dims are
 *         debug-checked against the tag once, here, then trusted. `etc` = the erased
 *         batch; the dims after it are the static tail (see `anyshape`). The optional
 *         layout tag chooses the trailing strides (like `recast`'s 2nd arg): `keep_strides`
 *         (default — strides stay runtime), `ccontiguous`/`fcontiguous` (fold the
 *         contiguous inner block — the "input is contiguous" precondition, checked here),
 *         or `strides<S...>` (impose them). Wraps the caller's arrays (no copy). */
template <storage Space = storage::view, class T, class offset_t, class S, class Layout = keep_strides,
          cs::enable_if_t<_is_anyshape<S>::value, int> = 0>
_TNY_HOST auto as_anyrank(T * data, offset_t * shape, offset_t * stride, int ndim, S, Layout = {}) {
    static_assert(!cs::is_const<offset_t>::value,
        "as_anyrank: the default WRAPS mutable shape/stride arrays; for `const` arrays "
        "(or to build a device-passable carrier) pass the `copy_meta` tag");
    using TailR  = _reindex_extents_t<offset_t, typename S::tail>;
    using TailSt = typename _tail_strides_of<TailR, Layout>::type;
    using HeadR  = _reindex_extents_t<offset_t, typename S::head>;
    // Head EXTENTS fold; head STRIDES stay runtime — a leading dim's contiguous stride
    // spans the (dynamic-rank) middle, so it isn't a compile-time constant in general.
    using HeadSt = _runtime_strides_t<HeadR::rank()>;
    anyrank<T, offset_t, _meta_view<offset_t>, Space, TailR, TailSt, HeadR, HeadSt> t;
    t.data = data; t.ndim = ndim;
    cs::dextents<offset_t, 1> e{ static_cast<offset_t>(ndim) };
    t.shape  = _meta_view<offset_t>(shape,  e);
    t.stride = _meta_view<offset_t>(stride, e);
    _check_ends<HeadR, HeadSt, TailR, TailSt>(shape, stride, ndim);
    return t;
}

/** @brief `as_anyrank(..., copy_meta, anyshape<etc,c,c>{}[, layout])` — the static-tail
 *         carrier over an INLINE (device-passable) meta store. */
template <cs::size_t MaxRank = TNY_MAX_RANK, storage Space = storage::view, class T, class offset_t, class S,
          class Layout = keep_strides, cs::enable_if_t<_is_anyshape<S>::value, int> = 0>
_TNY_HOST auto as_anyrank(T * data, const offset_t * shape, const offset_t * stride, int ndim, copy_meta_t, S, Layout = {}) {
    using TailR  = _reindex_extents_t<offset_t, typename S::tail>;
    using TailSt = typename _tail_strides_of<TailR, Layout>::type;
    using HeadR  = _reindex_extents_t<offset_t, typename S::head>;
    using HeadSt = _runtime_strides_t<HeadR::rank()>;   // head strides stay runtime (span the dynamic middle)
    anyrank<T, offset_t, _meta_store<offset_t, MaxRank>, Space, TailR, TailSt, HeadR, HeadSt> t;
    t.data = data; t.ndim = ndim;
    _TNY_CHECK(ndim <= static_cast<int>(MaxRank), "as_anyrank(copy_meta): ndim exceeds MaxRank (raise -DTNY_MAX_RANK)");
    const int n = ndim < static_cast<int>(MaxRank) ? ndim : static_cast<int>(MaxRank);
    for (int i = 0; i < n; ++i) { t.shape(i) = shape[i]; t.stride(i) = stride[i]; }
    _check_ends<HeadR, HeadSt, TailR, TailSt>(shape, stride, ndim);
    return t;
}

/**
 * @brief Narrow a view's — or a whole `anyrank` carrier's — OFFSET INDEX WIDTH to
 *        `Idx2` (default `int32_t`) when its element span fits, then call `f` — else
 *        call `f` with the argument as-is.
 *
 * The kernel-boundary primitive behind the int32 fast path (#115): it instantiates
 * `f` for BOTH widths and picks at run time via `index_fits`/`reindex`, so a genuinely
 * dynamic view runs its offset math in 32-bit (half the by-value footprint, fewer
 * device registers) exactly when that is lossless. `_TNY_HOST`; preserves the view's
 * mutability. Use it standalone on a known-rank view (or a `peel_front` batch cell), or
 * via `dispatch_rank<narrow_index>` to fuse it with the rank dispatch.
 *
 *     for (auto cell : at.peel_front<-Sr>()) dispatch_index(cell, [&](auto c){ kernel<Sr>(c); });
 *
 * An `anyrank` carries the same `index_fits`/`reindex` pair (#467), so the very same
 * call narrows the CARRIER before the rank is fixed — the GPU spelling, since narrowing
 * has to happen host-side, before the launch, while the batch idiom keeps `ndim` runtime.
 * `dispatch_index(at, f)` is then just the two arms written out:
 *
 *     if (at.index_fits<cs::int32_t>()) launch(at.reindex<cs::int32_t>());
 *     else                              launch(at);
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
    if      (v.template is_dense<ccontiguous>()) f(tensor<T, E, ccontiguous, O>(v.data(), v.shape()));  // C-order strides fold
    else if (v.template is_dense<fcontiguous>()) f(tensor<T, E, fcontiguous, O>(v.data(), v.shape()));  // F-order strides fold
    else                                         f(v);                                                    // genuinely strided
}
/** @brief The spelling for `dispatch_rank`'s opt-in flag: `dispatch_rank<narrow_index>(at, f)`. */
inline constexpr bool narrow_index = true;

namespace _detail {
template <cs::size_t R, bool Narrow, class T, class offset_t, class Meta, storage Space,
          class Tail, class TailS, class Head, class HeadS, class F>
_TNY_HOST bool dispatch_from(const anyrank<T, offset_t, Meta, Space, Tail, TailS, Head, HeadS> & t, F & f) {
    if constexpr (R <= anyrank<T, offset_t, Meta, Space, Tail, TailS, Head, HeadS>::max_rank) {
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
template <bool Narrow = false, class T, class offset_t, class Meta, storage Space,
          class Tail, class TailS, class Head, class HeadS, class F>
_TNY_HOST bool dispatch_rank(const anyrank<T, offset_t, Meta, Space, Tail, TailS, Head, HeadS> & t, F && f) {
    // Start the recursion at `ends_rank` (= head_rank + tail_rank): a rank below the
    // static Head+Tail can never equal `ndim` (the boundary check guaranteed
    // `ndim >= ends_rank`), and `fixed<R < ends_rank>()` is a hard static_assert. For an
    // empty Head/Tail this is 0 (unchanged — R=0 still handles a rank-0 scalar ndarray).
    using AR = anyrank<T, offset_t, Meta, Space, Tail, TailS, Head, HeadS>;
    return _detail::dispatch_from<AR::ends_rank, Narrow>(t, f);
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

/** @brief One parameter's candidate list for `dispatch_values` — the runtime value
 *         paired with the compile-time values it is allowed to be. You never spell
 *         this type; `candidates<Vs...>(v)` builds it. */
template <int... Vs> struct candidates_t { int value; };

/**
 * @brief `candidates<1,2,3>(d)` — one `dispatch_values` parameter: the compile-time
 *        candidates as template arguments, the runtime value as the argument.
 *
 * `v` may be any integer **or enum** type — a `bound`/`order` enum dispatches without a
 * hand-written `static_cast` at the call site (the candidates are plain `int`s, and so
 * is the `integral_constant` `f` receives, exactly as with `dispatch_value`).
 */
template <int... Vs, class V>
_TNY_API constexpr candidates_t<Vs...> candidates(V v) noexcept {
    static_assert(cs::is_integral<V>::value || cs::is_enum<V>::value,
                  "candidates(v): the runtime value must be an integer or an enum type");
    return { static_cast<int>(v) };
}

/** @brief `_is_candidates<X>::value` is true iff `X` is a `candidates<...>(v)` list —
 *         the guard `dispatch_values` puts on every argument after `f`. */
template <class> struct _is_candidates : cs::false_type {};
template <int... Vs> struct _is_candidates<candidates_t<Vs...>> : cs::true_type {};

namespace _detail {
// One nesting level of `dispatch_values`, lambda-free (no raw lambda in the engine, so
// it instantiates under nvcc without --extended-lambda). `values_step<F, Bound...>` holds
// a reference to the caller's `f` plus the constants matched SO FAR — as types only, an
// `integral_constant` being empty — and `run(...)` consumes the next candidate list with
// the very same fold `dispatch_value` uses. So this IS the hand-written nesting: one
// match test per parameter (not per combination), and `f` instantiated once per
// combination of candidates — the caller's candidate lists remain the whole budget.
template <class F, class... Bound>
struct values_step {
    F & f;
    _TNY_HOST bool run() const { f(Bound{}...); return true; }        // no lists left -> fire
    template <int... Vs, class... Rest>
    _TNY_HOST bool run(candidates_t<Vs...> c, Rest... rest) const {
        bool matched = false;
        ( (c.value == Vs
              ? (matched = values_step<F, Bound..., cs::integral_constant<int, Vs>>{f}.run(rest...))
              : false), ... );
        return matched;
    }
};
} // namespace _detail

/**
 * @brief The product form of `dispatch_value`: turn SEVERAL runtime values into
 *        compile-time ones in one call, one candidate list per parameter.
 *
 * `f` is called with one `integral_constant` per list, in list order, when every value
 * matched one of its own candidates; the return value says whether it ran. Pure sugar
 * over the nesting — same per-parameter match test, same instantiation count (once per
 * combination of candidates), and the same failure contract per parameter: a value
 * outside its list simply doesn't fire (no assert, no abort), so `f` is not called and
 * the call returns false.
 *
 *     dispatch_values([&](auto D, auto O, auto B){ kernel<D.value, O.value, B.value>(v); },
 *                     candidates<1,2,3>(spatial_ndim),      // spatial rank
 *                     candidates<0,1,2,3>(order),           // interpolation order
 *                     candidates<0,1,2,3,4,5,6,7>(bnd));    // boundary condition (an enum)
 *
 * instead of the three-deep `dispatch_value` pyramid. The candidate lists sit next to
 * each other, so the instantiation budget (3 × 4 × 8 here) is visible in one place.
 * `f` comes FIRST because the candidate lists are variadic — the one place teeny puts
 * the callable ahead of its arguments.
 */
template <class F, class... Ls>
_TNY_HOST bool dispatch_values(F && f, Ls... lists) {
    static_assert((_is_candidates<Ls>::value && ...),
                  "dispatch_values(f, candidates<...>(v), ...): every argument after `f` "
                  "must be a `candidates<...>(v)` list");
    return _detail::values_step<cs::remove_reference_t<F>>{ f }.run(lists...);
}

_TNY_NAMESPACE_END(tny)

#endif // TNY_DYNAMIC_H
