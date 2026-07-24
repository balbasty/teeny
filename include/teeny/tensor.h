#ifndef TNY_MD_TENSOR
#define TNY_MD_TENSOR
#include <cuda/std/mdspan>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <teeny/defines.h>
#include <teeny/alias.h>
#include <teeny/storage.h>
#include <teeny/layout.h>
#include <teeny/indexing.h>
#include <teeny/axis.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

// Forward declarations so the tensor's structural members can name as_tensor
// (its argument is a cuda::std::mdspan, so ADL would not find it).
template <class T, class Shape, class Layout = ccontiguous, own O = own::view>
struct tensor;
template <own OW = own::view, class MD>
_TNY_API tensor<typename MD::element_type, typename MD::extents_type,
                typename MD::layout_type, OW>
as_tensor(const MD & m);

/* --- shared axis dispatch: a STATIC index (`Int<k>()`) folds to an
 *     `integral_constant` where the value is known at compile time; a runtime index
 *     stays a value. Lives ONCE here and is reused by `tensor::extent`/`stride` AND
 *     the `shape()`/`strides()` accessor views below, so the fold rule can't drift. */
template <class Shape, class Layout, long Ax, class Map>
_TNY_API constexpr auto _axis_extent(const Map & m) noexcept {
    using Idx = typename Shape::index_type;
    constexpr cs::size_t D = _norm_axis(Ax, Shape::rank());      // -1 = last axis
    if constexpr (Shape::static_extent(D) != cs::dynamic_extent)
        return cs::integral_constant<Idx, static_cast<Idx>(Shape::static_extent(D))>{};
    else
        return static_cast<Idx>(m.extents().extent(D));
}
template <class Shape, class Layout, long Ax, class Map>
_TNY_API constexpr auto _axis_stride(const Map & m) noexcept {
    using Idx = typename Shape::index_type;
    constexpr cs::size_t D = _norm_axis(Ax, Shape::rank());
    if constexpr (_is_strides<Layout>::value && _static_stride_at<D, Layout>::value != dynamic_stride)
        return cs::integral_constant<Idx, static_cast<Idx>(_static_stride_at<D, Layout>::value)>{};
    else if constexpr (Shape::rank_dynamic() == 0 && _contiguous_layout<Layout>::value)
        return cs::integral_constant<Idx, static_cast<Idx>(Map{}.stride(D))>{};
    // the UNIT stride of a contiguous layout folds even with dynamic extents
    else if constexpr (cs::is_same<Layout, ccontiguous>::value && D + 1 == Shape::rank())
        return cs::integral_constant<Idx, 1>{};
    else if constexpr (cs::is_same<Layout, fcontiguous>::value && D == 0)
        return cs::integral_constant<Idx, 1>{};
    else
        return static_cast<Idx>(m.stride(D));
}

/* --- `shape()` / `strides()` accessor views: array-like, but static-index-aware.
 *     `x[Int<k>()]` folds to a compile-time value where derivable, `x[i]` (runtime)
 *     stays a value — the same rule `extent()`/`stride()` use. `Stride` selects
 *     strides vs extents. Holds the mapping BY VALUE (small, POD-ish) so it never
 *     dangles on a temporary; the shape view converts to the raw extents. */
template <bool Stride, class Shape, class Layout>
struct _geom_view {
    using mapping_type = typename Layout::template mapping<Shape>;
    using index_type   = typename Shape::index_type;
    mapping_type m;

    static constexpr cs::size_t rank() noexcept { return Shape::rank(); }

    template <class Idx, cs::enable_if_t<_is_ic<Idx>::value, int> = 0>
    _TNY_API constexpr auto operator[](Idx) const noexcept {
        if constexpr (Stride) return _axis_stride<Shape, Layout, static_cast<long>(Idx::value)>(m);
        else                  return _axis_extent<Shape, Layout, static_cast<long>(Idx::value)>(m);
    }
    template <class Idx, cs::enable_if_t<!_is_ic<Idx>::value, int> = 0>
    _TNY_API constexpr index_type operator[](Idx d) const noexcept {
        if constexpr (Stride) return static_cast<index_type>(m.stride(static_cast<cs::size_t>(d)));
        else                  return static_cast<index_type>(m.extents().extent(static_cast<cs::size_t>(d)));
    }
    // named spellings, parity with extent()/stride() (only the relevant one exists)
    template <bool S = Stride, class Idx, cs::enable_if_t<!S, int> = 0>
    _TNY_API constexpr auto extent(Idx d) const noexcept { return (*this)[d]; }
    template <bool S = Stride, class Idx, cs::enable_if_t<S, int> = 0>
    _TNY_API constexpr auto stride(Idx d) const noexcept { return (*this)[d]; }

    // the SHAPE view is interchangeable with the raw extents for interop.
    template <bool S = Stride, cs::enable_if_t<!S, int> = 0>
    _TNY_API constexpr operator const Shape &() const noexcept { return m.extents(); }

    // range-for yields runtime values over [0, rank).
    struct iterator {
        const _geom_view * v; cs::size_t i;
        _TNY_API constexpr index_type operator*() const noexcept { return (*v)[static_cast<index_type>(i)]; }
        _TNY_API constexpr iterator & operator++() noexcept { ++i; return *this; }
        _TNY_API constexpr bool operator!=(const iterator & o) const noexcept { return i != o.i; }
    };
    _TNY_API constexpr iterator begin() const noexcept { return { this, 0 }; }
    _TNY_API constexpr iterator end()   const noexcept { return { this, rank() }; }
};

/**
 * @brief Accumulate `v` into `*p`, atomic **on the device only**.
 *
 * The scatter/"push" write: on the device many threads accumulate into
 * overlapping outputs, which a plain `+=` would race. Device -> `atomicAdd`
 * (`double` needs sm_60+, `__half` sm_70+; not all integer widths have an
 * overload — that surfaces as an nvcc error at instantiation). Use via
 * `t.at(i...).add_<true>(v)` (scatter into one cell) / `a.add_<true>(b)`.
 *
 * WARNING: on the **host** this is a plain `*p += v` — NOT atomic. A push kernel
 * parallelised with std::thread over overlapping outputs races; guard those
 * writes yourself (per-thread partials, a mutex, or std::atomic_ref).
 */
template <class T>
_TNY_API void fetch_add(T * p, T v) noexcept {
#ifdef __CUDA_ARCH__
    atomicAdd(p, v);
#else
    *p += v;
#endif
}

/**
 * @brief One N-dimensional tensor, parameterised by ownership.
 *
 * The layout / extents / offset mapping is delegated to `cuda::std::mdspan`
 * (the mapping lives in an empty base, so a fully-static tensor is exactly the
 * size of its data). Ownership is a policy: `own::view` (non-owning, trivially
 * copyable, kernel-passable), `own::stack` (inline storage, static shape),
 * `own::heap` (host-only, move-only), the CUDA owners `own::gpu`/`pinned`/`mapped`
 * (from `cuda.h`), and the space-carrying views `own::gpu_view`/`pinned_view`/
 * `mapped_view` (a view of device / page-locked memory keeps its space). The
 * tensor's copy/move semantics are induced by the storage member, not hand-written.
 *
 * @tparam T        Element type.
 * @tparam Shape    The shape: any `cuda::std::extents<Idx, E...>` (static or
 *                  dynamic per dim). Spell it with the `shape<...>` alias.
 * @tparam Layout   mdspan layout policy (default `ccontiguous`).
 * @tparam O        Ownership kind (default `own::view`).
 */
template <class T, class Shape, class Layout, own O>
struct tensor : private Layout::template mapping<Shape> {
    using element_type = T;
    using extents_type = Shape;   // the shape (a cuda::std::extents); `shape_type` is a synonym
    using shape_type   = Shape;
    using layout_type  = Layout;
    using index_type   = typename Shape::index_type;
    using mapping_type = typename Layout::template mapping<Shape>;
    using view_type       = cs::mdspan<T, Shape, Layout>;
    using const_view_type = cs::mdspan<const T, Shape, Layout>;

    static constexpr own  ownership = O;
    static constexpr bool is_static = (Shape::rank_dynamic() == 0);
    // memory-space flags (mirror the own_* helpers, as compile-time constants):
    static constexpr bool is_view            = own_is_view(O);             // view / gpu_view / pinned_view / mapped_view
    static constexpr bool is_owning          = own_is_owning(O);           // heap/gpu/pinned/mapped
    static constexpr bool is_device          = own_is_device(O);           // gpu or gpu_view
    static constexpr bool is_host_accessible = own_is_host_accessible(O);  // dereferenceable on the host
    static constexpr cs::size_t buffer_size = storage_size<mapping_type, O == own::stack>::value;
    static_assert(O != own::stack || is_static, "stack tensor needs a fully static shape");

    storage<T, O, buffer_size> store_{};

    /* --- constructors --------------------------------------------- */
    tensor() = default;
    // Copy/move are induced by `store_` (view = shallow pointer copy; heap = move
    // of the buffer). They are spelled out because the assignment operators below
    // are — declaring an assignment operator would otherwise suppress the implicit
    // constructors. `= default` keeps them trivial (a view stays trivially copyable
    // / kernel-passable) and keeps a heap tensor move-only (its copy is deleted).
    tensor(const tensor &) = default;
    tensor(tensor &&)      = default;

    /** @brief View constructor: wrap `p` with the given mapping. */
    template <own OO = O, cs::enable_if_t<own_is_view(OO), int> = 0>
    _TNY_API tensor(T * p, mapping_type m) : mapping_type(m), store_(p) {}

    /** @brief View constructor from a pointer alone — for a fully-static geometry
     *         (static extents AND a fully determined layout: contiguous, or an
     *         all-static `strides<...>`). e.g. `tensor<float, shape<3,4>, strides<4,1>>(ptr)`. */
    template <own OO = O, cs::enable_if_t<own_is_view(OO) && is_static &&
              (_contiguous_layout<Layout>::value || _strides_all_static<Layout>::value), int> = 0>
    _TNY_API explicit tensor(T * p) : mapping_type(), store_(p) {}   // explicit: no silent T* -> tensor

    /** @brief View constructor from a pointer + extents (contiguous / static-stride layouts). */
    template <own OO = O, cs::enable_if_t<own_is_view(OO) && cs::is_constructible<mapping_type, Shape>::value, int> = 0>
    _TNY_API tensor(T * p, Shape e) : mapping_type(e), store_(p) {}

    // Allocation size from a mapping, guarded: a negative required_span_size
    // (negative strides — which are for VIEWS, not owning storage) would cast to
    // a huge size_t and try to allocate the universe. Assert, and clamp to 0.
    _TNY_HOST static cs::size_t _alloc_size(const mapping_type & m) {
        const auto n = m.required_span_size();
        _TNY_CHECK(n >= 0, "owning tensor: negative span size (negative strides are for views, not owners)");
        return n < 0 ? cs::size_t(0) : static_cast<cs::size_t>(n);
    }
    /** @brief Owning constructor: allocate storage for `m` (heap/device/host/pinned). */
    template <own OO = O, cs::enable_if_t<own_is_owning(OO), int> = 0>
    _TNY_HOST explicit tensor(mapping_type m)
        : mapping_type(m), store_(_alloc_size(m)) {}

    /** @brief Owning constructor from extents (contiguous / static-stride layouts). */
    template <own OO = O, cs::enable_if_t<own_is_owning(OO) && cs::is_constructible<mapping_type, Shape>::value, int> = 0>
    _TNY_HOST explicit tensor(Shape e)
        : mapping_type(e), store_(_alloc_size(mapping_type(e))) {}

    /* --- geometry ------------------------------------------------- */
    static constexpr cs::size_t rank() noexcept { return Shape::rank(); }
    _TNY_API constexpr const mapping_type & mapping() const noexcept { return *this; }
    _TNY_API constexpr const Shape & extents() const noexcept { return mapping_type::extents(); }
    static constexpr bool is_strides_layout    = _is_strides<Layout>::value;
    static constexpr bool is_contiguous_layout = _contiguous_layout<Layout>::value;

    /** @brief Extent of an axis given by a STATIC index (`extent(Int<0>())`):
     *         a compile-time `integral_constant` when that extent is static,
     *         else a runtime `index_type`. */
    template <class Idx, cs::enable_if_t<_is_ic<Idx>::value, int> = 0>
    _TNY_API constexpr auto extent(Idx) const noexcept
    { return _axis_extent<Shape, Layout, static_cast<long>(Idx::value)>(mapping()); }
    /** @brief Extent of an axis given by a RUNTIME index (`extent(0)`). */
    template <class Idx, cs::enable_if_t<!_is_ic<Idx>::value, int> = 0>
    _TNY_API constexpr index_type extent(Idx d) const noexcept
    { return mapping_type::extents().extent(static_cast<cs::size_t>(d)); }

    /** @brief `shape()` — the extents as an array-like accessor: `shape()[Int<k>()]`
     *         folds to a compile-time value where static, `shape()[i]` (runtime) is a
     *         value, and it converts to the raw `extents()` for interop. `shape(d)`
     *         is the per-axis shorthand (== `extent(d)`). */
    _TNY_API constexpr auto shape() const noexcept { return _geom_view<false, Shape, Layout>{ mapping() }; }
    template <class Idx> _TNY_API constexpr auto shape(Idx d) const noexcept { return extent(d); }
    /** @brief `strides()` — the strides as an array-like accessor (twin of `shape()`):
     *         `strides()[Int<k>()]` folds where the layout makes the stride derivable,
     *         `strides()[i]` (runtime) is a value. `strides(d)` == `stride(d)`. */
    _TNY_API constexpr auto strides() const noexcept { return _geom_view<true, Shape, Layout>{ mapping() }; }

    /** @brief Stride of an axis given by a STATIC index (`stride(Int<0>())`):
     *         a compile-time `integral_constant` when known statically (static-
     *         stride layout; a contiguous layout over static extents; or the
     *         always-unit stride of a contiguous layout even for dynamic shapes). */
    template <class Idx, cs::enable_if_t<_is_ic<Idx>::value, int> = 0>
    _TNY_API constexpr auto stride(Idx) const noexcept
    { return _axis_stride<Shape, Layout, static_cast<long>(Idx::value)>(mapping()); }
    /** @brief Stride of an axis given by a RUNTIME index (`stride(0)`). */
    template <class Idx, cs::enable_if_t<!_is_ic<Idx>::value, int> = 0>
    _TNY_API constexpr index_type stride(Idx d) const noexcept
    { return mapping_type::stride(static_cast<cs::size_t>(d)); }
private:
    static constexpr index_type _static_numel() noexcept {
        index_type n = 1;
        for (cs::size_t r = 0; r < rank(); ++r) n *= static_cast<index_type>(Shape::static_extent(r));
        return n;
    }
public:
    /** @brief Number of elements. A **fully static** shape folds to an
     *         `integral_constant` (so it propagates into later compile-time
     *         arithmetic, like `extent(Int<k>())`); any dynamic dim -> a runtime
     *         `index_type`. */
    _TNY_API constexpr auto numel() const noexcept {
        if constexpr (is_static)
            return cs::integral_constant<index_type, _static_numel()>{};
        else {
            index_type n = 1;
            for (cs::size_t r = 0; r < rank(); ++r) n *= extent(r);
            return n;
        }
    }
    /** @brief Whether the elements occupy a **dense block of memory**, in *some*
     *         axis order — true for a C- or F-contiguous tensor, and also for a
     *         permuted one (a permuted C-contiguous view still packs the same
     *         memory densely). Formally: the strides are a permutation of a dense
     *         nested packing (`1, e0, e0·e1, ...`). Size-1 axes are ignored (their
     *         stride is unconstrained); an empty tensor is trivially contiguous.
     *         Negative strides (flips) are *not* dense in this sense -> false.
     *
     *         Pass a layout for an **exact** check: `is_contiguous<ccontiguous>()`
     *         / `is_contiguous<fcontiguous>()` test C- /
     *         F-contiguity specifically — or any layout whose mapping is derivable
     *         from the extents. */
    _TNY_API constexpr bool is_contiguous() const noexcept {
        constexpr cs::size_t R = rank();
        if constexpr (R == 0) {
            return true;                           // a rank-0 tensor is one element -> trivially dense
        } else {
            bool used[R] = {};
            index_type prod = 1;                   // expected stride of the next-smallest axis
            for (cs::size_t step = 0; step < R; ++step) {
                int best = -1; index_type bs = 0;  // pick the unused extent>1 axis of least stride
                for (cs::size_t r = 0; r < R; ++r) {
                    if (used[r] || extent(r) <= 1) continue;
                    const index_type s = static_cast<index_type>(stride(r));
                    if (best < 0 || s < bs) { best = static_cast<int>(r); bs = s; }
                }
                if (best < 0) break;               // no more constraining axes
                if (bs != prod) return false;      // gap (or a negative/duplicate stride)
                used[best] = true;
                prod *= static_cast<index_type>(extent(best));
            }
            return true;
        }
    }
    /** @brief Exact contiguity in layout `L` (e.g. `ccontiguous`/`fcontiguous`): the
     *         actual strides equal what `L` produces for these extents. Two spellings —
     *         `t.is_contiguous<ccontiguous>()` (type form) and `t.is_contiguous(ccontiguous())`
     *         (value form, layout deduced from the argument). */
    template <class L>
    _TNY_API bool is_contiguous() const noexcept {
        if constexpr (rank() == 0) {
            return true;                           // rank-0: no strides -> matches any layout
        } else {
            typename L::template mapping<extents_type> m(extents());
            for (cs::size_t r = 0; r < rank(); ++r)
                if (static_cast<index_type>(stride(r)) != static_cast<index_type>(m.stride(r))) return false;
            return true;
        }
    }
    template <class L>
    _TNY_API bool is_contiguous(L) const noexcept { return is_contiguous<L>(); }

    /* --- data / views -------------------------------------------- */
    _TNY_API T *       data()       noexcept { return store_.data(); }
    _TNY_API const T * data() const noexcept { return store_.data(); }
    /** @brief The raw `cuda::std::mdspan` over this tensor's storage. */
    _TNY_API view_type       mdspan()       noexcept { return view_type(store_.data(), *this); }
    _TNY_API const_view_type mdspan() const noexcept { return const_view_type(store_.data(), *this); }

    /** @brief A non-owning teeny **view** of this tensor's storage — a `view` (or
     *         `gpu_view`, for a device tensor) that aliases the same memory (no
     *         copy), keeping the source layout. On an already-non-owning tensor it
     *         re-wraps the same pointer (an equivalent view). For the raw mdspan,
     *         use `mdspan()`. */
    _TNY_API auto view()       noexcept { return tensor<T,       Shape, Layout, own_view_of(O)>(store_.data(), mapping()); }
    _TNY_API auto view() const noexcept { return tensor<const T, Shape, Layout, own_view_of(O)>(store_.data(), mapping()); }

    /* --- element access / slicing -------------------------------- */
private:
    // wrap a negative index python-style for axis Ax (see free `_wrap_idx`).
    // `Wrap=false` (the unchecked `uget`/`uat` path) skips the wrap for
    // a runtime signed index — the caller promises it is already non-negative.
    template <cs::size_t Ax, bool Wrap = true, class Arg>
    _TNY_API constexpr index_type _wrap(Arg a) const {
        const index_type n = static_cast<index_type>(extent(cs::integral_constant<cs::size_t, Ax>{}));
        const index_type i = _wrap_idx<index_type, Wrap>(a, n, index_type(0));
        // Checked accessors bounds-check under -DTNY_HARDENED (off by default, off
        // on device); the `u`-accessors (Wrap == false) skip it. Catches both an
        // over-range positive index and a too-negative one (checked after the wrap).
        if constexpr (Wrap) _TNY_BOUND(i >= index_type(0) && i < n, "index out of range");
        return i;
    }
    template <bool Wrap = true, cs::size_t... Ax, class... Args>
    _TNY_API constexpr index_type _offset(cs::index_sequence<Ax...>, Args... a) const {
        return mapping_type::operator()(_wrap<Ax, Wrap>(a)...);
    }
    // resolve one slice bound against the axis extent n (none -> default).
    template <bool Wrap = true, class V> _TNY_API index_type _sl_bound(V v, index_type dflt, index_type n) const {
        return _wrap_idx<index_type, Wrap>(v, n, dflt);
    }
    // ---- the ONE sub-view builder (gather) ------------------------------------
    // Every slicing/take_along call routes here: per axis an integer DROPS the
    // axis (into the base offset), `all` KEEPS it, a range keeps a strided window
    // (optional negative step). Output is teeny's strides<...> layout, folding
    // each kept stride to a compile-time value where derivable — so it works on
    // ANY source layout (no submdspan) AND static shapes stay folded.
    // `stop` default for a negative step: `none` -> -1 (go past index 0), python-style.
    template <bool Wrap = true, class V> _TNY_API index_type _stop_neg(V v, index_type n) const {
        return _wrap_idx<index_type, Wrap>(v, n, index_type(-1));
    }
    template <cs::size_t Ax, bool Wrap = true, class Arg>
    _TNY_API void _sl_axis(Arg a, index_type & off, index_type * ext, index_type * str, cs::size_t & k) const {
        const index_type sd = static_cast<index_type>(stride(Ax));
        const index_type n  = static_cast<index_type>(extent(cs::integral_constant<cs::size_t, Ax>{}));
        if constexpr (_is_index<Arg>::value) {
            off += _wrap<Ax, Wrap>(a) * sd;                         // integer: drop this axis
        } else if constexpr (_is_slice_spec<Arg>::value) {
            const index_type step = static_cast<index_type>(a.step);
            _TNY_CHECK(step != index_type(0), "slice step cannot be 0");  // runtime twin of the compile-time guard (avoids /0)
            // Resolve the (start, stop) defaults per step sign (forward: [0..n];
            // backward: start at the last, stop before index 0), then clamp + count
            // via the shared `_range_count` — the SAME body the compile-time fold
            // `_static_range_len` uses, so the folded static extent can't diverge.
            index_type st, sp;
            if (step >= index_type(0)) { st = _sl_bound<Wrap>(a.start, index_type(0), n); sp = _sl_bound<Wrap>(a.stop, n, n); }
            else                       { st = _sl_bound<Wrap>(a.start, n - 1, n);         sp = _stop_neg<Wrap>(a.stop, n); }
            const index_type cnt = _range_count(st, sp, step, n);
            // An empty axis makes the whole view empty, so its offset is never read;
            // zero it so the accumulated base pointer stays in-bounds — a negative
            // start (step<0) would go BEFORE the buffer, and summed positive starts
            // (step>0, several empty axes) BEYOND one-past-the-end. Both are UB to
            // even form (#67 neg branch, #80 pos branch). #80.
            if (cnt <= index_type(0)) st = index_type(0);
            off += st * sd; ext[k] = cnt; str[k] = step * sd; ++k;  // stride may be negative
        } else {                                                    // full_extent (all)
            ext[k] = n; str[k] = sd; ++k;
        }
    }
    // static output extent for one axis: DROP (integer), the input static extent
    // (an `all`/full_extent OR a folded `slice(none,none)` kept axis), or dynamic.
    template <class Arg, cs::size_t Se> static constexpr cs::size_t _out_static() {
        if constexpr (_is_index<Arg>::value)                            return _drop_axis;
        else if constexpr (cs::is_same<Arg, cs::full_extent_t>::value)  return Se;
        else if constexpr (_is_full_slice<Arg>::value)                  return Se;
        // a compile-time range folds its extent too: source static + static
        // start/stop/step -> the length is computable now (mirrors _sl_axis, incl.
        // the TNY_NO_NEGATIVE_INDEX no-wrap case in _bound_static). Gated to a
        // SIGNED index_type: an unsigned one casts a negative STEP to a huge
        // positive at runtime (forward branch, empty) while the fold reverses, so
        // there we fall back to dynamic (correct, just unfolded).
        else if constexpr (_is_slice_spec<Arg>::value && Se != cs::dynamic_extent &&
                           cs::is_signed<index_type>::value &&
                           _static_bound<typename _slice_start<Arg>::type>::value &&
                           _static_bound<typename _slice_stop<Arg>::type>::value &&
                           _is_ic<typename _slice_step<Arg>::type>::value)
            return _static_range_len<typename _slice_start<Arg>::type,
                                     typename _slice_stop<Arg>::type,
                                     typename _slice_step<Arg>::type>(static_cast<long>(Se));
        else                                                            return cs::dynamic_extent;
    }
    template <bool Wrap = true, class P, cs::size_t... Ax, class... Args>
    _TNY_API auto _slice_range(P p, cs::index_sequence<Ax...>, Args... a) const {
        using Vt = cs::remove_pointer_t<P>;
        constexpr cs::size_t Nk = (cs::size_t(0) + ... + (_is_index<Args>::value ? cs::size_t(0) : cs::size_t(1)));
        // output extents (static where a kept axis is static) and output strides
        // (static where source-stride × step is known) — folded into strides<...>.
        using OE = typename _compact<index_type, _out_static<Args, Shape::static_extent(Ax)>()...>::type;
        using SF = typename _str_compact<_out_sstride<Args, Ax, Layout, Shape>()...>::type;
        index_type ext[Nk ? Nk : 1] = {}, str[Nk ? Nk : 1] = {}, off = 0; cs::size_t k = 0;
        ( _sl_axis<Ax, Wrap>(a, off, ext, str, k), ... );
        cs::array<index_type, Nk> ea{};
        for (cs::size_t i = 0; i < Nk; ++i) ea[i] = ext[i];
        // fold the kept strides into the strides<...> mapping (EBO when all static,
        // else fill the dynamic slots from `str`); Nk == OE::rank().
        return tensor<Vt, OE, SF, own_view_of(O)>(p + off, _detail::fold_mapping<SF>(OE(ea), str));
    }
    // For output axis `out_ax`: pick the front arg, one of the inserted `all`s,
    // or the back arg (shifted past the `fill` inserted `all`s). One ellipsis at
    // position `ell_pos` expands to `fill = rank - (n_args - 1)` copies of `all`.
    template <cs::size_t out_ax, cs::size_t ell_pos, cs::size_t fill, class Tup>
    _TNY_API static auto _ellip_arg(const Tup & t) {
        if constexpr (out_ax < ell_pos)         return cs::get<out_ax>(t);
        else if constexpr (out_ax < ell_pos + fill) return cs::full_extent;  // == `all`, but usable
        else                                    return cs::get<out_ax - fill + 1>(t);
    }
    template <class Tup, cs::size_t... I>
    _TNY_API static constexpr cs::size_t _tup_ellipsis_pos(cs::index_sequence<I...>) {
        return _ellipsis_pos<cs::tuple_element_t<I, Tup>...>();
    }
    // `Wrap` re-dispatches the expanded args through the CHECKED `operator()`
    // (Wrap=true) or the UNCHECKED `uget` (Wrap=false) — so `t.uget(1, ellipsis)`
    // stays wrap-free after the ellipsis is filled with `all`s.
    template <bool Wrap = true, class Tup, cs::size_t... out_ax>
    _TNY_API decltype(auto) _ellip_call(Tup t, cs::index_sequence<out_ax...>) {
        constexpr cs::size_t n_args   = cs::tuple_size<Tup>::value;
        static_assert(n_args - 1 <= rank(), "too many indices for ellipsis expansion");
        constexpr cs::size_t ell_pos  = _tup_ellipsis_pos<Tup>(cs::make_index_sequence<n_args>{});
        constexpr cs::size_t fill     = rank() - (n_args - 1);
        if constexpr (Wrap) return (*this)(_ellip_arg<out_ax, ell_pos, fill>(t)...);
        else                return uget(_ellip_arg<out_ax, ell_pos, fill>(t)...);
    }
    template <bool Wrap = true, class Tup, cs::size_t... out_ax>
    _TNY_API decltype(auto) _ellip_call(Tup t, cs::index_sequence<out_ax...>) const {
        constexpr cs::size_t n_args   = cs::tuple_size<Tup>::value;
        static_assert(n_args - 1 <= rank(), "too many indices for ellipsis expansion");
        constexpr cs::size_t ell_pos  = _tup_ellipsis_pos<Tup>(cs::make_index_sequence<n_args>{});
        constexpr cs::size_t fill     = rank() - (n_args - 1);
        if constexpr (Wrap) return (*this)(_ellip_arg<out_ax, ell_pos, fill>(t)...);
        else                return uget(_ellip_arg<out_ax, ell_pos, fill>(t)...);
    }
public:
    /** @brief Element access when every argument is an integer (negatives wrap). */
    template <class... Args, cs::enable_if_t<(_is_index<Args>::value && ...), int> = 0>
    _TNY_API T & operator()(Args... a) noexcept
    { return store_.data()[_offset(cs::make_index_sequence<rank()>{}, a...)]; }
    template <class... Args, cs::enable_if_t<(_is_index<Args>::value && ...), int> = 0>
    _TNY_API const T & operator()(Args... a) const noexcept
    { return store_.data()[_offset(cs::make_index_sequence<rank()>{}, a...)]; }

    /** @brief `at(i...)` — a single element as a **rank-0 VIEW** (all-integer
     *         args; negatives wrap). Unlike `operator()`, which returns a plain
     *         `T&`, this is a view, so the whole tensor API applies to one
     *         element: `x.at(i,j) = 3` writes it, `float v = x.at(i,j)` reads it
     *         (rank-0 tensors convert to/from `T`), and `x.at(i,j).add_<true>(v)`
     *         is an atomic scatter. */
    template <class... Args, cs::enable_if_t<(_is_index<Args>::value && ...), int> = 0>
    _TNY_API auto at(Args... a) noexcept {
        using E0 = cs::extents<index_type>;   // rank 0
        return tensor<T, E0, ccontiguous, own_view_of(O)>(&store_.data()[_offset(cs::make_index_sequence<rank()>{}, a...)]);
    }
    template <class... Args, cs::enable_if_t<(_is_index<Args>::value && ...), int> = 0>
    _TNY_API auto at(Args... a) const noexcept {
        using E0 = cs::extents<index_type>;
        return tensor<const T, E0, ccontiguous, own_view_of(O)>(&store_.data()[_offset(cs::make_index_sequence<rank()>{}, a...)]);
    }

    /** @brief Sub-view when any argument is a slice (`all`, `slice(a,b[,step])`).
     *         Integer args drop their axis, `all` keeps it, a range keeps a strided
     *         window — all via the one gather (folds static strides into
     *         `strides<...>`; works on any source layout). */
    template <class... Args, cs::enable_if_t<!(_is_index<Args>::value && ...) && !_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API auto operator()(Args... a) noexcept
    { return _slice_range(store_.data(), cs::make_index_sequence<rank()>{}, a...); }
    template <class... Args, cs::enable_if_t<!(_is_index<Args>::value && ...) && !_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API auto operator()(Args... a) const noexcept
    { return _slice_range(store_.data(), cs::make_index_sequence<rank()>{}, a...); }

    /* --- unchecked accessors: skip the negative-index wrap ------------------ *
     * `uget` is the `u`-prefixed twin of `operator()`, and `uat` of `at`, for
     * the hot path where every runtime index is known non-negative: they take
     * runtime signed indices AS-IS (no `i < 0 ? i + n : i` branch per axis) —
     * the per-call form of `-DTNY_NO_NEGATIVE_INDEX`. Passing a negative runtime
     * index is UB (the caller's promise). Static (`Int<>`) bounds and `none` are
     * unaffected, so a compile-time slice still folds identically; the result
     * TYPE matches the checked op exactly.
     *
     * `uget` mirrors `operator()` in full — one entry point, three forms chosen
     * by the argument types, exactly like `operator()`:
     *   - all-integer args  -> element `T&` (teeny has no element bounds check,
     *                          so this is simply the wrap-free read/write);
     *   - any slice arg     -> a VIEW (ranges are still clamped to a valid
     *                          extent — only the negative wrap is dropped);
     *   - one `ellipsis`    -> expand to `all`s and re-dispatch through `uget`,
     *                          so the filled call stays unchecked too.
     * (A negative runtime slice bound is taken as-is and then clamped, so with a
     * forward step a negative stop collapses to an empty axis — "no wrap", not
     * "wrap then clamp".) */
    // element: every arg is an index
    template <class... Args, cs::enable_if_t<(_is_index<Args>::value && ...), int> = 0>
    _TNY_API T & uget(Args... a) noexcept
    { return store_.data()[_offset<false>(cs::make_index_sequence<rank()>{}, a...)]; }
    template <class... Args, cs::enable_if_t<(_is_index<Args>::value && ...), int> = 0>
    _TNY_API const T & uget(Args... a) const noexcept
    { return store_.data()[_offset<false>(cs::make_index_sequence<rank()>{}, a...)]; }

    // slice: at least one slice arg, no ellipsis -> a VIEW
    template <class... Args, cs::enable_if_t<!(_is_index<Args>::value && ...) && !_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API auto uget(Args... a) noexcept
    { return _slice_range<false>(store_.data(), cs::make_index_sequence<rank()>{}, a...); }
    template <class... Args, cs::enable_if_t<!(_is_index<Args>::value && ...) && !_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API auto uget(Args... a) const noexcept
    { return _slice_range<false>(store_.data(), cs::make_index_sequence<rank()>{}, a...); }

    // ellipsis: expand to `all`s, then re-dispatch through `uget` (unchecked)
    template <class... Args, cs::enable_if_t<_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API decltype(auto) uget(Args... a) noexcept
    { return _ellip_call<false>(cs::make_tuple(a...), cs::make_index_sequence<rank()>{}); }
    template <class... Args, cs::enable_if_t<_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API decltype(auto) uget(Args... a) const noexcept
    { return _ellip_call<false>(cs::make_tuple(a...), cs::make_index_sequence<rank()>{}); }

    /** @brief Unchecked `at`: a single element as a rank-0 VIEW, no negative wrap. */
    template <class... Args, cs::enable_if_t<(_is_index<Args>::value && ...), int> = 0>
    _TNY_API auto uat(Args... a) noexcept {
        using E0 = cs::extents<index_type>;
        return tensor<T, E0, ccontiguous, own_view_of(O)>(&store_.data()[_offset<false>(cs::make_index_sequence<rank()>{}, a...)]);
    }
    template <class... Args, cs::enable_if_t<(_is_index<Args>::value && ...), int> = 0>
    _TNY_API auto uat(Args... a) const noexcept {
        using E0 = cs::extents<index_type>;
        return tensor<const T, E0, ccontiguous, own_view_of(O)>(&store_.data()[_offset<false>(cs::make_index_sequence<rank()>{}, a...)]);
    }

    /** @brief Ellipsis form: exactly one `ellipsis` in the args expands to
     *         `rank - (#other args)` copies of `all`, then the call re-runs — so
     *         `t(1, ellipsis, 2)` on rank 5 is `t(1, all, all, all, 2)`. What
     *         remains decides the result (all integers -> element, else view). */
    template <class... Args, cs::enable_if_t<_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API decltype(auto) operator()(Args... a) noexcept
    { return _ellip_call(cs::make_tuple(a...), cs::make_index_sequence<rank()>{}); }
    template <class... Args, cs::enable_if_t<_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API decltype(auto) operator()(Args... a) const noexcept
    { return _ellip_call(cs::make_tuple(a...), cs::make_index_sequence<rank()>{}); }

#if defined(__cpp_multidimensional_subscript)
    /** @brief C++23 multidimensional subscript: `t[i, j, k]` / `t[0, all, slice(1,4)]`
     *         — an exact alias of `operator()` (same element/view result, same
     *         unchecked semantics, same `all`/range/ellipsis args). Available only
     *         when the compiler provides the multidimensional subscript (C++23,
     *         `mdspan`'s own spelling); use `operator()` on C++17/20. */
    template <class... Args>
    _TNY_API decltype(auto) operator[](Args... a)       noexcept { return (*this)(a...); }
    template <class... Args>
    _TNY_API decltype(auto) operator[](Args... a) const noexcept { return (*this)(a...); }
#endif

    /* --- rank-0 <-> scalar interop -------------------------------- *
     * A rank-0 tensor holds exactly one element, so it acts like a scalar:
     * it converts to `T` (read) and is assignable from `T` (write). Together
     * with the math API this makes `at(i...)` a drop-in for a scalar lvalue. */
    template <cs::size_t R = rank(), cs::enable_if_t<R == 0, int> = 0>
    _TNY_API operator T() const noexcept { return store_.data()[0]; }
    template <cs::size_t R = rank(), cs::enable_if_t<R == 0, int> = 0>
    _TNY_API tensor & operator=(T v) noexcept { store_.data()[0] = v; return *this; }
    /** @brief The single element of a rank-0 tensor (explicit reader). */
    template <cs::size_t R = rank(), cs::enable_if_t<R == 0, int> = 0>
    _TNY_API T item() const noexcept { return store_.data()[0]; }

    /* --- assign INTO a slice/temporary view: copies CONTENTS ------- *
     * `a = b` on a NAMED view (an lvalue) rebinds it (shallow — the C++ default).
     * The result of a slice, e.g. `a(ellipsis) = b` or `a(0, all) = b`, is a
     * TEMPORARY (rvalue) view; assigning to it copies b's elements into the viewed
     * region (b broadcasts), the numpy `a[:] = b` meaning. The ref-qualifier is
     * what tells the two apart. A scalar rhs fills.
     *
     * The rebind operators are `&`-qualified (lvalue-only) and defaulted — this is
     * load-bearing, not cosmetic. Left implicit, the compiler-generated copy/move
     * assignment is unqualified, so for an rvalue `*this` with a SAME-TYPE rhs it
     * out-ranks the templated deep-copy below (a non-template beats a template),
     * and `a(slice) = b` silently rebinds a discarded temporary — a no-op. Being
     * `&`-qualified, these apply only to the lvalue rebind and leave every rvalue
     * assignment to the deep-copy template. `= default` keeps them trivial. */
    tensor & operator=(const tensor &) &  = default;   // lvalue: rebind (shallow)
    tensor & operator=(tensor &&)      &  = default;
    template <class B, class E2, class L2, own O2>
    _TNY_API void operator=(const tensor<B,E2,L2,O2> & rhs) && { this->copy_(rhs); }
    template <cs::size_t R = rank(), cs::enable_if_t<(R > 0), int> = 0>
    _TNY_API void operator=(T v) && { this->fill_(v); }

    /* --- structural views (return teeny views) --------------- */

private:
    // per output axis A: the matching take_along arg if A is named, else `all`
    // (keep the axis). Feeds the gather, so index/all/range all work uniformly.
    template <cs::size_t A, cs::size_t... Axes, class Tup>
    _TNY_API auto _ta_raw(const Tup & t) const {
        constexpr int p = _pos_in<A, Axes...>();
        if constexpr (p < 0) return cs::full_extent;
        else                 return cs::get<static_cast<cs::size_t>(p)>(t);
    }
    template <cs::size_t... Axes, class P, class Tup, cs::size_t... A>
    _TNY_API auto _ta_range(P p, const Tup & t, cs::index_sequence<A...> seq) const {
        return _slice_range(p, seq, _ta_raw<A, Axes...>(t)...);
    }
public:
    /**
     * @brief Index/slice one or more named axes; other axes are kept.
     *
     * `take_along<Axes...>(args...)` applies `args[k]` to axis `Axes[k]` (each an
     * integer -- negatives wrap -- or a slice `all`/`rng`) and keeps every other
     * axis, returning a view. e.g. `t.take_along<1>(2)` drops axis 1 at index 2;
     * `t.take_along<0,2>(i, rng(1,4))` binds axes 0 and 2 at once.
     */
    template <long... Axes, class... Args>
    _TNY_API auto take_along(Args... args) noexcept {
        static_assert(sizeof...(Axes) == sizeof...(Args), "take_along: one index per named axis");
        static_assert((_axis_in_range(Axes, rank()) && ...), "take_along: axis out of range");
        static_assert(_all_distinct<_norm_axis(Axes, rank())...>(), "take_along: axes must be distinct");
        return _ta_range<_norm_axis(Axes, rank())...>(store_.data(), cs::make_tuple(args...), cs::make_index_sequence<rank()>{});
    }
    template <long... Axes, class... Args>
    _TNY_API auto take_along(Args... args) const noexcept {
        static_assert(sizeof...(Axes) == sizeof...(Args), "take_along: one index per named axis");
        static_assert((_axis_in_range(Axes, rank()) && ...), "take_along: axis out of range");
        static_assert(_all_distinct<_norm_axis(Axes, rank())...>(), "take_along: axes must be distinct");
        return _ta_range<_norm_axis(Axes, rank())...>(store_.data(), cs::make_tuple(args...), cs::make_index_sequence<rank()>{});
    }

    /** @brief Reorder the axes (a permutation of 0..N-1; negatives wrap) -> a rank-N view. */
    template <long... Perm>
    _TNY_API auto permute() noexcept
    { static_assert(sizeof...(Perm) == rank(), "permute: need N axes"); static_assert(_is_perm<_norm_axis(Perm, rank())...>(), "permute: axes must be a permutation of 0..N-1 (in range, no repeats)"); return as_tensor<own_view_of(O)>(_detail::perm_md(mdspan(), cs::index_sequence<_norm_axis(Perm, rank())...>{})); }
    template <long... Perm>
    _TNY_API auto permute() const noexcept
    { static_assert(sizeof...(Perm) == rank(), "permute: need N axes"); static_assert(_is_perm<_norm_axis(Perm, rank())...>(), "permute: axes must be a permutation of 0..N-1 (in range, no repeats)"); return as_tensor<own_view_of(O)>(_detail::perm_md(mdspan(), cs::index_sequence<_norm_axis(Perm, rank())...>{})); }

    /** @brief Reverse axis `Ax` (negatives wrap) -> a view (numpy `flip`). Uses a
     *         negative stride, so the index type must be signed (`shape<...>` is). */
    template <long Ax = 0>
    _TNY_API auto flip() noexcept
    { static_assert(_axis_in_range(Ax, rank()), "flip: axis out of range"); return as_tensor<own_view_of(O)>(_detail::flip_md<_norm_axis(Ax, rank())>(mdspan(), cs::make_index_sequence<rank()>{})); }
    template <long Ax = 0>
    _TNY_API auto flip() const noexcept
    { static_assert(_axis_in_range(Ax, rank()), "flip: axis out of range"); return as_tensor<own_view_of(O)>(_detail::flip_md<_norm_axis(Ax, rank())>(mdspan(), cs::make_index_sequence<rank()>{})); }

    /** @brief A dense, row-major OWNING copy of this tensor (materialise a view /
     *         non-contiguous / permuted / flipped tensor). Static shape -> stack
     *         (host+device); dynamic -> heap (host only). */
    template <bool S = is_static, cs::enable_if_t<S, int> = 0>
    _TNY_API auto clone() const { tensor<T, Shape, ccontiguous, own::stack> c{}; c.copy_(*this); return c; }
    template <bool S = is_static, cs::enable_if_t<!S, int> = 0>
    _TNY_HOST auto clone() const { tensor<T, Shape, ccontiguous, own::heap> c(extents()); c.copy_(*this); return c; }

    /** @brief pytorch-like `.to<T2>()`: convert the element type to `T2`.
     *
     *  **No copy when it already matches** — if `T2` is the current element type
     *  and `Force` is false, this returns a (read-only) *view* of `*this`, no
     *  allocation, keeping the source layout. So `x.to<>()` is a zero-cost borrow,
     *  not a clone. Because it borrows, the result must not outlive the storage it
     *  points at — the same lifetime rule as `view()`/`permute()`/slicing. Pass
     *  `Force = true` to always materialise a fresh owning copy even when the dtype
     *  already matches (`x.to<float, true>()` force-clones a `float` tensor);
     *  `x.clone()` is the unconditional-copy spelling.
     *
     *  **On a temporary** the borrow is only taken when it *cannot* dangle: a
     *  non-owning **view** rvalue (`view`/`gpu_view` — e.g. a slice or `.to<>()`
     *  result) points at storage owned elsewhere, so borrowing from it is safe and
     *  stays zero-cost. An **owning** rvalue (`stack`/`heap`/`gpu`/`pinned`/
     *  `mapped`) would carry its storage off, so its matching-dtype case forces a
     *  fresh owning copy instead of a dangling borrow (mirroring the free
     *  `to<Space>(tensor&&)`).
     *
     *  When a conversion IS needed (`T2` differs, or `Force`), the result is a
     *  dense, row-major OWNING copy cast elementwise (via `copy_`): static shape
     *  -> stack (host+device), dynamic -> heap (host only). To also move across
     *  memory spaces (host <-> CUDA) use the `to<own::gpu, T2, Force>(x)` free
     *  functions from `<teeny/cuda.h>`. */
    template <class T2 = element_type, bool Force = false,
              cs::enable_if_t<!Force && cs::is_same<T2, element_type>::value, int> = 0>
    _TNY_API auto to() const & {
        return tensor<const element_type, Shape, Layout, own_view_of(O)>(data(), mapping());  // already that dtype -> borrow (gpu_view if device)
    }
    template <class T2 = element_type, bool Force = false, bool S = is_static,
              cs::enable_if_t<(Force || !cs::is_same<T2, element_type>::value) && S, int> = 0>
    _TNY_API auto to() const & { tensor<cs::remove_cv_t<T2>, Shape, ccontiguous, own::stack> c{}; c.copy_(*this); return c; }
    template <class T2 = element_type, bool Force = false, bool S = is_static,
              cs::enable_if_t<(Force || !cs::is_same<T2, element_type>::value) && !S, int> = 0>
    _TNY_HOST auto to() const & { tensor<cs::remove_cv_t<T2>, Shape, ccontiguous, own::heap> c(extents()); c.copy_(*this); return c; }
    // Rvalue overloads. A non-owning VIEW temporary (view/gpu_view) borrows
    // storage owned elsewhere, so a borrow from it is as safe as from an lvalue
    // (and stays _TNY_API even for a dynamic shape — it carries only a pointer).
    // An OWNING temporary (stack/heap/gpu/pinned/mapped) would take its storage
    // with it, so its matching-dtype case must force a fresh owning copy instead
    // of a dangling borrow. A differing dtype / Force copies either way. NB a
    // gpu/pinned/mapped *owning* temporary copies via the host `copy_` path (which
    // host-derefs the buffer) — the same host-oriented limitation the member
    // `.to<>()` dtype-convert already has; use the free `to<Space>(x)` in
    // <teeny/cuda.h> for a real memory-space move.
    template <class T2 = element_type, bool Force = false,
              cs::enable_if_t<own_is_view(O) && !Force && cs::is_same<T2, element_type>::value, int> = 0>
    _TNY_API auto to() const && {
        return tensor<const element_type, Shape, Layout, own_view_of(O)>(data(), mapping());  // view temp -> safe borrow (external storage)
    }
    template <class T2 = element_type, bool Force = false, bool S = is_static,
              cs::enable_if_t<!(own_is_view(O) && !Force && cs::is_same<T2, element_type>::value) && S, int> = 0>
    _TNY_API auto to() const && { tensor<cs::remove_cv_t<T2>, Shape, ccontiguous, own::stack> c{}; c.copy_(*this); return c; }
    template <class T2 = element_type, bool Force = false, bool S = is_static,
              cs::enable_if_t<!(own_is_view(O) && !Force && cs::is_same<T2, element_type>::value) && !S, int> = 0>
    _TNY_HOST auto to() const && { tensor<cs::remove_cv_t<T2>, Shape, ccontiguous, own::heap> c(extents()); c.copy_(*this); return c; }

private:
    // shared reshape body: one axis may be `-1` (numpy-style, inferred from numel).
    template <class El, long... NewExt>
    _TNY_API auto _reshape(El * p) const noexcept {
        static_assert(((NewExt < 0 ? 1 : 0) + ... + 0) <= 1, "reshape: at most one inferred (-1) dimension");
        using NE = cs::extents<index_type, (NewExt < 0 ? cs::dynamic_extent : static_cast<cs::size_t>(NewExt))...>;
        constexpr index_type known = (index_type(1) * ... * (NewExt < 0 ? index_type(1) : index_type(NewExt)));
        constexpr bool has_inferred = ((NewExt < 0) || ...);
        _TNY_CHECK(is_contiguous<ccontiguous>(), "reshape: needs a C-contiguous tensor (clone() first)");
        // With a `-1` the given extents must DIVIDE numel (the rest is inferred);
        // without one they must equal it exactly — else reshape<2,3> of a 24-elem
        // tensor would silently view only the first 6 (a divisor is not a reshape).
        if constexpr (has_inferred)
            _TNY_CHECK(known != 0 && numel() % known == 0, "reshape: numel not divisible by given extents");
        else
            _TNY_CHECK(known == numel(), "reshape: element count must match the given extents (no -1 to infer)");
        const index_type inferred = numel() / (known ? known : index_type(1));
        cs::array<index_type, sizeof...(NewExt)> ea{ (NewExt < 0 ? inferred : index_type(NewExt))... };
        return tensor<El, NE, ccontiguous, own_view_of(O)>(p, typename ccontiguous::template mapping<NE>(NE(ea)));
    }
public:
    /** @brief View this tensor as a new shape — requires it be C-contiguous
     *         (`clone()` first otherwise) and the element count to match. One
     *         extent may be **`-1`** (numpy-style), inferred from the total size:
     *         `t.reshape<6,-1>()`. */
    template <long... NewExt> _TNY_API auto reshape() noexcept       { return _reshape<T, NewExt...>(store_.data()); }
    template <long... NewExt> _TNY_API auto reshape() const noexcept { return _reshape<const T, NewExt...>(store_.data()); }

private:
    template <class El, class NewE, class NewL, cs::size_t... D>
    _TNY_API auto _recast(El * p, cs::index_sequence<D...>) const {
        static_assert(NewE::rank() == rank(), "recast: rank must match");
        // recast re-types the EXTENTS (to recover statically-known dims). Each static
        // dim of NewE must equal the runtime extent (a genuine mismatch is a bug —
        // validated host-debug). The STRIDES come from `NewL`:
        ( _TNY_CHECK(NewE::static_extent(D) == cs::dynamic_extent ||
                     static_cast<index_type>(NewE::static_extent(D)) == static_cast<index_type>(extent(D)),
                     "recast: a static dim does not match the actual extent"), ... );
        NewE oe(cs::array<index_type, rank()>{ static_cast<index_type>(extent(D))... });
        if constexpr (cs::is_same<NewL, keep_strides>::value) {
            // DEFAULT — PRESERVE the source strides: folded via NewE's (now richer)
            // static extents where the source layout makes them derivable, else
            // carried from the actual runtime strides. Works for ANY source layout
            // (contiguous, transposed, broadcast, strided) and can never mis-address.
            using SF = strides< _src_sstride<D, Layout, NewE>()... >;
            const index_type rstr[rank() ? rank() : 1] = { static_cast<index_type>(stride(D))... };
            return tensor<El, NewE, SF, own_view_of(O)>(p, _detail::fold_mapping<SF>(oe, rstr));
        } else if constexpr (cs::is_same<NewL, ccontiguous>::value || cs::is_same<NewL, fcontiguous>::value) {
            // EXPLICIT contiguous — reinterpret AS row/col-major: the strides are
            // DERIVED FROM THE EXTENTS (the "I promise this is contiguous" form, e.g.
            // to fold a dynamic_strides cell's inner unit stride). UB if the data is
            // not actually contiguous in that order — the caller's promise. Emit the
            // folded `strides<...>` (contiguous products, static where the trailing
            // extents are), filling any dynamic slot from the contiguous mapping.
            using SF = strides< _src_sstride<D, NewL, NewE>()... >;
            typename NewL::template mapping<NewE> cm(oe);
            const index_type rstr[rank() ? rank() : 1] = { static_cast<index_type>(cm.stride(D))... };
            return tensor<El, NewE, SF, own_view_of(O)>(p, _detail::fold_mapping<SF>(oe, rstr));
        } else {
            // EXPLICIT `strides<S...>` — bake its static strides; a dynamic_stride
            // slot is filled from the SOURCE stride for that axis. (To impose wholly
            // new runtime strides, `wrap(t.data(), shape, strides)` instead.)
            using SF = NewL;
            const index_type rstr[rank() ? rank() : 1] = { static_cast<index_type>(stride(D))... };
            return tensor<El, NewE, SF, own_view_of(O)>(p, _detail::fold_mapping<SF>(oe, rstr));
        }
    }
public:
    /** @brief Reinterpret with a MORE-STATIC extents type of the same rank —
     *         recover statically-known inner dims at the dynamic (ndarray)
     *         boundary: a runtime `(n,3,3)` view -> `.recast<shape<-1,3,3>>()` so
     *         the `3`s (extents) fold.
     *
     *         `NewLayout` chooses the STRIDES (default `keep_strides`):
     *         - **`keep_strides`** (default) — PRESERVE the source strides; works on
     *           ANY layout (no copy, no contiguity requirement), a strided/transposed
     *           source keeps its strides, a `dynamic_strides` source keeps them at
     *           run time. Never mis-addresses.
     *         - **`ccontiguous`/`fcontiguous`** — reinterpret AS that order, deriving
     *           the strides from the extents (folds the inner unit stride). The
     *           "I promise this is contiguous" form — UB if it isn't.
     *         - **`strides<S...>`** — impose those (static) strides; a `dynamic_stride`
     *           slot comes from the source.
     *
     *         Each static dim of `NewShape` is validated against the actual extent.
     *         Functional form: `recast(shape_value, layout_value)` (both may mix
     *         static/dynamic; the runtime values only deduce the types). */
    template <class NewShape, class NewLayout = keep_strides>
    _TNY_API auto recast()       { return _recast<T,       NewShape, NewLayout>(store_.data(), cs::make_index_sequence<rank()>{}); }
    template <class NewShape, class NewLayout = keep_strides>
    _TNY_API auto recast() const { return _recast<const T, NewShape, NewLayout>(store_.data(), cs::make_index_sequence<rank()>{}); }

    /** @brief View as 1-D (`ravel`) — requires C-contiguous (`clone()` first). Just
     *         `reshape<-1>()` (one inferred dim), spelled out for discoverability. */
    _TNY_API auto flatten() noexcept       { return reshape<-1>(); }
    _TNY_API auto flatten() const noexcept { return reshape<-1>(); }

    /** @brief Insert a size-1 axis at position `Ax` (numpy `newaxis`/`unsqueeze`)
     *         -> a rank-(N+1) view. Negative `Ax` counts from the back, so
     *         `.unsqueeze<-1>()` appends a trailing axis: `(H,W)` -> `(H,W,1)`. */
    template <long Ax = 0>
    _TNY_API auto unsqueeze() noexcept
    { constexpr cs::size_t A = _norm_axis(Ax, rank() + 1); static_assert(A <= rank(), "unsqueeze: axis out of range"); return as_tensor<own_view_of(O)>(_detail::unsqueeze_md<A>(mdspan(), cs::make_index_sequence<rank() + 1>{})); }
    template <long Ax = 0>
    _TNY_API auto unsqueeze() const noexcept
    { constexpr cs::size_t A = _norm_axis(Ax, rank() + 1); static_assert(A <= rank(), "unsqueeze: axis out of range"); return as_tensor<own_view_of(O)>(_detail::unsqueeze_md<A>(mdspan(), cs::make_index_sequence<rank() + 1>{})); }

private:
    static constexpr long _ax_all = cs::numeric_limits<long>::min();   // squeeze() sentinel: "all singletons"
    // gather arg for axis D when squeezing all singletons: drop a STATIC size-1
    // axis (index 0), keep the rest. (A dynamic axis that is 1 only at run time
    // can't be dropped — the rank must stay static.)
    template <cs::size_t D> static _TNY_API auto _sq_arg() noexcept {
        if constexpr (Shape::static_extent(D) == 1) return cs::integral_constant<index_type, 0>{};
        else                                          return cs::full_extent;
    }
    template <class P, cs::size_t... D>
    _TNY_API auto _squeeze_all(P p, cs::index_sequence<D...>) const noexcept
    { return _slice_range(p, cs::make_index_sequence<rank()>{}, _sq_arg<D>()...); }
public:
    /** @brief Drop a size-1 axis `Ax` (negatives wrap) -> a rank-(N-1) view.
     *         `squeeze()` (no axis) drops EVERY statically-size-1 axis. */
    template <long Ax = _ax_all>
    _TNY_API auto squeeze() noexcept {
        if constexpr (Ax == _ax_all) return _squeeze_all(store_.data(), cs::make_index_sequence<rank()>{});
        else { constexpr cs::size_t A = _norm_axis(Ax, rank()); static_assert(A < rank() && rank() > 0, "squeeze: axis out of range");
               static_assert(Shape::static_extent(A) == cs::dynamic_extent || Shape::static_extent(A) == 1,
                             "squeeze: axis must have extent 1");
               _TNY_CHECK(extent(A) == index_type(1), "squeeze: axis must have extent 1");   // runtime check for a dynamic extent
               return as_tensor<own_view_of(O)>(_detail::squeeze_md<A>(mdspan(), cs::make_index_sequence<rank() - 1>{})); }
    }
    template <long Ax = _ax_all>
    _TNY_API auto squeeze() const noexcept {
        if constexpr (Ax == _ax_all) return _squeeze_all(store_.data(), cs::make_index_sequence<rank()>{});
        else { constexpr cs::size_t A = _norm_axis(Ax, rank()); static_assert(A < rank() && rank() > 0, "squeeze: axis out of range");
               static_assert(Shape::static_extent(A) == cs::dynamic_extent || Shape::static_extent(A) == 1,
                             "squeeze: axis must have extent 1");
               _TNY_CHECK(extent(A) == index_type(1), "squeeze: axis must have extent 1");   // runtime check for a dynamic extent
               return as_tensor<own_view_of(O)>(_detail::squeeze_md<A>(mdspan(), cs::make_index_sequence<rank() - 1>{})); }
    }

    /* --- value-form axis args: x.squeeze(Int<1>()) == x.squeeze<1>() ---- *
     * Accept a static integer (`integral_constant`, e.g. `Int<k>()`) in place
     * of the `<k>` template argument, for call sites that prefer the value
     * spelling. `recast(shape<...>{})` deduces the target extents from a value. */
    template <class I, cs::enable_if_t<_is_ic<I>::value, int> = 0> _TNY_API auto flip(I)            noexcept { return flip<static_cast<long>(I::value)>(); }
    template <class I, cs::enable_if_t<_is_ic<I>::value, int> = 0> _TNY_API auto flip(I)      const noexcept { return flip<static_cast<long>(I::value)>(); }
    template <class I, cs::enable_if_t<_is_ic<I>::value, int> = 0> _TNY_API auto squeeze(I)         noexcept { return squeeze<static_cast<long>(I::value)>(); }
    template <class I, cs::enable_if_t<_is_ic<I>::value, int> = 0> _TNY_API auto squeeze(I)   const noexcept { return squeeze<static_cast<long>(I::value)>(); }
    template <class I, cs::enable_if_t<_is_ic<I>::value, int> = 0> _TNY_API auto unsqueeze(I)       noexcept { return unsqueeze<static_cast<long>(I::value)>(); }
    template <class I, cs::enable_if_t<_is_ic<I>::value, int> = 0> _TNY_API auto unsqueeze(I) const noexcept { return unsqueeze<static_cast<long>(I::value)>(); }
    template <class... I, cs::enable_if_t<(sizeof...(I) > 0) && (_is_ic<I>::value && ...), int> = 0> _TNY_API auto permute(I...)       noexcept { return permute<static_cast<long>(I::value)...>(); }
    template <class... I, cs::enable_if_t<(sizeof...(I) > 0) && (_is_ic<I>::value && ...), int> = 0> _TNY_API auto permute(I...) const noexcept { return permute<static_cast<long>(I::value)...>(); }
    template <class... I, cs::enable_if_t<(sizeof...(I) > 0) && (_is_ic<I>::value && ...), int> = 0> _TNY_API auto reshape(I...)       noexcept { return reshape<static_cast<long>(I::value)...>(); }
    template <class... I, cs::enable_if_t<(sizeof...(I) > 0) && (_is_ic<I>::value && ...), int> = 0> _TNY_API auto reshape(I...) const noexcept { return reshape<static_cast<long>(I::value)...>(); }
    template <class NewE> _TNY_API auto recast(NewE)       { return recast<NewE>(); }
    template <class NewE> _TNY_API auto recast(NewE) const { return recast<NewE>(); }
    // functional form with an explicit layout: recast(shape<...>{}, ccontiguous{}),
    // recast(shp, strides<S...>{}). Deduces both types; the runtime values pick the
    // types only (dynamic extents/strides come from the source — spell an exact
    // runtime layout with `wrap(t.data(), shape, strides)`).
    template <class NewE, class NewL> _TNY_API auto recast(NewE, NewL)       { return recast<NewE, NewL>(); }
    template <class NewE, class NewL> _TNY_API auto recast(NewE, NewL) const { return recast<NewE, NewL>(); }

    /* --- in-place elementwise math (declared here, defined in math.h) --- *
     * tensor rhs broadcasts; a scalar rhs applies to all. add_/sub_ take a
     * bool `Atomic` flag (default false): `a.add_<true>(b)` commits with
     * fetch_add — the atomic-on-device scatter/"push" write. */
    template <bool Atomic = false, class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & add_(const B & b);
    template <bool Atomic = false, class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & sub_(const B & b);
    template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & mul_(const B & b);
    template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & div_(const B & b);
    template <bool Atomic = false> _TNY_API tensor & add_(T s);
    template <bool Atomic = false> _TNY_API tensor & sub_(T s);
    _TNY_API tensor & mul_(T s);
    _TNY_API tensor & div_(T s);

    /* --- compound assignment (sugar over the in-place ops; broadcasts a
     *     tensor rhs, applies a scalar rhs) ------------------------------- */
    template <class B> _TNY_API tensor & operator+=(const B & b) { if constexpr (cs::is_arithmetic<B>::value) add_(static_cast<T>(b)); else add_(b); return *this; }
    template <class B> _TNY_API tensor & operator-=(const B & b) { if constexpr (cs::is_arithmetic<B>::value) sub_(static_cast<T>(b)); else sub_(b); return *this; }
    template <class B> _TNY_API tensor & operator*=(const B & b) { if constexpr (cs::is_arithmetic<B>::value) mul_(static_cast<T>(b)); else mul_(b); return *this; }
    template <class B> _TNY_API tensor & operator/=(const B & b) { if constexpr (cs::is_arithmetic<B>::value) div_(static_cast<T>(b)); else div_(b); return *this; }

    /* --- assignment / fill (broadcasting) ------------------------- */
    template <class B> _TNY_API tensor & copy_(const B & b);   // *this = b (broadcasts)
    _TNY_API tensor & fill_(T s);                              // *this = s
    _TNY_API tensor & zero_();                                 // *this = 0
    _TNY_API tensor & iota_(T start = T(0), T step = T(1));    // start, start+step, ... (row-major)

    /* --- out-of-place elementwise (tensor OR scalar rhs) -> new tensor --- */
    template <class B> _TNY_API auto add(const B & b) const;
    template <class B> _TNY_API auto sub(const B & b) const;
    template <class B> _TNY_API auto mul(const B & b) const;
    template <class B> _TNY_API auto div(const B & b) const;
    template <class B> _TNY_API auto pow(const B & b) const;

    /* --- generic elementwise with a user functor (device-safe) ---- */
    template <class F> _TNY_API tensor & map_(F f);                    // *this = f(*this)
    template <class G, class B> _TNY_API tensor & zip_with_(G g, const B & b);  // *this = g(*this, b) (broadcasts)
    template <class F> _TNY_API auto map(F f) const;                   // -> new tensor = f(*this)

    /* --- boolean reductions (numpy-style; `all` is the slice keyword, so
     *     these are members, and chain after a comparison: (a<b).all()) ---- */
    _TNY_API bool all() const;   // true iff every element is nonzero
    _TNY_API bool any() const;   // true iff any element is nonzero

    /* --- in-place unary math (element-wise) ----------------------- */
    _TNY_API tensor & neg_();
    _TNY_API tensor & abs_();
    _TNY_API tensor & exp_();
    _TNY_API tensor & log_();
    _TNY_API tensor & sin_();
    _TNY_API tensor & cos_();
    _TNY_API tensor & sqrt_();
    _TNY_API tensor & tanh_();
    _TNY_API tensor & floor_();
    _TNY_API tensor & ceil_();
    _TNY_API tensor & round_();
    _TNY_API tensor & trunc_();
    _TNY_API tensor & sign_();                 // -1 / 0 / +1
    _TNY_API tensor & pow_(T e);
    _TNY_API tensor & clamp_(T lo, T hi);      // clamp each element to [lo, hi]

    /* --- increment / decrement --------------------------------------- *
     * Prefix ++/-- mutate in place (add/subtract 1 from every element).
     * Postfix returns the pre-value, so it must allocate a copy -> only a
     * STATIC shape (stack copy, host+device); a dynamic shape has no postfix. */
    _TNY_API tensor & operator++() { return add_(T(1)); }
    _TNY_API tensor & operator--() { return sub_(T(1)); }
    template <bool S = is_static, cs::enable_if_t<S, int> = 0>
    _TNY_API tensor<T, Shape, ccontiguous, own::stack> operator++(int) { auto old = clone(); add_(T(1)); return old; }
    template <bool S = is_static, cs::enable_if_t<S, int> = 0>
    _TNY_API tensor<T, Shape, ccontiguous, own::stack> operator--(int) { auto old = clone(); sub_(T(1)); return old; }
};

/* ------------------------------------------------------------------ *
 *     Factories                                                      *
 * ------------------------------------------------------------------ */

/** @brief Wrap `p` as a non-owning view with a contiguous layout (default
 *         C-order). This is the factory; the `view<T,E>` alias is the type it
 *         produces, and the member `t.view()` re-views an existing tensor. */
template <class Layout = ccontiguous, class T, class Shape>
_TNY_API tensor<T, Shape, Layout, own::view> wrap(T * p, Shape e) {
    using Tn = tensor<T, Shape, Layout, own::view>;
    return Tn(p, typename Tn::mapping_type(e));
}

/** @brief Wrap `p` as a non-owning view with explicit **runtime strides** (a
 *         `layout_stride` view). Pass one stride per dimension — an `array` or a
 *         braced list — in ELEMENTS; strides may be negative (a reversed view).
 *
 *         `wrap(p, shape<2,3>{}, {3, 1})` is the row-major view; `{1, 2}` the
 *         column-major one. For strides known at compile time pass a
 *         `strides<S...>{}` instead (overload below) so they fold into the type. */
template <class T, class Shape>
_TNY_API tensor<T, Shape, cs::layout_stride, own::view>
wrap(T * p, Shape e, cs::array<typename Shape::index_type, Shape::rank()> st) {
    using Tn = tensor<T, Shape, cs::layout_stride, own::view>;
    return Tn(p, typename Tn::mapping_type(e, st));
}

/** @brief Wrap `p` as a non-owning view with per-dimension **compile-time
 *         strides** (may be negative): pass a `strides<S...>{}` as the third
 *         argument. `wrap(p, shape<3,3>{}, strides<4,1>{})` folds the strides into
 *         the type (`strides<S...>` layout, EBO). Every stride must be a
 *         compile-time value — a `strides<...>` tag is a *stateless* layout, so it
 *         cannot carry runtime strides. For a **mix** of static and runtime
 *         strides, use the template form below; for all-runtime strides the
 *         `{s...}` overload above (a `layout_stride` view) is simplest. */
template <cs::int64_t... Strides, class T, class Shape>
_TNY_API tensor<T, Shape, strides<Strides...>, own::view>
wrap(T * p, Shape e, strides<Strides...>) {
    static_assert(strides<Strides...>::all_static(),
        "wrap(ptr, shape, strides<...>{}): a strides<> tag carries only COMPILE-TIME "
        "strides; for mixed strides use wrap<S...>(ptr, shape, {runtime slots}), or "
        "for all-runtime strides pass the values as `{s0, s1, ...}`");
    using Tn = tensor<T, Shape, strides<Strides...>, own::view>;
    return Tn(p, typename Tn::mapping_type(e));
}

/** @brief Wrap `p` with a **mix of static and runtime strides** — the exact
 *         analogue of `shape<-1,2,3,-1>{d0,d1}` for strides. Give the per-dim
 *         pattern as template args (a compile-time stride, or `dynamic_stride`
 *         for a runtime one) and the runtime strides for the `dynamic_stride`
 *         slots as a braced list, in order:
 *
 *             wrap<dynamic_stride, 1>(ptr, shape<3,3>{}, {4});   // outer=4 (runtime), inner=1 (folds)
 *             wrap<dynamic_stride, dynamic_stride>(ptr, sh, {4,1}); // both runtime (a strides<> layout)
 *
 *         The static slots fold into the type; only the runtime ones are stored. */
template <cs::int64_t S0, cs::int64_t... Srest, class T, class Shape>   // S0 forces explicit <...>
_TNY_API tensor<T, Shape, strides<S0, Srest...>, own::view>
wrap(T * p, Shape e, cs::array<typename Shape::index_type, strides<S0, Srest...>::ndyn()> dyn) {
    using Tn = tensor<T, Shape, strides<S0, Srest...>, own::view>;
    return Tn(p, typename Tn::mapping_type(e, dyn));
}

/** @brief Compile-time memory-space traits (SFINAE-friendly free forms of the
 *         tensor's `is_view`/`is_device`/… members): `is_view_v<decltype(x)>`. */
template <class Tn> inline constexpr bool is_view_v            = Tn::is_view;
template <class Tn> inline constexpr bool is_owning_v          = Tn::is_owning;
template <class Tn> inline constexpr bool is_device_v          = Tn::is_device;
template <class Tn> inline constexpr bool is_host_accessible_v = Tn::is_host_accessible;

/** @brief A non-owning view type. Construct as `view<T,E>(ptr, extents)`. */
template <class T, class Shape, class Layout = ccontiguous>
using view = tensor<T, Shape, Layout, own::view>;

/** @brief Stack-owned tensor (fully static shape). Use `local<T,E>{}`. */
template <class T, class Shape, class Layout = ccontiguous>
using local = tensor<T, Shape, Layout, own::stack>;

/** @brief Heap-owned tensor (host only, move-only). Use `owned<T,E>(extents)`. */
template <class T, class Shape, class Layout = ccontiguous>
using owned = tensor<T, Shape, Layout, own::heap>;

/* --- functional factories (deduce the Shape type from the argument) ------ *
 * Complements the type aliases above; the `make_` prefix keeps them distinct.
 * Element type `T` is explicit (it can't be deduced from a shape); the extents
 * type is deduced, so a runtime-built shape needs no `decltype` spelling.       */

/** @brief `make_view<L>(ptr, extents)` — a non-owning view (alias of `wrap`). */
template <class Layout = ccontiguous, class T, class Shape>
_TNY_API auto make_view(T * p, Shape e) { return wrap<Layout>(p, e); }

/** @brief `empty<T>(extents)` — a new UNINITIALISED tensor. The one factory the
 *  `make_*` family fuses into: ownership is **deduced** from the shape (fully
 *  static -> `stack` (host+device); any dynamic extent -> `heap` (host)) unless a
 *  backend is named — `empty<T, own::gpu>(extents)`, or the value-tag spelling
 *  `empty<T>(extents, own_c<own::gpu>{})`. `gpu`/`pinned`/`mapped` require
 *  `<teeny/cuda.h>` (their storage lives there). `T` defaults to `float`. Split
 *  by the resolved ownership so the `stack` case stays `_TNY_API` (host+device)
 *  while the allocating cases are `_TNY_HOST`. */
template <class T = float, own O = own_deduce, class Layout = ccontiguous, class Shape,
          cs::enable_if_t<own_resolve(O, Shape::rank_dynamic() == 0) == own::stack, int> = 0>
_TNY_API auto empty(Shape = Shape{}) { return tensor<T, Shape, Layout, own::stack>{}; }
template <class T = float, own O = own_deduce, class Layout = ccontiguous, class Shape,
          cs::enable_if_t<own_resolve(O, Shape::rank_dynamic() == 0) != own::stack, int> = 0>
_TNY_HOST auto empty(Shape e) {
    constexpr own R = own_resolve(O, Shape::rank_dynamic() == 0);
    static_assert(!own_is_view(R), "empty(): a non-owning view kind (view/gpu_view/pinned_view/mapped_view) has no storage to allocate — use wrap()/make_view() for a view.");
    return tensor<T, Shape, Layout, R>(e);
}
/** @brief Value-tag backend form: `empty<T>(extents, own_c<own::gpu>{})`. Always
 *  `_TNY_HOST` (a host-side convenience); for a device-usable static-shape build
 *  spell the backend as a template arg — `empty<T, own::stack>(extents)`. */
template <class T = float, class Layout = ccontiguous, class Shape, own O>
_TNY_HOST auto empty(Shape e, own_c<O>) { return empty<T, O, Layout>(e); }

/** @brief `make_local<T>(extents)` — a stack-owned tensor (static shape).
 *         `T` defaults to `float` (numpy's default float dtype). Thin spelling of
 *         `empty<T, own::stack>`. */
template <class T = float, class Layout = ccontiguous, class Shape>
_TNY_API auto make_local(Shape = Shape{}) { return empty<T, own::stack, Layout>(Shape{}); }

/** @brief `make_heap<T>(extents)` — a heap-owned tensor (host, move-only).
 *         `T` defaults to `float`. Thin spelling of `empty<T, own::heap>`. */
template <class T = float, class Layout = ccontiguous, class Shape>
_TNY_HOST auto make_heap(Shape e) { return empty<T, own::heap, Layout>(e); }

/* --- numpy-style creation factories: static shape -> stack (host+device),   *
 *     dynamic shape -> heap (host only), mirroring the out-of-place ops.       */

/** @brief `full(extents, v)` — a new tensor filled with `v`. The element type
 *         defaults to the **value's** type (numpy/pytorch: `full(s, 3)` is int,
 *         `full(s, 3.0)` is float); pass `full<T>(...)` to override. Unlike the
 *         value-less `zeros`/`ones` (which default to `float`), there is a value
 *         here to infer from, so we do.
 *
 *  Ownership is deduced from the shape (static -> stack, dynamic -> heap) unless a
 *  **backend** is named — `full<T, own::pinned>(s, v)` or the value-tag
 *  `full<T>(s, v, own_c<own::pinned>{})`. Because it fills host-side, only
 *  host-accessible backends (stack/heap/pinned/mapped) are allowed; a device
 *  (`gpu`) fill needs a kernel launch, so it is a `static_assert` steering you to
 *  `to<own::gpu>(full<T>(s, v))`. Split by resolved ownership for the
 *  `_TNY_API`/`_TNY_HOST` annotation. */
template <class T = void, own O = own_deduce, class Layout = ccontiguous, class Shape, class V,
          class ET = cs::conditional_t<cs::is_same<T, void>::value, V, T>,
          cs::enable_if_t<own_resolve(O, Shape::rank_dynamic() == 0) == own::stack, int> = 0>
_TNY_API auto full(Shape e, V v) { auto t = empty<ET, O, Layout>(e); t.fill_(static_cast<ET>(v)); return t; }
template <class T = void, own O = own_deduce, class Layout = ccontiguous, class Shape, class V,
          class ET = cs::conditional_t<cs::is_same<T, void>::value, V, T>,
          cs::enable_if_t<own_resolve(O, Shape::rank_dynamic() == 0) != own::stack, int> = 0>
_TNY_HOST auto full(Shape e, V v) {
    static_assert(own_is_host_accessible(own_resolve(O, Shape::rank_dynamic() == 0)),
        "zeros/ones/full<..., own::gpu>: a device fill needs a kernel launch; use to<own::gpu>(full<T>(shape, v)) (or to<own::gpu>(zeros<T>(shape))), or empty<T, own::gpu>(shape) then a memset.");
    auto t = empty<ET, O, Layout>(e); t.fill_(static_cast<ET>(v)); return t;
}
/** @brief Value-tag backend form: `full<T>(extents, v, own_c<own::pinned>{})`. */
template <class T = void, class Layout = ccontiguous, class Shape, class V, own O>
_TNY_HOST auto full(Shape e, V v, own_c<O>) { return full<T, O, Layout>(e, v); }

/** @brief `zeros<T>(extents)` / `ones<T>(extents)` — a new tensor of 0s / 1s.
 *         `T` defaults to `float`. Same ownership deduction and backend selector
 *         as `full` (a device backend `static_assert`s — fill via
 *         `to<own::gpu>(zeros<T>(shape))`); the annotation is split to match the
 *         `full` overload each routes to. */
template <class T = float, own O = own_deduce, class Layout = ccontiguous, class Shape,
          cs::enable_if_t<own_resolve(O, Shape::rank_dynamic() == 0) == own::stack, int> = 0>
_TNY_API  auto zeros(Shape e) { return full<T, O, Layout>(e, T(0)); }
template <class T = float, own O = own_deduce, class Layout = ccontiguous, class Shape,
          cs::enable_if_t<own_resolve(O, Shape::rank_dynamic() == 0) != own::stack, int> = 0>
_TNY_HOST auto zeros(Shape e) { return full<T, O, Layout>(e, T(0)); }
template <class T = float, class Layout = ccontiguous, class Shape, own O>
_TNY_HOST auto zeros(Shape e, own_c<O>) { return zeros<T, O, Layout>(e); }
template <class T = float, own O = own_deduce, class Layout = ccontiguous, class Shape,
          cs::enable_if_t<own_resolve(O, Shape::rank_dynamic() == 0) == own::stack, int> = 0>
_TNY_API  auto ones(Shape e) { return full<T, O, Layout>(e, T(1)); }
template <class T = float, own O = own_deduce, class Layout = ccontiguous, class Shape,
          cs::enable_if_t<own_resolve(O, Shape::rank_dynamic() == 0) != own::stack, int> = 0>
_TNY_HOST auto ones(Shape e) { return full<T, O, Layout>(e, T(1)); }
template <class T = float, class Layout = ccontiguous, class Shape, own O>
_TNY_HOST auto ones(Shape e, own_c<O>) { return ones<T, O, Layout>(e); }

/** @brief `arange<T>(n)` — a 1-D tensor `[0, 1, ..., n-1]` (heap, host). `T`
 *         defaults to `int64_t` (an integer range, like numpy `arange(n)`). A
 *         host-accessible backend may be named — `arange<T, own::pinned>(n)` or
 *         `arange<T>(n, own_c<own::pinned>{})`; a device backend `static_assert`s
 *         (use `to<own::gpu>(arange<T>(n))`). The static-N forms below stay stack. */
template <class T = cs::int64_t, own O = own_deduce>
_TNY_HOST auto arange(long n) {
    using E = cs::dextents<cs::int64_t, 1>;
    static_assert(own_is_host_accessible(own_resolve(O, false)),
        "arange<..., own::gpu>: a device fill needs a kernel launch; use to<own::gpu>(arange<T>(n)).");
    auto t = empty<T, O, ccontiguous>(E{n}); t.iota_(); return t;
}
/** @brief Value-tag backend form: `arange<T>(n, own_c<own::pinned>{})`. */
template <class T = cs::int64_t, own O>
_TNY_HOST auto arange(long n, own_c<O>) { return arange<T, O>(n); }
/** @brief Static `arange<T, N>()` — a stack `[0..N-1]` (host+device, folds). */
template <class T = cs::int64_t, long N>
_TNY_API auto arange() { tensor<T, cs::extents<cs::int64_t, static_cast<cs::size_t>(N)>, ccontiguous, own::stack> t{}; t.iota_(); return t; }
/** @brief `arange<T>(Int<N>())` — the static form spelled with a static integer. */
template <class T = cs::int64_t, class V, V N>
_TNY_API auto arange(cs::integral_constant<V, N>) { return arange<T, static_cast<long>(N)>(); }

/** @brief Wrap any `cuda::std::mdspan` (e.g. a `submdspan` result) as a
 *         non-owning `tny::tensor` view, so the tensor API applies to it. */
template <own OW, class MD>
_TNY_API tensor<typename MD::element_type, typename MD::extents_type,
                typename MD::layout_type, OW>
as_tensor(const MD & m) {
    using Tn = tensor<typename MD::element_type, typename MD::extents_type,
                      typename MD::layout_type, OW>;
    return Tn(m.data_handle(), m.mapping());
}

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_TENSOR
