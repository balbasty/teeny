#ifndef TNY_MD_TENSOR
#define TNY_MD_TENSOR
#include <cuda/std/mdspan>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <teeny/_core/defines.h>
#include <teeny/storage.h>
#include <teeny/layout.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

// Forward declarations so the tensor's structural members can name as_tensor
// (its argument is a cuda::std::mdspan, so ADL would not find it).
template <class T, class Extents, class Layout = cs::layout_right, own O = own::view>
struct tensor;
template <class MD>
_TNY_API tensor<typename MD::element_type, typename MD::extents_type,
                typename MD::layout_type, own::view>
as_tensor(const MD & m);

/* --- detail: single-axis bind and permutation on a raw mdspan ----- */
namespace _detail {
template <cs::size_t D, cs::size_t A, class I>
_TNY_API auto take_arg(I i) { if constexpr (A == D) return i; else { (void)i; return cs::full_extent; } }
template <cs::size_t D, class MD, cs::size_t... A>
_TNY_API auto take_md(const MD & v, typename MD::index_type i, cs::index_sequence<A...>) {
    return cs::submdspan(v, take_arg<D, A>(i)...);
}
template <class MD, cs::size_t... P>
_TNY_API auto perm_md(const MD & v, cs::index_sequence<P...>) {
    using El  = typename MD::element_type;
    using Idx = typename MD::index_type;
    using E   = typename MD::extents_type;
    using PE  = cs::extents<Idx, E::static_extent(P)...>;
    cs::layout_stride::mapping<PE> m(
        PE(static_cast<Idx>(v.extent(P))...),
        cs::array<Idx, sizeof...(P)>{ static_cast<Idx>(v.stride(P))... });
    return cs::mdspan<El, PE, cs::layout_stride>(v.data_handle(), m);
}
// insert a size-1 axis at position AX (output rank = N+1). The new axis gets
// stride 1 (its index is always 0, so the value is irrelevant to the offset).
// J... = 0..N ; input axis for output j is j (j<AX) or j-1 (j>AX).
template <cs::size_t AX, class MD, cs::size_t... J>
_TNY_API auto unsqueeze_md(const MD & v, cs::index_sequence<J...>) {
    using El  = typename MD::element_type;
    using Idx = typename MD::index_type;
    using E   = typename MD::extents_type;
    using OE  = cs::extents<Idx, (J == AX ? cs::size_t(1) : E::static_extent(J < AX ? J : J - 1))...>;
    cs::layout_stride::mapping<OE> m(
        OE(static_cast<Idx>(J == AX ? Idx(1) : v.extent(J < AX ? J : J - 1))...),
        cs::array<Idx, sizeof...(J)>{ static_cast<Idx>(J == AX ? Idx(1) : v.stride(J < AX ? J : J - 1))... });
    return cs::mdspan<El, OE, cs::layout_stride>(v.data_handle(), m);
}
// drop axis AX (must have extent 1) -> output rank = N-1. J... = 0..N-2 ;
// input axis for output j is j (j<AX) or j+1 (j>=AX).
template <cs::size_t AX, class MD, cs::size_t... J>
_TNY_API auto squeeze_md(const MD & v, cs::index_sequence<J...>) {
    using El  = typename MD::element_type;
    using Idx = typename MD::index_type;
    using E   = typename MD::extents_type;
    using OE  = cs::extents<Idx, E::static_extent(J < AX ? J : J + 1)...>;
    cs::layout_stride::mapping<OE> m(
        OE(static_cast<Idx>(v.extent(J < AX ? J : J + 1))...),
        cs::array<Idx, sizeof...(J)>{ static_cast<Idx>(v.stride(J < AX ? J : J + 1))... });
    return cs::mdspan<El, OE, cs::layout_stride>(v.data_handle(), m);
}
} // namespace _detail

// traits: which layouts expose compile-time strides
template <class L> struct _static_strides : cs::false_type {};
template <cs::size_t... S> struct _static_strides<layout_static_stride<S...>> : cs::true_type {};
template <class L> struct _contiguous_layout : cs::false_type {};
template <> struct _contiguous_layout<cs::layout_right> : cs::true_type {};
template <> struct _contiguous_layout<cs::layout_left>  : cs::true_type {};

// D-th value of a size_t pack (recursive, so it is a constant expression)
template <cs::size_t D, cs::size_t S0, cs::size_t... S>
struct _nth_size { static constexpr cs::size_t value = _nth_size<D - 1, S...>::value; };
template <cs::size_t S0, cs::size_t... S>
struct _nth_size<0, S0, S...> { static constexpr cs::size_t value = S0; };

template <cs::size_t D, class L> struct _static_stride_at;
template <cs::size_t D, cs::size_t... S>
struct _static_stride_at<D, layout_static_stride<S...>> { static constexpr cs::size_t value = _nth_size<D, S...>::value; };

// an "index" argument to operator() is a (static or runtime) integer; anything
// else (all / rng, i.e. a slice specifier) turns operator() into a view.
template <class T> struct _is_ic : cs::false_type {};
template <class T, T V> struct _is_ic<cs::integral_constant<T,V>> : cs::true_type {};
template <class A> struct _is_index
    : cs::integral_constant<bool, cs::is_integral<A>::value || _is_ic<A>::value> {};

/** @brief Open-ended slice sentinel — teeny's `None` (python `a[:n]` / `a[m:]`).
 *
 * `none` is **static**: `slice(none, n)` starts at 0, `slice(m, none)` runs to
 * the end, and `slice(none, none)` folds to `full_extent` — i.e. `all` *is* the
 * static-none whole-axis slice (`all == slice(none, none)`), keeping the axis
 * and preserving its static extent. Combined with runtime bounds it resolves at
 * run time, so the one sentinel serves both the static (folding) and runtime
 * (dynamic) cases. */
struct none_t {};
constexpr none_t none{};

// a python-like slice spec `[start : stop : step]`; start/stop may be `none`.
template <class A, class B, class S>
struct slice_spec { A start; B stop; S step; };
template <class T> struct _is_slice_spec : cs::false_type {};
template <class A, class B, class S> struct _is_slice_spec<slice_spec<A,B,S>> : cs::true_type {};

// step == 1 (as a static integral constant)?
template <class S> _TNY_API constexpr bool _step1() {
    if constexpr (_is_ic<S>::value) return static_cast<long>(S::value) == 1; else return false;
}
// a "full" slice is `slice(none, none)` with unit step: it is exactly `all`
// (keeps the whole axis, folds, preserves static extents). `all == slice(none,none)`.
template <class Arg> struct _is_full_slice : cs::false_type {};
template <class S> struct _is_full_slice<slice_spec<none_t,none_t,S>> : cs::integral_constant<bool, _step1<S>()> {};

// a "real" range (needs the layout_stride path) is a slice that is not a full slice
template <class Arg> struct _is_real_range
    : cs::integral_constant<bool, _is_slice_spec<Arg>::value && !_is_full_slice<Arg>::value> {};
template <class... Args> struct _any_range : cs::integral_constant<bool, (_is_real_range<Args>::value || ... || false)> {};

/**
 * @brief A python-like slice `[start : stop : step)` for `operator()` /
 *        `take_along`. `none` marks an open end; negative bounds wrap
 *        (count from the back); `step` defaults to 1 and may exceed 1.
 *
 * `slice(1, 4)` = `[1,4)`; `slice(none, 4)` = `[0,4)`; `slice(2, none)` =
 * `[2,end)`; `slice(0, none, 2)` = every other element; `slice(none, none)`
 * keeps the whole axis (== `all`, which is preferable when you want the axis
 * kept — it folds and preserves static extents). A ranged axis is resolved at
 * run time (its extent becomes dynamic); axes kept with `all` stay static.
 */
template <class A, class B>
_TNY_API auto slice(A start, B stop) { return slice_spec<A, B, cs::integral_constant<long,1>>{ start, stop, {} }; }
template <class A, class B, class S>
_TNY_API auto slice(A start, B stop, S step) { return slice_spec<A, B, S>{ start, stop, step }; }

// position of axis A within the pack Axes... (-1 if absent)
template <cs::size_t A, cs::size_t... Axes>
_TNY_API constexpr int _pos_in() {
    cs::size_t axs[] = { Axes..., static_cast<cs::size_t>(-1) };
    for (cs::size_t p = 0; p < sizeof...(Axes); ++p) if (axs[p] == A) return static_cast<int>(p);
    return -1;
}

/**
 * @brief Accumulate `v` into `*p`, **atomically on the device**.
 *
 * The one primitive scatter/push kernels need that a plain `+=` cannot give:
 * on the device many threads accumulate into overlapping outputs, which races.
 * On the host this is a plain `+=`; on the device it is `atomicAdd` (which for
 * `double` needs `sm_60`+). Use via `t.add_at(v, i...)`.
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
 * copyable, kernel-passable), `own::stack` (inline storage, static shape), or
 * `own::heap` (host-only, move-only). The tensor's copy/move semantics are
 * induced by the storage member, not hand-written.
 *
 * @tparam T        Element type.
 * @tparam Extents  `cuda::std::extents<Idx, E...>` (static or dynamic per dim).
 * @tparam Layout   mdspan layout policy (default `layout_right`).
 * @tparam O        Ownership kind (default `own::view`).
 */
template <class T, class Extents, class Layout, own O>
struct tensor : private Layout::template mapping<Extents> {
    using element_type = T;
    using extents_type = Extents;
    using layout_type  = Layout;
    using index_type   = typename Extents::index_type;
    using mapping_type = typename Layout::template mapping<Extents>;
    using view_type       = cs::mdspan<T, Extents, Layout>;
    using const_view_type = cs::mdspan<const T, Extents, Layout>;

    static constexpr own  ownership = O;
    static constexpr bool is_static = (Extents::rank_dynamic() == 0);
    static constexpr cs::size_t buffer_size = storage_size<mapping_type, O == own::stack>::value;
    static_assert(O != own::stack || is_static, "stack tensor needs a fully static shape");

    storage<T, O, buffer_size> store_{};

    /* --- constructors --------------------------------------------- */
    tensor() = default;

    /** @brief View constructor: wrap `p` with the given mapping. */
    template <own OO = O, cs::enable_if_t<OO == own::view, int> = 0>
    _TNY_API tensor(T * p, mapping_type m) : mapping_type(m), store_(p) {}

    /** @brief View constructor from a pointer + extents (contiguous / static-stride layouts). */
    template <own OO = O, cs::enable_if_t<OO == own::view && cs::is_constructible<mapping_type, Extents>::value, int> = 0>
    _TNY_API tensor(T * p, Extents e) : mapping_type(e), store_(p) {}

    /** @brief Owning constructor: allocate storage for `m` (heap/device/host/pinned). */
    template <own OO = O, cs::enable_if_t<own_is_owning(OO), int> = 0>
    _TNY_HOST explicit tensor(mapping_type m)
        : mapping_type(m), store_(static_cast<cs::size_t>(m.required_span_size())) {}

    /** @brief Owning constructor from extents (contiguous / static-stride layouts). */
    template <own OO = O, cs::enable_if_t<own_is_owning(OO) && cs::is_constructible<mapping_type, Extents>::value, int> = 0>
    _TNY_HOST explicit tensor(Extents e)
        : mapping_type(e), store_(static_cast<cs::size_t>(mapping_type(e).required_span_size())) {}

    /* --- geometry ------------------------------------------------- */
    static constexpr cs::size_t rank() noexcept { return Extents::rank(); }
    _TNY_API constexpr const mapping_type & mapping() const noexcept { return *this; }
    _TNY_API constexpr const Extents & extents() const noexcept { return mapping_type::extents(); }
    static constexpr bool has_static_strides   = _static_strides<Layout>::value;
    static constexpr bool is_contiguous_layout = _contiguous_layout<Layout>::value;

    /** @brief Extent of an axis given by a STATIC index (`extent(Int<0>())`):
     *         a compile-time `integral_constant` when that extent is static,
     *         else a runtime `index_type`. */
    template <class Idx, cs::enable_if_t<_is_ic<Idx>::value, int> = 0>
    _TNY_API constexpr auto extent(Idx) const noexcept {
        constexpr cs::size_t D = static_cast<cs::size_t>(Idx::value);
        if constexpr (Extents::static_extent(D) != cs::dynamic_extent)
            return cs::integral_constant<index_type, static_cast<index_type>(Extents::static_extent(D))>{};
        else
            return mapping_type::extents().extent(D);
    }
    /** @brief Extent of an axis given by a RUNTIME index (`extent(0)`). */
    template <class Idx, cs::enable_if_t<!_is_ic<Idx>::value, int> = 0>
    _TNY_API constexpr index_type extent(Idx d) const noexcept
    { return mapping_type::extents().extent(static_cast<cs::size_t>(d)); }

    /** @brief Stride of an axis given by a STATIC index (`stride(Int<0>())`):
     *         a compile-time `integral_constant` when known statically (static-
     *         stride layout, or a contiguous layout over static extents). */
    template <class Idx, cs::enable_if_t<_is_ic<Idx>::value, int> = 0>
    _TNY_API constexpr auto stride(Idx) const noexcept {
        constexpr cs::size_t D = static_cast<cs::size_t>(Idx::value);
        if constexpr (has_static_strides)
            return cs::integral_constant<index_type, static_cast<index_type>(_static_stride_at<D, Layout>::value)>{};
        else if constexpr (is_static && is_contiguous_layout)
            return cs::integral_constant<index_type, static_cast<index_type>(mapping_type{}.stride(D))>{};
        else
            return mapping_type::stride(D);
    }
    /** @brief Stride of an axis given by a RUNTIME index (`stride(0)`). */
    template <class Idx, cs::enable_if_t<!_is_ic<Idx>::value, int> = 0>
    _TNY_API constexpr index_type stride(Idx d) const noexcept
    { return mapping_type::stride(static_cast<cs::size_t>(d)); }
    _TNY_API constexpr index_type numel() const noexcept {
        index_type n = 1;
        for (cs::size_t r = 0; r < rank(); ++r) n *= extent(r);
        return n;
    }

    /* --- data / views -------------------------------------------- */
    _TNY_API T *       data()       noexcept { return store_.data(); }
    _TNY_API const T * data() const noexcept { return store_.data(); }
    _TNY_API view_type       view()       noexcept { return view_type(store_.data(), *this); }
    _TNY_API const_view_type view() const noexcept { return const_view_type(store_.data(), *this); }

    /* --- element access / slicing -------------------------------- */
private:
    // wrap a negative index python-style; fold when the index is static
    template <cs::size_t Ax, class Arg>
    _TNY_API constexpr index_type _wrap(Arg a) const {
        index_type i = static_cast<index_type>(a);
        return i < index_type(0)
             ? static_cast<index_type>(i + extent(cs::integral_constant<cs::size_t, Ax>{}))
             : i;
    }
    template <cs::size_t... Ax, class... Args>
    _TNY_API constexpr index_type _offset(cs::index_sequence<Ax...>, Args... a) const {
        return mapping_type::operator()(_wrap<Ax>(a)...);
    }
    // resolve one slice bound against the axis extent n (none -> default; wrap negatives)
    template <class V> _TNY_API index_type _sl_bound(V v, index_type dflt, index_type n) const {
        if constexpr (cs::is_same<V, none_t>::value) { (void)v; (void)n; return dflt; }
        else { index_type i = static_cast<index_type>(v); return i < index_type(0) ? static_cast<index_type>(i + n) : i; }
    }
    // turn a slice_spec into a submdspan `strided_slice{offset, width, stride}`.
    // `width` is measured in the original index space; submdspan yields
    // ceil(width/stride). NOTE: ranges are always applied to a layout_stride
    // source (see _src_for) because CCCL's submdspan mis-deduces exhaustiveness
    // for a range on layout_right/left and would produce wrong strides.
    template <cs::size_t Ax, class A, class B, class S>
    _TNY_API auto _resolve(slice_spec<A,B,S> s) const {
        const index_type n    = static_cast<index_type>(extent(cs::integral_constant<cs::size_t,Ax>{}));
        const index_type st   = _sl_bound(s.start, index_type(0), n);
        const index_type sp   = _sl_bound(s.stop,  n,             n);
        const index_type step = static_cast<index_type>(s.step);
        return cs::strided_slice<index_type,index_type,index_type>{ st, static_cast<index_type>(sp - st), step };
    }
    // slice specifier for axis Ax: integer -> wrap; `slice(none,none)` -> full_extent
    // (folds, == `all`); any other slice -> strided_slice; `all` -> passed through.
    template <cs::size_t Ax, class Arg>
    _TNY_API auto _spec(Arg a) const {
        if constexpr (_is_index<Arg>::value)          return _wrap<Ax>(a);
        else if constexpr (_is_full_slice<Arg>::value) { (void)a; return cs::full_extent; }
        else if constexpr (_is_slice_spec<Arg>::value) return _resolve<Ax>(a);
        else                                           return a;   // full_extent (all)
    }
    // reinterpret the data as an explicit layout_stride view (strides materialised)
    template <class P, cs::size_t... Ax>
    _TNY_API auto _sv(P p, cs::index_sequence<Ax...>) const {
        cs::layout_stride::mapping<Extents> m(
            extents(), cs::array<index_type, rank()>{ static_cast<index_type>(stride(Ax))... });
        return cs::mdspan<cs::remove_pointer_t<P>, Extents, cs::layout_stride>(p, m);
    }
    _TNY_API auto _stride_view()       noexcept { return _sv(store_.data(), cs::make_index_sequence<rank()>{}); }
    _TNY_API auto _stride_view() const noexcept { return _sv(store_.data(), cs::make_index_sequence<rank()>{}); }
    // integer-drop / full_extent slicing is correct on the native layout and keeps
    // static strides; a range must go through the layout_stride view.
    template <class... Args> _TNY_API auto _src_for()       { if constexpr (_any_range<Args...>::value) return _stride_view(); else return view(); }
    template <class... Args> _TNY_API auto _src_for() const { if constexpr (_any_range<Args...>::value) return _stride_view(); else return view(); }
    template <class V, cs::size_t... Ax, class... Args>
    _TNY_API auto _slice(V v, cs::index_sequence<Ax...>, Args... a) const {
        return as_tensor(cs::submdspan(v, _spec<Ax>(a)...));
    }
public:
    /** @brief Element access when every argument is an integer (negatives wrap). */
    template <class... Args, cs::enable_if_t<(_is_index<Args>::value && ...), int> = 0>
    _TNY_API T & operator()(Args... a) noexcept
    { return store_.data()[_offset(cs::make_index_sequence<rank()>{}, a...)]; }
    template <class... Args, cs::enable_if_t<(_is_index<Args>::value && ...), int> = 0>
    _TNY_API const T & operator()(Args... a) const noexcept
    { return store_.data()[_offset(cs::make_index_sequence<rank()>{}, a...)]; }

    /** @brief Scatter-accumulate: `(*this)(i...) += v`, atomic on the device.
     *         The write half of a "push"/splat kernel. Integer indices only
     *         (negatives wrap, like `operator()`). */
    template <class... Args, cs::enable_if_t<(_is_index<Args>::value && ...), int> = 0>
    _TNY_API void add_at(T v, Args... a) noexcept
    { fetch_add(&store_.data()[_offset(cs::make_index_sequence<rank()>{}, a...)], v); }

    /** @brief Sub-view when any argument is a slice (`all`, `slice(a,b)`). */
    template <class... Args, cs::enable_if_t<!(_is_index<Args>::value && ...), int> = 0>
    _TNY_API auto operator()(Args... a) noexcept
    { return _slice(_src_for<Args...>(), cs::make_index_sequence<rank()>{}, a...); }
    template <class... Args, cs::enable_if_t<!(_is_index<Args>::value && ...), int> = 0>
    _TNY_API auto operator()(Args... a) const noexcept
    { return _slice(_src_for<Args...>(), cs::make_index_sequence<rank()>{}, a...); }

    /* --- structural views (return md::tensor views) --------------- */

private:
    // specifier for output axis A: the matching take_along arg (index -> wrap,
    // slice -> pass through) if A is named, else `all` (keep the axis).
    template <cs::size_t A, cs::size_t... Axes, class Tup>
    _TNY_API auto _ta_spec(const Tup & t) const {
        constexpr int p = _pos_in<A, Axes...>();
        if constexpr (p < 0) return cs::full_extent;
        else                 return _spec<A>(cs::get<static_cast<cs::size_t>(p)>(t));
    }
    template <cs::size_t... Axes, class V, class Tup, cs::size_t... A>
    _TNY_API auto _ta(V v, const Tup & t, cs::index_sequence<A...>) const {
        return as_tensor(cs::submdspan(v, _ta_spec<A, Axes...>(t)...));
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
    template <cs::size_t... Axes, class... Args>
    _TNY_API auto take_along(Args... args) noexcept {
        static_assert(sizeof...(Axes) == sizeof...(Args), "take_along: one index per named axis");
        return _ta<Axes...>(_src_for<Args...>(), cs::make_tuple(args...), cs::make_index_sequence<rank()>{});
    }
    template <cs::size_t... Axes, class... Args>
    _TNY_API auto take_along(Args... args) const noexcept {
        static_assert(sizeof...(Axes) == sizeof...(Args), "take_along: one index per named axis");
        return _ta<Axes...>(_src_for<Args...>(), cs::make_tuple(args...), cs::make_index_sequence<rank()>{});
    }

    /** @brief Reorder the axes (a permutation of 0..N-1) -> a rank-N view. */
    template <cs::size_t... Perm>
    _TNY_API auto permute() noexcept
    { static_assert(sizeof...(Perm) == rank(), "permute: need N axes"); return as_tensor(_detail::perm_md(view(), cs::index_sequence<Perm...>{})); }
    template <cs::size_t... Perm>
    _TNY_API auto permute() const noexcept
    { static_assert(sizeof...(Perm) == rank(), "permute: need N axes"); return as_tensor(_detail::perm_md(view(), cs::index_sequence<Perm...>{})); }

    /** @brief Insert a size-1 axis at position `Ax` (numpy `newaxis`/`unsqueeze`)
     *         -> a rank-(N+1) view. e.g. a `(H,W)` grid -> `(H,W,1)` with
     *         `.unsqueeze<2>()`. */
    template <cs::size_t Ax = 0>
    _TNY_API auto unsqueeze() noexcept
    { static_assert(Ax <= rank(), "unsqueeze: axis out of range"); return as_tensor(_detail::unsqueeze_md<Ax>(view(), cs::make_index_sequence<rank() + 1>{})); }
    template <cs::size_t Ax = 0>
    _TNY_API auto unsqueeze() const noexcept
    { static_assert(Ax <= rank(), "unsqueeze: axis out of range"); return as_tensor(_detail::unsqueeze_md<Ax>(view(), cs::make_index_sequence<rank() + 1>{})); }

    /** @brief Drop axis `Ax` (which must have extent 1) -> a rank-(N-1) view. */
    template <cs::size_t Ax>
    _TNY_API auto squeeze() noexcept
    { static_assert(Ax < rank() && rank() > 0, "squeeze: axis out of range"); return as_tensor(_detail::squeeze_md<Ax>(view(), cs::make_index_sequence<rank() - 1>{})); }
    template <cs::size_t Ax>
    _TNY_API auto squeeze() const noexcept
    { static_assert(Ax < rank() && rank() > 0, "squeeze: axis out of range"); return as_tensor(_detail::squeeze_md<Ax>(view(), cs::make_index_sequence<rank() - 1>{})); }

    /* --- in-place elementwise math (declared here, defined in math.h) --- */
    template <class B> _TNY_API tensor & add_(const B & b);
    template <class B> _TNY_API tensor & sub_(const B & b);
    template <class B> _TNY_API tensor & mul_(const B & b);
    template <class B> _TNY_API tensor & div_(const B & b);
    _TNY_API tensor & add_(T s);
    _TNY_API tensor & sub_(T s);
    _TNY_API tensor & mul_(T s);
    _TNY_API tensor & div_(T s);

    /* --- assignment / fill (broadcasting) ------------------------- */
    template <class B> _TNY_API tensor & copy_(const B & b);   // *this = b (broadcasts)
    _TNY_API tensor & fill_(T s);                              // *this = s
    _TNY_API tensor & zero_();                                 // *this = 0

    /* --- out-of-place elementwise (tensor OR scalar rhs) -> new tensor --- */
    template <class B> _TNY_API auto add(const B & b) const;
    template <class B> _TNY_API auto sub(const B & b) const;
    template <class B> _TNY_API auto mul(const B & b) const;
    template <class B> _TNY_API auto div(const B & b) const;
    template <class B> _TNY_API auto pow(const B & b) const;

    /* --- in-place unary math (element-wise) ----------------------- */
    _TNY_API tensor & neg_();
    _TNY_API tensor & abs_();
    _TNY_API tensor & exp_();
    _TNY_API tensor & log_();
    _TNY_API tensor & sin_();
    _TNY_API tensor & cos_();
    _TNY_API tensor & sqrt_();
    _TNY_API tensor & tanh_();
    _TNY_API tensor & pow_(T e);
};

/* ------------------------------------------------------------------ *
 *     Factories                                                      *
 * ------------------------------------------------------------------ */

/** @brief Non-owning view over `p` with a contiguous layout (default C-order). */
template <class Layout = cs::layout_right, class T, class Extents>
_TNY_API tensor<T, Extents, Layout, own::view> view(T * p, Extents e) {
    using Tn = tensor<T, Extents, Layout, own::view>;
    return Tn(p, typename Tn::mapping_type(e));
}

/** @brief Non-owning view with per-dimension compile-time strides. */
template <cs::size_t... Strides, class T, class Extents>
_TNY_API tensor<T, Extents, layout_static_stride<Strides...>, own::view>
view_strided(T * p, Extents e) {
    using Tn = tensor<T, Extents, layout_static_stride<Strides...>, own::view>;
    return Tn(p, typename Tn::mapping_type(e));
}

/** @brief A non-owning view type. Construct as `view_t<T,E>(ptr, extents)`. */
template <class T, class Extents, class Layout = cs::layout_right>
using view_t = tensor<T, Extents, Layout, own::view>;

/** @brief Stack-owned tensor (fully static shape). Use `local<T,E>{}`. */
template <class T, class Extents, class Layout = cs::layout_right>
using local = tensor<T, Extents, Layout, own::stack>;

/** @brief Heap-owned tensor (host only, move-only). Use `owned<T,E>(extents)`. */
template <class T, class Extents, class Layout = cs::layout_right>
using owned = tensor<T, Extents, Layout, own::heap>;

/** @brief Wrap any `cuda::std::mdspan` (e.g. a `submdspan` result) as a
 *         non-owning `md::tensor` view, so the tensor API applies to it. */
template <class MD>
_TNY_API tensor<typename MD::element_type, typename MD::extents_type,
                typename MD::layout_type, own::view>
as_tensor(const MD & m) {
    using Tn = tensor<typename MD::element_type, typename MD::extents_type,
                      typename MD::layout_type, own::view>;
    return Tn(m.data_handle(), m.mapping());
}

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_TENSOR
