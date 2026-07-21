#ifndef TNY_MD_TENSOR
#define TNY_MD_TENSOR
#include <cuda/std/mdspan>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <teeny/defines.h>
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
// reverse axis AX: keep the extents, negate that axis' stride, and shift the
// data handle to the last element along AX (so index 0 maps to the old last).
template <cs::size_t AX, class MD, cs::size_t... D>
_TNY_API auto flip_md(const MD & v, cs::index_sequence<D...>) {
    using El  = typename MD::element_type;
    using Idx = typename MD::index_type;
    using E   = typename MD::extents_type;
    static_assert(cs::is_signed<Idx>::value, "flip needs a signed index type (e.g. shape<...>)");
    cs::layout_stride::mapping<E> m(
        v.extents(),
        cs::array<Idx, sizeof...(D)>{ static_cast<Idx>(D == AX ? -static_cast<Idx>(v.stride(D)) : static_cast<Idx>(v.stride(D)))... });
    const Idx off = static_cast<Idx>((static_cast<Idx>(v.extent(AX)) - 1) * static_cast<Idx>(v.stride(AX)));
    return cs::mdspan<El, E, cs::layout_stride>(v.data_handle() + off, m);
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

// traits: layout classification for stride folding
template <class L> struct _is_strides : cs::false_type {};       // teeny's strides<S...>
template <cs::int64_t... S> struct _is_strides<strides<S...>> : cs::true_type {};
template <class L> struct _strides_all_static : cs::false_type {};  // and every stride known?
template <cs::int64_t... S> struct _strides_all_static<strides<S...>> : cs::integral_constant<bool, strides<S...>::all_static()> {};
template <class L> struct _contiguous_layout : cs::false_type {};
template <> struct _contiguous_layout<cs::layout_right> : cs::true_type {};
template <> struct _contiguous_layout<cs::layout_left>  : cs::true_type {};

// per-dim static stride from a strides<...> layout (signed; `dynamic_stride` if
// runtime, or for any non-strides layout so callers fall through).
template <cs::size_t D, class L> struct _static_stride_at { static constexpr cs::int64_t value = dynamic_stride; };
template <cs::size_t D, cs::int64_t... S> struct _static_stride_at<D, strides<S...>> { static constexpr cs::int64_t value = strides<S...>::S_[D]; };

// an "index" argument to operator() is a (static or runtime) integer; anything
// else (all / rng, i.e. a slice specifier) turns operator() into a view.
template <class T> struct _is_ic : cs::false_type {};
template <class T, T V> struct _is_ic<cs::integral_constant<T,V>> : cs::true_type {};
template <class A> struct _is_index
    : cs::integral_constant<bool, cs::is_integral<A>::value || _is_ic<A>::value> {};

// normalise a python-style AXIS index (negatives count from the back) against a
// rank. Axis template parameters are signed (`long`) so `-1` is the last axis.
_TNY_API constexpr cs::size_t _norm_axis(long a, cs::size_t rank) noexcept {
    return static_cast<cs::size_t>(a < 0 ? a + static_cast<long>(rank) : a);
}

// ---- compile-time output extents for the manual range slicer --------------
// Each axis maps to: DROP (integer arg, no output axis), its static extent (an
// `all`/full_extent kept axis), or dynamic_extent (a range). `_compact` removes
// the DROPs and builds the resulting extents type (static where derivable).
inline constexpr cs::size_t _drop_axis = cs::size_t(-2);
template <cs::size_t V, class E> struct _ext_prepend;
template <cs::size_t V, class Idx, cs::size_t... E> struct _ext_prepend<V, cs::extents<Idx, E...>> { using type = cs::extents<Idx, V, E...>; };
template <class Idx, cs::size_t... V> struct _compact { using type = cs::extents<Idx>; };
template <class Idx, cs::size_t V0, cs::size_t... Vs>
struct _compact<Idx, V0, Vs...> {
    using rest = typename _compact<Idx, Vs...>::type;
    using type = cs::conditional_t<V0 == _drop_axis, rest, typename _ext_prepend<V0, rest>::type>;
};

/** @brief Open-ended slice sentinel — teeny's `None` (python `a[:n]` / `a[m:]`).
 *
 * `slice(none, n)` starts at 0, `slice(m, none)` runs to the end, and
 * `slice(none, none)` **folds** to `full_extent` — so `all == slice(none, none)`,
 * keeping the axis and its static extent (`all` is built from it). Combined with
 * runtime bounds it resolves at run time, so the one sentinel covers both. */
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

    /** @brief View constructor from a pointer alone — for a fully-static geometry
     *         (static extents AND a fully determined layout: contiguous, or an
     *         all-static `strides<...>`). e.g. `tensor<float, shape<3,4>, strides<4,1>>(ptr)`. */
    template <own OO = O, cs::enable_if_t<OO == own::view && is_static &&
              (_contiguous_layout<Layout>::value || _strides_all_static<Layout>::value), int> = 0>
    _TNY_API tensor(T * p) : mapping_type(), store_(p) {}

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
    static constexpr bool is_strides_layout    = _is_strides<Layout>::value;
    static constexpr bool is_contiguous_layout = _contiguous_layout<Layout>::value;

    /** @brief Extent of an axis given by a STATIC index (`extent(Int<0>())`):
     *         a compile-time `integral_constant` when that extent is static,
     *         else a runtime `index_type`. */
    template <class Idx, cs::enable_if_t<_is_ic<Idx>::value, int> = 0>
    _TNY_API constexpr auto extent(Idx) const noexcept {
        constexpr cs::size_t D = _norm_axis(static_cast<long>(Idx::value), rank());   // -1 = last axis
        if constexpr (Extents::static_extent(D) != cs::dynamic_extent)
            return cs::integral_constant<index_type, static_cast<index_type>(Extents::static_extent(D))>{};
        else
            return mapping_type::extents().extent(D);
    }
    /** @brief Extent of an axis given by a RUNTIME index (`extent(0)`). */
    template <class Idx, cs::enable_if_t<!_is_ic<Idx>::value, int> = 0>
    _TNY_API constexpr index_type extent(Idx d) const noexcept
    { return mapping_type::extents().extent(static_cast<cs::size_t>(d)); }

    /** @brief `shape()` / `shape(d)` — python-friendly aliases of `extents()` /
     *         `extent(d)` (static index -> integral_constant, runtime -> value). */
    _TNY_API constexpr const Extents & shape() const noexcept { return extents(); }
    template <class Idx> _TNY_API constexpr auto shape(Idx d) const noexcept { return extent(d); }

    /** @brief Stride of an axis given by a STATIC index (`stride(Int<0>())`):
     *         a compile-time `integral_constant` when known statically (static-
     *         stride layout; a contiguous layout over static extents; or the
     *         always-unit stride of a contiguous layout even for dynamic shapes). */
    template <class Idx, cs::enable_if_t<_is_ic<Idx>::value, int> = 0>
    _TNY_API constexpr auto stride(Idx) const noexcept {
        constexpr cs::size_t D = _norm_axis(static_cast<long>(Idx::value), rank());   // -1 = last axis
        // a strides<...> layout folds per-dim: static value if known, else runtime
        if constexpr (is_strides_layout && _static_stride_at<D, Layout>::value != dynamic_stride)
            return cs::integral_constant<index_type, static_cast<index_type>(_static_stride_at<D, Layout>::value)>{};
        else if constexpr (is_static && is_contiguous_layout)
            return cs::integral_constant<index_type, static_cast<index_type>(mapping_type{}.stride(D))>{};
        // The unit stride of a contiguous layout is 1 regardless of dynamic
        // extents: layout_right's last axis, layout_left's first axis.
        else if constexpr (cs::is_same<Layout, cs::layout_right>::value && D + 1 == rank())
            return cs::integral_constant<index_type, 1>{};
        else if constexpr (cs::is_same<Layout, cs::layout_left>::value && D == 0)
            return cs::integral_constant<index_type, 1>{};
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
    /** @brief Whether the strides are dense row-major (C-contiguous). */
    _TNY_API constexpr bool is_contiguous() const noexcept {
        index_type expect = 1;
        for (int d = static_cast<int>(rank()) - 1; d >= 0; --d) {
            if (static_cast<index_type>(stride(d)) != expect) return false;
            expect *= static_cast<index_type>(extent(d));
        }
        return true;
    }

    /* --- data / views -------------------------------------------- */
    _TNY_API T *       data()       noexcept { return store_.data(); }
    _TNY_API const T * data() const noexcept { return store_.data(); }
    _TNY_API view_type       view()       noexcept { return view_type(store_.data(), *this); }
    _TNY_API const_view_type view() const noexcept { return const_view_type(store_.data(), *this); }

    /* --- element access / slicing -------------------------------- */
private:
    // wrap a negative index python-style. The wrap is done in a SIGNED domain so
    // that negative indices work even when index_type is unsigned (e.g. a raw
    // extents<size_t,...>): casting `a` to index_type first would turn -1 into a
    // huge value before the `< 0` test could catch it.
    template <cs::size_t Ax, class Arg>
    _TNY_API constexpr index_type _wrap(Arg a) const {
        using S = cs::make_signed_t<index_type>;
        const S i = static_cast<S>(a);
        const S n = static_cast<S>(extent(cs::integral_constant<cs::size_t, Ax>{}));
        return static_cast<index_type>(i < S(0) ? i + n : i);
    }
    template <cs::size_t... Ax, class... Args>
    _TNY_API constexpr index_type _offset(cs::index_sequence<Ax...>, Args... a) const {
        return mapping_type::operator()(_wrap<Ax>(a)...);
    }
    // resolve one slice bound against the axis extent n (none -> default; wrap
    // negatives in a signed domain, so it works even for an unsigned index_type)
    template <class V> _TNY_API index_type _sl_bound(V v, index_type dflt, index_type n) const {
        if constexpr (cs::is_same<V, none_t>::value) { (void)v; (void)n; return dflt; }
        else {
            using S = cs::make_signed_t<index_type>;
            const S i = static_cast<S>(v);
            return static_cast<index_type>(i < S(0) ? i + static_cast<S>(n) : i);
        }
    }
    // slice specifier for a NON-range arg (integer -> wrap; else `all`/full_extent).
    // Range slices don't use submdspan (see _slice_range) so never reach here.
    template <cs::size_t Ax, class Arg>
    _TNY_API auto _spec(Arg a) const {
        if constexpr (_is_index<Arg>::value)           return _wrap<Ax>(a);
        else if constexpr (_is_full_slice<Arg>::value) { (void)a; return cs::full_extent; }
        else                                           return a;   // full_extent (all)
    }
    // native-layout slicing for the non-range case (int drops + full_extent):
    // submdspan is correct here and preserves static extents.
    template <class V, cs::size_t... Ax, class... Args>
    _TNY_API auto _slice(V v, cs::index_sequence<Ax...>, Args... a) const {
        return as_tensor(cs::submdspan(v, _spec<Ax>(a)...));
    }

    // ---- manual layout_stride sub-view builder for the RANGE path -------------
    // CCCL's submdspan mis-deduces exhaustiveness for a range on layout_right/left
    // AND cannot express a negative stride, so ranges are built by hand: per axis,
    // accumulate the base offset and (for kept axes) the output extent + stride.
    // `stop` default for a negative step: `none` -> -1 (go past index 0), python-style.
    template <class V> _TNY_API index_type _stop_neg(V v, index_type n) const {
        if constexpr (cs::is_same<V, none_t>::value) { (void)v; (void)n; return index_type(-1); }
        else { using S = cs::make_signed_t<index_type>; S i = static_cast<S>(v); return static_cast<index_type>(i < S(0) ? i + static_cast<S>(n) : i); }
    }
    template <cs::size_t Ax, class Arg>
    _TNY_API void _sl_axis(Arg a, index_type & off, index_type * ext, index_type * str, cs::size_t & k) const {
        const index_type sd = static_cast<index_type>(stride(Ax));
        const index_type n  = static_cast<index_type>(extent(cs::integral_constant<cs::size_t, Ax>{}));
        if constexpr (_is_index<Arg>::value) {
            off += _wrap<Ax>(a) * sd;                                // integer: drop this axis
        } else if constexpr (_is_slice_spec<Arg>::value) {
            const index_type step = static_cast<index_type>(a.step);
            index_type st, cnt;
            if (step >= index_type(0)) {
                st = _sl_bound(a.start, index_type(0), n);
                const index_type sp = _sl_bound(a.stop, n, n);
                const index_type w = sp - st; cnt = w <= index_type(0) ? index_type(0) : (w + step - 1) / step;
            } else {
                const index_type ns = -step;
                st = _sl_bound(a.start, n - 1, n);                   // default start = last
                const index_type sp = _stop_neg(a.stop, n);         // default stop = before-0
                const index_type w = st - sp; cnt = w <= index_type(0) ? index_type(0) : (w + ns - 1) / ns;
            }
            off += st * sd; ext[k] = cnt; str[k] = step * sd; ++k;  // stride may be negative
        } else {                                                    // full_extent (all)
            ext[k] = n; str[k] = sd; ++k;
        }
    }
    // static output extent for one axis: DROP (integer), the input static extent
    // (an `all`/full_extent kept axis), or dynamic (a range).
    template <class Arg> static constexpr cs::size_t _out_static(cs::size_t se) {
        if constexpr (_is_index<Arg>::value)                       return _drop_axis;
        else if constexpr (cs::is_same<Arg, cs::full_extent_t>::value) return se;
        else                                                       return cs::dynamic_extent;
    }
    template <class P, cs::size_t... Ax, class... Args>
    _TNY_API auto _slice_range(P p, cs::index_sequence<Ax...>, Args... a) const {
        constexpr cs::size_t Nk = (cs::size_t(0) + ... + (_is_index<Args>::value ? cs::size_t(0) : cs::size_t(1)));
        // output extents: static for `all`-kept static axes, dynamic for ranges
        using OE = typename _compact<index_type, _out_static<Args>(Extents::static_extent(Ax))...>::type;
        index_type ext[Nk ? Nk : 1] = {}, str[Nk ? Nk : 1] = {}, off = 0; cs::size_t k = 0;
        ( _sl_axis<Ax>(a, off, ext, str, k), ... );
        cs::array<index_type, Nk> ea{}, sa{};
        for (cs::size_t i = 0; i < Nk; ++i) { ea[i] = ext[i]; sa[i] = str[i]; }
        cs::layout_stride::mapping<OE> m(OE(ea), sa);
        return as_tensor(cs::mdspan<cs::remove_pointer_t<P>, OE, cs::layout_stride>(p + off, m));
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

    /** @brief Sub-view when any argument is a slice (`all`, `slice(a,b[,step])`).
     *         A real range (with an optional negative step) builds a layout_stride
     *         view by hand; pure integer/`all` slicing stays on the native layout. */
    template <class... Args, cs::enable_if_t<!(_is_index<Args>::value && ...), int> = 0>
    _TNY_API auto operator()(Args... a) noexcept {
        if constexpr (_any_range<Args...>::value) return _slice_range(store_.data(), cs::make_index_sequence<rank()>{}, a...);
        else                                      return _slice(view(), cs::make_index_sequence<rank()>{}, a...);
    }
    template <class... Args, cs::enable_if_t<!(_is_index<Args>::value && ...), int> = 0>
    _TNY_API auto operator()(Args... a) const noexcept {
        if constexpr (_any_range<Args...>::value) return _slice_range(store_.data(), cs::make_index_sequence<rank()>{}, a...);
        else                                      return _slice(view(), cs::make_index_sequence<rank()>{}, a...);
    }

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
    // raw per-axis arg for the manual range path: the named arg, else full_extent.
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
        if constexpr (_any_range<Args...>::value) return _ta_range<_norm_axis(Axes, rank())...>(store_.data(), cs::make_tuple(args...), cs::make_index_sequence<rank()>{});
        else                                      return _ta<_norm_axis(Axes, rank())...>(view(), cs::make_tuple(args...), cs::make_index_sequence<rank()>{});
    }
    template <long... Axes, class... Args>
    _TNY_API auto take_along(Args... args) const noexcept {
        static_assert(sizeof...(Axes) == sizeof...(Args), "take_along: one index per named axis");
        if constexpr (_any_range<Args...>::value) return _ta_range<_norm_axis(Axes, rank())...>(store_.data(), cs::make_tuple(args...), cs::make_index_sequence<rank()>{});
        else                                      return _ta<_norm_axis(Axes, rank())...>(view(), cs::make_tuple(args...), cs::make_index_sequence<rank()>{});
    }

    /** @brief Reorder the axes (a permutation of 0..N-1; negatives wrap) -> a rank-N view. */
    template <long... Perm>
    _TNY_API auto permute() noexcept
    { static_assert(sizeof...(Perm) == rank(), "permute: need N axes"); return as_tensor(_detail::perm_md(view(), cs::index_sequence<_norm_axis(Perm, rank())...>{})); }
    template <long... Perm>
    _TNY_API auto permute() const noexcept
    { static_assert(sizeof...(Perm) == rank(), "permute: need N axes"); return as_tensor(_detail::perm_md(view(), cs::index_sequence<_norm_axis(Perm, rank())...>{})); }

    /** @brief Reverse axis `Ax` (negatives wrap) -> a view (numpy `flip`). Uses a
     *         negative stride, so the index type must be signed (`shape<...>` is). */
    template <long Ax = 0>
    _TNY_API auto flip() noexcept
    { return as_tensor(_detail::flip_md<_norm_axis(Ax, rank())>(view(), cs::make_index_sequence<rank()>{})); }
    template <long Ax = 0>
    _TNY_API auto flip() const noexcept
    { return as_tensor(_detail::flip_md<_norm_axis(Ax, rank())>(view(), cs::make_index_sequence<rank()>{})); }

    /** @brief A dense, row-major OWNING copy of this tensor (materialise a view /
     *         non-contiguous / permuted / flipped tensor). Static shape -> stack
     *         (host+device); dynamic -> heap (host only). */
    template <bool S = is_static, cs::enable_if_t<S, int> = 0>
    _TNY_API auto clone() const { tensor<T, Extents, cs::layout_right, own::stack> c{}; c.copy_(*this); return c; }
    template <bool S = is_static, cs::enable_if_t<!S, int> = 0>
    _TNY_HOST auto clone() const { tensor<T, Extents, cs::layout_right, own::heap> c(extents()); c.copy_(*this); return c; }

    /** @brief View this tensor as a new static shape — requires it be C-contiguous
     *         (`clone()` first otherwise) and the element count to match. */
    template <cs::size_t... NewExt>
    _TNY_API auto reshape() noexcept {
        using NE = cs::extents<index_type, NewExt...>;
        _TNY_CHECK(is_contiguous(), "reshape: needs a C-contiguous tensor (clone() first)");
        _TNY_CHECK(numel() == static_cast<index_type>((index_type(1) * ... * index_type(NewExt))), "reshape: numel mismatch");
        return tensor<T, NE, cs::layout_right, own::view>(store_.data(), typename cs::layout_right::template mapping<NE>(NE{}));
    }
    template <cs::size_t... NewExt>
    _TNY_API auto reshape() const noexcept {
        using NE = cs::extents<index_type, NewExt...>;
        _TNY_CHECK(is_contiguous(), "reshape: needs a C-contiguous tensor (clone() first)");
        return tensor<const T, NE, cs::layout_right, own::view>(store_.data(), typename cs::layout_right::template mapping<NE>(NE{}));
    }

    /** @brief View as 1-D (`ravel`) — requires C-contiguous (`clone()` first). */
    _TNY_API auto flatten() noexcept {
        using NE = cs::dextents<index_type, 1>;
        _TNY_CHECK(is_contiguous(), "flatten: needs a C-contiguous tensor (clone() first)");
        return tensor<T, NE, cs::layout_right, own::view>(store_.data(), typename cs::layout_right::template mapping<NE>(NE{ numel() }));
    }
    _TNY_API auto flatten() const noexcept {
        using NE = cs::dextents<index_type, 1>;
        _TNY_CHECK(is_contiguous(), "flatten: needs a C-contiguous tensor (clone() first)");
        return tensor<const T, NE, cs::layout_right, own::view>(store_.data(), typename cs::layout_right::template mapping<NE>(NE{ numel() }));
    }

    /** @brief Insert a size-1 axis at position `Ax` (numpy `newaxis`/`unsqueeze`)
     *         -> a rank-(N+1) view. Negative `Ax` counts from the back, so
     *         `.unsqueeze<-1>()` appends a trailing axis: `(H,W)` -> `(H,W,1)`. */
    template <long Ax = 0>
    _TNY_API auto unsqueeze() noexcept
    { constexpr cs::size_t A = _norm_axis(Ax, rank() + 1); static_assert(A <= rank(), "unsqueeze: axis out of range"); return as_tensor(_detail::unsqueeze_md<A>(view(), cs::make_index_sequence<rank() + 1>{})); }
    template <long Ax = 0>
    _TNY_API auto unsqueeze() const noexcept
    { constexpr cs::size_t A = _norm_axis(Ax, rank() + 1); static_assert(A <= rank(), "unsqueeze: axis out of range"); return as_tensor(_detail::unsqueeze_md<A>(view(), cs::make_index_sequence<rank() + 1>{})); }

    /** @brief Drop axis `Ax` (extent 1; negatives wrap) -> a rank-(N-1) view. */
    template <long Ax>
    _TNY_API auto squeeze() noexcept
    { constexpr cs::size_t A = _norm_axis(Ax, rank()); static_assert(A < rank() && rank() > 0, "squeeze: axis out of range"); return as_tensor(_detail::squeeze_md<A>(view(), cs::make_index_sequence<rank() - 1>{})); }
    template <long Ax>
    _TNY_API auto squeeze() const noexcept
    { constexpr cs::size_t A = _norm_axis(Ax, rank()); static_assert(A < rank() && rank() > 0, "squeeze: axis out of range"); return as_tensor(_detail::squeeze_md<A>(view(), cs::make_index_sequence<rank() - 1>{})); }

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

/** @brief Non-owning view with per-dimension compile-time strides (may be negative). */
template <cs::int64_t... Strides, class T, class Extents>
_TNY_API tensor<T, Extents, strides<Strides...>, own::view>
view_strided(T * p, Extents e) {
    using Tn = tensor<T, Extents, strides<Strides...>, own::view>;
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

/* --- functional factories (deduce the Extents type from the argument) ------ *
 * Complements the type aliases above; the `make_` prefix keeps them distinct.
 * Element type `T` is explicit (it can't be deduced from a shape); the extents
 * type is deduced, so a runtime-built shape needs no `decltype` spelling.       */

/** @brief `make_view<L>(ptr, extents)` — a non-owning view (alias of `view`). */
template <class Layout = cs::layout_right, class T, class Extents>
_TNY_API auto make_view(T * p, Extents e) { return view<Layout>(p, e); }

/** @brief `make_local<T>(extents)` — a stack-owned tensor (static shape). */
template <class T, class Layout = cs::layout_right, class Extents>
_TNY_API auto make_local(Extents = Extents{}) { return tensor<T, Extents, Layout, own::stack>{}; }

/** @brief `make_heap<T>(extents)` — a heap-owned tensor (host, move-only). */
template <class T, class Layout = cs::layout_right, class Extents>
_TNY_HOST auto make_heap(Extents e) { return tensor<T, Extents, Layout, own::heap>(e); }

/* --- numpy-style creation factories: static shape -> stack (host+device),   *
 *     dynamic shape -> heap (host only), mirroring the out-of-place ops.       */

/** @brief `full<T>(extents, v)` — a new tensor filled with `v`. */
template <class T, class Layout = cs::layout_right, class Extents, cs::enable_if_t<Extents::rank_dynamic() == 0, int> = 0>
_TNY_API auto full(Extents, T v) { tensor<T, Extents, Layout, own::stack> t{}; t.fill_(v); return t; }
template <class T, class Layout = cs::layout_right, class Extents, cs::enable_if_t<Extents::rank_dynamic() != 0, int> = 0>
_TNY_HOST auto full(Extents e, T v) { tensor<T, Extents, Layout, own::heap> t(e); t.fill_(v); return t; }

/** @brief `zeros<T>(extents)` / `ones<T>(extents)` — a new tensor of 0s / 1s. */
template <class T, class Layout = cs::layout_right, class Extents>
_TNY_API auto zeros(Extents e) { return full<T, Layout>(e, T(0)); }
template <class T, class Layout = cs::layout_right, class Extents>
_TNY_API auto ones(Extents e) { return full<T, Layout>(e, T(1)); }

/** @brief `arange<T>(n)` — a 1-D tensor `[0, 1, ..., n-1]` (heap, host). */
template <class T>
_TNY_HOST auto arange(long n) {
    using E = cs::dextents<cs::int64_t, 1>;
    tensor<T, E, cs::layout_right, own::heap> t(E{n}); t.iota_(); return t;
}

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
