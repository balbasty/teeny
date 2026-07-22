#ifndef TNY_MD_TENSOR
#define TNY_MD_TENSOR
#include <cuda/std/mdspan>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <teeny/defines.h>
#include <teeny/storage.h>
#include <teeny/layout.h>
#include <teeny/indexing.h>
#include <teeny/axis.h>

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
    // wrap a negative index python-style for axis Ax (see free `_wrap_idx`).
    template <cs::size_t Ax, class Arg>
    _TNY_API constexpr index_type _wrap(Arg a) const {
        return _wrap_idx<index_type>(a, static_cast<index_type>(extent(cs::integral_constant<cs::size_t, Ax>{})), index_type(0));
    }
    template <cs::size_t... Ax, class... Args>
    _TNY_API constexpr index_type _offset(cs::index_sequence<Ax...>, Args... a) const {
        return mapping_type::operator()(_wrap<Ax>(a)...);
    }
    // resolve one slice bound against the axis extent n (none -> default).
    template <class V> _TNY_API index_type _sl_bound(V v, index_type dflt, index_type n) const {
        return _wrap_idx<index_type>(v, n, dflt);
    }
    // ---- the ONE sub-view builder (gather) ------------------------------------
    // Every slicing/take_along call routes here: per axis an integer DROPS the
    // axis (into the base offset), `all` KEEPS it, a range keeps a strided window
    // (optional negative step). Output is teeny's strides<...> layout, folding
    // each kept stride to a compile-time value where derivable — so it works on
    // ANY source layout (no submdspan) AND static shapes stay folded.
    // `stop` default for a negative step: `none` -> -1 (go past index 0), python-style.
    template <class V> _TNY_API index_type _stop_neg(V v, index_type n) const {
        return _wrap_idx<index_type>(v, n, index_type(-1));
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
    // (an `all`/full_extent OR a folded `slice(none,none)` kept axis), or dynamic.
    template <class Arg> static constexpr cs::size_t _out_static(cs::size_t se) {
        if constexpr (_is_index<Arg>::value)                            return _drop_axis;
        else if constexpr (cs::is_same<Arg, cs::full_extent_t>::value)  return se;
        else if constexpr (_is_full_slice<Arg>::value)                  return se;
        else                                                            return cs::dynamic_extent;
    }
    template <class P, cs::size_t... Ax, class... Args>
    _TNY_API auto _slice_range(P p, cs::index_sequence<Ax...>, Args... a) const {
        using Vt = cs::remove_pointer_t<P>;
        constexpr cs::size_t Nk = (cs::size_t(0) + ... + (_is_index<Args>::value ? cs::size_t(0) : cs::size_t(1)));
        // output extents (static where a kept axis is static) and output strides
        // (static where source-stride × step is known) — folded into strides<...>.
        using OE = typename _compact<index_type, _out_static<Args>(Extents::static_extent(Ax))...>::type;
        using SF = typename _str_compact<_out_sstride<Args, Ax, Layout, Extents>()...>::type;
        using Map = typename SF::template mapping<OE>;
        index_type ext[Nk ? Nk : 1] = {}, str[Nk ? Nk : 1] = {}, off = 0; cs::size_t k = 0;
        ( _sl_axis<Ax>(a, off, ext, str, k), ... );
        cs::array<index_type, Nk> ea{};
        for (cs::size_t i = 0; i < Nk; ++i) ea[i] = ext[i];
        if constexpr (SF::ndyn() == 0) {   // every kept stride folded -> EBO mapping
            return tensor<Vt, OE, SF, own::view>(p + off, Map(OE(ea)));
        } else {                           // supply the runtime strides for the dynamic slots
            cs::array<index_type, SF::ndyn()> dyn{};
            for (cs::size_t i = 0; i < Nk; ++i) if (SF::S_[i] == dynamic_stride) dyn[SF::slot(i)] = str[i];
            return tensor<Vt, OE, SF, own::view>(p + off, Map(OE(ea), dyn));
        }
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
        return tensor<T, E0, cs::layout_right, own::view>(&store_.data()[_offset(cs::make_index_sequence<rank()>{}, a...)]);
    }
    template <class... Args, cs::enable_if_t<(_is_index<Args>::value && ...), int> = 0>
    _TNY_API auto at(Args... a) const noexcept {
        using E0 = cs::extents<index_type>;
        return tensor<const T, E0, cs::layout_right, own::view>(&store_.data()[_offset(cs::make_index_sequence<rank()>{}, a...)]);
    }

    /** @brief Scatter-accumulate: `(*this)(i...) += v`, atomic on the device —
     *         the write half of a "push"/splat kernel. Shorthand for
     *         `at(i...).add_<true>(v)` (integer indices only; negatives wrap). */
    template <class... Args, cs::enable_if_t<(_is_index<Args>::value && ...), int> = 0>
    _TNY_API void add_at(T v, Args... a) noexcept
    { at(a...).template add_<true>(v); }

    /** @brief Sub-view when any argument is a slice (`all`, `slice(a,b[,step])`).
     *         Integer args drop their axis, `all` keeps it, a range keeps a strided
     *         window — all via the one gather (folds static strides into
     *         `strides<...>`; works on any source layout). */
    template <class... Args, cs::enable_if_t<!(_is_index<Args>::value && ...), int> = 0>
    _TNY_API auto operator()(Args... a) noexcept
    { return _slice_range(store_.data(), cs::make_index_sequence<rank()>{}, a...); }
    template <class... Args, cs::enable_if_t<!(_is_index<Args>::value && ...), int> = 0>
    _TNY_API auto operator()(Args... a) const noexcept
    { return _slice_range(store_.data(), cs::make_index_sequence<rank()>{}, a...); }

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

    /* --- structural views (return md::tensor views) --------------- */

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
        return _ta_range<_norm_axis(Axes, rank())...>(store_.data(), cs::make_tuple(args...), cs::make_index_sequence<rank()>{});
    }
    template <long... Axes, class... Args>
    _TNY_API auto take_along(Args... args) const noexcept {
        static_assert(sizeof...(Axes) == sizeof...(Args), "take_along: one index per named axis");
        return _ta_range<_norm_axis(Axes, rank())...>(store_.data(), cs::make_tuple(args...), cs::make_index_sequence<rank()>{});
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

private:
    // shared reshape body: one axis may be `-1` (numpy-style, inferred from numel).
    template <class El, long... NewExt>
    _TNY_API auto _reshape(El * p) const noexcept {
        static_assert(((NewExt < 0 ? 1 : 0) + ... + 0) <= 1, "reshape: at most one inferred (-1) dimension");
        using NE = cs::extents<index_type, (NewExt < 0 ? cs::dynamic_extent : static_cast<cs::size_t>(NewExt))...>;
        constexpr index_type known = (index_type(1) * ... * (NewExt < 0 ? index_type(1) : index_type(NewExt)));
        _TNY_CHECK(is_contiguous(), "reshape: needs a C-contiguous tensor (clone() first)");
        _TNY_CHECK(known != 0 && numel() % known == 0, "reshape: numel not divisible by given extents");
        const index_type inferred = numel() / (known ? known : index_type(1));
        cs::array<index_type, sizeof...(NewExt)> ea{ (NewExt < 0 ? inferred : index_type(NewExt))... };
        return tensor<El, NE, cs::layout_right, own::view>(p, typename cs::layout_right::template mapping<NE>(NE(ea)));
    }
public:
    /** @brief View this tensor as a new shape — requires it be C-contiguous
     *         (`clone()` first otherwise) and the element count to match. One
     *         extent may be **`-1`** (numpy-style), inferred from the total size:
     *         `t.reshape<6,-1>()`. */
    template <long... NewExt> _TNY_API auto reshape() noexcept       { return _reshape<T, NewExt...>(store_.data()); }
    template <long... NewExt> _TNY_API auto reshape() const noexcept { return _reshape<const T, NewExt...>(store_.data()); }

private:
    template <class El, class NewE, cs::size_t... D>
    _TNY_API auto _recast(El * p, cs::index_sequence<D...>) const {
        static_assert(NewE::rank() == rank(), "recast: rank must match");
        return tensor<El, NewE, cs::layout_right, own::view>(
            p, typename cs::layout_right::template mapping<NewE>(NewE(cs::array<index_type, rank()>{ static_cast<index_type>(extent(D))... })));
    }
public:
    /** @brief Reinterpret with a MORE-STATIC extents type of the same rank —
     *         recover statically-known inner dims at the dynamic (ndarray)
     *         boundary: a runtime `(n,3,3)` view -> `.recast<shape<-1,3,3>>()` so
     *         the `3`s fold. Static dims of `NewE` are validated against the
     *         actual extents; requires a C-contiguous tensor. */
    template <class NewE> _TNY_API auto recast()       { return _recast<T,       NewE>(store_.data(), cs::make_index_sequence<rank()>{}); }
    template <class NewE> _TNY_API auto recast() const { return _recast<const T, NewE>(store_.data(), cs::make_index_sequence<rank()>{}); }

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

private:
    static constexpr long _ax_all = cs::numeric_limits<long>::min();   // squeeze() sentinel: "all singletons"
    // gather arg for axis D when squeezing all singletons: drop a STATIC size-1
    // axis (index 0), keep the rest. (A dynamic axis that is 1 only at run time
    // can't be dropped — the rank must stay static.)
    template <cs::size_t D> static _TNY_API auto _sq_arg() noexcept {
        if constexpr (Extents::static_extent(D) == 1) return cs::integral_constant<index_type, 0>{};
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
               return as_tensor(_detail::squeeze_md<A>(view(), cs::make_index_sequence<rank() - 1>{})); }
    }
    template <long Ax = _ax_all>
    _TNY_API auto squeeze() const noexcept {
        if constexpr (Ax == _ax_all) return _squeeze_all(store_.data(), cs::make_index_sequence<rank()>{});
        else { constexpr cs::size_t A = _norm_axis(Ax, rank()); static_assert(A < rank() && rank() > 0, "squeeze: axis out of range");
               return as_tensor(_detail::squeeze_md<A>(view(), cs::make_index_sequence<rank() - 1>{})); }
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
    _TNY_API tensor<T, Extents, cs::layout_right, own::stack> operator++(int) { auto old = clone(); add_(T(1)); return old; }
    template <bool S = is_static, cs::enable_if_t<S, int> = 0>
    _TNY_API tensor<T, Extents, cs::layout_right, own::stack> operator--(int) { auto old = clone(); sub_(T(1)); return old; }
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
