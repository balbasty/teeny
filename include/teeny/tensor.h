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

/** @brief A half-open slice `[start, stop)` for `operator()` / `take_along`. */
template <class A, class B>
_TNY_API auto rng(A start, B stop) { return cs::tuple<long,long>{ static_cast<long>(start), static_cast<long>(stop) }; }

// position of axis A within the pack Axes... (-1 if absent)
template <cs::size_t A, cs::size_t... Axes>
_TNY_API constexpr int _pos_in() {
    cs::size_t axs[] = { Axes..., static_cast<cs::size_t>(-1) };
    for (cs::size_t p = 0; p < sizeof...(Axes); ++p) if (axs[p] == A) return static_cast<int>(p);
    return -1;
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
    // slice specifier for axis Ax: wrap an integer, pass a slice through
    template <cs::size_t Ax, class Arg>
    _TNY_API auto _spec(Arg a) const {
        if constexpr (_is_index<Arg>::value) return _wrap<Ax>(a);
        else                                 return a;
    }
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

    /** @brief Sub-view when any argument is a slice (`all`, `rng(a,b)`). */
    template <class... Args, cs::enable_if_t<!(_is_index<Args>::value && ...), int> = 0>
    _TNY_API auto operator()(Args... a) noexcept
    { return _slice(view(), cs::make_index_sequence<rank()>{}, a...); }
    template <class... Args, cs::enable_if_t<!(_is_index<Args>::value && ...), int> = 0>
    _TNY_API auto operator()(Args... a) const noexcept
    { return _slice(view(), cs::make_index_sequence<rank()>{}, a...); }

    /* --- structural views (return md::tensor views) --------------- */

private:
    // specifier for output axis A: the matching take_along arg (index -> wrap,
    // slice -> pass through) if A is named, else `all` (keep the axis).
    template <cs::size_t A, cs::size_t... Axes, class Tup>
    _TNY_API auto _ta_spec(const Tup & t) const {
        constexpr int p = _pos_in<A, Axes...>();
        if constexpr (p < 0) return cs::full_extent;
        else {
            auto arg = cs::get<static_cast<cs::size_t>(p)>(t);
            if constexpr (_is_index<decltype(arg)>::value) return _wrap<A>(arg);
            else                                           return arg;
        }
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
        return _ta<Axes...>(view(), cs::make_tuple(args...), cs::make_index_sequence<rank()>{});
    }
    template <cs::size_t... Axes, class... Args>
    _TNY_API auto take_along(Args... args) const noexcept {
        static_assert(sizeof...(Axes) == sizeof...(Args), "take_along: one index per named axis");
        return _ta<Axes...>(view(), cs::make_tuple(args...), cs::make_index_sequence<rank()>{});
    }

    /** @brief Reorder the axes (a permutation of 0..N-1) -> a rank-N view. */
    template <cs::size_t... Perm>
    _TNY_API auto permute() noexcept
    { static_assert(sizeof...(Perm) == rank(), "permute: need N axes"); return as_tensor(_detail::perm_md(view(), cs::index_sequence<Perm...>{})); }
    template <cs::size_t... Perm>
    _TNY_API auto permute() const noexcept
    { static_assert(sizeof...(Perm) == rank(), "permute: need N axes"); return as_tensor(_detail::perm_md(view(), cs::index_sequence<Perm...>{})); }

    /* --- in-place elementwise math (declared here, defined in math.h) --- */
    template <class B> _TNY_API tensor & add_(const B & b);
    template <class B> _TNY_API tensor & sub_(const B & b);
    template <class B> _TNY_API tensor & mul_(const B & b);
    template <class B> _TNY_API tensor & div_(const B & b);
    _TNY_API tensor & add_(T s);
    _TNY_API tensor & sub_(T s);
    _TNY_API tensor & mul_(T s);
    _TNY_API tensor & div_(T s);

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
