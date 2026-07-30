#ifndef TNY_TENSOR_H
#define TNY_TENSOR_H
#include <cuda/std/mdspan>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/atomic>
#include <teeny/defines.h>
#include <teeny/alias.h>
#include <teeny/kwargs.h>
#include <teeny/storage.h>
#include <teeny/layout.h>
#include <teeny/indexing.h>
#include <teeny/axis.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

// Forward declarations so the tensor's structural members can name as_tensor
// (its argument is a cuda::std::mdspan, so ADL would not find it).
template <class T, class Shape, class Layout = ccontiguous, storage O = storage::view>
struct tensor;

// Output-destination tag for the out-of-place math producers (math.h). `f(args,
// into(y))` writes the result INTO the caller's `y` — one fused pass, no
// allocation — and returns `y&`, instead of allocating a fresh result. A DISTINCT
// type, so a destination can never be confused with a scalar argument (e.g. the
// fused `alpha` of `add(b, alpha)`). Declared here so the tensor's out-of-place
// members can name `into_t<D>`; the `into()` factory is defined after the class.
template <class D> struct into_t { D & dest; };
namespace _kw { template <class D> struct is_keyword<into_t<D>> : cs::true_type {}; }
/** @brief `_is_into_tag<X>::value` is true iff `X` is an `into_t<D>` instantiation —
 *  lets a generic trailing keyword bag (`_kw`) find the destination tag among
 *  `dtype<...>`/`axis<...>`/`keepdims_t` siblings. */
template <class> struct _is_into_tag : cs::false_type {};
template <class D> struct _is_into_tag<into_t<D>> : cs::true_type {};

template <storage OW = storage::view, class MD>
_TNY_API tensor<typename MD::element_type, typename MD::extents_type,
                typename MD::layout_type, OW>
as_tensor(const MD & m);

// Whether a shape is fully static. A free helper rather than inline in tensor's
// body: MSVC's two-phase lookup can mis-resolve an unqualified `Shape::rank_dynamic()`
// inside `tensor`, whose private base `Layout::mapping<Shape>` (layout.h) itself
// privately inherits from `Shape` for EBO — a private-inheritance/dependent-base
// disambiguation quirk (clang/gcc resolve it fine). A namespace-scope function
// template has no enclosing class, so there is no base-class scope to conflict with.
template <class Shape>
_TNY_API constexpr bool _is_static_shape() { return Shape::rank_dynamic() == 0; }

// `_shape_rank<E>()`/`_shape_static_extent<E>(d)` (the same MSVC quirk as
// `_is_static_shape` above, for `E::rank()`/`E::static_extent(d)`) live in
// layout.h instead of here -- it needs them too (`strides<...>::mapping`'s own
// body), and layout.h is included before this point, so tensor.h just reuses
// that one definition (#315: also fixes `_recast`'s `NewE::rank()`, an
// UNRELATED type parameter that trips the same misattribution).

namespace _md {
/* --- extents of an AXIS reduction: the input extents with `Axes...` dropped
 *     (static where the input is). Lives here rather than next to the reduction
 *     engines in math.h because the reduction METHOD declarations below must
 *     SFINAE on it — a fully static result is stack-owned (host+device,
 *     _TNY_API), a dynamic one is heap-owned (host-only, _TNY_HOST) — the same
 *     split the free reductions in math.h use. -------------------------------- */
// static output extent for input axis D when reducing Axes... (drop if reduced).
// A VALUE-yielding class template (`::value`), not a function template called
// in place -- MSVC's trailing-return-type parser doesn't reliably fold a
// function CALL used as a non-type template argument pack element (it treats
// the callee itself as the argument rather than the call's result), so
// `red_ext<D,E,Axes...>()...` inside `_compact<...>`'s argument list silently
// fails to specialize on real MSVC (part of #296's investigation). A class
// template's `::value` is an ordinary non-type template argument, which every
// compiler folds the same way. (A SIBLING defect, not this one: the "type
// pack + deduced pack" trap documented in CLAUDE.md, #334 -- both converge on
// the same class-template-::value fix shape, which is why they're easy to
// conflate.)
template <cs::size_t D, class E, long... Axes>
struct _red_ext_v {
    static constexpr cs::size_t value =
        _pos_in<D, _norm_axis(Axes, E::rank())...>() >= 0 ? _drop_axis : E::static_extent(D);
};
template <class E, long... Axes, cs::size_t... D>
auto reduced_ext_(cs::index_sequence<D...>)
    -> typename _compact<typename E::index_type, _red_ext_v<D, E, Axes...>::value...>::type;
template <class E, long... Axes>
using reduced_extents = decltype(reduced_ext_<E, Axes...>(cs::make_index_sequence<E::rank()>{}));

/* --- index_select's output extents (#326): the input extents with axis `Axis`'s
 * static extent REPLACED by `NewExt` (the gather index tensor's own static
 * extent(0) — a numeric value when the index tensor has a static shape, else
 * `dynamic_extent`), every other axis unchanged. Same class-template `::value`
 * shape as `_red_ext_v` above (not a function call in place), for the same
 * MSVC non-type-template-argument-pack quirk noted there. ---------------- */
template <cs::size_t D, class E, cs::size_t Axis, cs::size_t NewExt>
struct _repl_ext_v {
    static constexpr cs::size_t value = (D == Axis) ? NewExt : E::static_extent(D);
};
template <class E, cs::size_t Axis, cs::size_t NewExt, cs::size_t... D>
auto index_select_ext_(cs::index_sequence<D...>)
    -> cs::extents<typename E::index_type, _repl_ext_v<D, E, Axis, NewExt>::value...>;
template <class E, cs::size_t Axis, cs::size_t NewExt>
using index_select_extents = decltype(index_select_ext_<E, Axis, NewExt>(cs::make_index_sequence<E::rank()>{}));

/* --- shared by the generic "trailing keyword bag" reduction entry points
 * (math.h): whether reducing over the axes named by an `axis<...>` TAG (rather
 * than an explicit `Axes...` template pack) would leave a dynamic result — the
 * same static(stack,_TNY_API)/dynamic(heap,_TNY_HOST) split every axis
 * reduction needs, computed from the TAG so the tag-driven entry point can
 * SFINAE on it exactly like the explicit-Axes one does. `axis<>` (no axes
 * given -- the bare, all-axes reduction) is never dynamic (a full reduction is
 * always a scalar, never allocates) -- that is the primary template below;
 * the partial specialization below handles a real (non-empty) axis list. */
// Same MSVC two-phase-lookup quirk `_is_static_shape`/`_shape_rank` (above, in
// the enclosing `tny` scope) work around for `tensor`'s own body: MSVC can
// mis-resolve `reduced_extents<...>::rank_dynamic()` when it's evaluated
// directly in a class TEMPLATE's static-member initializer rather than inside
// an ordinary function body. Route it through a plain function template
// (only instantiated when actually called, unlike a static member initializer)
// so `_red_dyn`'s partial specialization below doesn't retrigger it.
template <class E, long... Axes>
_TNY_API constexpr cs::size_t _red_dyn_value() { return reduced_extents<E, Axes...>::rank_dynamic(); }

template <class E, class AxisTag> struct _red_dyn { static constexpr cs::size_t value = 0; };
template <class E, long A0, long... Rest> struct _red_dyn<E, axis<A0, Rest...>> {
    static constexpr auto value = _red_dyn_value<E, A0, Rest...>();
};
}  // namespace _md

/* --- shared axis dispatch: a STATIC index (`Int<k>()`) folds to an
 *     `integral_constant` where the value is known at compile time; a runtime index
 *     stays a value. Lives ONCE here and is reused by `tensor::extent`/`stride` AND
 *     the `shape()`/`strides()` accessor views below, so the fold rule can't drift. */
template <class Shape, class Layout, long Ax, class Map>
_TNY_API constexpr auto _axis_extent(const Map & m) noexcept {
    using Idx = typename Shape::index_type;
    constexpr cs::size_t D = _norm_axis(Ax, _shape_rank<Shape>());      // -1 = last axis
    if constexpr (_shape_static_extent<Shape>(D) != cs::dynamic_extent)
        return cs::integral_constant<Idx, static_cast<Idx>(_shape_static_extent<Shape>(D))>{};
    else
        return static_cast<Idx>(m.extents().extent(D));
}
template <class Shape, class Layout, long Ax, class Map>
_TNY_API constexpr auto _axis_stride(const Map & m) noexcept {
    using Idx = typename Shape::index_type;
    constexpr cs::size_t D = _norm_axis(Ax, _shape_rank<Shape>());
    // The compile-time stride is exactly what the layout + extents derive: a
    // strides<> baked value, or the contiguous product of the trailing (C) /
    // leading (F) STATIC extents — so it folds even for a partially-dynamic
    // contiguous shape (shape<-1,3,3>'s stride(0) = 9, the unit stride = 1).
    // `dynamic_stride` means only known at run time -> read it off the mapping.
    constexpr cs::int64_t SS = _src_sstride<D, Layout, Shape>();
    if constexpr (SS != dynamic_stride)
        return cs::integral_constant<Idx, static_cast<Idx>(SS)>{};
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

    static constexpr cs::size_t rank() noexcept { return _shape_rank<Shape>(); }

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
 * @brief Accumulate `v` into `*p`, atomic on both host and device (#257).
 *
 * INTERNAL primitive behind the atomic accumulate ops — prefer
 * `a.atomic_add_(x)` / `t.at(i...).atomic_add_(v)` in user code.
 *
 * The scatter/"push" write: many threads may accumulate into overlapping
 * outputs, which a plain `+=` would race. Device -> `atomicAdd` (`double`
 * needs sm_60+, `__half` sm_70+; not all integer widths have an overload —
 * that surfaces as an nvcc error at instantiation). Host, arithmetic `T`
 * EXCLUDING `bool`/`long double` -> `cuda::std::atomic_ref<T>` (libcu++'s
 * C++17-usable backport of `std::atomic_ref`) so a push kernel parallelised
 * with `std::thread`/OpenMP over overlapping outputs is genuinely race-free,
 * matching the device semantics instead of merely documenting the caller
 * must work around it.
 *
 * The remaining element types keep the old plain `*p += v` (still not
 * thread-safe there — same as before this fix, not a regression): `bool`
 * (libcu++'s `atomic_ref<bool>` has no `fetch_add`) and `long double`
 * (`atomic_ref<long double>` needs a 16-byte atomic RMW, which pulls in
 * `libatomic` and fails to LINK on common toolchains that don't provide it
 * — a working build must not start failing to link just because a caller
 * touches `atomic_add_` on a `long double` tensor). Non-arithmetic `T` (a
 * portable software `half`/`bfloat16` struct, OR the native `__half`/
 * `__nv_bfloat16` CUDA types under `__CUDACC__` on a host translation unit)
 * has no atomic representation to route through `atomic_ref` either way.
 */
template <class T>
_TNY_API void fetch_add(T * p, T v) noexcept {
#ifdef __CUDA_ARCH__
    atomicAdd(p, v);
#else
    if constexpr (cs::is_arithmetic<T>::value
                  && !cs::is_same<T, bool>::value
                  && !cs::is_same<T, long double>::value)
        cs::atomic_ref<T>(*p).fetch_add(v);
    else *p += v;
#endif
}

namespace _detail {
/* --- multi-axis unsqueeze/squeeze folds (see tensor::unsqueeze / ::squeeze) ------ *
 * Both take ALREADY-NORMALISED (non-negative), strictly ascending axis positions and
 * apply the single-axis view op one step at a time. `t` is taken by forwarding
 * reference so the FIRST step keeps the source's constness (and never copies an
 * owning tensor); every later step runs on the view the previous one returned.
 *
 * unsqueeze: positions are relative to the FINAL rank, so insert SMALLEST-first —
 *   each insert lands left of the not-yet-processed positions and leaves them valid.
 *   (H,W) unsqueeze<1,3>: insert 1 -> (H,1,W), insert 3 -> (H,1,W,1).
 * squeeze:   positions are relative to the SOURCE rank, so drop LARGEST-first —
 *   dropping a later axis never shifts an earlier one.
 *   rank-4 squeeze<0,2>: drop 2 -> (0,1,3), drop 0 -> (1,3). */
template <long A0, class Tn> _TNY_API auto _unsqueeze_fold(Tn && t) noexcept
{ return t.template unsqueeze<A0>(); }
template <long A0, long A1, long... Rest, class Tn> _TNY_API auto _unsqueeze_fold(Tn && t) noexcept
{ return _unsqueeze_fold<A1, Rest...>(t.template unsqueeze<A0>()); }

template <long A0, class Tn> _TNY_API auto _squeeze_fold(Tn && t) noexcept
{ return t.template squeeze<A0>(); }
template <long A0, long A1, long... Rest, class Tn> _TNY_API auto _squeeze_fold(Tn && t) noexcept
{ return _squeeze_fold<A1, Rest...>(t).template squeeze<A0>(); }

// Re-expand a `_sorted_axes<...>` (indexing.h, #275) into the fold above, so
// unsqueeze<Ax...>/squeeze<Ax...> can accept their axes in ANY order: the member
// functions sort the (normalised, distinctness-checked) axes into `Sorted`, then
// call here with `I = 0..N-1` to unpack `Sorted::value[I]...` back into the fold's
// `long...` template pack, now guaranteed ascending regardless of caller order.
template <class Sorted, cs::size_t... I, class Tn>
_TNY_API auto _unsqueeze_sorted(Tn && t, cs::index_sequence<I...>) noexcept
{ return _unsqueeze_fold<Sorted::value[I]...>(t); }
template <class Sorted, cs::size_t... I, class Tn>
_TNY_API auto _squeeze_sorted(Tn && t, cs::index_sequence<I...>) noexcept
{ return _squeeze_fold<Sorted::value[I]...>(t); }
} // namespace _detail

/**
 * @brief One N-dimensional tensor, parameterised by ownership.
 *
 * The layout / extents / offset mapping is delegated to `cuda::std::mdspan`
 * (the mapping lives in an empty base, so a fully-static tensor is exactly the
 * size of its data). Ownership is a policy: `storage::view` (non-owning, trivially
 * copyable, kernel-passable), `storage::stack` (inline storage, static shape),
 * `storage::heap` (host-only, move-only), the CUDA owners `storage::gpu`/`pinned`/`mapped`
 * (from `cuda.h`), and the space-carrying views `storage::gpu_view`/`pinned_view`/
 * `mapped_view` (a view of device / page-locked memory keeps its space). The
 * tensor's copy/move semantics are induced by the storage member, not hand-written.
 *
 * @tparam T        Element type.
 * @tparam Shape    The shape: any `cuda::std::extents<Idx, E...>` (static or
 *                  dynamic per dim). Spell it with the `shape<...>` alias.
 * @tparam Layout   mdspan layout policy (default `ccontiguous`).
 * @tparam O        Ownership kind (default `storage::view`).
 */
template <class T, class Shape, class Layout, storage O>
struct tensor : private Layout::template mapping<Shape> {
    using element_type = T;
    using extents_type = Shape;   // the shape (a cuda::std::extents); `shape_type` is a synonym
    using shape_type   = Shape;
    using layout_type  = Layout;
    using index_type   = typename Shape::index_type;
    using mapping_type = typename Layout::template mapping<Shape>;
    using view_type       = cs::mdspan<T, Shape, Layout>;
    using const_view_type = cs::mdspan<const T, Shape, Layout>;

    static constexpr storage  ownership = O;
    static constexpr bool is_static = _is_static_shape<Shape>();
    // memory-space flags (mirror the storage_* helpers, as compile-time constants):
    static constexpr bool is_view            = storage_is_view(O);             // view / gpu_view / pinned_view / mapped_view
    static constexpr bool is_owning          = storage_is_owning(O);           // heap/gpu/pinned/mapped
    static constexpr bool is_device          = storage_is_device(O);           // gpu or gpu_view
    static constexpr bool is_host_accessible = storage_is_host_accessible(O);  // dereferenceable on the host
    static constexpr cs::size_t buffer_size = storage_size<mapping_type, O == storage::stack>::value;
    static_assert(O != storage::stack || is_static, "stack tensor needs a fully static shape");

    storage_policy<T, O, buffer_size> store_{};

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
    template <storage OO = O, cs::enable_if_t<storage_is_view(OO), int> = 0>
    _TNY_API tensor(T * p, mapping_type m) : mapping_type(m), store_(p) {}

    /** @brief View constructor from a pointer alone — for a fully-static geometry
     *         (static extents AND a fully determined layout: contiguous, or an
     *         all-static `strides<...>`). e.g. `tensor<float, shape<3,4>, strides<4,1>>(ptr)`. */
    template <storage OO = O, cs::enable_if_t<storage_is_view(OO) && is_static &&
              (_contiguous_layout<Layout>::value || _strides_all_static<Layout>::value), int> = 0>
    _TNY_API explicit tensor(T * p) : mapping_type(), store_(p) {}   // explicit: no silent T* -> tensor

    /** @brief View constructor from a pointer + extents (contiguous / static-stride layouts). */
    template <storage OO = O, cs::enable_if_t<storage_is_view(OO) && cs::is_constructible<mapping_type, Shape>::value, int> = 0>
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
    template <storage OO = O, cs::enable_if_t<storage_is_owning(OO), int> = 0>
    _TNY_HOST explicit tensor(mapping_type m)
        : mapping_type(m), store_(_alloc_size(m)) {}

    /** @brief Owning constructor from extents (contiguous / static-stride layouts). */
    template <storage OO = O, cs::enable_if_t<storage_is_owning(OO) && cs::is_constructible<mapping_type, Shape>::value, int> = 0>
    _TNY_HOST explicit tensor(Shape e)
        : mapping_type(e), store_(_alloc_size(mapping_type(e))) {}

    /** @brief UNINITIALISED constructors (numpy `np.empty`) used by `empty()`: the
     *         buffer is left indeterminate — fill before reading. `local<...>{}` and
     *         `zeros(...)` keep their zero-fill; this is the opt-out. */
    template <storage OO = O, cs::enable_if_t<OO == storage::stack, int> = 0>
    _TNY_API explicit tensor(_uninit_t) : store_(_uninit) {}
    template <storage OO = O, cs::enable_if_t<storage_is_owning(OO) && cs::is_constructible<mapping_type, Shape>::value, int> = 0>
    _TNY_HOST tensor(Shape e, _uninit_t) : mapping_type(e), store_(_alloc_size(mapping_type(e)), _uninit) {}

    /* --- geometry ------------------------------------------------- */
    static constexpr cs::size_t rank() noexcept { return _shape_rank<Shape>(); }
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
    _TNY_API constexpr index_type stride(Idx d) const noexcept {
        // A rank-0 (scalar) tensor has no axes; `mapping::stride(r)` is constrained to
        // rank > 0 in a spec-conformant mdspan (CCCL 2.x enforces it, 3.x tolerated the
        // instantiation), so guard it — same pattern as the dlpack exporter.
        if constexpr (rank() == 0) { (void)d; return index_type(0); }
        else                       return mapping_type::stride(static_cast<cs::size_t>(d));
    }
private:
    static constexpr index_type _static_numel() noexcept {
        index_type n = 1;
        for (cs::size_t r = 0; r < rank(); ++r) n *= static_cast<index_type>(_shape_static_extent<Shape>(r));
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
     *         stride is unconstrained); an empty tensor is trivially dense.
     *         Negative strides (flips) are *not* dense in this sense -> false.
     *
     *         Pass a layout for an **exact** check: `is_dense<ccontiguous>()` /
     *         `is_dense<fcontiguous>()` test C-/F-contiguity specifically (or any
     *         layout whose mapping is derivable from the extents). For the C-order
     *         question specifically, `is_contiguous()` (below) reads clearer. */
    _TNY_API constexpr bool is_dense() const noexcept {
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
    /** @brief Exact denseness in layout `L` (e.g. `ccontiguous`/`fcontiguous`): the
     *         actual strides equal what `L` produces for these extents. Two spellings —
     *         `t.is_dense<ccontiguous>()` (type form) and `t.is_dense(ccontiguous())`
     *         (value form, layout deduced from the argument). */
    template <class L>
    _TNY_API bool is_dense() const noexcept {
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
    _TNY_API bool is_dense(L) const noexcept { return is_dense<L>(); }

    /** @brief Whether the elements are **contiguous in a specific order** — **C-order
     *         by default** (numpy/pytorch's `is_contiguous`), or F-order via
     *         `is_contiguous<fcontiguous>()`. A thin alias of `is_dense<Layout>()`;
     *         this (not `is_dense()`) is what `reshape`/`flatten` need. Value form:
     *         `is_contiguous(ccontiguous{})`. */
    template <class L = ccontiguous>
    _TNY_API bool is_contiguous() const noexcept { return is_dense<L>(); }
    template <class L>
    _TNY_API bool is_contiguous(L) const noexcept { return is_dense<L>(); }

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
    _TNY_API auto view()       noexcept { return tensor<T,       Shape, Layout, storage_view_of(O)>(store_.data(), mapping()); }
    _TNY_API auto view() const noexcept { return tensor<const T, Shape, Layout, storage_view_of(O)>(store_.data(), mapping()); }

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
        if constexpr (_is_newaxis<Arg>::value) {                        // newaxis: insert a size-1 axis, consume no source axis
            (void)a; ext[k] = index_type(1); str[k] = index_type(0); ++k;
            return;
        } else {
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
    }
    // static output extent for one axis: DROP (integer), the input static extent
    // (an `all`/full_extent OR a folded `slice(none,none)` kept axis), or dynamic.
    template <class Arg, cs::size_t Se> static constexpr cs::size_t _out_static() {
        if constexpr (_is_newaxis<Arg>::value)                          return 1;   // newaxis: inserted axis, static extent 1
        else if constexpr (_is_index<Arg>::value)                       return _drop_axis;
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
    // The SOURCE axis that arg `i` binds: the count of source-CONSUMING args before
    // it. Every arg consumes one source axis EXCEPT a bare `none` (newaxis), which
    // consumes none — so a `none` shifts the source axis of every later arg by 0 and
    // itself maps to whatever axis count precedes it (an out-of-range value it never
    // reads). This is the arg -> source-axis accounting that decouples the arg
    // position (which grows with each `none`) from the source rank.
    template <class... A2>
    _TNY_API static constexpr cs::size_t _src_axis_at(cs::size_t i) noexcept {
        const bool consumes[] = { (!_is_newaxis<A2>::value)..., false };
        cs::size_t ax = 0;
        for (cs::size_t j = 0; j < i; ++j) ax += consumes[j] ? 1 : 0;
        return ax;
    }
    // Source static extent feeding `_out_static` for arg `Arg` at source axis `SrcAx`.
    // A newaxis has no source axis (SrcAx may be out of range), so short-circuit it —
    // `_out_static` returns the fixed extent 1 and never reads this.
    template <class Arg, cs::size_t SrcAx>
    _TNY_API static constexpr cs::size_t _arg_src_ext() noexcept {
        if constexpr (_is_newaxis<Arg>::value) return cs::dynamic_extent;   // unused
        else                                   return _shape_static_extent<Shape>(SrcAx);
    }
    template <bool Wrap = true, class P, cs::size_t... Ax, class... Args>
    _TNY_API auto _slice_range(P p, cs::index_sequence<Ax...> argseq, Args... a) const {
        // The index sequence is one entry per ARG (grows with each `none`); map each
        // to its source axis so the gather still consumes exactly `rank()` axes.
        return _slice_gather<Wrap>(p, cs::index_sequence<_src_axis_at<Args...>(Ax)...>{}, a...);
    }
    template <bool Wrap = true, class P, cs::size_t... SA, class... Args>
    _TNY_API auto _slice_gather(P p, cs::index_sequence<SA...>, Args... a) const {
        using Vt = cs::remove_pointer_t<P>;
        // exactly one source-consuming arg (int/all/range) per axis; `none` is extra.
        static_assert((cs::size_t(0) + ... + (_is_newaxis<Args>::value ? cs::size_t(0) : cs::size_t(1))) == rank(),
                      "slice: one index per axis (any number of `none`/newaxis is extra)");
        constexpr cs::size_t Nk = (cs::size_t(0) + ... + (_is_index<Args>::value ? cs::size_t(0) : cs::size_t(1)));
        // output extents (static where a kept axis is static; `none` -> static 1) and
        // output strides (static where derivable; `none` -> static 0) -> strides<...>.
        using OE = typename _compact<index_type, _out_static<Args, _arg_src_ext<Args, SA>()>()...>::type;
        using SF = typename _str_compact<_out_sstride<Args, SA, Layout, Shape>()...>::type;
        index_type ext[Nk ? Nk : 1] = {}, str[Nk ? Nk : 1] = {}, off = 0; cs::size_t k = 0;
        ( _sl_axis<SA, Wrap>(a, off, ext, str, k), ... );
        cs::array<index_type, Nk> ea{};
        for (cs::size_t i = 0; i < Nk; ++i) ea[i] = ext[i];
        // fold the kept strides into the strides<...> mapping (EBO when all static,
        // else fill the dynamic slots from `str`); Nk == OE::rank().
        return tensor<Vt, OE, SF, storage_view_of(O)>(p + off, _detail::fold_mapping<SF>(OE(ea), str));
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
    // #newaxis (`none`) args in the tuple — they don't consume a source axis, so the
    // ellipsis `fill` excludes them and the expanded arg count grows by this many.
    template <class Tup, cs::size_t... I>
    _TNY_API static constexpr cs::size_t _tup_newaxis_count(cs::index_sequence<I...>) {
        return (cs::size_t(0) + ... + (_is_newaxis<cs::tuple_element_t<I, Tup>>::value ? cs::size_t(1) : cs::size_t(0)));
    }
    template <class... Args> _TNY_API static constexpr cs::size_t _n_newaxis() {
        return (cs::size_t(0) + ... + (_is_newaxis<Args>::value ? cs::size_t(1) : cs::size_t(0)));
    }
    // `Wrap` re-dispatches the expanded args through the CHECKED `operator()`
    // (Wrap=true) or the UNCHECKED `uget` (Wrap=false) — so `t.uget(1, ellipsis)`
    // stays wrap-free after the ellipsis is filled with `all`s.
    template <bool Wrap = true, class Tup, cs::size_t... out_ax>
    _TNY_API decltype(auto) _ellip_call(Tup t, cs::index_sequence<out_ax...>) {
        constexpr cs::size_t n_args   = cs::tuple_size<Tup>::value;
        constexpr cs::size_t n_new     = _tup_newaxis_count<Tup>(cs::make_index_sequence<n_args>{});
        static_assert(n_args - 1 - n_new <= rank(), "too many indices for ellipsis expansion");
        constexpr cs::size_t ell_pos  = _tup_ellipsis_pos<Tup>(cs::make_index_sequence<n_args>{});
        constexpr cs::size_t fill     = rank() - (n_args - 1 - n_new);  // ellipsis fills only source-consuming args (excl. `none`)
        if constexpr (Wrap) return (*this)(_ellip_arg<out_ax, ell_pos, fill>(t)...);
        else                return uget(_ellip_arg<out_ax, ell_pos, fill>(t)...);
    }
    template <bool Wrap = true, class Tup, cs::size_t... out_ax>
    _TNY_API decltype(auto) _ellip_call(Tup t, cs::index_sequence<out_ax...>) const {
        constexpr cs::size_t n_args   = cs::tuple_size<Tup>::value;
        constexpr cs::size_t n_new     = _tup_newaxis_count<Tup>(cs::make_index_sequence<n_args>{});
        static_assert(n_args - 1 - n_new <= rank(), "too many indices for ellipsis expansion");
        constexpr cs::size_t ell_pos  = _tup_ellipsis_pos<Tup>(cs::make_index_sequence<n_args>{});
        constexpr cs::size_t fill     = rank() - (n_args - 1 - n_new);  // ellipsis fills only source-consuming args (excl. `none`)
        if constexpr (Wrap) return (*this)(_ellip_arg<out_ax, ell_pos, fill>(t)...);
        else                return uget(_ellip_arg<out_ax, ell_pos, fill>(t)...);
    }
public:
    /** @brief Element access when every argument is an integer (negatives wrap). */
    template <class... Args, cs::enable_if_t<_all_index<Args...>::value, int> = 0>
    _TNY_API T & operator()(Args... a) noexcept
    { return store_.data()[_offset(cs::make_index_sequence<rank()>{}, a...)]; }
    template <class... Args, cs::enable_if_t<_all_index<Args...>::value, int> = 0>
    _TNY_API const T & operator()(Args... a) const noexcept
    { return store_.data()[_offset(cs::make_index_sequence<rank()>{}, a...)]; }

    /** @brief `at(i...)` — a single element as a **rank-0 VIEW** (all-integer
     *         args; negatives wrap). Unlike `operator()`, which returns a plain
     *         `T&`, this is a view, so the whole tensor API applies to one
     *         element: `x.at(i,j) = 3` writes it, `float v = x.at(i,j)` reads it
     *         (rank-0 tensors convert to/from `T`), and `x.at(i,j).atomic_add_(v)`
     *         is an atomic scatter. */
    template <class... Args, cs::enable_if_t<_all_index<Args...>::value, int> = 0>
    _TNY_API auto at(Args... a) noexcept {
        using E0 = cs::extents<index_type>;   // rank 0
        return tensor<T, E0, ccontiguous, storage_view_of(O)>(&store_.data()[_offset(cs::make_index_sequence<rank()>{}, a...)]);
    }
    template <class... Args, cs::enable_if_t<_all_index<Args...>::value, int> = 0>
    _TNY_API auto at(Args... a) const noexcept {
        using E0 = cs::extents<index_type>;
        return tensor<const T, E0, ccontiguous, storage_view_of(O)>(&store_.data()[_offset(cs::make_index_sequence<rank()>{}, a...)]);
    }

    /** @brief Sub-view when any argument is a slice (`all`, `slice(a,b[,step])`)
     *         or a bare `none` (numpy `newaxis`). Integer args drop their axis,
     *         `all` keeps it, a range keeps a strided window, and a bare `none`
     *         inserts a size-1 axis (static extent 1, stride 0) at its position —
     *         all via the one gather (folds static strides into `strides<...>`;
     *         works on any source layout). `t(none,all,all)` == `unsqueeze<0>()`. */
    template <class... Args, cs::enable_if_t<!_all_index<Args...>::value && !_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API auto operator()(Args... a) noexcept
    { return _slice_range(store_.data(), cs::make_index_sequence<sizeof...(a)>{}, a...); }
    template <class... Args, cs::enable_if_t<!_all_index<Args...>::value && !_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API auto operator()(Args... a) const noexcept
    { return _slice_range(store_.data(), cs::make_index_sequence<sizeof...(a)>{}, a...); }

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
    template <class... Args, cs::enable_if_t<_all_index<Args...>::value, int> = 0>
    _TNY_API T & uget(Args... a) noexcept
    { return store_.data()[_offset<false>(cs::make_index_sequence<rank()>{}, a...)]; }
    template <class... Args, cs::enable_if_t<_all_index<Args...>::value, int> = 0>
    _TNY_API const T & uget(Args... a) const noexcept
    { return store_.data()[_offset<false>(cs::make_index_sequence<rank()>{}, a...)]; }

    // slice: at least one slice arg, no ellipsis -> a VIEW
    template <class... Args, cs::enable_if_t<!_all_index<Args...>::value && !_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API auto uget(Args... a) noexcept
    { return _slice_range<false>(store_.data(), cs::make_index_sequence<sizeof...(a)>{}, a...); }
    template <class... Args, cs::enable_if_t<!_all_index<Args...>::value && !_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API auto uget(Args... a) const noexcept
    { return _slice_range<false>(store_.data(), cs::make_index_sequence<sizeof...(a)>{}, a...); }

    // ellipsis: expand to `all`s, then re-dispatch through `uget` (unchecked)
    template <class... Args, cs::enable_if_t<_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API decltype(auto) uget(Args... a) noexcept
    { return _ellip_call<false>(cs::make_tuple(a...), cs::make_index_sequence<rank() + _n_newaxis<Args...>()>{}); }
    template <class... Args, cs::enable_if_t<_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API decltype(auto) uget(Args... a) const noexcept
    { return _ellip_call<false>(cs::make_tuple(a...), cs::make_index_sequence<rank() + _n_newaxis<Args...>()>{}); }

    /** @brief Unchecked `at`: a single element as a rank-0 VIEW, no negative wrap. */
    template <class... Args, cs::enable_if_t<_all_index<Args...>::value, int> = 0>
    _TNY_API auto uat(Args... a) noexcept {
        using E0 = cs::extents<index_type>;
        return tensor<T, E0, ccontiguous, storage_view_of(O)>(&store_.data()[_offset<false>(cs::make_index_sequence<rank()>{}, a...)]);
    }
    template <class... Args, cs::enable_if_t<_all_index<Args...>::value, int> = 0>
    _TNY_API auto uat(Args... a) const noexcept {
        using E0 = cs::extents<index_type>;
        return tensor<const T, E0, ccontiguous, storage_view_of(O)>(&store_.data()[_offset<false>(cs::make_index_sequence<rank()>{}, a...)]);
    }

    /** @brief Ellipsis form: exactly one `ellipsis` in the args expands to
     *         `rank - (#other args)` copies of `all`, then the call re-runs — so
     *         `t(1, ellipsis, 2)` on rank 5 is `t(1, all, all, all, 2)`. What
     *         remains decides the result (all integers -> element, else view). */
    template <class... Args, cs::enable_if_t<_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API decltype(auto) operator()(Args... a) noexcept
    { return _ellip_call(cs::make_tuple(a...), cs::make_index_sequence<rank() + _n_newaxis<Args...>()>{}); }
    template <class... Args, cs::enable_if_t<_has_ellipsis<Args...>::value, int> = 0>
    _TNY_API decltype(auto) operator()(Args... a) const noexcept
    { return _ellip_call(cs::make_tuple(a...), cs::make_index_sequence<rank() + _n_newaxis<Args...>()>{}); }

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
    template <class B, class E2, class L2, storage O2>
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
    /** @brief Value form: `t.take_along(axis<0,2>{}, i, slice(1,4))` ==
     *         `t.take_along<0,2>(i, slice(1,4))`. The leading `axis<...>` selector is a
     *         single distinct-typed argument, so it needs no `.template` on a dependent
     *         receiver AND disambiguates cleanly from the template form. */
    template <long... Axes, class... Args>
    _TNY_API auto take_along(axis<Axes...>, Args... args) noexcept       { return take_along<Axes...>(args...); }
    template <long... Axes, class... Args>
    _TNY_API auto take_along(axis<Axes...>, Args... args) const noexcept { return take_along<Axes...>(args...); }

    /**
     * @brief Subsample a coloured/strided sub-lattice: bind named axes to a
     *        `slice(start,none,k)` each, sharing one STEP `k` across all of
     *        them but taking a separate START per axis — sugar for
     *        `take_along` (#258), for the "every `k`-th voxel, offset per
     *        axis" pattern coloured Gauss-Seidel relaxation needs
     *        (`loc[d] % k == digit_d(n)`). Pure sugar, no new addressing
     *        power: `t.subsample<0,1>(k, s0, s1)` ==
     *        `t.take_along<0,1>(slice(s0,none,k), slice(s1,none,k))`.
     *        `k` and each `start` accept a runtime value OR a compile-time
     *        one (`Int<k>()`) — folds through `slice()`'s own static-range
     *        machinery, so a fully-static `(start,k)` pair keeps a folded
     *        static output extent/stride, same as a hand-written `slice()`.
     */
    template <long... Axes, class K, class... Starts>
    _TNY_API auto subsample(K k, Starts... starts) noexcept {
        static_assert(sizeof...(Axes) == sizeof...(Starts), "subsample: one start per named axis");
        return take_along<Axes...>(slice(starts, none, k)...);
    }
    template <long... Axes, class K, class... Starts>
    _TNY_API auto subsample(K k, Starts... starts) const noexcept {
        static_assert(sizeof...(Axes) == sizeof...(Starts), "subsample: one start per named axis");
        return take_along<Axes...>(slice(starts, none, k)...);
    }
    /** @brief Value form: `t.subsample(axis<0,1>{}, k, s0, s1)` ==
     *         `t.subsample<0,1>(k, s0, s1)` — leading `axis<...>` selector,
     *         same placement as `take_along`'s own value form (a second
     *         variadic pack, the starts, needs the disambiguating tag up
     *         front rather than trailing). */
    template <long... Axes, class K, class... Starts>
    _TNY_API auto subsample(axis<Axes...>, K k, Starts... starts) noexcept       { return subsample<Axes...>(k, starts...); }
    template <long... Axes, class K, class... Starts>
    _TNY_API auto subsample(axis<Axes...>, K k, Starts... starts) const noexcept { return subsample<Axes...>(k, starts...); }

private:
    // whether a compile-time-known (size, step) pair is in range for axis A —
    // only checkable when the axis extent AND both size/step are static; a
    // dynamic participant defers entirely to unfold's runtime _TNY_CHECK.
    template <cs::size_t A, class Sz, class St> static _TNY_API constexpr bool _unfold_static_ok() {
        if constexpr (_shape_static_extent<Shape>(A) == cs::dynamic_extent || !_is_ic<Sz>::value || !_is_ic<St>::value) return true;
        else return static_cast<long>(Sz::value) >= 1 && static_cast<long>(St::value) >= 1
                  && static_cast<long>(Sz::value) <= static_cast<long>(_shape_static_extent<Shape>(A));
    }
public:
    /**
     * @brief Sliding/strided window along axis `Axis` (pytorch `Tensor.unfold`):
     *        appends a NEW trailing axis of width `size`, stepped by `step`
     *        along `Axis` -> a rank-(N+1) view. `Axis`'s own extent shrinks to
     *        the window COUNT `(shape(Axis) - size) / step + 1`, e.g.
     *        `t.unfold<0>(K, s)` == pytorch's `t.unfold(0, K, s)`. `size`/`step`
     *        accept a runtime value OR a compile-time one (`Int<k>()`), folding
     *        the output extent/stride to static where derivable (like `slice()`).
     *        `size` must be in `[1, shape(Axis)]` and `step >= 1` — a
     *        `static_assert` when both are known at compile time, a debug-time
     *        check otherwise. ND windows compose by chaining:
     *        `t.unfold<0>(K0,s0).unfold<1>(K1,s1)` appends TWO window axes at
     *        the end (nitorch's nd-unfold pattern) — no separate nd-unfold
     *        primitive is needed.
     */
    template <long Axis, class Sz, class St = cs::integral_constant<long,1>>
    _TNY_API auto unfold(Sz size, St step = St{}) noexcept {
        static_assert(_axis_in_range(Axis, rank()), "unfold: axis out of range");
        constexpr cs::size_t A = _norm_axis(Axis, rank());
        static_assert(_unfold_static_ok<A, Sz, St>(), "unfold: size must be in [1, extent(Axis)] and step >= 1");
        // compare in a SIGNED type: index_type may be unsigned, and a negative runtime
        // step/size would otherwise wrap to a huge unsigned value and pass `>= 1` (#339 review).
        using _UfSIdx = cs::make_signed_t<index_type>;
        _TNY_CHECK(static_cast<_UfSIdx>(size) >= _UfSIdx(1) && static_cast<_UfSIdx>(step) >= _UfSIdx(1)
                   && static_cast<_UfSIdx>(size) <= static_cast<_UfSIdx>(extent(A)), "unfold: size must be in [1, extent(Axis)] and step >= 1");
        return as_tensor<storage_view_of(O)>(_detail::unfold_md<A>(mdspan(), size, step, cs::make_index_sequence<rank()>{}));
    }
    template <long Axis, class Sz, class St = cs::integral_constant<long,1>>
    _TNY_API auto unfold(Sz size, St step = St{}) const noexcept {
        static_assert(_axis_in_range(Axis, rank()), "unfold: axis out of range");
        constexpr cs::size_t A = _norm_axis(Axis, rank());
        static_assert(_unfold_static_ok<A, Sz, St>(), "unfold: size must be in [1, extent(Axis)] and step >= 1");
        // compare in a SIGNED type: index_type may be unsigned, and a negative runtime
        // step/size would otherwise wrap to a huge unsigned value and pass `>= 1` (#339 review).
        using _UfSIdx = cs::make_signed_t<index_type>;
        _TNY_CHECK(static_cast<_UfSIdx>(size) >= _UfSIdx(1) && static_cast<_UfSIdx>(step) >= _UfSIdx(1)
                   && static_cast<_UfSIdx>(size) <= static_cast<_UfSIdx>(extent(A)), "unfold: size must be in [1, extent(Axis)] and step >= 1");
        return as_tensor<storage_view_of(O)>(_detail::unfold_md<A>(mdspan(), size, step, cs::make_index_sequence<rank()>{}));
    }
    /** @brief Value form: `t.unfold(Int<0>(), K, s)` == `t.unfold<0>(K, s)` —
     *         a single-axis selector (like `flip`/`squeeze`/`unsqueeze`'s own
     *         `Int<k>()` twin), so no `.template` is needed on a dependent
     *         receiver. */
    template <class I, class Sz, class St = cs::integral_constant<long,1>, cs::enable_if_t<_is_ic<I>::value, int> = 0>
    _TNY_API auto unfold(I, Sz size, St step = St{})       noexcept { return unfold<static_cast<long>(I::value)>(size, step); }
    template <class I, class Sz, class St = cs::integral_constant<long,1>, cs::enable_if_t<_is_ic<I>::value, int> = 0>
    _TNY_API auto unfold(I, Sz size, St step = St{}) const noexcept { return unfold<static_cast<long>(I::value)>(size, step); }

private:
    // Build the runtime index_select output extents: axis `Axis`'s extent is
    // `newExt` (the gather index tensor's own numel), every other axis copies
    // this tensor's own extent.
    template <cs::size_t Axis, class OutE, cs::size_t... D>
    _TNY_API OutE _idxsel_shape(cs::index_sequence<D...>, index_type newExt) const {
        return OutE(static_cast<index_type>(D == Axis ? newExt : extent(D))...);
    }
public:
    /**
     * @brief Gather along axis `Axis` using a rank-1 integer index TENSOR `idx`
     *        (numpy/pytorch `index_select`/`take`): `out(...,j,...) = a(...,idx(j),...)`
     *        for `j` in `[0, idx.numel())` — axis `Axis`'s extent becomes `idx`'s
     *        (static when `idx`'s own shape is static). `idx`'s values wrap negative
     *        like any other teeny index (it's built on `take_along`, which already
     *        wraps). Distinct from `take_along` (compile-time indices/ranges): `idx`'s
     *        VALUES are runtime DATA, so this always materialises a copy — an
     *        arbitrary data-dependent gather isn't expressible as an affine mdspan
     *        view. Prefer the `into(dest)` form (`_TNY_API`, no allocation, device-safe)
     *        in a kernel; this allocating form is `_TNY_HOST` convenience and copies
     *        on the HOST, so `*this` must be host-accessible (a `gpu`/`gpu_view`
     *        source: gather into a preallocated device `into(dest)` instead).
     */
    template <long Axis, class Ti,class Ei,class Li,storage Oi,
              cs::enable_if_t<_md::index_select_extents<Shape, _norm_axis(Axis, rank()), Ei::static_extent(0)>::rank_dynamic() == 0, int> = 0>
    _TNY_API auto index_select(const tensor<Ti,Ei,Li,Oi> & idx) const {
        static_assert(cs::is_integral<Ti>::value, "index_select: idx must have an integer element type");
        static_assert(Ei::rank() == 1, "index_select: idx must be rank-1");
        static_assert(_axis_in_range(Axis, rank()), "index_select: axis out of range");
        static_assert(storage_is_host_accessible(O),
            "index_select()'s allocating form copies on the host and cannot dereference device "
            "memory; for a gpu/gpu_view source, gather into a preallocated device into(dest) instead.");
        using OutE = _md::index_select_extents<Shape, _norm_axis(Axis, rank()), Ei::static_extent(0)>;
        tensor<T, OutE, ccontiguous, storage::stack> out{};
        index_select<Axis>(idx, into(out));
        return out;
    }
    template <long Axis, class Ti,class Ei,class Li,storage Oi,
              cs::enable_if_t<_md::index_select_extents<Shape, _norm_axis(Axis, rank()), Ei::static_extent(0)>::rank_dynamic() != 0, int> = 0>
    _TNY_HOST auto index_select(const tensor<Ti,Ei,Li,Oi> & idx) const {
        static_assert(cs::is_integral<Ti>::value, "index_select: idx must have an integer element type");
        static_assert(Ei::rank() == 1, "index_select: idx must be rank-1");
        static_assert(_axis_in_range(Axis, rank()), "index_select: axis out of range");
        static_assert(storage_is_host_accessible(O),
            "index_select()'s allocating form copies on the host and cannot dereference device "
            "memory; for a gpu/gpu_view source, gather into a preallocated device into(dest) instead.");
        constexpr cs::size_t A = _norm_axis(Axis, rank());
        using OutE = _md::index_select_extents<Shape, A, Ei::static_extent(0)>;
        OutE oe = _idxsel_shape<A, OutE>(cs::make_index_sequence<rank()>{}, static_cast<index_type>(idx.extent(0)));
        tensor<T, OutE, ccontiguous, storage::heap> out(oe);
        index_select<Axis>(idx, into(out));
        return out;
    }
    /** @brief Value form: `t.index_select(idx, axis<Axis>{})` == `t.index_select<Axis>(idx)`.
     *         Deduces `Axis` from the tag, so no `.template` disambiguator is needed
     *         on a type-dependent receiver (the primary reason this form exists —
     *         the mesh-distance kernels this feature targets call it from templates).
     *         SPLIT IN TWO on the same key as the `<Axis>` pair it forwards to (#375):
     *         a static result is stack-owned (host+device) so the forwarder is
     *         `_TNY_API`; a dynamic result is heap-owned (host only) so it is
     *         `_TNY_HOST` — else nvcc's device pass would see a `_TNY_API` forwarder
     *         call a `__host__` allocator. */
    template <class Ti,class Ei,class Li,storage Oi, long Axis,
              cs::enable_if_t<_md::index_select_extents<Shape, _norm_axis(Axis, rank()), Ei::static_extent(0)>::rank_dynamic() == 0, int> = 0>
    _TNY_API auto index_select(const tensor<Ti,Ei,Li,Oi> & idx, axis<Axis>) const { return index_select<Axis>(idx); }
    template <class Ti,class Ei,class Li,storage Oi, long Axis,
              cs::enable_if_t<_md::index_select_extents<Shape, _norm_axis(Axis, rank()), Ei::static_extent(0)>::rank_dynamic() != 0, int> = 0>
    _TNY_HOST auto index_select(const tensor<Ti,Ei,Li,Oi> & idx, axis<Axis>) const { return index_select<Axis>(idx); }
    template <class Ti,class Ei,class Li,storage Oi, long Axis, class D>
    _TNY_API auto & index_select(const tensor<Ti,Ei,Li,Oi> & idx, axis<Axis>, into_t<D> out) const { return index_select<Axis>(idx, out); }

    /** @brief `into(dest)` form: writes the gather straight into `dest` — one pass,
     *         no allocation, `_TNY_API` (device-safe). Returns `dest&`. `dest`'s
     *         extents must match (axis `Axis` == `idx.numel()`, checked; every other
     *         axis == this tensor's own, checked by the underlying `copy_`). `dest`
     *         must not ALIAS this tensor's storage — an aliased in-place gather is
     *         unsupported (each `j` overwrites a slot of `dest` that a LATER `j` may
     *         still need to read from `*this`) and silently reorders instead of
     *         erroring. */
    template <long Axis, class Ti,class Ei,class Li,storage Oi, class D>
    _TNY_API auto & index_select(const tensor<Ti,Ei,Li,Oi> & idx, into_t<D> out) const {
        static_assert(cs::is_integral<Ti>::value, "index_select: idx must have an integer element type");
        static_assert(Ei::rank() == 1, "index_select: idx must be rank-1");
        static_assert(_axis_in_range(Axis, rank()), "index_select: axis out of range");
        constexpr cs::size_t A = _norm_axis(Axis, rank());
        using DstE = typename D::extents_type;
        constexpr cs::size_t dstA = DstE::static_extent(A);
        constexpr cs::size_t idxA = Ei::static_extent(0);
        static_assert(dstA == cs::dynamic_extent || idxA == cs::dynamic_extent || dstA == idxA,
            "index_select: dest's axis Axis extent must equal idx's extent(0)");
        _TNY_CHECK(static_cast<index_type>(out.dest.extent(A)) == static_cast<index_type>(idx.extent(0)),
            "index_select: dest's axis Axis extent must equal idx.extent(0)");
        // idx(j)'s VALUE (unlike the loop bound) can be negative -- must stay
        // SIGNED so take_along's wrap (_wrap_idx) takes its negative-index branch
        // rather than reinterpreting a negative value as a huge unsigned index
        // when this tensor's own index_type happens to be unsigned (#326 review).
        const index_type n = static_cast<index_type>(idx.extent(0));
        for (index_type j = 0; j < n; ++j)
            out.dest.template take_along<(long)A>(j).copy_(take_along<(long)A>(static_cast<cs::make_signed_t<index_type>>(idx(j))));
        return out.dest;
    }

    /** @brief Reorder the axes (a permutation of 0..N-1; negatives wrap) -> a rank-N view. */
    template <long... Perm>
    _TNY_API auto permute() noexcept
    { static_assert(sizeof...(Perm) == rank(), "permute: need N axes"); static_assert(_is_perm<_norm_axis(Perm, rank())...>(), "permute: axes must be a permutation of 0..N-1 (in range, no repeats)"); return as_tensor<storage_view_of(O)>(_detail::perm_md(mdspan(), cs::index_sequence<_norm_axis(Perm, rank())...>{})); }
    template <long... Perm>
    _TNY_API auto permute() const noexcept
    { static_assert(sizeof...(Perm) == rank(), "permute: need N axes"); static_assert(_is_perm<_norm_axis(Perm, rank())...>(), "permute: axes must be a permutation of 0..N-1 (in range, no repeats)"); return as_tensor<storage_view_of(O)>(_detail::perm_md(mdspan(), cs::index_sequence<_norm_axis(Perm, rank())...>{})); }

    /** @brief Reverse axis `Ax` (negatives wrap) -> a view (numpy `flip`). Uses a
     *         negative stride, so the index type must be signed (`shape<...>` is). */
    template <long Ax = 0>
    _TNY_API auto flip() noexcept
    { static_assert(_axis_in_range(Ax, rank()), "flip: axis out of range"); return as_tensor<storage_view_of(O)>(_detail::flip_md<_norm_axis(Ax, rank())>(mdspan(), cs::make_index_sequence<rank()>{})); }
    template <long Ax = 0>
    _TNY_API auto flip() const noexcept
    { static_assert(_axis_in_range(Ax, rank()), "flip: axis out of range"); return as_tensor<storage_view_of(O)>(_detail::flip_md<_norm_axis(Ax, rank())>(mdspan(), cs::make_index_sequence<rank()>{})); }

    /** @brief A dense, row-major OWNING copy of this tensor (materialise a view /
     *         non-contiguous / permuted / flipped tensor). Static shape -> stack
     *         (host+device); dynamic -> heap (host only).
     *
     *  Copies on the HOST via `copy_`, so it cannot dereference DEVICE memory: the
     *  dynamic-shape (`_TNY_HOST`) overload `static_assert`s that the source is
     *  host-accessible. For a `gpu`/`gpu_view` tensor use the free `to<Space>(x)`
     *  from `<teeny/cuda.h>` (e.g. `to<storage::heap>(x)`), which copies device-aware. */
    template <bool S = is_static, cs::enable_if_t<S, int> = 0>
    _TNY_API auto clone() const { tensor<T, Shape, ccontiguous, storage::stack> c{}; c.copy_(*this); return c; }
    template <bool S = is_static, cs::enable_if_t<!S, int> = 0>
    _TNY_HOST auto clone() const {
        static_assert(storage_is_host_accessible(O),
            "clone()/to() copies on the host and cannot dereference device memory; "
            "for a gpu/gpu_view tensor use the free to<Space>(x) (e.g. to<storage::heap>(x) "
            "or to<storage::gpu>(x), from <teeny/cuda.h>) which does a device-aware copy.");
        tensor<T, Shape, ccontiguous, storage::heap> c(extents()); c.copy_(*this); return c;
    }

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
     *  -> stack (host+device), dynamic -> heap (host only). The copy runs on the
     *  HOST, so it cannot dereference DEVICE memory: the dynamic-shape (`_TNY_HOST`)
     *  copying overload `static_assert`s that the source is host-accessible. To also
     *  move across memory spaces (host <-> CUDA) — or to convert a `gpu`/`gpu_view`
     *  tensor at all — use the `to<storage::gpu, T2, Force>(x)` free functions from
     *  `<teeny/cuda.h>`, which copy device-aware. */
    template <class T2 = element_type, bool Force = false,
              cs::enable_if_t<!Force && cs::is_same<T2, element_type>::value, int> = 0>
    _TNY_API auto to() const & {
        return tensor<const element_type, Shape, Layout, storage_view_of(O)>(data(), mapping());  // already that dtype -> borrow (gpu_view if device)
    }
    template <class T2 = element_type, bool Force = false, bool S = is_static,
              cs::enable_if_t<(Force || !cs::is_same<T2, element_type>::value) && S, int> = 0>
    _TNY_API auto to() const & { tensor<cs::remove_cv_t<T2>, Shape, ccontiguous, storage::stack> c{}; c.copy_(*this); return c; }
    template <class T2 = element_type, bool Force = false, bool S = is_static,
              cs::enable_if_t<(Force || !cs::is_same<T2, element_type>::value) && !S, int> = 0>
    _TNY_HOST auto to() const & {
        static_assert(storage_is_host_accessible(O),
            "clone()/to() copies on the host and cannot dereference device memory; "
            "for a gpu/gpu_view tensor use the free to<Space>(x) (e.g. to<storage::heap>(x) "
            "or to<storage::gpu>(x), from <teeny/cuda.h>) which does a device-aware copy.");
        tensor<cs::remove_cv_t<T2>, Shape, ccontiguous, storage::heap> c(extents()); c.copy_(*this); return c;
    }
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
              cs::enable_if_t<storage_is_view(O) && !Force && cs::is_same<T2, element_type>::value, int> = 0>
    _TNY_API auto to() const && {
        return tensor<const element_type, Shape, Layout, storage_view_of(O)>(data(), mapping());  // view temp -> safe borrow (external storage)
    }
    template <class T2 = element_type, bool Force = false, bool S = is_static,
              cs::enable_if_t<!(storage_is_view(O) && !Force && cs::is_same<T2, element_type>::value) && S, int> = 0>
    _TNY_API auto to() const && { tensor<cs::remove_cv_t<T2>, Shape, ccontiguous, storage::stack> c{}; c.copy_(*this); return c; }
    template <class T2 = element_type, bool Force = false, bool S = is_static,
              cs::enable_if_t<!(storage_is_view(O) && !Force && cs::is_same<T2, element_type>::value) && !S, int> = 0>
    _TNY_HOST auto to() const && {
        static_assert(storage_is_host_accessible(O),
            "clone()/to() copies on the host and cannot dereference device memory; "
            "for a gpu/gpu_view tensor use the free to<Space>(x) (e.g. to<storage::heap>(x) "
            "or to<storage::gpu>(x), from <teeny/cuda.h>) which does a device-aware copy.");
        tensor<cs::remove_cv_t<T2>, Shape, ccontiguous, storage::heap> c(extents()); c.copy_(*this); return c;
    }

    /** @brief Value-tag form: `x.to(dtype<T2>{})` == `x.to<T2>()`. Deduces `T2` from
     *  the tag instead of an explicit `<T2>` template argument, so a type-dependent
     *  receiver spells it with no `.template`. `Force` is a LEADING explicit template
     *  arg (since `T2` is now deduced) — `x.to<true>(dtype<T2>{})`. A thin forwarder:
     *  the enable_if mirrors exactly which of `to<T2,Force>()`'s branches is
     *  `_TNY_API` (the matching-dtype borrow is `_TNY_API` regardless of a dynamic
     *  shape; only a differing-dtype/`Force` copy of a dynamic shape is `_TNY_HOST`),
     *  so the annotation always matches — never `_TNY_API` calling `_TNY_HOST`. */
    template <bool Force = false, class T2, bool S = is_static,
              cs::enable_if_t<S || (!Force && cs::is_same<T2, element_type>::value), int> = 0>
    _TNY_API auto to(dtype<T2>) const & { return to<T2, Force>(); }
    template <bool Force = false, class T2, bool S = is_static,
              cs::enable_if_t<!(S || (!Force && cs::is_same<T2, element_type>::value)), int> = 0>
    _TNY_HOST auto to(dtype<T2>) const & { return to<T2, Force>(); }
    template <bool Force = false, class T2, bool S = is_static,
              cs::enable_if_t<S || (storage_is_view(O) && !Force && cs::is_same<T2, element_type>::value), int> = 0>
    _TNY_API auto to(dtype<T2>) const && { return cs::move(*this).template to<T2, Force>(); }
    template <bool Force = false, class T2, bool S = is_static,
              cs::enable_if_t<!(S || (storage_is_view(O) && !Force && cs::is_same<T2, element_type>::value)), int> = 0>
    _TNY_HOST auto to(dtype<T2>) const && { return cs::move(*this).template to<T2, Force>(); }

private:
    // POD the compile-time reshape solver returns (fully-static sources): whether the
    // reshape is viewable without a copy, plus the resolved target extents and strides.
    template <cs::size_t M> struct _rs_static_t {
        bool ok;
        index_type  te[M ? M : 1];
        cs::int64_t ts[M ? M : 1];
    };
    // Solve the reshape at COMPILE TIME from the fully-static source geometry (static
    // extents + statically-known strides). Mirrors the runtime body: resolve a `-1`,
    // check the element count, then run numpy's C-order no-copy walk. Only meaningful
    // when the source strides are all static (the caller gates on that).
    template <long... NewExt, cs::size_t... Rs>
    static constexpr _rs_static_t<sizeof...(NewExt)> _reshape_static_impl(cs::index_sequence<Rs...>) {
        constexpr cs::size_t R = sizeof...(Rs), M = sizeof...(NewExt);
        cs::array<index_type, R ? R : 1> se{ static_cast<index_type>(_shape_static_extent<Shape>(Rs))... };
        cs::array<index_type, R ? R : 1> ss{ static_cast<index_type>(_src_sstride<Rs, Layout, Shape>())... };
        const long ne[M ? M : 1] = { NewExt... };
        index_type n = 1; for (cs::size_t r = 0; r < R; ++r) n *= se[r];
        index_type known = 1; bool has_inf = false;
        for (cs::size_t k = 0; k < M; ++k) { if (ne[k] < 0) has_inf = true; else known *= static_cast<index_type>(ne[k]); }
        _rs_static_t<M> out{}; out.ok = true; index_type inferred = 1;
        // A `-1` must DIVIDE numel; without one the counts must match exactly.
        if (has_inf) { if (known == 0 || n % known != 0) out.ok = false; else inferred = n / known; }
        else         { if (known != n)                   out.ok = false; }
        cs::array<index_type, M ? M : 1> te{}, ts{};
        for (cs::size_t k = 0; k < M; ++k) te[k] = ne[k] < 0 ? inferred : static_cast<index_type>(ne[k]);
        if (out.ok) out.ok = _detail::reshape_view_strides(se, ss, te, ts);
        for (cs::size_t k = 0; k < M; ++k) { out.te[k] = te[k]; out.ts[k] = static_cast<cs::int64_t>(ts[k]); }
        return out;
    }
    template <long... NewExt>
    static constexpr _rs_static_t<sizeof...(NewExt)> _reshape_static() {
        return _reshape_static_impl<NewExt...>(cs::make_index_sequence<rank()>{});
    }
    // Build the folded static-geometry view (extents + strides both compile-time) from
    // the solved result — a pure static `strides<...>` view (pointer-only ctor).
    template <class El, long... NewExt, cs::size_t... Ks>
    static _TNY_API auto _make_static_reshape(El * p, cs::index_sequence<Ks...>) {
        constexpr _rs_static_t<sizeof...(NewExt)> RS = _reshape_static<NewExt...>();
        using NE = cs::extents<index_type, static_cast<cs::size_t>(RS.te[Ks])...>;
        using SF = ::tny::strides< RS.ts[Ks]... >;   // qualify: the member strides() shadows the type
        return tensor<El, NE, SF, storage_view_of(O)>(p);
    }

    // shared reshape body: one axis may be `-1` (numpy-style, inferred from numel).
    // View-when-stride-compatible (numpy semantics): a reshape is a VIEW whenever the
    // source layout can be regrouped without a copy — any C-contiguous run split or
    // merged — not only when the whole tensor is C-contiguous. The output strides come
    // from numpy's no-copy walk and land in a folded `strides<...>` layout: folded to
    // compile-time immediates when the SOURCE geometry is fully static (extents AND
    // strides), all-runtime otherwise. A statically-decidable non-viewable reshape is a
    // compile error (`static_assert`); the dynamic case is a debug `_TNY_CHECK`.
    template <class El, long... NewExt>
    _TNY_API auto _reshape(El * p) const noexcept {
        static_assert(((NewExt < 0 ? 1 : 0) + ... + 0) <= 1, "reshape: at most one inferred (-1) dimension");
        constexpr cs::size_t M = sizeof...(NewExt);
        constexpr bool src_static = is_static &&
            (_contiguous_layout<Layout>::value || _strides_all_static<Layout>::value);
        if constexpr (src_static) {
            // fully-static source: solve + fold entirely at compile time.
            static_assert(_reshape_static<NewExt...>().ok,
                "reshape: this static shape/layout cannot be viewed as the requested shape "
                "without a copy — the element counts must match AND the layout must be "
                "regroupable in C-order (clone() first, or query can_reshape_without_copy<...>()).");
            return _make_static_reshape<El, NewExt...>(p, cs::make_index_sequence<M>{});
        } else {
            // dynamic source (or dynamic strides): compute the strides at run time. The
            // target extents are still static where given as a literal; the inferred
            // `-1` dim is dynamic (numel is runtime).
            using NE = cs::extents<index_type, (NewExt < 0 ? cs::dynamic_extent : static_cast<cs::size_t>(NewExt))...>;
            constexpr index_type known = (index_type(1) * ... * (NewExt < 0 ? index_type(1) : index_type(NewExt)));
            constexpr bool has_inferred = ((NewExt < 0) || ...);
            if constexpr (has_inferred)
                _TNY_CHECK(known != 0 && numel() % known == 0, "reshape: numel not divisible by given extents");
            else
                _TNY_CHECK(known == numel(), "reshape: element count must match the given extents (no -1 to infer)");
            const index_type inferred = numel() / (known ? known : index_type(1));
            cs::array<index_type, rank()> se{}, ss{};
            for (cs::size_t r = 0; r < rank(); ++r) { se[r] = static_cast<index_type>(extent(r)); ss[r] = static_cast<index_type>(stride(r)); }
            cs::array<index_type, M> te{ (NewExt < 0 ? inferred : index_type(NewExt))... }, ts{};
            const bool ok = _detail::reshape_view_strides(se, ss, te, ts);
            _TNY_CHECK(ok, "reshape: this layout cannot be viewed as the requested shape without a "
                           "copy (clone() first, or query can_reshape_without_copy<...>()).");
            using SF = _runtime_strides_t<M>;
            NE oe(te);
            return tensor<El, NE, SF, storage_view_of(O)>(p, _detail::fold_mapping<SF>(oe, ts.data()));
        }
    }
public:
    /** @brief View this tensor as a new shape — numpy semantics: a **VIEW** whenever
     *         the layout can be regrouped without a copy (not only when C-contiguous;
     *         a strided/permuted source often still views — split a contiguous axis,
     *         merge a contiguous run). The output is a folded `strides<...>` view
     *         (compile-time strides when the source is fully static). One extent may
     *         be **`-1`** (numpy-style), inferred from the total size: `t.reshape<6,-1>()`.
     *         A non-viewable reshape is a compile error (static source) or a debug
     *         check (dynamic) — `clone()` first, or query `can_reshape_without_copy`. */
    template <long... NewExt> _TNY_API auto reshape() noexcept       { return _reshape<T, NewExt...>(store_.data()); }
    template <long... NewExt> _TNY_API auto reshape() const noexcept { return _reshape<const T, NewExt...>(store_.data()); }

    /** @brief Whether `reshape<NewExt...>()` can produce a VIEW (no copy) of this
     *         tensor's actual layout — numpy's rule: not just C-contiguity, but any
     *         stride-compatible regrouping (splitting an axis, merging a contiguous
     *         run). One `-1` may be inferred. `false` -> the reshape needs a
     *         `clone()`. (The result type of a viewable `reshape` is a folded
     *         `strides<...>` view.) */
    template <long... NewExt>
    _TNY_API bool can_reshape_without_copy() const noexcept {
        static_assert(((NewExt < 0 ? 1 : 0) + ... + 0) <= 1, "reshape: at most one inferred (-1) dimension");
        constexpr cs::size_t M = sizeof...(NewExt);
        constexpr index_type known = (index_type(1) * ... * (NewExt < 0 ? index_type(1) : index_type(NewExt)));
        constexpr bool has_inferred = ((NewExt < 0) || ...);
        const index_type n = numel();
        index_type inferred = 1;
        if constexpr (has_inferred) { if (known == 0 || n % known != 0) return false; inferred = n / (known ? known : index_type(1)); }
        else                        { if (known != n) return false; }
        cs::array<index_type, rank()> se{}, ss{};
        for (cs::size_t r = 0; r < rank(); ++r) { se[r] = static_cast<index_type>(extent(r)); ss[r] = static_cast<index_type>(stride(r)); }
        cs::array<index_type, M> te{ (NewExt < 0 ? inferred : index_type(NewExt))... }, ts{};
        return _detail::reshape_view_strides(se, ss, te, ts);
    }

private:
    template <class El, class NewE, class NewL, cs::size_t... D>
    _TNY_API auto _recast(El * p, cs::index_sequence<D...>) const {
        static_assert(_shape_rank<NewE>() == rank(), "recast: rank must match");
        // recast re-types the EXTENTS (to recover statically-known dims). Each static
        // dim of NewE must equal the runtime extent (a genuine mismatch is a bug —
        // validated host-debug). The STRIDES come from `NewL`:
        ( _TNY_CHECK(_shape_static_extent<NewE>(D) == cs::dynamic_extent ||
                     static_cast<index_type>(_shape_static_extent<NewE>(D)) == static_cast<index_type>(extent(D)),
                     "recast: a static dim does not match the actual extent"), ... );
        NewE oe(cs::array<index_type, rank()>{ static_cast<index_type>(extent(D))... });
        // The OUTPUT layout: `keep_strides` (default) PRESERVES the source layout —
        // its mapping already merges layout + extents, so we just retype the extents
        // and let the strides be derived/carried (contiguous ones then fold in the
        // accessor under NewE's richer static extents). An explicit `NewL` overrides:
        // `ccontiguous`/`fcontiguous` reinterpret AS that order (strides derived from
        // the extents — the "I promise it's contiguous" form, UB if it isn't), a
        // `strides<S...>` imposes those strides (dynamic slots filled from the source).
        using OL = cs::conditional_t<cs::is_same<NewL, keep_strides>::value, Layout, NewL>;
        const index_type rstr[rank() ? rank() : 1] = { static_cast<index_type>(stride(D))... };
        auto m = _detail::retype_mapping<OL>(oe, rstr);
        // An explicit layout OVERRIDE replaces the strides (derived for a contiguous
        // NewL, imposed for a strides<...> one). Verify — host-debug, symmetric with
        // the extent check above — that the imposed stride actually matches the
        // source's: a false "I promise it's contiguous" would otherwise SILENTLY
        // mis-address in every build. Axes of extent <= 1 impose no constraint (their
        // stride is unobservable). keep_strides carries the real strides, so no check.
        if constexpr (!cs::is_same<NewL, keep_strides>::value) {
            ( _TNY_CHECK(static_cast<index_type>(oe.extent(D)) <= index_type(1) ||
                         static_cast<index_type>(m.stride(D)) == rstr[D],
                         "recast<E, Layout>: the imposed layout's stride does not match the "
                         "source's actual stride — the data is not laid out as promised; "
                         "use recast<E> (keep_strides) to preserve the real strides"), ... );
        }
        return tensor<El, NewE, OL, storage_view_of(O)>(p, m);
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

    /** @brief Does every element offset of this view fit the index type `Idx2`?
     *         Computes the SIGNED reach directly (teeny has negative-stride views, so
     *         `required_span_size`'s non-negative assumption doesn't apply):
     *         `max = Σ_{s>0}(e−1)·s`, `min = Σ_{s<0}(e−1)·s`; fits ⟺ `min..max` ⊆
     *         `Idx2`. Accumulates in a wide type; a broadcast (stride-0) axis adds 0.
     *         The precondition `reindex<Idx2>()` debug-checks. */
    template <class Idx2>
    _TNY_API bool index_fits() const noexcept {
        using W = long long;
        W maxo = 0, mino = 0;
        for (cs::size_t r = 0; r < rank(); ++r) {
            const W e = static_cast<W>(extent(r));
            if (e <= W(1)) continue;                              // size-1/0 axis: no reach
            const W reach = (e - W(1)) * static_cast<W>(stride(r));
            if (reach > 0) maxo += reach; else mino += reach;     // reach sign == stride sign (e>1)
        }
        return maxo <= static_cast<W>((cs::numeric_limits<Idx2>::max)())
            && mino >= static_cast<W>((cs::numeric_limits<Idx2>::min)());
    }

    /** @brief No-copy, **layout-preserving** retype of the offset index width to
     *         `Idx2`: same pointer, same layout KIND, the extents' `index_type` and any
     *         dynamic strides narrowed to `Idx2` (a `strides<...>` literal pack is
     *         unchanged). Narrowing the boundary view to `shape32` halves the by-value
     *         footprint and runs offset math in 32-bit (big device win). Orthogonal to
     *         `recast` (which staticizes the extent VALUES) — they compose. Debug-checks
     *         `index_fits<Idx2>()`; UB if the caller lies (same contract as `u*`). */
    template <class Idx2>
    _TNY_API auto reindex() {
        _TNY_CHECK(index_fits<Idx2>(), "reindex: element offsets don't fit the target index type (span exceeds its range)");
        return _recast<T, _reindex_extents_t<Idx2, Shape>, keep_strides>(store_.data(), cs::make_index_sequence<rank()>{});
    }
    template <class Idx2>
    _TNY_API auto reindex() const {
        _TNY_CHECK(index_fits<Idx2>(), "reindex: element offsets don't fit the target index type (span exceeds its range)");
        return _recast<const T, _reindex_extents_t<Idx2, Shape>, keep_strides>(store_.data(), cs::make_index_sequence<rank()>{});
    }

    /** @brief View as 1-D (`ravel`) — a VIEW whenever the layout is mergeable into a
     *         single contiguous run without a copy (numpy semantics; `clone()` first
     *         otherwise). Just `reshape<-1>()` (one inferred dim), spelled out for
     *         discoverability. */
    _TNY_API auto flatten() noexcept       { return reshape<-1>(); }
    _TNY_API auto flatten() const noexcept { return reshape<-1>(); }

    /** @brief Insert a size-1 axis at position `Ax` (numpy `newaxis`/`unsqueeze`)
     *         -> a rank-(N+1) view. Negative `Ax` counts from the back, so
     *         `.unsqueeze<-1>()` appends a trailing axis: `(H,W)` -> `(H,W,1)`. */
    template <long Ax = 0>
    _TNY_API auto unsqueeze() noexcept
    { constexpr cs::size_t A = _norm_axis(Ax, rank() + 1); static_assert(A <= rank(), "unsqueeze: axis out of range"); return as_tensor<storage_view_of(O)>(_detail::unsqueeze_md<A>(mdspan(), cs::make_index_sequence<rank() + 1>{})); }
    template <long Ax = 0>
    _TNY_API auto unsqueeze() const noexcept
    { constexpr cs::size_t A = _norm_axis(Ax, rank() + 1); static_assert(A <= rank(), "unsqueeze: axis out of range"); return as_tensor<storage_view_of(O)>(_detail::unsqueeze_md<A>(mdspan(), cs::make_index_sequence<rank() + 1>{})); }

    /** @brief Insert size-1 axes at SEVERAL positions at once (numpy
     *         `expand_dims(a, axis=(...))`) -> a rank-(N+k) view. The positions are
     *         relative to the **final** rank `N + k` (negatives count from the back
     *         of it), and must be distinct (in ANY order — sorted internally, #275)
     *         — e.g. `(H,W).unsqueeze<1,3>()` -> `(H,1,W,1)`, `(H,W).unsqueeze<0,-1>()`
     *         -> `(1,H,W,1)`. Arity picks this overload; one axis still means
     *         `unsqueeze<Ax>()` above. */
    template <long Ax0, long Ax1, long... Rest>
    _TNY_API auto unsqueeze() noexcept {
        constexpr cs::size_t NR = rank() + 2 + sizeof...(Rest);   // final (post-insert) rank
        static_assert(_axis_in_range(Ax0, NR) && _axis_in_range(Ax1, NR) && (_axis_in_range(Rest, NR) && ...),
                      "unsqueeze: axis out of range");
        static_assert(_all_distinct<_norm_axis(Ax0, NR), _norm_axis(Ax1, NR), _norm_axis(Rest, NR)...>(),
                      "unsqueeze: axes must be distinct");
        using Sorted = _sorted_axes<(long)_norm_axis(Ax0, NR), (long)_norm_axis(Ax1, NR), (long)_norm_axis(Rest, NR)...>;
        return _detail::_unsqueeze_sorted<Sorted>(*this, cs::make_index_sequence<2 + sizeof...(Rest)>{});
    }
    template <long Ax0, long Ax1, long... Rest>
    _TNY_API auto unsqueeze() const noexcept {
        constexpr cs::size_t NR = rank() + 2 + sizeof...(Rest);   // final (post-insert) rank
        static_assert(_axis_in_range(Ax0, NR) && _axis_in_range(Ax1, NR) && (_axis_in_range(Rest, NR) && ...),
                      "unsqueeze: axis out of range");
        static_assert(_all_distinct<_norm_axis(Ax0, NR), _norm_axis(Ax1, NR), _norm_axis(Rest, NR)...>(),
                      "unsqueeze: axes must be distinct");
        using Sorted = _sorted_axes<(long)_norm_axis(Ax0, NR), (long)_norm_axis(Ax1, NR), (long)_norm_axis(Rest, NR)...>;
        return _detail::_unsqueeze_sorted<Sorted>(*this, cs::make_index_sequence<2 + sizeof...(Rest)>{});
    }

private:
    static constexpr long _ax_all = (cs::numeric_limits<long>::min)();   // squeeze() sentinel: "all singletons"
    // gather arg for axis D when squeezing all singletons: drop a STATIC size-1
    // axis (index 0), keep the rest. (A dynamic axis that is 1 only at run time
    // can't be dropped — the rank must stay static.)
    template <cs::size_t D> static _TNY_API auto _sq_arg() noexcept {
        if constexpr (_shape_static_extent<Shape>(D) == 1) return cs::integral_constant<index_type, 0>{};
        else                                          return cs::full_extent;
    }
    template <class P, cs::size_t... D>
    _TNY_API auto _squeeze_all(P p, cs::index_sequence<D...>) const noexcept
    { return _slice_range(p, cs::make_index_sequence<rank()>{}, _sq_arg<D>()...); }
    // Every named (already-normalised) axis is size-1 as far as the TYPE knows — a
    // DYNAMIC extent passes here and is checked at run time by the per-axis squeeze
    // the multi-axis fold calls. An out-of-range axis short-circuits (its own
    // static_assert reports it) so `static_extent` is never asked for a bad rank.
    template <cs::size_t... A> static _TNY_API constexpr bool _all_extent1() noexcept
    { return ((A >= rank() || _shape_static_extent<Shape>(A) == cs::dynamic_extent || _shape_static_extent<Shape>(A) == 1) && ...); }
public:
    /** @brief Drop a size-1 axis `Ax` (negatives wrap) -> a rank-(N-1) view.
     *         `squeeze()` (no axis) drops EVERY statically-size-1 axis. */
    template <long Ax = _ax_all>
    _TNY_API auto squeeze() noexcept {
        if constexpr (Ax == _ax_all) return _squeeze_all(store_.data(), cs::make_index_sequence<rank()>{});
        else { constexpr cs::size_t A = _norm_axis(Ax, rank()); static_assert(A < rank() && rank() > 0, "squeeze: axis out of range");
               static_assert(_shape_static_extent<Shape>(A) == cs::dynamic_extent || _shape_static_extent<Shape>(A) == 1,
                             "squeeze: axis must have extent 1");
               _TNY_CHECK(extent(A) == index_type(1), "squeeze: axis must have extent 1");   // runtime check for a dynamic extent
               return as_tensor<storage_view_of(O)>(_detail::squeeze_md<A>(mdspan(), cs::make_index_sequence<rank() - 1>{})); }
    }
    template <long Ax = _ax_all>
    _TNY_API auto squeeze() const noexcept {
        if constexpr (Ax == _ax_all) return _squeeze_all(store_.data(), cs::make_index_sequence<rank()>{});
        else { constexpr cs::size_t A = _norm_axis(Ax, rank()); static_assert(A < rank() && rank() > 0, "squeeze: axis out of range");
               static_assert(_shape_static_extent<Shape>(A) == cs::dynamic_extent || _shape_static_extent<Shape>(A) == 1,
                             "squeeze: axis must have extent 1");
               _TNY_CHECK(extent(A) == index_type(1), "squeeze: axis must have extent 1");   // runtime check for a dynamic extent
               return as_tensor<storage_view_of(O)>(_detail::squeeze_md<A>(mdspan(), cs::make_index_sequence<rank() - 1>{})); }
    }

    /** @brief Drop SEVERAL size-1 axes at once (numpy `squeeze(axis=(...))`) -> a
     *         rank-(N-k) view. The positions are relative to the **source** rank
     *         (negatives count from the back) and must be distinct (in ANY order —
     *         sorted internally, #275); every named axis must have extent 1
     *         (`static_assert` where the extent is static, `_TNY_CHECK` where it is
     *         dynamic). e.g. a `(1,H,1,W)` view `.squeeze<0,2>()` -> `(H,W)`. Arity
     *         picks this overload; one axis (or none) still means `squeeze<Ax>()`
     *         above. */
    template <long Ax0, long Ax1, long... Rest>
    _TNY_API auto squeeze() noexcept {
        static_assert(_axis_in_range(Ax0, rank()) && _axis_in_range(Ax1, rank()) && (_axis_in_range(Rest, rank()) && ...),
                      "squeeze: axis out of range");
        static_assert(_all_distinct<_norm_axis(Ax0, rank()), _norm_axis(Ax1, rank()), _norm_axis(Rest, rank())...>(),
                      "squeeze: axes must be distinct");
        static_assert(_all_extent1<_norm_axis(Ax0, rank()), _norm_axis(Ax1, rank()), _norm_axis(Rest, rank())...>(),
                      "squeeze: every named axis must have extent 1");
        using Sorted = _sorted_axes<(long)_norm_axis(Ax0, rank()), (long)_norm_axis(Ax1, rank()), (long)_norm_axis(Rest, rank())...>;
        return _detail::_squeeze_sorted<Sorted>(*this, cs::make_index_sequence<2 + sizeof...(Rest)>{});   // each step re-checks a dynamic extent
    }
    template <long Ax0, long Ax1, long... Rest>
    _TNY_API auto squeeze() const noexcept {
        static_assert(_axis_in_range(Ax0, rank()) && _axis_in_range(Ax1, rank()) && (_axis_in_range(Rest, rank()) && ...),
                      "squeeze: axis out of range");
        static_assert(_all_distinct<_norm_axis(Ax0, rank()), _norm_axis(Ax1, rank()), _norm_axis(Rest, rank())...>(),
                      "squeeze: axes must be distinct");
        static_assert(_all_extent1<_norm_axis(Ax0, rank()), _norm_axis(Ax1, rank()), _norm_axis(Rest, rank())...>(),
                      "squeeze: every named axis must have extent 1");
        using Sorted = _sorted_axes<(long)_norm_axis(Ax0, rank()), (long)_norm_axis(Ax1, rank()), (long)_norm_axis(Rest, rank())...>;
        return _detail::_squeeze_sorted<Sorted>(*this, cs::make_index_sequence<2 + sizeof...(Rest)>{});   // each step re-checks a dynamic extent
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
    template <class... I, cs::enable_if_t<(sizeof...(I) > 0) && _all_ic<I...>::value, int> = 0> _TNY_API auto permute(I...)       noexcept { return permute<static_cast<long>(I::value)...>(); }
    template <class... I, cs::enable_if_t<(sizeof...(I) > 0) && _all_ic<I...>::value, int> = 0> _TNY_API auto permute(I...) const noexcept { return permute<static_cast<long>(I::value)...>(); }

    /** @brief Value form: `t.squeeze(axis<0,2>{})` == `t.squeeze<0,2>()`, likewise
     *         `unsqueeze`/`permute`. `squeeze`/`unsqueeze`/`permute` are axis-LIST
     *         ops (like `peel`/`take_along`/the reductions), so — unlike the
     *         single-axis `Int<k>()` form above — they take the `axis<...>` tag: a
     *         single distinct-typed argument, so no `.template` is needed on a
     *         dependent receiver.
     *
     *         An EMPTY list — `axis<>{}` — names no axis, so it is a **no-op**: the
     *         same shape and strides back, as a view (numpy's own rule for an empty
     *         axis tuple, `np.squeeze(a, axis=())` / `np.expand_dims(a, axis=())`;
     *         same identity `_keepdims<>`/`take_along(axis<>{})`/`peel(t, axis<>{})`
     *         already have). It is NOT the same as the no-argument `squeeze()`
     *         (drop EVERY statically-size-1 axis) or `unsqueeze()` (insert at axis
     *         0) — those keep their meanings; only the axis-LIST spelling reads an
     *         empty list as "no axes named" (#369). Generic code that computes an
     *         axis list therefore stays correct when the list comes out empty.
     *
     *         `permute` is the exception, and needs nothing added: it takes a FULL
     *         permutation, so its own `sizeof...(Perm) == rank()` check already
     *         accepts `axis<>{}` for a rank-0 tensor only (a no-op there — the one
     *         permutation of no axes) and rejects it at compile time for any other
     *         rank, rather than silently doing something else. */
    template <long... Axes> _TNY_API auto squeeze(axis<Axes...>)       noexcept
    { if constexpr (sizeof...(Axes) == 0) return view(); else return squeeze<Axes...>(); }
    template <long... Axes> _TNY_API auto squeeze(axis<Axes...>) const noexcept
    { if constexpr (sizeof...(Axes) == 0) return view(); else return squeeze<Axes...>(); }
    template <long... Axes> _TNY_API auto unsqueeze(axis<Axes...>)       noexcept
    { if constexpr (sizeof...(Axes) == 0) return view(); else return unsqueeze<Axes...>(); }
    template <long... Axes> _TNY_API auto unsqueeze(axis<Axes...>) const noexcept
    { if constexpr (sizeof...(Axes) == 0) return view(); else return unsqueeze<Axes...>(); }
    template <long... Axes> _TNY_API auto permute(axis<Axes...>)       noexcept { return permute<Axes...>(); }
    template <long... Axes> _TNY_API auto permute(axis<Axes...>) const noexcept { return permute<Axes...>(); }
    template <class... I, cs::enable_if_t<(sizeof...(I) > 0) && _all_ic<I...>::value, int> = 0> _TNY_API auto reshape(I...)       noexcept { return reshape<static_cast<long>(I::value)...>(); }
    template <class... I, cs::enable_if_t<(sizeof...(I) > 0) && _all_ic<I...>::value, int> = 0> _TNY_API auto reshape(I...) const noexcept { return reshape<static_cast<long>(I::value)...>(); }
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
     * bool `Atomic` flag (default false): when true the write is an
     * atomic-on-device scatter/"push" accumulate — spelled `a.atomic_add_(b)`
     * (see below); `add_<Atomic>`/`sub_<Atomic>` is the underlying form. */
    template <bool Atomic = false, class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & add_(const B & b);
    template <bool Atomic = false, class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & sub_(const B & b);
    template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & mul_(const B & b);
    template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & div_(const B & b);
    template <bool Atomic = false> _TNY_API tensor & add_(T s);
    template <bool Atomic = false> _TNY_API tensor & sub_(T s);
    _TNY_API tensor & mul_(T s);
    _TNY_API tensor & div_(T s);

    /* --- in-place running min/max: *this = min/max(*this, b) --- *
     * tensor rhs broadcasts; a scalar rhs applies to all. For the running-  *
     * min/max update idiom (e.g. best.minimum_(candidate) in a nearest-     *
     * distance search) — mirrors add_/mul_'s tensor-or-scalar rhs shape. */
    template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & minimum_(const B & b);
    template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & maximum_(const B & b);
    _TNY_API tensor & minimum_(T s);
    _TNY_API tensor & maximum_(T s);

    /* --- fused scaled accumulate (BLAS axpy): *this += alpha*b / *this -= *
     * alpha*b; the tensor rhs `b` broadcasts. `y.add_(x, a)` is axpy; a       *
     * scaled copy `y = a*x` is `y.zero_().add_(x, a)`. --------------------- */
    template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & add_(const B & b, T alpha);
    template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & sub_(const B & b, T alpha);

    /* --- atomic accumulate aliases: readable spelling of the atomic scatter *
     * "push" write. `atomic_add_(x)` == `add_<true>(x)`, `atomic_sub_(x)` == *
     * `sub_<true>(x)` — atomic on both host and device (#257; a delta        *
     * commit, not a read-modify-write). Both a broadcasting tensor rhs and   *
     * a scalar rhs, mirroring the add_/sub_ overloads. Works on a rank-0     *
     * at(i...) result, so `a.at(i,j).atomic_add_(v)` is the scatter-         *
     * accumulate idiom. The underlying form is add_<Atomic>/sub_<Atomic>. */
    template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & atomic_add_(const B & b);
    template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API tensor & atomic_sub_(const B & b);
    _TNY_API tensor & atomic_add_(T s);
    _TNY_API tensor & atomic_sub_(T s);

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
    /* --- ... into a caller-provided destination (one fused pass, no alloc) -> dest& --- */
    template <class B, class D> _TNY_API auto & add(const B & b, into_t<D> out) const;
    template <class B, class D> _TNY_API auto & sub(const B & b, into_t<D> out) const;
    template <class B, class D> _TNY_API auto & mul(const B & b, into_t<D> out) const;
    template <class B, class D> _TNY_API auto & div(const B & b, into_t<D> out) const;
    template <class B, class D> _TNY_API auto & pow(const B & b, into_t<D> out) const;
    /* --- fused out-of-place axpy: a + alpha*b / a - alpha*b (b tensor, broadcasts);
     *     the in-place twin is add_(b, alpha). `alpha` is arithmetic and `into_t` a
     *     distinct type, so the scalar and the destination never collide. --------- */
    template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API auto add(const B & b, T alpha) const;
    template <class B, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API auto sub(const B & b, T alpha) const;
    template <class B, class D, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API auto & add(const B & b, T alpha, into_t<D> out) const;
    template <class B, class D, cs::enable_if_t<!cs::is_arithmetic<B>::value, int> = 0> _TNY_API auto & sub(const B & b, T alpha, into_t<D> out) const;

    /* --- out-of-place unary math AS METHODS: a.exp(), a.sqrt(), … -> new tensor,
     *     or a.exp(into(y)) -> y&. The free forms (exp(a)) and in-place (a.exp_())
     *     still exist; these give method parity with a.add(b). ------------------- */
#define _TNY_OOP_UNARY_DECL(NAME) \
    _TNY_API auto NAME() const;   \
    template <class D> _TNY_API auto & NAME(into_t<D> out) const;
    _TNY_OOP_UNARY_DECL(neg)  _TNY_OOP_UNARY_DECL(abs)   _TNY_OOP_UNARY_DECL(exp)
    _TNY_OOP_UNARY_DECL(log)  _TNY_OOP_UNARY_DECL(sin)   _TNY_OOP_UNARY_DECL(cos)
    _TNY_OOP_UNARY_DECL(sqrt) _TNY_OOP_UNARY_DECL(tanh)  _TNY_OOP_UNARY_DECL(floor)
    _TNY_OOP_UNARY_DECL(ceil) _TNY_OOP_UNARY_DECL(round) _TNY_OOP_UNARY_DECL(trunc)
    _TNY_OOP_UNARY_DECL(sign)
#undef _TNY_OOP_UNARY_DECL

    /* --- min/max/clamp/normalize/cross AS METHODS (free forms exist too) --- */
    template <class B> _TNY_API auto minimum(const B & b) const;                          // tensor (broadcasts) or scalar rhs
    template <class B> _TNY_API auto maximum(const B & b) const;
    template <class B, class D> _TNY_API auto & minimum(const B & b, into_t<D> out) const;
    template <class B, class D> _TNY_API auto & maximum(const B & b, into_t<D> out) const;
    _TNY_API auto clamp(T lo, T hi) const;
    template <class D> _TNY_API auto & clamp(T lo, T hi, into_t<D> out) const;
    _TNY_API auto normalize() const;                                                      // unit vector (a.normalize_() is in place)
    template <class D> _TNY_API auto & normalize(into_t<D> out) const;
    template <class Tb,class Eb,class Lb,storage Ob> _TNY_API auto cross(const tensor<Tb,Eb,Lb,Ob> & b) const;   // 3D (a.cross_(b) is in place)
    template <class Tb,class Eb,class Lb,storage Ob, class D> _TNY_API auto & cross(const tensor<Tb,Eb,Lb,Ob> & b, into_t<D> out) const;

    /* --- generic elementwise with a user functor (device-safe) ---- */
    // `f` is applied per element; its APPLICATION ORDER is unspecified (a dense view is
    // walked in physical, not logical, order — a transposed in-place map_ vectorizes).
    // Use a pure functor; don't rely on a stateful `f` observing a particular order.
    template <class F> _TNY_API tensor & map_(F f);                    // *this = f(*this)
    template <class G, class B> _TNY_API tensor & zip_with_(G g, const B & b);  // *this = g(*this, b) (broadcasts)
    template <class F> _TNY_API auto map(F f) const;                   // -> new tensor = f(*this)

    /* --- boolean reductions (numpy-style; `all` is the slice keyword, so
     *     these are members, and chain after a comparison: (a<b).all()) ---- */
    _TNY_API bool all() const;   // true iff every element is nonzero
    _TNY_API bool any() const;   // true iff any element is nonzero

    /* --- reductions as methods (parity with the free sum(a)/mean(a)/norm(a)/...):
     *     thin forwarders to the free forms, DEFINED in math.h. Same overload
     *     shapes as the free functions — full (all axes; leading TYPE =
     *     accumulator), explicit axis (`m.sum<0>()`), and a generic trailing
     *     keyword bag (`m.sum(axis<0>{})`, `m.sum(dtype<double>{}, into(d))`,
     *     `m.sum<0>(keepdims)`, ... any subset/order of `dtype`/`axis`/`into`/
     *     `keepdims` — see math.h's `_TNY_RED_TAGGED`). */
// The AXIS forms allocate their (lower-rank) result, so — exactly like the free
// axis reductions in math.h — each is TWO overloads keyed on whether that result
// is fully static: static -> stack, host+device (_TNY_API); dynamic -> heap, host
// only (_TNY_HOST). A single _TNY_API forwarder would make a __host__ __device__
// method call the __host__-only free overload for a dynamic shape. The FULL
// reductions never allocate, so they stay single _TNY_API overloads. The
// tag-driven form needs the SAME split, keyed off `_md::_red_dyn` (computed from
// whichever `axis<...>` tag -- if any -- turns up in the trailing bag) instead of
// an explicit `Ax...` pack. `CMP` is `==` (static -> _TNY_API) or `!=` (dynamic ->
// _TNY_HOST). The out-of-line definitions in math.h repeat both keys (without the
// `= 0` default).
#define _TNY_RED_AXIS_IF(E, CMP)                                                                            \
    cs::enable_if_t<(sizeof...(Ax) > 0) && _md::reduced_extents<E,Ax...>::rank_dynamic() CMP 0, int> = 0
#define _TNY_RED_TAGGED_IF(E, CMP)                                                                          \
    class AxisTag = _kw::find_t<_is_axis_tag, axis<>, Tag0, Tags...>,                                       \
    cs::enable_if_t<_md::_red_dyn<E,AxisTag>::value CMP 0, int> = 0
#define _TNY_RED_METHOD_DECL(NAME)                                                                          \
    template <class Acc = void> _TNY_API auto NAME() const;                                                 \
    template <long... Ax, class... Tags, _TNY_RED_AXIS_IF(Shape, ==)> _TNY_API  decltype(auto) NAME(Tags... tags) const; \
    template <long... Ax, class... Tags, _TNY_RED_AXIS_IF(Shape, !=)> _TNY_HOST decltype(auto) NAME(Tags... tags) const; \
    template <class Acc, long... Ax, class... Tags, _TNY_RED_AXIS_IF(Shape, ==)> _TNY_API  decltype(auto) NAME(Tags... tags) const; \
    template <class Acc, long... Ax, class... Tags, _TNY_RED_AXIS_IF(Shape, !=)> _TNY_HOST decltype(auto) NAME(Tags... tags) const; \
    template <class Acc = void, class Tag0, class... Tags, _TNY_RED_TAGGED_IF(Shape, ==)> _TNY_API  decltype(auto) NAME(Tag0 tag0, Tags... tags) const; \
    template <class Acc = void, class Tag0, class... Tags, _TNY_RED_TAGGED_IF(Shape, !=)> _TNY_HOST decltype(auto) NAME(Tag0 tag0, Tags... tags) const;
    _TNY_RED_METHOD_DECL(sum)    _TNY_RED_METHOD_DECL(prod)  _TNY_RED_METHOD_DECL(max)
    _TNY_RED_METHOD_DECL(min)    _TNY_RED_METHOD_DECL(mean)  _TNY_RED_METHOD_DECL(sqnorm)
    _TNY_RED_METHOD_DECL(norm)
#undef _TNY_RED_METHOD_DECL
#undef _TNY_RED_TAGGED_IF
#undef _TNY_RED_AXIS_IF
    // dot is binary (no axis form): m.dot(b) / m.dot<Acc>(b) / m.dot(b, dtype<Acc>{},
    // into(cell)) — a generic trailing keyword bag (dtype/into, any subset/order),
    // requiring at least one tag so it never competes with the bare form above.
    template <class Acc = void, class Tb,class Eb,class Lb,storage Ob>
    _TNY_API auto dot(const tensor<Tb,Eb,Lb,Ob> & b) const;
    template <class Acc = void, class Tb,class Eb,class Lb,storage Ob, class Tag0, class... Tags>
    _TNY_API decltype(auto) dot(const tensor<Tb,Eb,Lb,Ob> & b, Tag0 tag0, Tags... tags) const;
    // sqdist/dist: same binary (no axis) shape as dot.
    template <class Acc = void, class Tb,class Eb,class Lb,storage Ob>
    _TNY_API auto sqdist(const tensor<Tb,Eb,Lb,Ob> & b) const;
    template <class Acc = void, class Tb,class Eb,class Lb,storage Ob, class Tag0, class... Tags>
    _TNY_API decltype(auto) sqdist(const tensor<Tb,Eb,Lb,Ob> & b, Tag0 tag0, Tags... tags) const;
    template <class Acc = void, class Tb,class Eb,class Lb,storage Ob>
    _TNY_API auto dist(const tensor<Tb,Eb,Lb,Ob> & b) const;
    template <class Acc = void, class Tb,class Eb,class Lb,storage Ob, class Tag0, class... Tags>
    _TNY_API decltype(auto) dist(const tensor<Tb,Eb,Lb,Ob> & b, Tag0 tag0, Tags... tags) const;

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
    _TNY_API tensor & normalize_();            // *this /= norm(*this)  (L2; floating element types)
    template <long... Axes> _TNY_API tensor & normalize_();   // ...over the named axes (keepdim); axes distinct & ascending
    template <class Tb, class Eb, class Lb, storage Ob>
    _TNY_API tensor & cross_(const tensor<Tb,Eb,Lb,Ob> & b);   // *this = (*this) × b  (3D; rank-1, length 3)

    /* --- increment / decrement --------------------------------------- *
     * Prefix ++/-- mutate in place (add/subtract 1 from every element).
     * Postfix returns the pre-value, so it must allocate a copy -> only a
     * STATIC shape (stack copy, host+device); a dynamic shape has no postfix. */
    _TNY_API tensor & operator++() { return add_(T(1)); }
    _TNY_API tensor & operator--() { return sub_(T(1)); }
    template <bool S = is_static, cs::enable_if_t<S, int> = 0>
    _TNY_API tensor<T, Shape, ccontiguous, storage::stack> operator++(int) { auto old = clone(); add_(T(1)); return old; }
    template <bool S = is_static, cs::enable_if_t<S, int> = 0>
    _TNY_API tensor<T, Shape, ccontiguous, storage::stack> operator--(int) { auto old = clone(); sub_(T(1)); return old; }
};

/** @brief `into(y)` — the output-destination tag: pass it as the last argument to
 *         an out-of-place math producer (`a.add(b, into(y))`, `cross(a,b,into(y))`,
 *         `exp(a, into(y))`, …) to write the result into `y` (one fused pass, no
 *         allocation) and get `y&` back, instead of a freshly allocated result.
 *         `y`'s dtype need not match: the arithmetic runs in the OPERANDS' own
 *         precision (a scalar rhs and the fused `alpha` included) and only the
 *         RESULT is cast to `y`, so `a.op(b, into(y))` gives exactly the numbers
 *         `y.copy_(a.op(b))` would (#379) — including for a `half`/`bfloat16`
 *         operand, where `into(y)` rounds through the twin's own `promote_t`
 *         (a 16-bit float there) before casting to `y`, not straight from the
 *         float compute value. `scan(t, init, f, into(y))` is the one
 *         deliberate exception — see its own doc-comment in iterate.h. */
template <class T, class E, class L, storage O>
_TNY_API into_t<tensor<T,E,L,O>> into(tensor<T,E,L,O> & d) noexcept { return into_t<tensor<T,E,L,O>>{ d }; }

/** @brief `into(y)` over a TEMPORARY **view** — the destination may be written
 *         straight out of a view-producing op, with no named intermediate:
 *         `cross(a, b, into(N(i, all)))`, `sum(a, into(cells.at(i, j)))`,
 *         `x.add(y, into(z.permute<1,0>()))`. Every view-producing op (slicing,
 *         `at`, `permute`, `unsqueeze`, `take_along`, `peel_at`, …) returns its
 *         view BY VALUE, so without this overload the most natural destination
 *         there is — a slot of a bigger output — had to be given a name first,
 *         which is exactly the boilerplate `into(dest)` exists to remove.
 *
 *         Restricted to the non-owning VIEW storages (`view`/`gpu_view`/
 *         `pinned_view`/`mapped_view`): a temporary view aliases backing storage
 *         the caller owns elsewhere, so the write outlives the call, and the view
 *         itself lives to the end of the full expression that contains the
 *         producer. A temporary OWNING tensor (`into(zeros<double>(shape<3>{}))`,
 *         `into(local<double,shape<3>>{})`) is rejected instead: its storage dies
 *         with the expression, so the result would be computed and thrown away.
 *
 *         The one sharp edge: use the call for its EFFECT, don't keep the `dest&`
 *         it returns — `auto & r = cross(a, b, into(N(i, all)))` dangles once the
 *         temporary view goes away (same rule as `for (auto v : peel<0>(t))` and
 *         the other temporaries in the library). */
template <class T, class E, class L, storage O>
_TNY_API into_t<tensor<T,E,L,O>> into(tensor<T,E,L,O> && d) noexcept {
    static_assert(storage_is_view(O),
        "into(dest): a TEMPORARY destination must be a non-owning VIEW (a slice / at() / "
        "permute / peel cell of storage owned elsewhere). A temporary OWNING tensor dies "
        "with the expression, so the result would be discarded -- name the destination "
        "first, then pass into(that).");
    return into_t<tensor<T,E,L,O>>{ d };
}

/** @brief Free forms of `reindex`/`index_fits` — deduce the tensor, so a
 *         type-dependent receiver avoids `.template`: `reindex<int32_t>(t)`,
 *         `index_fits<int32_t>(t)`. (`Idx2` is a TYPE, so there is no value form.) */
template <class Idx2, class T, class E, class L, storage O>
_TNY_API auto reindex(tensor<T,E,L,O> & t)       { return t.template reindex<Idx2>(); }
template <class Idx2, class T, class E, class L, storage O>
_TNY_API auto reindex(const tensor<T,E,L,O> & t) { return t.template reindex<Idx2>(); }
template <class Idx2, class T, class E, class L, storage O>
_TNY_API bool index_fits(const tensor<T,E,L,O> & t) { return t.template index_fits<Idx2>(); }

/* ------------------------------------------------------------------ *
 *     Factories                                                      *
 * ------------------------------------------------------------------ */

/** @brief Wrap `p` as a non-owning view with a contiguous layout (default
 *         C-order). This is the factory; the `view<T,E>` alias is the type it
 *         produces, and the member `t.view()` re-views an existing tensor.
 *
 *         MEMORY SPACE: `p` is a **host** pointer unless a trailing `storage_c<Space>{}`
 *         (or `storage_v<Space>`) tag names where it lives — pass the plain BACKEND the
 *         memory is in (`storage::gpu` for a device pointer, `storage::pinned`/`storage::mapped`
 *         for page-locked host memory). Since `wrap` always yields a VIEW, the space
 *         folds to its view kind (`gpu -> gpu_view`, …) via `storage_view_of` — you
 *         never spell the `_view` kinds. Symmetric with `as_anyrank<Space>` /
 *         `from_dlpack<T,Space>`.
 *
 *         `storage::heap`/`storage::stack` name no distinct memory space (they are
 *         *ownership* kinds, not backends), so passing one here just folds to a plain
 *         `storage::view`, same as leaving the tag off — it does NOT make `wrap` return
 *         an owning tensor. For an owning heap/stack tensor, copy into one with
 *         `empty<T, storage::heap>(e)`/`make_heap<T>(e)` instead.
 *
 *         The trailing argument is a keyword-tag bag (#277/#282), not a fixed
 *         `storage_c<Space>` parameter — today the only recognised keyword is
 *         `storage_c`/`storage_v`, but a future keyword (e.g. a `stream` tag) lands
 *         on all four `wrap` positional forms without touching any of them again. */
template <class Layout = ccontiguous, storage Space = storage_deduce, class T, class Shape, class... Tags>
_TNY_API auto wrap(T * p, Shape e, Tags... /*tags*/) {
    using ok = _kw::accepts<_is_storage_tag>;
    static_assert(ok::known<Tags...>(), "wrap(): unrecognised trailing argument — expected storage_c<...>{}/storage_v<...>");
    static_assert(ok::unique<Tags...>(), "wrap(): the same keyword was given twice");
    constexpr storage R = storage_arg<Space, storage::view, Tags...>();
    using Tn = tensor<T, Shape, Layout, storage_view_of(R)>;
    return Tn(p, typename Tn::mapping_type(e));
}

/** @brief Value-tag layout form: `wrap(p, e, fcontiguous{})` == `wrap<fcontiguous>(p, e)` —
 *  deduces the layout from a bare `ccontiguous{}`/`fcontiguous{}` argument instead of an
 *  explicit `<Layout>` template argument, so a type-dependent receiver needs no
 *  `.template`. Composes with a trailing `storage_c<Space>{}` exactly like the template
 *  form. (`strides<S...>{}` keeps its own dedicated overload above — it carries the
 *  static strides themselves, not just a layout kind, so it is not a `Layout` here.)
 *  A SECOND layout tag after this one — `wrap(p, e, fcontiguous{}, fcontiguous{})`,
 *  agreeing or not — is a `static_assert` (#394): `Layout` is a single positional slot,
 *  not a composable keyword, so it can only be given once per call. */
template <class Layout, storage Space = storage_deduce, class T, class Shape, class... Tags,
          cs::enable_if_t<cs::is_same<Layout, ccontiguous>::value || cs::is_same<Layout, fcontiguous>::value, int> = 0>
_TNY_API auto wrap(T * p, Shape e, Layout, Tags... /*tags*/) {
    static_assert(!_kw::has<_is_layout_tag, Tags...>(),
        "wrap(): a layout tag was already given positionally — remove the duplicate");
    using ok = _kw::accepts<_is_storage_tag>;
    static_assert(ok::known<Tags...>(), "wrap(): unrecognised trailing argument — expected storage_c<...>{}/storage_v<...>");
    static_assert(ok::unique<Tags...>(), "wrap(): the same keyword was given twice");
    constexpr storage R = storage_arg<Space, storage::view, Tags...>();
    using Tn = tensor<T, Shape, Layout, storage_view_of(R)>;
    return Tn(p, typename Tn::mapping_type(e));
}

/** @brief Does `MD` look like an `mdspan`? (has `data_handle()`/`mapping()` and the
 *         `element_type`/`extents_type`/`layout_type` typedefs `as_tensor` reads).
 *         Constrains `wrap(mdspan)` so a 1-argument `wrap(x)` on something that is
 *         NOT an mdspan — `wrap(some_teeny_tensor)`, say — fails as a clean "no
 *         matching function for call to wrap(...)" at the call site instead of an
 *         unrelated-looking error deep inside `as_tensor` (#370). */
template <class MD, class = void>
struct _is_mdspan_like : cs::false_type {};
template <class MD>
struct _is_mdspan_like<MD, cs::void_t<decltype(cs::declval<const MD &>().data_handle()),
                                      decltype(cs::declval<const MD &>().mapping()),
                                      typename MD::element_type,
                                      typename MD::extents_type,
                                      typename MD::layout_type>> : cs::true_type {};

/** @brief `wrap(mdspan)` — a spelling of `as_tensor(mdspan)` under the one factory
 *  name users already reach for. Wrap any `cuda::std::mdspan`/`submdspan` result as
 *  a non-owning view; the element type, extents and layout all come from the mdspan.
 *
 *  MEMORY SPACE: same contract as the pointer forms — the mdspan wraps a **host**
 *  pointer unless a trailing `storage_c<Space>{}` (or `storage_v<Space>`) tag names
 *  where it lives, and since `wrap` always yields a VIEW the plain backend folds to
 *  its view kind (`storage::gpu -> gpu_view`, …) via `storage_view_of`, so you never
 *  spell the `_view` kinds: `wrap(md, storage_v<storage::gpu>)` is a `gpu_view`.
 *  Takes the same trailing keyword-tag bag as the four positional forms (#282/#370).
 *
 *  NB the explicit template argument of THIS overload is the memory SPACE (an
 *  `storage` value: `wrap<storage::gpu>(md)`), not a layout **type** as in
 *  `wrap<fcontiguous>(p, e)` — the layout is already carried by the mdspan, so
 *  there is nothing to name. Prefer the value-tag spelling above, which reads the
 *  same on every `wrap` form.
 *
 *  `as_tensor` stays available (it is what teeny's own view-producing ops —
 *  `permute`/`flip`/`squeeze`/… — call internally, with a pre-folded space). */
template <storage Space = storage_deduce, class MD, class... Tags,
          cs::enable_if_t<_is_mdspan_like<MD>::value, int> = 0>
_TNY_API auto wrap(const MD & md, Tags... /*tags*/) {
    using ok = _kw::accepts<_is_storage_tag>;
    static_assert(ok::known<Tags...>(), "wrap(): unrecognised trailing argument — expected storage_c<...>{}/storage_v<...>");
    static_assert(ok::unique<Tags...>(), "wrap(): the same keyword was given twice");
    constexpr storage R = storage_arg<Space, storage::view, Tags...>();
    return as_tensor<storage_view_of(R)>(md);
}

/** @brief Wrap `p` as a non-owning view with explicit **runtime strides** (a
 *         `layout_stride` view). Pass one stride per dimension — an `array` or a
 *         braced list — in ELEMENTS; strides may be negative (a reversed view).
 *
 *         `wrap(p, shape<2,3>{}, {3, 1})` is the row-major view; `{1, 2}` the
 *         column-major one. For strides known at compile time pass a
 *         `strides<S...>{}` instead (overload below) so they fold into the type.
 *         A trailing `storage_c<Space>{}` tags the memory space (default host; the plain
 *         backend folds to its view kind, e.g. `storage::gpu -> gpu_view`).
 *
 *         `wrap` TRUSTS the strides you give it: a **stride 0** (or a stride smaller
 *         than an inner extent) makes a SELF-OVERLAPPING view where several indices
 *         alias one element. Reading such a view is fine (that is how a broadcast
 *         works), but an **in-place write** into it (`v.add_(b)`, `v.iota_(...)`)
 *         applies the update to the same element repeatedly — a host-debug check
 *         rejects an in-place write whose destination has an `extent > 1` axis with
 *         stride 0. `clone()` to a dense tensor first if you need to write. */
template <storage Space = storage_deduce, class T, class Shape, class... Tags>
_TNY_API auto wrap(T * p, Shape e, cs::array<typename Shape::index_type, _shape_rank<Shape>()> st, Tags... /*tags*/) {
    using ok = _kw::accepts<_is_storage_tag>;
    static_assert(ok::known<Tags...>(), "wrap(): unrecognised trailing argument — expected storage_c<...>{}/storage_v<...>");
    static_assert(ok::unique<Tags...>(), "wrap(): the same keyword was given twice");
    constexpr storage R = storage_arg<Space, storage::view, Tags...>();
    using Tn = tensor<T, Shape, cs::layout_stride, storage_view_of(R)>;
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
template <cs::int64_t... Strides, storage Space = storage_deduce, class T, class Shape, class... Tags>
_TNY_API auto wrap(T * p, Shape e, strides<Strides...>, Tags... /*tags*/) {
    static_assert(strides<Strides...>::all_static(),
        "wrap(ptr, shape, strides<...>{}): a strides<> tag carries only COMPILE-TIME "
        "strides; for mixed strides use wrap<S...>(ptr, shape, {runtime slots}), or "
        "for all-runtime strides pass the values as `{s0, s1, ...}`");
    using ok = _kw::accepts<_is_storage_tag>;
    static_assert(ok::known<Tags...>(), "wrap(): unrecognised trailing argument — expected storage_c<...>{}/storage_v<...>");
    static_assert(ok::unique<Tags...>(), "wrap(): the same keyword was given twice");
    constexpr storage R = storage_arg<Space, storage::view, Tags...>();
    using Tn = tensor<T, Shape, strides<Strides...>, storage_view_of(R)>;
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
 *         The static slots fold into the type; only the runtime ones are stored.
 *         A trailing `storage_c<Space>{}` tags the memory space (default host; the plain
 *         backend folds to its view kind, e.g. `storage::gpu -> gpu_view`). */
template <cs::int64_t S0, cs::int64_t... Srest, storage Space = storage_deduce, class T, class Shape, class... Tags>   // S0 forces explicit <...>
_TNY_API auto wrap(T * p, Shape e, cs::array<typename Shape::index_type, strides<S0, Srest...>::ndyn()> dyn, Tags... /*tags*/) {
    using ok = _kw::accepts<_is_storage_tag>;
    static_assert(ok::known<Tags...>(), "wrap(): unrecognised trailing argument — expected storage_c<...>{}/storage_v<...>");
    static_assert(ok::unique<Tags...>(), "wrap(): the same keyword was given twice");
    constexpr storage R = storage_arg<Space, storage::view, Tags...>();
    using Tn = tensor<T, Shape, strides<S0, Srest...>, storage_view_of(R)>;
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
using view = tensor<T, Shape, Layout, storage::view>;

/** @brief Stack-owned tensor (fully static shape). Use `local<T,E>{}`. */
template <class T, class Shape, class Layout = ccontiguous>
using local = tensor<T, Shape, Layout, storage::stack>;

/** @brief Heap-owned tensor (host only, move-only). Use `owned<T,E>(extents)`. */
template <class T, class Shape, class Layout = ccontiguous>
using owned = tensor<T, Shape, Layout, storage::heap>;

/* --- functional factories (deduce the Shape type from the argument) ------ *
 * Complements the type aliases above; the `make_` prefix keeps them distinct.
 * Element type `T` is explicit (it can't be deduced from a shape); the extents
 * type is deduced, so a runtime-built shape needs no `decltype` spelling.       */

/** @brief `make_view<L>(ptr, extents)` — a non-owning view (alias of `wrap`).
 *         The layout may be an explicit `<L>` template argument or, like `wrap`,
 *         a positional value tag (`make_view(p, e, fcontiguous{})`) — see the
 *         overload below. Takes the same optional trailing keyword-tag bag as
 *         `wrap` (#282) — today just `storage_c<Space>{}`/`storage_v<Space>`. */
template <class Layout = ccontiguous, storage Space = storage_deduce, class T, class Shape, class... Tags>
_TNY_API auto make_view(T * p, Shape e, Tags... tags) { return wrap<Layout, Space>(p, e, tags...); }

/** @brief Value-tag layout form, mirroring `wrap`'s (#374):
 *  `make_view(p, e, fcontiguous{})` == `make_view<fcontiguous>(p, e)`, deduced from a
 *  bare `ccontiguous{}`/`fcontiguous{}` argument so a type-dependent receiver needs no
 *  `.template`. Composes with a trailing `storage_c<Space>{}` exactly like the template
 *  form. Without this overload only `ccontiguous{}` would work — it would reach `wrap`'s
 *  own positional layout overload by accident, because `make_view`'s `Layout` *defaults*
 *  to `ccontiguous` — while `fcontiguous{}` fell through to the keyword bag and was
 *  rejected as an unrecognised trailing argument.
 *  A SECOND layout tag — `make_view(p, e, fcontiguous{}, fcontiguous{})` — is a
 *  `static_assert` (#394), same as `wrap`'s. */
template <class Layout, storage Space = storage_deduce, class T, class Shape, class... Tags,
          cs::enable_if_t<cs::is_same<Layout, ccontiguous>::value || cs::is_same<Layout, fcontiguous>::value, int> = 0>
_TNY_API auto make_view(T * p, Shape e, Layout, Tags... tags) {
    // Checked HERE, before forwarding, not left to wrap()'s own assert: a second
    // layout tag lives in THIS pack right now, but wrap<Layout, Space>(p, e, tags...)
    // forwards Layout as an explicit template argument, and wrap's positional-layout
    // overload then re-consumes a lone matching tag through its own required Layout
    // parameter (partial ordering prefers that fixed parameter over its Tags... pack)
    // rather than through Tags — so wrap's assert never sees it there (#394).
    static_assert(!_kw::has<_is_layout_tag, Tags...>(),
        "make_view(): a layout tag was already given positionally — remove the duplicate");
    return wrap<Layout, Space>(p, e, tags...);
}

/** @brief `empty<T>(extents)` — a new UNINITIALISED tensor. The one factory the
 *  `make_*` family fuses into: ownership is **deduced** from the shape (fully
 *  static -> `stack` (host+device); any dynamic extent -> `heap` (host)) unless a
 *  backend is named — `empty<T, storage::gpu>(extents)`, or the value-tag spelling
 *  `empty<T>(extents, storage_c<storage::gpu>{})`. `gpu`/`pinned`/`mapped` require
 *  `<teeny/cuda.h>` (their storage lives there). `T` defaults to `float`. Split
 *  by the resolved ownership so the `stack` case stays `_TNY_API` (host+device)
 *  while the allocating cases are `_TNY_HOST`.
 *
 *  Element type, backend, and layout may each be given as a leading explicit
 *  template argument OR as a trailing value tag (`dtype<T>{}`/`storage_c<O>{}`/
 *  a layout tag), in ANY order and ANY subset: `empty<double>(e)`,
 *  `empty(e, dtype<double>{})`, `empty(e, storage_c<storage::gpu>{}, dtype<double>{})`
 *  all work. `_kw::accepts`/`dtype_arg_t`/`storage_arg`/`layout_arg_t` (`kwargs.h`
 *  and each tag's own header) resolve the trailing bag; an unrecognised or
 *  duplicated keyword fails on one clean `static_assert` instead of an
 *  overload-resolution wall (#279/#280). */
template <class T = void, storage O = storage_deduce, class Layout = void, class Shape, class... Tags,
          cs::enable_if_t<storage_resolve(storage_arg<O, storage_deduce, Tags...>(), Shape::rank_dynamic() == 0) == storage::stack, int> = 0>
_TNY_API auto empty(Shape = Shape{}, Tags... /*tags*/) {
    using ok = _kw::accepts<_is_dtype, _is_storage_tag, _is_layout_tag>;
    static_assert(ok::known<Tags...>(), "empty(): unrecognised trailing argument — expected dtype<T>{}, storage_c<...>{} or a layout tag (ccontiguous{}/fcontiguous{})");
    static_assert(ok::unique<Tags...>(), "empty(): the same keyword was given twice");
    using ET = dtype_arg_t<T, float, Tags...>;
    using LO = layout_arg_t<Layout, ccontiguous, Tags...>;
    return tensor<ET, Shape, LO, storage::stack>(_uninit);
}
template <class T = void, storage O = storage_deduce, class Layout = void, class Shape, class... Tags,
          cs::enable_if_t<storage_resolve(storage_arg<O, storage_deduce, Tags...>(), Shape::rank_dynamic() == 0) != storage::stack, int> = 0>
_TNY_HOST auto empty(Shape e, Tags... /*tags*/) {
    using ok = _kw::accepts<_is_dtype, _is_storage_tag, _is_layout_tag>;
    static_assert(ok::known<Tags...>(), "empty(): unrecognised trailing argument — expected dtype<T>{}, storage_c<...>{} or a layout tag (ccontiguous{}/fcontiguous{})");
    static_assert(ok::unique<Tags...>(), "empty(): the same keyword was given twice");
    using ET = dtype_arg_t<T, float, Tags...>;
    constexpr storage R = storage_resolve(storage_arg<O, storage_deduce, Tags...>(), Shape::rank_dynamic() == 0);
    static_assert(!storage_is_view(R), "empty(): a non-owning view kind (view/gpu_view/pinned_view/mapped_view) has no storage to allocate — use wrap()/make_view() for a view.");
    using LO = layout_arg_t<Layout, ccontiguous, Tags...>;
    return tensor<ET, Shape, LO, R>(e, _uninit);
}
/** @brief Legacy forwarder for the one spelling the generic entry point above
 *  cannot cover: a LEADING explicit template argument that means the BACKEND
 *  rather than the element type, because a `dtype<T>{}` tag makes `T` deducible
 *  from the call instead — `empty<storage::pinned>(e, dtype<double>{})`. A
 *  non-variadic overload is more specialised than a variadic one in partial
 *  ordering, so this wins whenever it applies (verified on both compilers with
 *  a standalone probe before this landed); every other spelling falls through
 *  to the generic entry point above. */
template <storage O = storage_deduce, class Layout = ccontiguous, class Shape, class T,
          cs::enable_if_t<storage_resolve(O, Shape::rank_dynamic() == 0) == storage::stack, int> = 0>
_TNY_API auto empty(Shape e, dtype<T>) { return empty<T, O, Layout>(e); }
template <storage O = storage_deduce, class Layout = ccontiguous, class Shape, class T,
          cs::enable_if_t<storage_resolve(O, Shape::rank_dynamic() == 0) != storage::stack, int> = 0>
_TNY_HOST auto empty(Shape e, dtype<T>) { return empty<T, O, Layout>(e); }

/** @brief `make_local<T>(extents)` — a stack-owned tensor (static shape).
 *         `T` defaults to `float` (numpy's default float dtype). Thin spelling of
 *         `empty<T, storage::stack>`. Takes the same trailing `dtype`/layout
 *         keyword-tag bag as `empty` (#282; no `storage_c` — the backend is fixed). */
template <class T = void, class Layout = void, class Shape, class... Tags>
_TNY_API auto make_local(Shape e = Shape{}, Tags... tags) { return empty<T, storage::stack, Layout>(e, tags...); }

/** @brief `make_heap<T>(extents)` — a heap-owned tensor (host, move-only).
 *         `T` defaults to `float`. Thin spelling of `empty<T, storage::heap>`. Takes
 *         the same trailing `dtype`/layout keyword-tag bag as `empty` (#282). */
template <class T = void, class Layout = void, class Shape, class... Tags>
_TNY_HOST auto make_heap(Shape e, Tags... tags) { return empty<T, storage::heap, Layout>(e, tags...); }

/* --- numpy-style creation factories: static shape -> stack (host+device),   *
 *     dynamic shape -> heap (host only), mirroring the out-of-place ops.       */

/** @brief `full(extents, v)` — a new tensor filled with `v`. The element type
 *         defaults to the **value's** type (numpy/pytorch: `full(s, 3)` is int,
 *         `full(s, 3.0)` is float); pass `full<T>(...)` to override. Unlike the
 *         value-less `zeros`/`ones` (which default to `float`), there is a value
 *         here to infer from, so we do.
 *
 *  Ownership is deduced from the shape (static -> stack, dynamic -> heap) unless a
 *  **backend** is named — `full<T, storage::pinned>(s, v)` or the value-tag
 *  `full<T>(s, v, storage_c<storage::pinned>{})`. Because it fills host-side, only
 *  host-accessible backends (stack/heap/pinned/mapped) are allowed; a device
 *  (`gpu`) fill needs a kernel launch, so it is a `static_assert` steering you to
 *  `to<storage::gpu>(full<T>(s, v))`. Split by resolved ownership for the
 *  `_TNY_API`/`_TNY_HOST` annotation.
 *
 *  Element type, backend, and layout may each be given as a leading explicit
 *  template argument OR as a trailing value tag, in ANY order and ANY subset,
 *  same as `empty` (#280/#281): `full(e, v, fcontiguous{})`,
 *  `full(e, v, storage_c<storage::heap>{}, dtype<double>{})`. */
template <class T = void, storage O = storage_deduce, class Layout = void, class Shape, class V, class... Tags,
          cs::enable_if_t<storage_resolve(storage_arg<O, storage_deduce, Tags...>(), Shape::rank_dynamic() == 0) == storage::stack, int> = 0>
_TNY_API auto full(Shape e, V v, Tags... /*tags*/) {
    static_assert(!_kw::is_keyword<V>::value, "full(shape, v, ...): the fill VALUE must come before any keyword tag — "
        "did you mean full(shape, v, dtype<T>{}, ...)?");
    using ok = _kw::accepts<_is_dtype, _is_storage_tag, _is_layout_tag>;
    static_assert(ok::known<Tags...>(), "full(): unrecognised trailing argument — expected dtype<T>{}, storage_c<...>{} or a layout tag (ccontiguous{}/fcontiguous{})");
    static_assert(ok::unique<Tags...>(), "full(): the same keyword was given twice");
    using ET = dtype_arg_t<T, V, Tags...>;
    using LO = layout_arg_t<Layout, ccontiguous, Tags...>;
    auto t = empty<ET, storage::stack, LO>(e); t.fill_(static_cast<ET>(v)); return t;
}
template <class T = void, storage O = storage_deduce, class Layout = void, class Shape, class V, class... Tags,
          cs::enable_if_t<storage_resolve(storage_arg<O, storage_deduce, Tags...>(), Shape::rank_dynamic() == 0) != storage::stack, int> = 0>
_TNY_HOST auto full(Shape e, V v, Tags... /*tags*/) {
    static_assert(!_kw::is_keyword<V>::value, "full(shape, v, ...): the fill VALUE must come before any keyword tag — "
        "did you mean full(shape, v, dtype<T>{}, ...)?");
    using ok = _kw::accepts<_is_dtype, _is_storage_tag, _is_layout_tag>;
    static_assert(ok::known<Tags...>(), "full(): unrecognised trailing argument — expected dtype<T>{}, storage_c<...>{} or a layout tag (ccontiguous{}/fcontiguous{})");
    static_assert(ok::unique<Tags...>(), "full(): the same keyword was given twice");
    using ET = dtype_arg_t<T, V, Tags...>;
    constexpr storage R = storage_resolve(storage_arg<O, storage_deduce, Tags...>(), Shape::rank_dynamic() == 0);
    static_assert(storage_is_host_accessible(R),
        "zeros/ones/full<..., storage::gpu>: a device fill needs a kernel launch; use to<storage::gpu>(full<T>(shape, v)) (or to<storage::gpu>(zeros<T>(shape))), or empty<T, storage::gpu>(shape) then a memset.");
    using LO = layout_arg_t<Layout, ccontiguous, Tags...>;
    auto t = empty<ET, R, LO>(e); t.fill_(static_cast<ET>(v)); return t;
}
/** @brief Legacy forwarder for the one spelling the generic entry point above
 *  cannot cover: a LEADING explicit template argument that means the BACKEND
 *  rather than the element type, because a `dtype<T>{}` tag makes `T` deducible
 *  from the call instead — `full<storage::pinned>(e, v, dtype<double>{})`. See
 *  `empty()`'s twin (tensor.h) for the full explanation; the same partial-
 *  ordering mechanism applies here. */
template <storage O = storage_deduce, class Layout = ccontiguous, class Shape, class V, class T,
          cs::enable_if_t<storage_resolve(O, Shape::rank_dynamic() == 0) == storage::stack, int> = 0>
_TNY_API auto full(Shape e, V v, dtype<T>) { return full<T, O, Layout>(e, v); }
template <storage O = storage_deduce, class Layout = ccontiguous, class Shape, class V, class T,
          cs::enable_if_t<storage_resolve(O, Shape::rank_dynamic() == 0) != storage::stack, int> = 0>
_TNY_HOST auto full(Shape e, V v, dtype<T>) { return full<T, O, Layout>(e, v); }

/** @brief `zeros<T>(extents)` / `ones<T>(extents)` — a new tensor of 0s / 1s.
 *         `T` defaults to `float`. Same ownership deduction, backend selector, and
 *         `_TNY_API`/`_TNY_HOST` split as `full`; also composes `dtype`/`storage_c`/a
 *         layout tag in ANY order/subset, same as `empty` (#280/#281):
 *         `zeros(e, dtype<double>{})`, `zeros(e, fcontiguous{}, storage_c<storage::heap>{})`.
 *         A device backend `static_assert`s — fill via `to<storage::gpu>(zeros<T>(shape))`. */
template <class T = void, storage O = storage_deduce, class Layout = void, class Shape, class... Tags,
          cs::enable_if_t<storage_resolve(storage_arg<O, storage_deduce, Tags...>(), Shape::rank_dynamic() == 0) == storage::stack, int> = 0>
_TNY_API  auto zeros(Shape e, Tags... /*tags*/) {
    using ok = _kw::accepts<_is_dtype, _is_storage_tag, _is_layout_tag>;
    static_assert(ok::known<Tags...>(), "zeros(): unrecognised trailing argument — expected dtype<T>{}, storage_c<...>{} or a layout tag (ccontiguous{}/fcontiguous{})");
    static_assert(ok::unique<Tags...>(), "zeros(): the same keyword was given twice");
    using ET = dtype_arg_t<T, float, Tags...>;
    using LO = layout_arg_t<Layout, ccontiguous, Tags...>;
    return full<ET, storage::stack, LO>(e, ET(0));
}
template <class T = void, storage O = storage_deduce, class Layout = void, class Shape, class... Tags,
          cs::enable_if_t<storage_resolve(storage_arg<O, storage_deduce, Tags...>(), Shape::rank_dynamic() == 0) != storage::stack, int> = 0>
_TNY_HOST auto zeros(Shape e, Tags... /*tags*/) {
    using ok = _kw::accepts<_is_dtype, _is_storage_tag, _is_layout_tag>;
    static_assert(ok::known<Tags...>(), "zeros(): unrecognised trailing argument — expected dtype<T>{}, storage_c<...>{} or a layout tag (ccontiguous{}/fcontiguous{})");
    static_assert(ok::unique<Tags...>(), "zeros(): the same keyword was given twice");
    using ET = dtype_arg_t<T, float, Tags...>;
    constexpr storage R = storage_resolve(storage_arg<O, storage_deduce, Tags...>(), Shape::rank_dynamic() == 0);
    using LO = layout_arg_t<Layout, ccontiguous, Tags...>;
    return full<ET, R, LO>(e, ET(0));
}
/** @brief Legacy forwarder — see `full()`'s twin above. */
template <storage O = storage_deduce, class Layout = ccontiguous, class Shape, class T,
          cs::enable_if_t<storage_resolve(O, Shape::rank_dynamic() == 0) == storage::stack, int> = 0>
_TNY_API  auto zeros(Shape e, dtype<T>) { return zeros<T, O, Layout>(e); }
template <storage O = storage_deduce, class Layout = ccontiguous, class Shape, class T,
          cs::enable_if_t<storage_resolve(O, Shape::rank_dynamic() == 0) != storage::stack, int> = 0>
_TNY_HOST auto zeros(Shape e, dtype<T>) { return zeros<T, O, Layout>(e); }

template <class T = void, storage O = storage_deduce, class Layout = void, class Shape, class... Tags,
          cs::enable_if_t<storage_resolve(storage_arg<O, storage_deduce, Tags...>(), Shape::rank_dynamic() == 0) == storage::stack, int> = 0>
_TNY_API  auto ones(Shape e, Tags... /*tags*/) {
    using ok = _kw::accepts<_is_dtype, _is_storage_tag, _is_layout_tag>;
    static_assert(ok::known<Tags...>(), "ones(): unrecognised trailing argument — expected dtype<T>{}, storage_c<...>{} or a layout tag (ccontiguous{}/fcontiguous{})");
    static_assert(ok::unique<Tags...>(), "ones(): the same keyword was given twice");
    using ET = dtype_arg_t<T, float, Tags...>;
    using LO = layout_arg_t<Layout, ccontiguous, Tags...>;
    return full<ET, storage::stack, LO>(e, ET(1));
}
template <class T = void, storage O = storage_deduce, class Layout = void, class Shape, class... Tags,
          cs::enable_if_t<storage_resolve(storage_arg<O, storage_deduce, Tags...>(), Shape::rank_dynamic() == 0) != storage::stack, int> = 0>
_TNY_HOST auto ones(Shape e, Tags... /*tags*/) {
    using ok = _kw::accepts<_is_dtype, _is_storage_tag, _is_layout_tag>;
    static_assert(ok::known<Tags...>(), "ones(): unrecognised trailing argument — expected dtype<T>{}, storage_c<...>{} or a layout tag (ccontiguous{}/fcontiguous{})");
    static_assert(ok::unique<Tags...>(), "ones(): the same keyword was given twice");
    using ET = dtype_arg_t<T, float, Tags...>;
    constexpr storage R = storage_resolve(storage_arg<O, storage_deduce, Tags...>(), Shape::rank_dynamic() == 0);
    using LO = layout_arg_t<Layout, ccontiguous, Tags...>;
    return full<ET, R, LO>(e, ET(1));
}
/** @brief Legacy forwarder — see `full()`'s twin above. */
template <storage O = storage_deduce, class Layout = ccontiguous, class Shape, class T,
          cs::enable_if_t<storage_resolve(O, Shape::rank_dynamic() == 0) == storage::stack, int> = 0>
_TNY_API  auto ones(Shape e, dtype<T>) { return ones<T, O, Layout>(e); }
template <storage O = storage_deduce, class Layout = ccontiguous, class Shape, class T,
          cs::enable_if_t<storage_resolve(O, Shape::rank_dynamic() == 0) != storage::stack, int> = 0>
_TNY_HOST auto ones(Shape e, dtype<T>) { return ones<T, O, Layout>(e); }

/** @brief `arange<T>(n)` — a 1-D tensor `[0, 1, ..., n-1]` (heap, host). `T`
 *         defaults to `int64_t` (an integer range, like numpy `arange(n)`). A
 *         host-accessible backend may be named — `arange<T, storage::pinned>(n)` or
 *         `arange<T>(n, storage_c<storage::pinned>{})`; a device backend `static_assert`s
 *         (use `to<storage::gpu>(arange<T>(n))`). The static-N forms below stay stack.
 *         `T`/backend compose via the generic keyword mechanism too (#280/#281):
 *         `arange(n, dtype<double>{}, storage_c<storage::pinned>{})`, either order. No
 *         layout keyword — a 1-D tensor has no C/F distinction. */
template <class T = void, storage O = storage_deduce, class... Tags>
_TNY_HOST auto arange(long n, Tags... /*tags*/) {
    using ok = _kw::accepts<_is_dtype, _is_storage_tag>;
    static_assert(ok::known<Tags...>(), "arange(): unrecognised trailing argument — expected dtype<T>{} or storage_c<...>{}");
    static_assert(ok::unique<Tags...>(), "arange(): the same keyword was given twice");
    using ET = dtype_arg_t<T, cs::int64_t, Tags...>;
    using E = cs::dextents<cs::int64_t, 1>;
    constexpr storage R = storage_resolve(storage_arg<O, storage_deduce, Tags...>(), false);
    static_assert(storage_is_host_accessible(R),
        "arange<..., storage::gpu>: a device fill needs a kernel launch; use to<storage::gpu>(arange<T>(n)).");
    auto t = empty<ET, R, ccontiguous>(E{n}); t.iota_(); return t;
}
/** @brief Legacy forwarder — see `full()`'s twin above (the analogous "leading
 *  O when a dtype tag is present" spelling: `arange<storage::pinned>(n, dtype<double>{})`). */
template <storage O = storage_deduce, class T>
_TNY_HOST auto arange(long n, dtype<T>) { return arange<T, O>(n); }
/** @brief Static `arange<T, N>()` — a stack `[0..N-1]` (host+device, folds). */
template <class T = cs::int64_t, long N>
_TNY_API auto arange() { tensor<T, cs::extents<cs::int64_t, static_cast<cs::size_t>(N)>, ccontiguous, storage::stack> t{}; t.iota_(); return t; }
/** @brief `arange<T>(Int<N>())` — the static form spelled with a static integer. */
template <class T = cs::int64_t, class V, V N>
_TNY_API auto arange(cs::integral_constant<V, N>) { return arange<T, static_cast<long>(N)>(); }

/** @brief Wrap any `cuda::std::mdspan` (e.g. a `submdspan` result) as a
 *         non-owning `tny::tensor` view, so the tensor API applies to it. */
template <storage OW, class MD>
_TNY_API tensor<typename MD::element_type, typename MD::extents_type,
                typename MD::layout_type, OW>
as_tensor(const MD & m) {
    using Tn = tensor<typename MD::element_type, typename MD::extents_type,
                      typename MD::layout_type, OW>;
    return Tn(m.data_handle(), m.mapping());
}

_TNY_NAMESPACE_END(tny)

#endif // TNY_TENSOR_H
