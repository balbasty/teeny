#ifndef TNY_MD_LAYOUT
#define TNY_MD_LAYOUT
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/array>
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>
#include <teeny/defines.h>
#include <teeny/alias.h>
#include <teeny/kwargs.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

// MSVC two-phase-lookup / private-inheritance workaround (#294, generalized by
// #315): MSVC can mis-resolve, or misattribute to an unrelated type's private
// EBO base, a qualified `E::rank()`/`E::static_extent(d)` call made directly
// inside a class that has a private multi-inheritance EBO base (like
// `strides<...>::mapping`, right below) -- or, per #315, even a call on a type
// with NO such relationship at all, elsewhere in the same translation unit. A
// namespace-scope function template has no enclosing class scope to conflict
// with. Defined here (rather than tensor.h, which needs the exact same
// helpers) because layout.h is the earliest-included header that needs them;
// tensor.h/axis.h reuse this single definition instead of duplicating it.
template <class E>
_TNY_API constexpr cs::size_t _shape_rank() { return E::rank(); }
template <class E>
_TNY_API constexpr cs::size_t _shape_static_extent(cs::size_t d) { return E::static_extent(d); }

/**
 * @brief Per-dimension dynamic-stride sentinel.
 *
 * Strides are **signed**: a negative stride is a legitimate value (reversed /
 * flipped views, and DLPack tensors carry them). So — unlike `shape<...>`, where
 * `-1` marks a dynamic extent — we cannot use `-1` to mean "runtime" for a
 * stride. Instead a reserved out-of-band value (`INT64_MIN`) marks a dynamic
 * stride, leaving every ordinary stride (including negatives) expressible.
 */
inline constexpr cs::int64_t dynamic_stride = (cs::numeric_limits<cs::int64_t>::min)();

// storage for the dynamic strides only — EMPTY (EBO) when there are none, so a
// fully-static `strides<...>` mapping carries no runtime stride data.
template <class Index, cs::size_t NDyn>
struct _dyn_strides {
    cs::array<Index, NDyn> v{};
    _TNY_API constexpr Index at(cs::size_t i) const noexcept { return v[i]; }
};
template <class Index>
struct _dyn_strides<Index, 0> {
    _TNY_API constexpr Index at(cs::size_t) const noexcept { return Index(0); }
};

/**
 * @brief An mdspan layout policy with **per-dimension static or dynamic strides**
 *        — the stride analogue of `extents`/`shape`.
 *
 * `ccontiguous`/`fcontiguous` (mdspan `layout_right`/`layout_left`) give contiguous (extent-derived) strides;
 * `layout_stride` stores every stride at run time. `strides<S...>` bakes the
 * KNOWN strides into the type (folding to immediates) — **including negative
 * strides** — while any dimension marked `dynamic_stride` is supplied at run time:
 *
 *     tensor<float, shape<3,4>, strides<4,1>>(ptr);                    // static, folds
 *     tensor<float, shape<3,4>, strides<-4,1>>(ptr);                   // reversed rows
 *     tensor<float, shape<-1,4>, strides<dynamic_stride,1>>(ptr, {n}); // outer stride runtime
 *
 * When every stride is static the mapping is empty (EBO), so a stack tensor is
 * still exactly `sizeof` its data. Only the *dynamic* strides are stored.
 *
 * Note: CCCL's `cs::submdspan` is only defined for the standard layouts, so it
 * does not apply here — but teeny's own slicing/`take_along`/`permute`/`flip`/
 * `peel` build their views by hand (no submdspan), so they all work on a
 * strides<...> source and in fact fold their output strides the same way. And
 * `required_span_size` assumes non-negative strides — negative strides are for
 * VIEWS into existing storage, not owning allocation.
 *
 * @tparam S  One stride per dimension: a compile-time value (may be negative),
 *            or `dynamic_stride` for a runtime stride.
 */
template <cs::int64_t... S>
struct strides {
    static constexpr cs::size_t  N = sizeof...(S);
    static constexpr cs::int64_t S_[N ? N : 1] = { S... };

    static constexpr cs::size_t ndyn() noexcept {
        cs::size_t c = 0; for (cs::size_t i = 0; i < N; ++i) if (S_[i] == dynamic_stride) ++c; return c;
    }
    static constexpr bool all_static() noexcept { return ndyn() == 0; }
    // index of dimension r within the dynamic-stride array (undefined if r is static)
    static constexpr cs::size_t slot(cs::size_t r) noexcept {
        cs::size_t c = 0; for (cs::size_t i = 0; i < r; ++i) if (S_[i] == dynamic_stride) ++c; return c;
    }

    // Shape is a private base (not a member) so the mapping is EMPTY (EBO)
    // when the shape is fully static, keeping strides<...> tensors sizeof-exact.
    // _TNY_EMPTY_BASES (defines.h): MSVC only auto-folds the FIRST empty base
    // of a class into zero extra bytes -- this mapping privately inherits TWO
    // (_dyn_strides and Shape), so without the tag MSVC leaves it non-empty.
    template <class Shape>
    struct _TNY_EMPTY_BASES mapping : private _dyn_strides<typename Shape::index_type, strides::ndyn()>, private Shape {
        using extents_type = Shape;
        using index_type   = typename Shape::index_type;
        using rank_type    = typename Shape::rank_type;
        using layout_type  = strides;
        using _dyn         = _dyn_strides<index_type, strides::ndyn()>;
        static_assert(N == _shape_rank<Shape>(), "strides: one stride per dimension");

        mapping() = default;

        /** @brief Fully-static strides: construct from extents only. */
        template <cs::size_t M = strides::ndyn(), cs::enable_if_t<M == 0, int> = 0>
        _TNY_API constexpr mapping(const Shape & e) : Shape(e) {}

        // Cast a dynamic-stride array to this mapping's index_type (each stride is a
        // real value, so a narrower target must fit — the caller guarantees it, e.g.
        // reindex via index_fits). Lets a reindex hand int64 strides to an int32 mapping.
        template <class OtherIdx>
        _TNY_API static constexpr cs::array<index_type, strides::ndyn()>
        _narrow_dyn(const cs::array<OtherIdx, strides::ndyn()> & a) {
            cs::array<index_type, strides::ndyn()> o{};
            for (cs::size_t i = 0; i < strides::ndyn(); ++i) o[i] = static_cast<index_type>(a[i]);
            return o;
        }
        /** @brief Mixed strides: extents + the runtime strides (dim order, dynamic ones
         *         only). Templated on the array's element type so a `reindex` (narrowing
         *         the offset index width) can pass its wider source strides — each is
         *         cast to `index_type`; symmetric with mdspan's `layout_stride`. */
        template <class OtherIdx>
        _TNY_API constexpr mapping(const Shape & e, const cs::array<OtherIdx, strides::ndyn()> & dyn)
            : _dyn{_narrow_dyn(dyn)}, Shape(e) {}

        _TNY_API constexpr const Shape & extents() const noexcept { return *this; }
        _TNY_API constexpr index_type stride(rank_type r) const noexcept {
            return S_[r] == dynamic_stride ? _dyn::at(strides::slot(r)) : static_cast<index_type>(S_[r]);
        }
        template <class... I>
        _TNY_API constexpr index_type operator()(I... i) const noexcept {
            // Array size floored to 1: a rank-0 tensor (e.g. squeeze<0>() on a
            // shape<1>) makes I... empty, and a genuine zero-length array is a
            // GCC/Clang extension MSVC rejects (C2466, same class of bug as #313).
            // The loop below is bounded by Shape::rank(), never the array's own
            // size, so the unused padding slot at rank 0 is never read.
            const index_type id[sizeof...(I) ? sizeof...(I) : 1] = { static_cast<index_type>(i)... };
            index_type off = 0;
            for (rank_type r = 0; r < _shape_rank<Shape>(); ++r) off += id[r] * stride(r);
            return off;
        }
        _TNY_API constexpr index_type required_span_size() const noexcept {
            index_type n = 1;
            for (rank_type r = 0; r < _shape_rank<Shape>(); ++r) {
                if (extents().extent(r) == 0) return 0;
                n += (static_cast<index_type>(extents().extent(r)) - 1) * stride(r);
            }
            return n;
        }
        static constexpr bool is_always_unique()     noexcept { return true; }
        static constexpr bool is_always_exhaustive() noexcept { return false; }
        static constexpr bool is_always_strided()    noexcept { return true; }
        _TNY_API constexpr bool is_unique()     const noexcept { return true; }
        _TNY_API constexpr bool is_exhaustive() const noexcept { return false; }
        _TNY_API constexpr bool is_strided()    const noexcept { return true; }
    };
};

/** @brief Back-compat alias: the original all-static-stride layout name. */
template <cs::int64_t... S> using layout_static_stride = strides<S...>;

// strides<dynamic_stride × M> — the all-runtime-strided layout of a view whose
// strides are only known at run time (e.g. a reshape of a dynamic source: the
// no-copy walk yields the strides at run time). Built by index-sequence expansion
// so its arity matches the view rank. (The name `_dyn_strides` above is the mapping's
// storage struct — this is the *layout type*.)
template <cs::size_t M, class Seq = cs::make_index_sequence<M>> struct _runtime_strides;
template <cs::size_t M, cs::size_t... I> struct _runtime_strides<M, cs::index_sequence<I...>> {
    using type = strides< ((void)I, dynamic_stride)... >;
};
template <cs::size_t M> using _runtime_strides_t = typename _runtime_strides<M>::type;

/** @brief Sentinel `Layout` selector for `recast<NewShape, keep_strides>()` (the
 *         default): PRESERVE the source strides (fold where the source layout makes
 *         them derivable, keep runtime otherwise). Contrast an explicit layout —
 *         `recast<NewShape, ccontiguous>()` reinterprets AS that layout, deriving
 *         the strides from the extents (the "I promise this is C-contiguous" form).
 *         Not a real layout (it has no mapping) — only a recast tag. */
struct keep_strides {};

/* --- layout classification (stride folding) ----------------------- *
 * Traits the tensor class uses to decide when a stride is a compile-time
 * constant. Live here (with the strides<...> definition) rather than in
 * tensor.h so the class body stays uncluttered.                        */

template <class L> struct _is_strides : cs::false_type {};       // teeny's strides<S...>
template <cs::int64_t... S> struct _is_strides<strides<S...>> : cs::true_type {};
template <class L> struct _strides_all_static : cs::false_type {};  // and every stride known?
template <cs::int64_t... S> struct _strides_all_static<strides<S...>> : cs::integral_constant<bool, strides<S...>::all_static()> {};
template <class L> struct _contiguous_layout : cs::false_type {};
template <> struct _contiguous_layout<ccontiguous> : cs::true_type {};
template <> struct _contiguous_layout<fcontiguous>  : cs::true_type {};
// kwargs-family name (matches _is_dtype/_is_storage_tag) for the same predicate --
// a layout tag, for THIS step (#279), is exactly a bare ccontiguous{}/fcontiguous{}
// value (strides<...>{} is wrap's own third positional argument, not a keyword --
// see the design note on #277 §4; composing it as a keyword is #283's job).
template <class L> using _is_layout_tag = _contiguous_layout<L>;

/** @brief layout_arg_t<Expl, Dflt, Tags...>: the Layout a call site should use --
 *         an explicit template argument (Expl != void) wins, else a bare
 *         ccontiguous{}/fcontiguous{} tag found in Tags..., else the library
 *         default Dflt. Unlike dtype_arg_t, no unwrapping is needed: the tag
 *         itself IS the layout type, so find_t's result is the answer directly.
 *         static_assert if BOTH an explicit Expl and a tag were supplied. */
template <class Expl, class Dflt, class... Tags>
struct _layout_resolve {
    static_assert(cs::is_same<Expl, void>::value || !_kw::has<_is_layout_tag, Tags...>(),
        "layout given both as an explicit template argument and as a layout tag "
        "(ccontiguous{}/fcontiguous{}) -- pick one");
    using type = cs::conditional_t<!cs::is_same<Expl, void>::value, Expl,
        _kw::find_t<_is_layout_tag, Dflt, Tags...>>;
};
template <class Expl, class Dflt, class... Tags>
using layout_arg_t = typename _layout_resolve<Expl, Dflt, Tags...>::type;

// per-dim static stride from a strides<...> layout (signed; `dynamic_stride` if
// runtime, or for any non-strides layout so callers fall through).
template <cs::size_t D, class L> struct _static_stride_at { static constexpr cs::int64_t value = dynamic_stride; };
template <cs::size_t D, cs::int64_t... S> struct _static_stride_at<D, strides<S...>> { static constexpr cs::int64_t value = strides<S...>::S_[D]; };

// Compile-time stride of SOURCE axis Ax under layout L over extents E, or
// `dynamic_stride` if only known at run time. Feeds the slice/gather stride
// folding: a static source stride (×static step) yields a static output stride.
//   - strides<...>       : the baked-in per-dim value.
//   - ccontiguous/fcontiguous : the contiguous product of the trailing/leading static
//                          extents (dynamic if any of them is dynamic).
//   - layout_stride etc. : always dynamic.
template <cs::size_t Ax, class L, class E>
_TNY_API constexpr cs::int64_t _src_sstride() {
    if constexpr (_is_strides<L>::value) return _static_stride_at<Ax, L>::value;
    else if constexpr (cs::is_same<L, ccontiguous>::value) {
        cs::int64_t s = 1;
        for (cs::size_t d = Ax + 1; d < _shape_rank<E>(); ++d) {
            if (_shape_static_extent<E>(d) == cs::dynamic_extent) return dynamic_stride;
            s *= static_cast<cs::int64_t>(_shape_static_extent<E>(d));
        }
        return s;
    } else if constexpr (cs::is_same<L, fcontiguous>::value) {
        cs::int64_t s = 1;
        for (cs::size_t d = 0; d < Ax; ++d) {
            if (_shape_static_extent<E>(d) == cs::dynamic_extent) return dynamic_stride;
            s *= static_cast<cs::int64_t>(_shape_static_extent<E>(d));
        }
        return s;
    } else return dynamic_stride;
}

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_LAYOUT
