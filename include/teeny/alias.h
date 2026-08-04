#ifndef TNY_ALIAS_H
#define TNY_ALIAS_H
// Convenience: pull the mdspan vocabulary teeny builds on into `tny`, so user
// code needs only `using namespace tny;`. These are the standard cuda::std
// names -- none collide with teeny's own, so exposing them is safe (not
// breaking): teeny defines `tensor`, `view`, `layout_static_stride`, ... which
// are all distinct from `extents`, `layout_right`, `array`, etc.
#include <cuda/std/mdspan>
#include <cuda/std/array>
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <teeny/defines.h>
#include <teeny/kwargs.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

// shapes / layouts / views
using cs::extents;
using cs::dextents;
using cs::dynamic_extent;
using cs::full_extent;
using cs::full_extent_t;
using cs::layout_right;
using cs::layout_left;
using cs::layout_stride;
/** @brief Names for the two contiguous layouts: `ccontiguous` is C-contiguous
 *         (row-major, `layout_right`), `fcontiguous` is Fortran-contiguous
 *         (column-major, `layout_left`). Use wherever a `Layout` is expected —
 *         this is teeny's default and preferred spelling. */
using ccontiguous = cs::layout_right;
using fcontiguous = cs::layout_left;
namespace _kw {
template <> struct is_keyword<ccontiguous> : cs::true_type {};
template <> struct is_keyword<fcontiguous> : cs::true_type {};
}
/** @brief Legacy aliases (`corder`/`forder`); prefer `ccontiguous`/`fcontiguous`. */
using corder = ccontiguous;
using forder = fcontiguous;
/** @brief `dynamic_strides` — the all-runtime strided layout (`cs::layout_stride`,
 *         a full runtime stride array). Prefer teeny's `strides<S...>` (folds known
 *         strides to immediates, and is what slicing produces); `dynamic_strides`
 *         is `strides<>` with every stride runtime. */
using dynamic_strides = cs::layout_stride;
using cs::mdspan;
using cs::submdspan;

// containers / utilities
using cs::array;
using cs::size_t;
using cs::ptrdiff_t;
using cs::index_sequence;
using cs::make_index_sequence;
using cs::integral_constant;

/* ------------------------------------------------------------------ *
 *     Static integral values (compile-time indices / extents)        *
 *                                                                    *
 *  cuda::std / std do not ship short names for integral_constant, so  *
 *  teeny provides them. Each converts implicitly to a runtime         *
 *  integral and carries `::value`, so it works as both a compile-time *
 *  and a runtime value. Pass one where a tensor wants a static index   *
 *  (e.g. `t(Int<1>(), j)`, `t.extent(Int<0>())`).                     *
 * ------------------------------------------------------------------ */
template <int            V> using Int    = cs::integral_constant<int, V>;
template <long           V> using Long   = cs::integral_constant<long, V>;
template <cs::size_t     V> using Size   = cs::integral_constant<cs::size_t, V>;
template <unsigned       V> using UInt   = cs::integral_constant<unsigned, V>;
template <cs::ptrdiff_t  V> using Diff   = cs::integral_constant<cs::ptrdiff_t, V>;
template <bool           V> using Bool   = cs::integral_constant<bool, V>;
// fixed-width integer value forms
template <cs::int8_t     V> using Int8   = cs::integral_constant<cs::int8_t, V>;
template <cs::int16_t    V> using Int16  = cs::integral_constant<cs::int16_t, V>;
template <cs::int32_t    V> using Int32  = cs::integral_constant<cs::int32_t, V>;
template <cs::int64_t    V> using Int64  = cs::integral_constant<cs::int64_t, V>;
template <cs::uint8_t    V> using UInt8  = cs::integral_constant<cs::uint8_t, V>;
template <cs::uint16_t   V> using UInt16 = cs::integral_constant<cs::uint16_t, V>;
template <cs::uint32_t   V> using UInt32 = cs::integral_constant<cs::uint32_t, V>;
template <cs::uint64_t   V> using UInt64 = cs::integral_constant<cs::uint64_t, V>;

// numpy short spellings of the fixed-width value forms (width in BYTES, like
// numpy: `I4` is a 4-byte int32 value — NOT Rust's bit-width `i4`). So
// `shape<I4<3>::value>` etc.; `I4<3>` is `Int32<3>`.
template <cs::int8_t   V> using I1 = Int8<V>;
template <cs::int16_t  V> using I2 = Int16<V>;
template <cs::int32_t  V> using I4 = Int32<V>;
template <cs::int64_t  V> using I8 = Int64<V>;
template <cs::uint8_t  V> using U1 = UInt8<V>;
template <cs::uint16_t V> using U2 = UInt16<V>;
template <cs::uint32_t V> using U4 = UInt32<V>;
template <cs::uint64_t V> using U8 = UInt64<V>;

/* ------------------------------------------------------------------ *
 *   Element dtype aliases (numpy short codes; width in BYTES)         *
 *                                                                    *
 *  Convenience TYPE names for the element type `T`, matching numpy's  *
 *  dtype codes (`i4` == int32, `f4` == float) — the width is the BYTE *
 *  count (numpy), not Rust's bit count. Use where you'd name a type:  *
 *  `local<f4, shape<3>>`, `zeros<i4>(sh)`. The floats `f2`/`bf16`     *
 *  live in half.h (where those types are defined).                    *
 * ------------------------------------------------------------------ */
using i1 = cs::int8_t;   using i2 = cs::int16_t;   using i4 = cs::int32_t;   using i8 = cs::int64_t;
using u1 = cs::uint8_t;  using u2 = cs::uint16_t;  using u4 = cs::uint32_t;  using u8 = cs::uint64_t;
using f4 = float;        using f8 = double;

// fold a per-dim size to an mdspan extent: any NEGATIVE value (numpy's -1) means
// dynamic; `dynamic_extent` itself passes through.
template <class T> _TNY_API constexpr cs::size_t _dyn_extent(T e) {
    if constexpr (cs::is_signed<T>::value) { if (e < T(0)) return cs::dynamic_extent; }
    return static_cast<cs::size_t>(e);
}

/** @brief User-friendly shape type: `shape<2,3,4>` == `extents<int64_t, 2,3,4>`.
 *
 * The fixed-size `int64_t` index type matches DLPack's `shape` exactly, so it
 * drops straight onto ndarray bindings. A dynamic dimension can be spelled
 * either `dynamic_extent` or, numpy-style, **`-1`** — so `shape<-1,2,3>` ==
 * `shape<dynamic_extent,2,3>` == `extents<int64_t, dynamic_extent, 2, 3>`. Use
 * it in place of `extents<...>`: `local<double, shape<3,3>>`,
 * `owned<float, shape<-1,4>>`. */
template <auto... E> using shape = cs::extents<cs::int64_t, _dyn_extent(E)...>;

/** @brief `shape<...>` with an explicit index type: `shape_as<int32_t, -1,3,3>`.
 *  `shape<>` is the int64 default (DLPack's index type); `shape32<...>` narrows the
 *  offset math to **int32** for the kernel-boundary view (see `reindex`). The `-1`
 *  == dynamic rule is the same. */
template <class Idx, auto... E> using shape_as = cs::extents<Idx, _dyn_extent(E)...>;
template <auto... E>            using shape32  = shape_as<cs::int32_t, E...>;

// Retype an extents' index_type to Idx2, keeping the same compile-time extent values
// (the primitive behind `reindex` — swap offset width, preserve the shape).
//
// COMPILE-TIME TWIN of `index_fits`'s extent-value check (#489). `index_fits`
// answers the same question for the RUNTIME extents; a STATIC one is known here,
// so it is rejected here — with a diagnostic, rather than silently emitting an
// `extents` whose own `extent(d)` reads a truncated (or negative) number:
// `shape<300,2>` retyped to `int8_t` used to compile clean and then report
// `extent(0) == 44`. mdspan itself mandates that every static extent be
// representable in the index type; CCCL 2.8.2 does not enforce it, so the check
// lives here. This is the one choke point every narrowing path routes through
// (`tensor::reindex` and `anyrank::reindexed`'s static Head/Tail geometry), so
// one `static_assert` covers them all. `dynamic_extent` is a placeholder, not a
// size — it passes, and its runtime value is what `index_fits` then checks.
template <class Idx2, class E, class Seq> struct _reindex_extents;
template <class Idx2, class E, cs::size_t... D>
struct _reindex_extents<Idx2, E, cs::index_sequence<D...>> {
    // `max()` is never negative, so widening it to `size_t` (the domain
    // `static_extent` reports in) is exact for every integral `Idx2`.
    static_assert(((E::static_extent(D) == cs::dynamic_extent ||
                    E::static_extent(D) <=
                        static_cast<cs::size_t>((cs::numeric_limits<Idx2>::max)())) && ...),
                  "reindex<Idx2>(): a static extent is too large to represent in the target "
                  "index type (it would be truncated in the narrowed extents)");
    using type = cs::extents<Idx2, E::static_extent(D)...>;
};
template <class Idx2, class E>
using _reindex_extents_t = typename _reindex_extents<Idx2, E, cs::make_index_sequence<E::rank()>>::type;

/** @brief Fully-dynamic shape of a given rank: `rank<3>` == `shape<-1,-1,-1>`
 *  == `extents<int64_t, dynamic_extent, dynamic_extent, dynamic_extent>`. Handy
 *  for a rank-N view whose sizes are all runtime: `view<float, rank<3>>`.
 *  `rank<0>` is the rank-0 (scalar) shape. */
template <cs::size_t N> using rank = cs::dextents<cs::int64_t, N>;

/** @brief The **ellipsis** marker (numpy `...`) for the unspecified middle axes. Its
 *  own type (an empty enum) keeps it distinct from any real extent value, and being an
 *  enum it is a valid non-type template argument for `anyshape<...>`.
 *
 *  It has two roles that never overlap, so one marker serves both:
 *  - when **indexing**, `t(1, ellipsis, 2)` stands for as many `all` as fill the rank;
 *  - in an `anyshape<...>` boundary tag it marks the rank-erased region — there it is
 *    conventionally spelled **`etc`** ("and so on"), an alias of `ellipsis`.
 *  So `ellipsis`/`ellipsis_t` is the primary name and `etc`/`etc_t` the alias:
 *  `t(1, etc, 2)` == `t(1, ellipsis, 2)` and `anyshape<ellipsis, 3>` == `anyshape<etc, 3>`.
 *  (The `_is_ellipsis` indexing trait lives in `indexing.h`.) */
enum class ellipsis_t {};
constexpr ellipsis_t ellipsis{};
/** @brief `etc` — the `anyshape<...>` spelling of `ellipsis` (same marker, same value). */
using etc_t = ellipsis_t;
constexpr ellipsis_t etc = ellipsis;

// Split an `anyshape<...>` pack at its single `etc` into the leading Head (before)
// and trailing Tail (after). Walks the heterogeneous `auto...` pack one element at a
// time, accumulating the head into a `_vpack` until the `etc` element is matched by
// the more-specialised second partial specialization.
template <auto... Vs> struct _vpack {};
template <class HeadAcc, auto... Es> struct _ashape_split;
template <auto... Hs, auto... Rest>   // hit `etc`: head = accumulated, tail = the rest
struct _ashape_split<_vpack<Hs...>, etc, Rest...> { using head = shape<Hs...>; using tail = shape<Rest...>; };
template <auto... Hs, auto E0, auto... Rest>   // any other element: append to head, recurse
struct _ashape_split<_vpack<Hs...>, E0, Rest...> : _ashape_split<_vpack<Hs..., E0>, Rest...> {};

/** @brief The shape spelling for the rank-erased `anyrank` boundary: exactly one
 *  `etc` marks the dynamic-rank region, the dims AFTER it are the static **Tail**
 *  (anchored at `ndim`), the dims BEFORE it are the static **Head** (anchored at 0).
 *  Each non-`etc` slot is a per-dim static extent or `-1` (dynamic), exactly like
 *  `shape<...>`. Hand it to `as_anyrank(..., anyshape<etc,-1,-1,3>{})` or
 *  `from_dlpack<T, anyshape<etc,-1,-1,3>>(m)` so the peeled cells fold those inner
 *  dims — `anyshape<etc,-1,-1,3>` == `(*batch, spatial, spatial, C=3)`.
 *
 *  A static leading **Head** (dims BEFORE `etc`) is allowed too:
 *  `anyshape<A, B, etc, C, D>` == `(A, B, *middle, C, D)` — e.g.
 *  `anyshape<3, etc, 5>` for `(C_in=3, *spatial, C_out=5)`. The Head folds in
 *  `fixed`/`dispatch_rank` (full-rank materialisation); `peel_front<-Sr>` stays
 *  trailing-oriented (a leading Head is normally peeled into the batch).
 *
 *  Unlike a plain `shape<...>` (a concrete fixed-rank `extents`), an `anyshape` is a
 *  SPEC, not a tensor type — a runtime-rank object needs the data + runtime arrays,
 *  not just a type. */
template <auto... Es>
struct anyshape {
    static_assert((cs::size_t(0) + ... + (cs::is_same<decltype(Es), etc_t>::value ? 1u : 0u)) == 1u,
        "anyshape: needs exactly one `etc` (the erased dynamic-rank region)");
    using _sp  = _ashape_split<_vpack<>, Es...>;
    using head = typename _sp::head;   // static LEADING dims (before etc), anchored at 0
    using tail = typename _sp::tail;   // static TRAILING dims (after etc), anchored at ndim
};
template <class> struct _is_anyshape : cs::false_type {};
template <auto... Es> struct _is_anyshape<anyshape<Es...>> : cs::true_type {};

/** @brief Compile-time **axis selector** — a value tag carrying a list of axes,
 *  the sibling of `shape<...>` for axis arguments. It lets axis-taking ops be
 *  spelled by VALUE (deducing the axes from the argument type) instead of an
 *  explicit template list, so on a type-dependent receiver they need no
 *  `.template`: `peel(t, axis<0,1>{})` == `peel<0,1>(t)`,
 *  `t.slice_along(axis<0,2>{}, i, slice(1,4))` == `t.slice_along<0,2>(i, slice(1,4))`.
 *
 *  Like numpy's `axis: int | list[int]`, one variadic tag covers both a single
 *  axis (`axis<0>{}`) and a list (`axis<0,2>{}`); axes are **signed** (negatives
 *  count from the back, as everywhere in teeny). `rank` is the axis count. */
template <long... Axes> struct axis { static constexpr cs::size_t rank = sizeof...(Axes); };
namespace _kw { template <long... Axes> struct is_keyword<axis<Axes...>> : cs::true_type {}; }
/** @brief `_is_axis_tag<X>::value` is true iff `X` is an `axis<Axes...>` instantiation —
 *  lets a generic trailing keyword bag (`_kw`) find the axis-selector tag among
 *  `dtype<...>`/`into_t<...>`/`keepdims_t`/... siblings. */
template <class> struct _is_axis_tag : cs::false_type {};
template <long... Axes> struct _is_axis_tag<axis<Axes...>> : cs::true_type {};
/** @brief `_is_empty_axis<X>::value` is true iff `X` is the EXPLICITLY EMPTY axis
 *  list `axis<>` — "over NO axis", which every axis-list op treats as the identity
 *  (numpy's `axis=()`). It is a REQUEST, and a different one from "no axis keyword
 *  was supplied at all" (a call site spells that absence with `_kw::unset`, whose
 *  meaning is that call site's own default — all axes, for a reduction). Using one
 *  type for both is exactly what made `sum(a, axis<>{})` silently reduce over
 *  everything (#398), so keep the two apart wherever an axis tag is optional. */
template <class> struct _is_empty_axis : cs::false_type {};
template <> struct _is_empty_axis<axis<>> : cs::true_type {};

/** @brief Compile-time **element-type tag** — a value carrier for `T`, the
 *  sibling of `axis<...>` for the dtype argument. It lets a type-parameterised
 *  call be spelled by VALUE (deducing `T` from the argument) instead of an
 *  explicit `<T>` template argument, so on a type-dependent receiver it needs
 *  no `.template`: `empty(shape<3,3>{}, dtype<double>{})` == `empty<double>(shape<3,3>{})`,
 *  `a.to(dtype<float>{})` == `a.to<float>()`. Numpy's `dtype=` keyword is the
 *  namesake — including reuse as the reduction accumulator/result type:
 *  `sum(a, dtype<double>{})` == `sum<double>(a)`, matching `np.sum(a, dtype=...)`. */
template <class T> struct dtype {};
/** @brief `_is_dtype<X>::value` is true iff `X` is a `dtype<T>` instantiation —
 *  guards a generically-typed value parameter (e.g. `full`'s fill value) against
 *  accidentally binding a misplaced `dtype<T>{}` tag instead of a real value. */
template <class> struct _is_dtype : cs::false_type {};
template <class T> struct _is_dtype<dtype<T>> : cs::true_type {};
namespace _kw { template <class T> struct is_keyword<dtype<T>> : cs::true_type {}; }

// dtype_arg_t<Expl, Dflt, Tags...>: the element/accumulator type a call site should
// use -- an explicit template argument (Expl != void) wins, else a dtype<T>{} tag
// found in Tags..., else the library default Dflt; supplying BOTH an explicit Expl
// and a dtype<...> tag is a static_assert. That precedence rule (and its wording)
// lives ONCE, in `_kw::resolve` (kwargs.h) -- shared with `storage_arg`/`layout_arg_t`.
// The only dtype-specific part is the unwrap step below: read the `T` out of the
// `dtype<T>` tag that was found (and fall back to `Dflt` when there was none).
template <class X, class D> struct _dtype_arg             { using type = D; };
template <class T, class D> struct _dtype_arg<dtype<T>, D> { using type = T; };
template <class Expl, class Dflt, class... Tags>
using dtype_arg_t = _kw::resolve_t<_dtype_arg, Expl, void, _is_dtype, Dflt, Tags...>;

/** @brief Keep-this-axis marker for slicing (an alias of `full_extent`). */
constexpr cs::full_extent_t all{};

/** @brief numpy/pytorch `keepdims=True` tag for axis reductions — pass as any
 *  trailing keyword (composes with `dtype<...>`/`axis<...>`/`into(dest)` in any
 *  order) to keep the reduced axes as size-1 instead of removing them, so the
 *  result broadcasts back against the input: `sum<0>(a, keepdims)`,
 *  `sum(a, axis<0,2>{}, keepdims)`. A distinct empty-tag type, like `all`/`none`,
 *  so it never collides with another argument. */
struct keepdims_t {};
constexpr keepdims_t keepdims{};
namespace _kw { template <> struct is_keyword<keepdims_t> : cs::true_type {}; }
/** @brief `_is_keepdims_tag<X>::value` is true iff `X` is `keepdims_t` — lets a
 *  generic trailing keyword bag (`_kw`) find it among `dtype<...>`/`axis<...>`/
 *  `into_t<...>` siblings. */
template <class> struct _is_keepdims_tag : cs::false_type {};
template <> struct _is_keepdims_tag<keepdims_t> : cs::true_type {};

_TNY_NAMESPACE_END(tny)

#endif // TNY_ALIAS_H
