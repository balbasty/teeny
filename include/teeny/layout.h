#ifndef TNY_MD_LAYOUT
#define TNY_MD_LAYOUT
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/array>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <teeny/defines.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/**
 * @brief Per-dimension dynamic-stride sentinel.
 *
 * Strides are **signed**: a negative stride is a legitimate value (reversed /
 * flipped views, and DLPack tensors carry them). So — unlike `shape<...>`, where
 * `-1` marks a dynamic extent — we cannot use `-1` to mean "runtime" for a
 * stride. Instead a reserved out-of-band value (`INT64_MIN`) marks a dynamic
 * stride, leaving every ordinary stride (including negatives) expressible.
 */
inline constexpr cs::int64_t dynamic_stride = cs::numeric_limits<cs::int64_t>::min();

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
 * `layout_right`/`layout_left` give contiguous (extent-derived) strides;
 * `layout_stride` stores every stride at run time. `strides<S...>` bakes the
 * KNOWN strides into the type (folding to immediates, like jitfields' posdef
 * `Pointer<T,S>`) — **including negative strides** — while any dimension marked
 * `dynamic_stride` is supplied at run time:
 *
 *     tensor<float, shape<3,4>, strides<4,1>>(ptr);                    // static, folds
 *     tensor<float, shape<3,4>, strides<-4,1>>(ptr);                   // reversed rows
 *     tensor<float, shape<-1,4>, strides<dynamic_stride,1>>(ptr, {n}); // outer stride runtime
 *
 * When every stride is static the mapping is empty (EBO), so a stack tensor is
 * still exactly `sizeof` its data. Only the *dynamic* strides are stored.
 *
 * Note: `submdspan` (and therefore `peel`/`take_along`/`permute`) is only
 * defined by CCCL for the standard layouts, so it does not apply here. And
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

    // Extents is a private base (not a member) so the mapping is EMPTY (EBO)
    // when the shape is fully static, keeping strides<...> tensors sizeof-exact.
    template <class Extents>
    struct mapping : private _dyn_strides<typename Extents::index_type, strides::ndyn()>, private Extents {
        using extents_type = Extents;
        using index_type   = typename Extents::index_type;
        using rank_type    = typename Extents::rank_type;
        using layout_type  = strides;
        using _dyn         = _dyn_strides<index_type, strides::ndyn()>;
        static_assert(N == Extents::rank(), "strides: one stride per dimension");

        mapping() = default;

        /** @brief Fully-static strides: construct from extents only. */
        template <cs::size_t M = strides::ndyn(), cs::enable_if_t<M == 0, int> = 0>
        _TNY_API constexpr mapping(const Extents & e) : Extents(e) {}

        /** @brief Mixed strides: extents + the runtime strides (dim order, dynamic ones only). */
        _TNY_API constexpr mapping(const Extents & e, const cs::array<index_type, strides::ndyn()> & dyn)
            : _dyn{dyn}, Extents(e) {}

        _TNY_API constexpr const Extents & extents() const noexcept { return *this; }
        _TNY_API constexpr index_type stride(rank_type r) const noexcept {
            return S_[r] == dynamic_stride ? _dyn::at(strides::slot(r)) : static_cast<index_type>(S_[r]);
        }
        template <class... I>
        _TNY_API constexpr index_type operator()(I... i) const noexcept {
            const index_type id[] = { static_cast<index_type>(i)... };
            index_type off = 0;
            for (rank_type r = 0; r < Extents::rank(); ++r) off += id[r] * stride(r);
            return off;
        }
        _TNY_API constexpr index_type required_span_size() const noexcept {
            index_type n = 1;
            for (rank_type r = 0; r < Extents::rank(); ++r) {
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

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_LAYOUT
