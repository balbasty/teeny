#ifndef TNY_MD_LAYOUT
#define TNY_MD_LAYOUT
#include <cuda/std/cstddef>
#include <teeny/_core/defines.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(md)

namespace cs = cuda::std;

/**
 * @brief An mdspan layout policy with per-dimension COMPILE-TIME strides.
 *
 * This is the one thing standard mdspan layouts don't offer: `layout_right`
 * / `layout_left` give contiguous (extent-derived) strides, `layout_stride`
 * stores every stride at run time. `layout_static_stride<S...>` bakes the
 * strides into the type, so a non-contiguous stride still folds to an
 * immediate (e.g. jitfields' posdef `Pointer<T, S>`).
 *
 * @tparam Strides  One compile-time stride per dimension.
 */
template <cs::size_t... Strides>
struct layout_static_stride {
    template <class Extents>
    struct mapping {
        using extents_type = Extents;
        using index_type   = typename Extents::index_type;
        using rank_type    = typename Extents::rank_type;
        using layout_type  = layout_static_stride;
        static_assert(sizeof...(Strides) == Extents::rank(),
                      "layout_static_stride: one stride per dimension");

        Extents ext_{};
        mapping() = default;
        _TNY_API constexpr mapping(const Extents & e) : ext_(e) {}
        _TNY_API constexpr const Extents & extents() const noexcept { return ext_; }

        static constexpr index_type strides_[] = { static_cast<index_type>(Strides)... };

        template <class... I>
        _TNY_API constexpr index_type operator()(I... i) const noexcept {
            const index_type id[] = { static_cast<index_type>(i)... };
            index_type off = 0;
            for (rank_type r = 0; r < extents_type::rank(); ++r) off += id[r] * strides_[r];
            return off;
        }
        _TNY_API constexpr index_type required_span_size() const noexcept {
            index_type n = 1;
            for (rank_type r = 0; r < extents_type::rank(); ++r) {
                if (ext_.extent(r) == 0) return 0;
                n += (static_cast<index_type>(ext_.extent(r)) - 1) * strides_[r];
            }
            return n;
        }
        _TNY_API constexpr index_type stride(rank_type r) const noexcept { return strides_[r]; }
        static constexpr bool is_always_unique()     noexcept { return true; }
        static constexpr bool is_always_exhaustive() noexcept { return false; }
        static constexpr bool is_always_strided()    noexcept { return true; }
        _TNY_API constexpr bool is_unique()     const noexcept { return true; }
        _TNY_API constexpr bool is_exhaustive() const noexcept { return false; }
        _TNY_API constexpr bool is_strided()    const noexcept { return true; }
    };
};

_TNY_NAMESPACE_END(md)
_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_LAYOUT
