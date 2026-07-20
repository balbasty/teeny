#ifndef TNY_MD_TENSOR
#define TNY_MD_TENSOR
#include <cuda/std/mdspan>
#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <teeny/_core/defines.h>
#include <teeny/md/storage.h>
#include <teeny/md/layout.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(md)

namespace cs = cuda::std;

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
template <class T, class Extents, class Layout = cs::layout_right, own O = own::view>
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

    /** @brief Heap constructor: allocate storage for `m`. */
    template <own OO = O, cs::enable_if_t<OO == own::heap, int> = 0>
    _TNY_HOST explicit tensor(mapping_type m)
        : mapping_type(m), store_(static_cast<cs::size_t>(m.required_span_size())) {}

    /* --- geometry ------------------------------------------------- */
    static constexpr cs::size_t rank() noexcept { return Extents::rank(); }
    _TNY_API constexpr const mapping_type & mapping() const noexcept { return *this; }
    _TNY_API constexpr const Extents & extents() const noexcept { return mapping_type::extents(); }
    _TNY_API constexpr index_type extent(cs::size_t r) const noexcept { return mapping_type::extents().extent(r); }
    _TNY_API constexpr index_type stride(cs::size_t r) const noexcept { return mapping_type::stride(r); }
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

    /* --- element access ------------------------------------------ */
    template <class... I> _TNY_API T &       operator()(I... i)       noexcept { return store_.data()[mapping_type::operator()(i...)]; }
    template <class... I> _TNY_API const T & operator()(I... i) const noexcept { return store_.data()[mapping_type::operator()(i...)]; }

    /* --- in-place elementwise math (declared here, defined in math.h) --- */
    template <class B> _TNY_API tensor & add_(const B & b);
    template <class B> _TNY_API tensor & sub_(const B & b);
    template <class B> _TNY_API tensor & mul_(const B & b);
    template <class B> _TNY_API tensor & div_(const B & b);
    _TNY_API tensor & add_(T s);
    _TNY_API tensor & sub_(T s);
    _TNY_API tensor & mul_(T s);
    _TNY_API tensor & div_(T s);
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

/** @brief Stack-owned tensor (fully static shape); elements value-initialised. */
template <class T, class Extents, class Layout = cs::layout_right>
_TNY_API tensor<T, Extents, Layout, own::stack> local() { return {}; }

/** @brief Heap-owned tensor (host only, move-only). */
template <class T, class Extents, class Layout = cs::layout_right>
_TNY_HOST tensor<T, Extents, Layout, own::heap> owned(Extents e) {
    using Tn = tensor<T, Extents, Layout, own::heap>;
    return Tn(typename Tn::mapping_type(e));
}

_TNY_NAMESPACE_END(md)
_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_TENSOR
