#ifndef TNY_MD_ITERATE
#define TNY_MD_ITERATE
#include <cuda/std/mdspan>
#include <cuda/std/utility>
#include <teeny/defines.h>
#include <teeny/tensor.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/* ================================================================== *
 *  nd-peel: iterate over a SUBSET of axes, yielding a lower-rank view *
 *  over the remaining axes.                                          *
 *                                                                    *
 *  Replaces the ndindex<->linear plumbing (jitfields index2offset /  *
 *  sub2offset): a linear index over the peeled axes is decoded for   *
 *  you, and the corresponding sub-view (a `md::tensor` view into the  *
 *  original data) is returned.                                       *
 *                                                                    *
 *  Two entry points:                                                 *
 *    peel_at<Axes...>(t, i)  -> the i-th sub-view  (grid-stride loop) *
 *    peel<Axes...>(t)       -> a range of them    (range-for)       *
 * ================================================================== */

namespace _md {

_TNY_API constexpr int pos_of(cs::size_t a, const cs::size_t * d, cs::size_t n) {
    for (cs::size_t p = 0; p < n; ++p) if (d[p] == a) return static_cast<int>(p);
    return -1;
}

// Argument for axis A of the submdspan call: the decoded index if A is peeled,
// otherwise `full_extent` (keep the axis).
template <cs::size_t A, class I, cs::size_t... Axes>
_TNY_API auto axis_arg(const I * idx, cs::index_sequence<Axes...>) {
    constexpr cs::size_t dd[] = { Axes..., static_cast<cs::size_t>(-1) };
    constexpr int p = pos_of(A, dd, sizeof...(Axes));
    if constexpr (p < 0) return cs::full_extent;
    else                 return idx[p];
}

template <class MD, class I, cs::size_t... Axes, cs::size_t... A>
_TNY_API auto submd(const MD & src, const I * idx,
                    cs::index_sequence<Axes...> axes, cs::index_sequence<A...>) {
    return cs::submdspan(src, axis_arg<A>(idx, axes)...);
}

} // namespace _md

/** @brief The `i`-th sub-view obtained by peeling `Axes...` (0 <= i < product
 *         of the peeled extents). Peeled axes vary in row-major order (the
 *         last listed axis fastest). Returns a `md::tensor` view. */
template <cs::size_t... Axes, class MD>
_TNY_API auto peel_at(const MD & src, typename MD::index_type i) {
    using I = typename MD::index_type;
    constexpr cs::size_t nd = sizeof...(Axes);
    const I e[nd ? nd : 1]   = { static_cast<I>(src.extent(Axes))... };
    I       idx[nd ? nd : 1] = {};
    I rem = i;
    for (int p = static_cast<int>(nd) - 1; p >= 0; --p) { idx[p] = rem % e[p]; rem /= e[p]; }
    return as_tensor(_md::submd(src, idx, cs::index_sequence<Axes...>{},
                                cs::make_index_sequence<MD::rank()>{}));
}
// convenience: peel from a md::tensor (uses its view). Non-const -> mutable
// peel; const -> read-only peel.
template <long... Axes, class T, class E, class L, own O>
_TNY_API auto peel_at(tensor<T,E,L,O> & t, typename tensor<T,E,L,O>::index_type i) {
    return peel_at<_norm_axis(Axes, tensor<T,E,L,O>::rank())...>(t.view(), i);
}
template <long... Axes, class T, class E, class L, own O>
_TNY_API auto peel_at(const tensor<T,E,L,O> & t, typename tensor<T,E,L,O>::index_type i) {
    return peel_at<_norm_axis(Axes, tensor<T,E,L,O>::rank())...>(t.view(), i);
}

/** @brief A range of sub-views obtained by peeling `Axes...`. Supports
 *         `size()`, `operator[]`, and range-for. */
template <class MD, cs::size_t... Axes>
struct peel_range {
    using index_type = typename MD::index_type;
    MD src;

    _TNY_API index_type size() const noexcept {
        const index_type e[] = { static_cast<index_type>(src.extent(Axes))..., index_type(1) };
        index_type n = 1;
        for (cs::size_t p = 0; p < sizeof...(Axes); ++p) n *= e[p];
        return n;
    }
    _TNY_API auto operator[](index_type i) const { return peel_at<Axes...>(src, i); }

    struct iterator {
        const peel_range * r;
        index_type i;
        _TNY_API auto operator*() const { return (*r)[i]; }
        _TNY_API iterator & operator++() { ++i; return *this; }
        _TNY_API bool operator!=(const iterator & o) const { return i != o.i; }
        _TNY_API bool operator==(const iterator & o) const { return i == o.i; }
    };
    _TNY_API iterator begin() const { return { this, 0 }; }
    _TNY_API iterator end()   const { return { this, size() }; }
};

/** @brief Build a range of sub-views by peeling `Axes...` of `t`. Non-const `t`
 *         yields mutable peel; const `t` yields read-only peel. */
template <long... Axes, class T, class E, class L, own O>
_TNY_API auto peel(tensor<T,E,L,O> & t) {
    return peel_range<decltype(t.view()), _norm_axis(Axes, tensor<T,E,L,O>::rank())...>{ t.view() };
}
template <long... Axes, class T, class E, class L, own O>
_TNY_API auto peel(const tensor<T,E,L,O> & t) {
    return peel_range<decltype(t.view()), _norm_axis(Axes, tensor<T,E,L,O>::rank())...>{ t.view() };
}
/** @brief Build a range of sub-views over a raw mdspan. */
template <cs::size_t... Axes, class MD>
_TNY_API peel_range<MD, Axes...> peel_of(const MD & m) { return { m }; }

namespace _md {
template <class T, cs::size_t... A> _TNY_API auto sfront(T & t, cs::index_sequence<A...>) { return peel<A...>(t); }
template <class T, cs::size_t... A> _TNY_API auto sfront_at(T & t, typename T::index_type i, cs::index_sequence<A...>) { return peel_at<A...>(t, i); }
} // namespace _md

/** @brief Peel the FIRST `N` axes (e.g. an arbitrary number of leading batch
 *         dims) -> a range of sub-views over the remaining axes. The
 *         runtime-batch-rank half of the `(*batch, *spatial, C)` pattern: pick
 *         `N` = number of batch dims and each sub-view is `(*spatial, C)`. */
template <cs::size_t N, class T, class E, class L, own O>
_TNY_API auto peel_front(tensor<T,E,L,O> & t)       { return _md::sfront(t, cs::make_index_sequence<N>{}); }
template <cs::size_t N, class T, class E, class L, own O>
_TNY_API auto peel_front(const tensor<T,E,L,O> & t) { return _md::sfront(t, cs::make_index_sequence<N>{}); }

/** @brief The `i`-th sub-view obtained by peeling the first `N` axes (grid-stride style). */
template <cs::size_t N, class T, class E, class L, own O>
_TNY_API auto peel_front_at(tensor<T,E,L,O> & t, typename tensor<T,E,L,O>::index_type i)
{ return _md::sfront_at(t, i, cs::make_index_sequence<N>{}); }
template <cs::size_t N, class T, class E, class L, own O>
_TNY_API auto peel_front_at(const tensor<T,E,L,O> & t, typename tensor<T,E,L,O>::index_type i)
{ return _md::sfront_at(t, i, cs::make_index_sequence<N>{}); }

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_ITERATE
