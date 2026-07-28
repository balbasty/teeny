// SCRATCH MSVC repro for #315 -- NOT a real teeny feature test. Isolates the
// "recast() misattributes a private-inheritance access error to an unrelated
// NewE type parameter" shape without CCCL, using a small hand-rolled stand-in
// for cuda::std::extents + teeny's strides<...>::mapping private double
// inheritance. Delete before merging; see #315 for context.
#include <cstddef>

// Stand-in for cuda::std::extents<Idx, E...>: has rank()/static_extent(d).
template <long... E>
struct FakeExtents {
    static constexpr std::size_t rank() { return sizeof...(E); }
    static constexpr long static_extent(std::size_t d) {
        constexpr long vals[] = { E..., 0 };
        return vals[d];
    }
};

// Stand-in for teeny's _dyn_strides (an empty helper base for EBO).
template <class Idx, std::size_t N> struct FakeDynStrides {};

// Stand-in for teeny's strides<S...>::mapping<Shape>: privately inherits from
// a dyn-strides helper AND from Shape itself (the exact double-private-base
// EBO shape that triggers the MSVC misattribution per #315's evidence).
template <class Shape>
struct FakeMapping : private FakeDynStrides<int, 0>, private Shape {
    static constexpr std::size_t rank() { return Shape::rank(); }
};

// Matches tensor.h's ALREADY-FIXED (#294) pattern: a free function template so
// there's no enclosing-class scope for MSVC's two-phase lookup to conflict with.
template <class Shape>
static constexpr std::size_t _shape_rank() { return Shape::rank(); }

// PROPOSED fix for #315: same pattern, for an EXTENTS-like type named from an
// unrelated context (recast's NewE).
template <class E>
static constexpr std::size_t _extents_rank() { return E::rank(); }

// Stand-in for tensor<T, Shape, strides<...>, O>: privately inherits from
// FakeMapping<Shape> (its OWN shape), same as the real tensor's EBO base.
// rank() ALREADY routes through _shape_rank<Shape>() here (mirroring real
// current teeny post-#294) -- so if the bug were about the bare `rank()` call
// alone, it would already be gone; the point of this repro is to isolate
// whether `NewE::rank()` ALSO needs its own wrapper.
template <class Shape>
struct FakeTensor : private FakeMapping<Shape> {
    static constexpr std::size_t rank() { return _shape_rank<Shape>(); }

    // BUGGY (matches current teeny's _recast exactly): NewE::rank() called directly.
    template <class NewE>
    static constexpr bool recast_rank_matches_buggy() {
        return NewE::rank() == rank();
    }

    // FIXED (proposed): NewE::rank() routed through the same free-function pattern.
    template <class NewE>
    static constexpr bool recast_rank_matches_fixed() {
        return _extents_rank<NewE>() == rank();
    }
};

using SrcShape = FakeExtents<5, 6>;   // matches this tensor's OWN shape
using NewShape = FakeExtents<5, 6>;   // the recast TARGET -- unrelated type, same rank

int main() {
    using T = FakeTensor<SrcShape>;
    static_assert(T::recast_rank_matches_buggy<NewShape>(), "recast: rank must match (buggy path)");
    static_assert(T::recast_rank_matches_fixed<NewShape>(), "recast: rank must match (fixed path)");
    return 0;
}
