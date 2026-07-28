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

// Stand-in for tensor<T, Shape, strides<...>, O>: privately inherits from
// FakeMapping<Shape> (its OWN shape), same as the real tensor's EBO base.
template <class Shape>
struct FakeTensor : private FakeMapping<Shape> {
    static constexpr std::size_t rank() { return Shape::rank(); }

    // Mirrors tensor::_recast<NewE,...>: NewE is UNRELATED to this tensor's
    // own Shape/FakeMapping -- the real bug is that MSVC's diagnostic for
    // `NewE::rank()` here blames the CURRENT instantiation's own private base
    // instead of (correctly) just evaluating the qualified name.
    template <class NewE>
    static constexpr bool recast_rank_matches() {
        return NewE::rank() == rank();
    }
};

using SrcShape = FakeExtents<5, 6>;   // matches this tensor's OWN shape
using NewShape = FakeExtents<5, 6>;   // the recast TARGET -- unrelated type, same rank

int main() {
    using T = FakeTensor<SrcShape>;
    static_assert(T::recast_rank_matches<NewShape>(), "recast: rank must match");
    return 0;
}
