#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

// A bare `none` argument to operator()/uget is numpy `newaxis` (`a[None]`): it
// inserts a size-1 axis (static extent 1, stride 0) at that position, consuming
// NO source axis. This test proves, via ELEMENT-IDENTITY, that `none`-insertion
// is the same view as `unsqueeze` at the same position, and that it composes with
// integers, ranges, ellipsis, and multiple `none`s.

// element-identity of two rank-3 views over the same (i,j,k) grid
template <class A, class B>
bool same3(const A & x, const B & y) {
    if (A::rank() != 3 || B::rank() != 3) return false;
    for (long i = 0; i < 3; ++i) if (x.extent(i) != y.extent(i)) return false;
    for (long i = 0; i < (long)x.extent(0); ++i)
        for (long j = 0; j < (long)x.extent(1); ++j)
            for (long k = 0; k < (long)x.extent(2); ++k)
                if (x(i,j,k) != y(i,j,k)) return false;
    return true;
}

int main() {
    double buf[12];
    for (long i = 0; i < 12; ++i) buf[i] = i;
    auto t = wrap(buf, shape<3,4>{});      // (H,W) = (3,4), strides (4,1)

    // ---- `newaxis` is a named alias of `none` (same type/value; numpy's np.newaxis
    //      vs None distinction, collapsed to one teeny type) ----
    static_assert(cs::is_same<decltype(newaxis), decltype(none)>::value, "newaxis IS none (same type)");

    // ---- t(none,all,all) == unsqueeze<0>()  : (3,4) -> (1,3,4) ----
    auto a0 = t(none, all, all);
    auto u0 = t.unsqueeze<0>();
    static_assert(decltype(a0)::rank() == 3, "newaxis at 0 -> rank 3");
    static_assert(decltype(a0)::extents_type::static_extent(0) == 1, "inserted axis is static 1");
    if (a0.extent(0) != 1 || a0.extent(1) != 3 || a0.extent(2) != 4) return 1;
    if (!same3(a0, u0)) return 2;

    // ---- t(all,none,all) == unsqueeze<1>()  : (3,4) -> (3,1,4) ----
    auto a1 = t(all, none, all);
    auto u1 = t.unsqueeze<1>();
    static_assert(decltype(a1)::rank() == 3, "newaxis at 1 -> rank 3");
    static_assert(decltype(a1)::extents_type::static_extent(1) == 1, "inserted axis is static 1");
    if (!same3(a1, u1)) return 3;

    // ---- t(all,all,none) == unsqueeze<-1>() : (3,4) -> (3,4,1) ----
    auto a2 = t(all, all, none);
    auto u2 = t.unsqueeze<-1>();
    static_assert(decltype(a2)::rank() == 3, "newaxis at end -> rank 3");
    static_assert(decltype(a2)::extents_type::static_extent(2) == 1, "inserted trailing axis is static 1");
    if (!same3(a2, u2)) return 4;

    // ---- inserted axis has stride 0 (folded) — writing through the source shows
    //      the single [.,0,.] plane IS the whole source (index into axis is 0) ----
    a1(2, 0, 3) = 999.0;
    if (t(2,3) != 999.0) return 5;
    a1(2, 0, 3) = 11.0;                         // restore
    if (t(2,3) != 11.0) return 6;

    // ---- ellipsis + none: t(ellipsis, none) appends a trailing size-1 axis ----
    auto e0 = t(ellipsis, none);                // (3,4) -> (3,4,1)
    static_assert(decltype(e0)::rank() == 3, "ellipsis+none -> rank 3");
    static_assert(decltype(e0)::extents_type::static_extent(2) == 1, "trailing newaxis static 1");
    if (!same3(e0, u2)) return 7;

    // ---- ellipsis + leading none: t(none, ellipsis) prepends ----
    auto e1 = t(none, ellipsis);                // (3,4) -> (1,3,4)
    static_assert(decltype(e1)::rank() == 3, "none+ellipsis -> rank 3");
    if (!same3(e1, u0)) return 8;

    // ---- int + newaxis: t(0, none, all) : (3,4) -> (1,4) [numpy a[0,None,:]] ----
    auto in0 = t(0, none, all);
    static_assert(decltype(in0)::rank() == 2, "int drops one axis, none adds one");
    static_assert(decltype(in0)::extents_type::static_extent(0) == 1, "inserted axis static 1");
    if (in0.extent(0) != 1 || in0.extent(1) != 4) return 9;
    for (long k = 0; k < 4; ++k) if (in0(0, k) != t(0, k)) return 10;

    // ---- multiple none: t(none, all, none, all, none) : (3,4)->(1,3,1,4,1) ----
    auto m0 = t(none, all, none, all, none);
    static_assert(decltype(m0)::rank() == 5, "3 newaxes on a rank-2 source -> rank 5");
    static_assert(decltype(m0)::extents_type::static_extent(0) == 1 &&
                  decltype(m0)::extents_type::static_extent(2) == 1 &&
                  decltype(m0)::extents_type::static_extent(4) == 1, "the three inserted axes are static 1");
    if (m0.extent(1) != 3 || m0.extent(3) != 4) return 11;
    for (long i = 0; i < 3; ++i) for (long j = 0; j < 4; ++j)
        if (m0(0, i, 0, j, 0) != t(i, j)) return 12;

    // ---- none composes with a range (non-full slice) ----
    auto r0 = t(none, all, slice(1,3));         // (3,4) -> (1,3,2)
    static_assert(decltype(r0)::rank() == 3, "none + all + range -> rank 3");
    static_assert(decltype(r0)::extents_type::static_extent(0) == 1, "inserted axis static 1");
    if (r0.extent(0) != 1 || r0.extent(1) != 3 || r0.extent(2) != 2) return 13;
    for (long i = 0; i < 3; ++i) for (long j = 0; j < 2; ++j)
        if (r0(0, i, j) != t(i, 1 + j)) return 14;

    // ---- the inserted axis' stride folds to a static 0 (strides<...> layout) ----
    static_assert(decltype(a0)::layout_type::S_[0] == 0,
                  "inserted axis has a folded static stride 0");

    // ---- uget mirrors operator(): same result, unchecked path ----
    auto g0 = t.uget(none, all, all);
    static_assert(decltype(g0)::rank() == 3, "uget newaxis -> rank 3");
    if (!same3(g0, u0)) return 15;

    // ---- `newaxis` works as an actual argument, identically to `none` ----
    auto an = t(newaxis, all, all);             // == t(none, all, all)
    static_assert(decltype(an)::rank() == 3, "newaxis arg -> rank 3");
    if (!same3(an, a0)) return 18;

    // ---- a dynamic-shape source: none still inserts a STATIC 1 axis ----
    auto td = wrap(buf, shape<cs::dynamic_extent,4>{3, 4});
    auto d0 = td(all, none, all);               // (3,?4) -> (3,1,4), axis0 dynamic
    static_assert(decltype(d0)::rank() == 3, "dynamic source, newaxis -> rank 3");
    static_assert(decltype(d0)::extents_type::static_extent(1) == 1, "newaxis static even on dynamic source");
    if (d0.extent(0) != 3 || d0.extent(1) != 1 || d0.extent(2) != 4) return 16;
    for (long i = 0; i < 3; ++i) for (long j = 0; j < 4; ++j)
        if (d0(i, 0, j) != t(i, j)) return 17;

    return 0;
}
