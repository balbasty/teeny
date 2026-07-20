#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    double buf[24];
    for (long i=0;i<24;++i) buf[i]=i;
    auto t = view(buf, extents<long,2,3,4>{});        // strides (12,4,1)

    // all-integer -> element access (T&)
    if (t(1,2,3) != 1*12+2*4+3) return 1;
    static_assert(cs::is_same<decltype(t(0,0,0)), double&>(), "int... -> element");

    // negative indices wrap python-style
    if (t(-1,-1,-1) != t(1,2,3)) return 2;
    if (t(-2,0,0)  != t(0,0,0)) return 3;

    // static index (Int<>) as an element index
    if (t(Int<1>(), 2, 3) != t(1,2,3)) return 4;

    // slice: any `all` / `slice` argument -> a sub-view (tensor)
    auto row = t(1, all, all);                        // fix axis 0 -> (3,4) view
    static_assert(decltype(row)::rank() == 2, "one integer + all,all -> rank 2");
    if (row(2,3) != t(1,2,3)) return 5;

    auto col = t(all, 2, all);                        // fix axis 1 -> (2,4) view
    static_assert(decltype(col)::rank() == 2, "middle fixed");
    if (col(1,3) != t(1,2,3)) return 6;

    // slice: half-open range keeps the axis
    auto sub = t(1, slice(1,3), all);                   // axis0 fixed, axis1 [1,3) -> (2,4)
    static_assert(decltype(sub)::rank() == 2, "slice keeps axis");
    if (sub(0,0) != t(1,1,0)) return 7;               // first row of the range
    if (sub(1,3) != t(1,2,3)) return 8;

    // peel are mutable views (write-through)
    row(0,0) = 999.0;
    if (t(1,0,0) != 999.0) return 9;
    row(0,0) = 0.0;                                   // restore

    // ---- python-like slice: none / negative / step --------------------
    // `none` open ends: slice(none, k) starts at 0; slice(k, none) runs to end.
    auto a = t(0, slice(none, 2), all);               // axis1 [0,2)
    if (a.extent(0) != 2 || a(0,0) != t(0,0,0) || a(1,3) != t(0,1,3)) return 10;
    auto b = t(0, slice(1, none), all);               // axis1 [1,3)
    if (b.extent(0) != 2 || b(0,0) != t(0,1,0)) return 11;
    // slice(none,none) folds to full_extent (== all): keeps the axis AND its
    // static extent (the static none case).
    auto c = t(0, slice(none,none), all);
    static_assert(decltype(c)::extents_type::static_extent(0) == 3, "slice(none,none)==all folds");
    if (c.extent(0) != 3 || c(1,2) != t(0,1,2)) return 12;

    // negative bounds wrap (count from the back)
    auto d = t(0, slice(-2, none), all);              // last two of axis1 -> [1,3)
    if (d.extent(0) != 2 || d(0,0) != t(0,1,0) || d(1,0) != t(0,2,0)) return 13;

    // step: every other element along the last axis (0,2) of 4 -> length 2
    auto e = t(0, 0, slice(0, 4, Int<2>()));
    if (e.extent(0) != 2 || e(0) != t(0,0,0) || e(1) != t(0,0,2)) return 14;
    auto f = t(0, 0, slice(none, none, Int<2>()));    // whole axis, stride 2
    if (f.extent(0) != 2 || f(1) != t(0,0,2)) return 15;

    // slice also works through take_along (same resolution)
    auto g = t.take_along<2>(slice(1, none));         // keep axes 0,1; axis2 [1,4)
    static_assert(decltype(g)::rank() == 3, "take_along keeps unnamed axes");
    if (g.extent(2) != 3 || g(1,2,0) != t(1,2,1)) return 16;

    return 0;
}
