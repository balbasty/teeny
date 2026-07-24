#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    double buf[24];
    auto t = wrap(buf, extents<long,2,3,4>{});
    // static axis (Int<D>) -> compile-time integral_constant when the extent is static
    static_assert(t.extent(Int<1>()) == 3, "static extent value");
    static_assert(cs::is_same<decltype(t.extent(Int<1>())), cs::integral_constant<long,3>>(), "static extent type");
    static_assert(t.stride(Int<0>()) == 12 && t.stride(Int<2>()) == 1, "static stride values");
    static_assert(cs::is_same<decltype(t.stride(Int<0>())), cs::integral_constant<long,12>>(), "static stride type");
    long s0 = t.stride(Int<0>());  if (s0 != 12) return 1;         // converts to runtime

    // runtime axis -> runtime index_type
    static_assert(cs::is_same<decltype(t.extent(0)), long>(), "runtime extent -> long");
    if (t.extent(1) != 3) return 2;

    // dynamic extent, static axis -> runtime
    using E = extents<long, dynamic_extent, 3>;
    auto d = wrap(buf, E{2});
    static_assert(d.extent(Int<1>()) == 3, "static in mixed");
    static_assert(cs::is_same<decltype(d.extent(Int<0>())), long>(), "dynamic extent, static axis -> runtime");
    if (d.extent(Int<0>()) != 2) return 3;

    // static-stride layout -> static stride even with dynamic extents
    auto vs = wrap(buf, extents<long,dynamic_extent,3,3>{2}, strides<16,3,1>{});
    static_assert(vs.stride(Int<0>()) == 16, "static-stride layout");

    // the UNIT stride of a contiguous layout is static 1 even with a dynamic shape:
    // layout_right -> last axis, layout_left -> first axis.
    using DynE = extents<long, dynamic_extent, dynamic_extent, dynamic_extent>;
    auto dr = wrap(buf, DynE{2,3,4});                      // layout_right
    static_assert(cs::is_same<decltype(dr.stride(Int<2>())), cs::integral_constant<long,1>>(), "right unit stride folds");
    static_assert(cs::is_same<decltype(dr.stride(Int<0>())), long>(), "right outer stride runtime");
    auto dl = wrap<cs::layout_left>(buf, DynE{2,3,4});     // layout_left
    static_assert(cs::is_same<decltype(dl.stride(Int<0>())), cs::integral_constant<long,1>>(), "left unit stride folds");

    // a PARTIALLY-dynamic contiguous shape folds every stride derivable from its
    // static extents (not just the unit one): shape<-1,3,3> row-major -> stride(0)
    // = 3*3 = 9 and stride(1) = 3 fold, even though the OUTER extent is dynamic.
    auto pc = wrap(buf, extents<long,dynamic_extent,3,3>{2});
    static_assert(cs::is_same<decltype(pc.stride(Int<0>())), cs::integral_constant<long,9>>(), "partial-dyn outer stride folds (3*3)");
    static_assert(cs::is_same<decltype(pc.stride(Int<1>())), cs::integral_constant<long,3>>(), "partial-dyn inner stride folds");
    static_assert(cs::is_same<decltype(pc.stride(Int<2>())), cs::integral_constant<long,1>>(), "partial-dyn unit stride folds");
    // but a dynamic INNER extent makes the outer stride genuinely runtime
    auto pd = wrap(buf, extents<long,3,dynamic_extent,3>{4});   // shape<3,-1,3>
    static_assert(cs::is_same<decltype(pd.stride(Int<0>())), long>(), "dyn inner extent -> outer stride runtime");
    static_assert(cs::is_same<decltype(pd.stride(Int<2>())), cs::integral_constant<long,1>>(), "unit still folds");

    // `shape` aliases `extent(s)` (python-friendly)
    static_assert(t.shape(Int<1>()) == 3, "shape(Int) == extent(Int)");
    if (t.shape(0) != 2 || t.shape().extent(2) != 4) return 4;

    // ---- shape()/strides() are array-like: Int<k>() folds, runtime stays runtime --
    static_assert(cs::is_same<decltype(t.shape()[Int<1>()]), cs::integral_constant<long,3>>(),
                  "shape()[Int] folds to integral_constant");
    static_assert(cs::is_same<decltype(t.strides()[Int<2>()]), cs::integral_constant<long,1>>(),
                  "strides()[Int] unit stride folds");
    static_assert(decltype(t.strides()[Int<0>()])::value == 12, "ccontiguous outer stride folds (3*4)");
    static_assert(decltype(t.shape()[Int<-1>()])::value == 4, "negative axis index folds (last)");
    if (t.shape()[1] != 3 || t.strides()[1] != 4) return 5;              // runtime indices
    if (t.shape().rank() != 3 || t.strides().rank() != 3) return 6;
    { long p = 1; for (auto e : t.shape())   p *= e; if (p != 24) return 7; }   // iterate (runtime)
    { long s = 0; for (auto v : t.strides()) s += v; if (s != 17) return 8; }
    { shape<2,3,4> e = t.shape(); if (e.extent(0) != 2) return 9; }      // converts to raw extents

    // dynamic_strides: no static fold, runtime values are correct
    auto dv = wrap(buf, DynE{2,3,4}, {12,4,1});
    static_assert(cs::is_same<decltype(dv.strides()[Int<1>()]), long>(), "dynamic stride stays runtime");
    if (dv.shape()[0] != 2 || dv.strides()[0] != 12 || dv.strides()[2] != 1) return 10;
    return 0;
}
