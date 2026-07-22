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

    // `shape` aliases `extent(s)` (python-friendly)
    static_assert(t.shape(Int<1>()) == 3, "shape(Int) == extent(Int)");
    if (t.shape(0) != 2 || t.shape().extent(2) != 4) return 4;
    return 0;
}
