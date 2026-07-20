#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;                      // extents, layout_*, ic, all now visible
namespace cs = cuda::std;

int main() {
    double buf[24];
    auto t = view(buf, extents<long,2,3,4>{});     // static extents, layout_right
    // extent<D>() -> compile-time integral_constant
    static_assert(t.extent<1>() == 3, "static extent value");
    static_assert(cs::is_same<decltype(t.extent<1>()), cs::integral_constant<long,3>>(), "extent<D> static type");
    // stride<D>() -> compile-time for contiguous static extents
    static_assert(t.stride<0>() == 12 && t.stride<2>() == 1, "static stride values");
    static_assert(cs::is_same<decltype(t.stride<0>()), cs::integral_constant<long,12>>(), "stride<D> static type");
    // integral_constant converts implicitly to a runtime integral
    long s0 = t.stride<0>();  if (s0 != 12) return 1;

    // dynamic extent -> runtime index_type (not integral_constant)
    using E = extents<long, dynamic_extent, 3>;
    auto d = view(buf, E{2});
    static_assert(d.extent<1>() == 3, "static in mixed");
    static_assert(cs::is_same<decltype(d.extent<0>()), long>(), "dynamic extent<D> -> runtime");
    if (d.extent<0>() != 2) return 2;

    // static-stride layout -> stride<D> static even with dynamic extents
    auto vs = view_strided<16,3,1>(buf, extents<long,dynamic_extent,3,3>{2});
    static_assert(vs.stride<0>() == 16, "static-stride layout stride<D>");

    return 0;
}
