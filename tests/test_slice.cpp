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

    // slice: any `all` / `rng` argument -> a sub-view (tensor)
    auto row = t(1, all, all);                        // fix axis 0 -> (3,4) view
    static_assert(decltype(row)::rank() == 2, "one integer + all,all -> rank 2");
    if (row(2,3) != t(1,2,3)) return 5;

    auto col = t(all, 2, all);                        // fix axis 1 -> (2,4) view
    static_assert(decltype(col)::rank() == 2, "middle fixed");
    if (col(1,3) != t(1,2,3)) return 6;

    // rng: half-open range keeps the axis
    auto sub = t(1, rng(1,3), all);                   // axis0 fixed, axis1 [1,3) -> (2,4)
    static_assert(decltype(sub)::rank() == 2, "rng keeps axis");
    if (sub(0,0) != t(1,1,0)) return 7;               // first row of the range
    if (sub(1,3) != t(1,2,3)) return 8;

    // slices are mutable views (write-through)
    row(0,0) = 999.0;
    if (t(1,0,0) != 999.0) return 9;

    return 0;
}
