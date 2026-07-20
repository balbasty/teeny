#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    double buf[120];
    for (long i=0;i<120;++i) buf[i]=i;
    auto t = view(buf, extents<long,2,3,4,5>{});      // rank 4

    // single named axis (integer) -> drop it
    auto a = t.take_along<1>(2);                       // axis1=2 -> (2,4,5)
    static_assert(decltype(a)::rank() == 3, "one axis dropped");
    if (a(1,3,4) != t(1,2,3,4)) return 1;

    // negative index wraps
    auto b = t.take_along<0>(-1);                      // axis0=1 -> (3,4,5)
    if (b(2,3,4) != t(1,2,3,4)) return 2;

    // multiple axes at once (a pack of dimensions)
    auto c = t.take_along<1,3>(2, 4);                  // axis1=2, axis3=4 -> (2,4)
    static_assert(decltype(c)::rank() == 2, "two axes dropped");
    if (c(1,3) != t(1,2,3,4)) return 3;

    // mix an integer and a slice across named axes; keep the rest
    auto d = t.take_along<0,2>(1, slice(1,3));           // axis0=1, axis2=[1,3) -> (3,2,5)
    static_assert(decltype(d)::rank() == 3, "int + slice");
    if (d(2,0,4) != t(1,2,1,4)) return 4;              // axis2 offset 1
    if (d(2,1,4) != t(1,2,2,4)) return 5;

    // static index arg
    auto e = t.take_along<2>(Int<3>());
    if (e(1,2,4) != t(1,2,3,4)) return 6;

    // write-through
    c(0,0) = 777.0;
    if (t(0,2,0,4) != 777.0) return 7;
    return 0;
}
