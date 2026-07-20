#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;

int main()
{
    double buf[24];
    for (long i = 0; i < 24; ++i) buf[i] = i;
    auto t = view(buf, extents<long,2,3,4>{});          // strides (12,4,1)

    // ---- sub<D>(i): bind an axis, drop it ------------------------------
    auto s1 = t.sub<1>(2);                              // fix axis 1 -> (2,4)
    static_assert(decltype(s1)::rank() == 2, "sub drops an axis");
    if (s1(1,3) != t(1,2,3)) return 1;
    if (s1(0,0) != t(0,2,0)) return 2;

    auto s0 = t.sub<0>(1);                              // fix axis 0 -> (3,4)
    if (s0(2,3) != t(1,2,3)) return 3;

    // sub is a mutable view: writing through it hits the original buffer
    s1(0,0) = 999.0;
    if (t(0,2,0) != 999.0) return 4;

    // ---- permute<...>() ------------------------------------------------
    auto p = t.permute<2,0,1>();                        // (2,3,4) -> (4,2,3)
    static_assert(decltype(p)::rank() == 3, "permute keeps rank");
    if (p.extent(0) != 4 || p.extent(1) != 2 || p.extent(2) != 3) return 5;
    for (long i=0;i<2;++i) for (long j=0;j<3;++j) for (long k=0;k<4;++k)
        if (p(k,i,j) != t(i,j,k)) return 6;

    // permute is a view too
    p(0,0,0) = 7.0;                                     // == t(0,0,0)
    if (t(0,0,0) != 7.0) return 7;

    // ---- on a stack tensor ---------------------------------------------
    auto m = local<double, extents<long,2,2>>();
    m(0,0)=1; m(0,1)=2; m(1,0)=3; m(1,1)=4;
    auto mt = m.permute<1,0>();                         // transpose view
    if (mt(0,1) != m(1,0) || mt(1,0) != m(0,1)) return 8;

    return 0;
}
