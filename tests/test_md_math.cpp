#include <teeny/md.h>
#include <cuda/std/type_traits>

using namespace tny::md;
namespace cs = cuda::std;
using cs::extents;
using cs::dynamic_extent;

int main()
{
    // ---- math on fully static tensors: a + b -> c (stack-owned) --------
    auto A = local<double, extents<long,2,2>>();
    auto B = local<double, extents<long,2,2>>();
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    B(0,0)=10;B(0,1)=20;B(1,0)=30;B(1,1)=40;

    auto C = A + B;
    static_assert(decltype(C)::ownership == own::stack, "out-of-place result is stack-owned");
    static_assert(decltype(C)::is_static, "result extent statically known");
    static_assert(sizeof(C) == 4*sizeof(double), "result stores exactly its data");
    if (C(0,0)!=11 || C(0,1)!=22 || C(1,0)!=33 || C(1,1)!=44) return 1;

    if ((A - B)(0,0) != -9)  return 2;
    if ((A * B)(1,1) != 160) return 3;
    if ((B / A)(1,1) != 10)  return 4;

    // result type follows common_type (int + double -> double)
    auto Ai = local<int, extents<long,2,2>>(); Ai(0,0)=5;
    auto Ad = local<double, extents<long,2,2>>(); Ad(0,0)=0.5;
    static_assert(cs::is_same<decltype(Ai + Ad)::element_type, double>(), "common_type result");
    if ((Ai + Ad)(0,0) != 5.5) return 5;

    // ---- in-place on a strided view into a bigger buffer ---------------
    double buf[9]; for (int i=0;i<9;++i) buf[i]=i;
    auto full = view(buf, extents<long,3,3>{});
    full.add_(100.0);                              // scalar in place
    if (buf[0]!=100 || buf[8]!=108) return 6;
    full.mul_(2.0);
    if (buf[4]!=208) return 7;

    auto ones = local<double, extents<long,3,3>>(); ones.add_(1.0);
    full.sub_(ones);                               // elementwise in place
    if (buf[4]!=207) return 8;

    // in-place works between a view and a stack tensor of matching shape
    auto twos = local<double, extents<long,3,3>>(); twos.add_(2.0);
    ones.mul_(twos);                               // ones now all 2
    if (ones(1,1) != 2) return 9;

    // ---- reductions ----------------------------------------------------
    auto R = local<double, extents<long,2,2>>();
    R(0,0)=1; R(0,1)=2; R(1,0)=3; R(1,1)=4;
    if (sum(R) != 10) return 10;
    if (dot(R, R) != 1+4+9+16) return 11;          // 30
    if (prod(R) != 24) return 20;                  // 1*2*3*4
    if (max(R) != 4)  return 21;
    if (min(R) != 1)  return 22;

    // reduction over the (contiguous) view: buf[i] == 2*i + 199, i = 0..8
    double expect = 0; for (int i=0;i<9;++i) expect += 2*i + 199;
    if (sum(full) != expect) return 12;

    // ---- out-of-place on DYNAMIC extents: allowed on the host, heap result
    using DynE = extents<long, dynamic_extent, dynamic_extent>;
    auto Da = owned<double, DynE>(DynE{2,2});
    auto Db = owned<double, DynE>(DynE{2,2});
    Da(0,0)=1; Da(0,1)=2; Da(1,0)=3; Da(1,1)=4;
    Db(0,0)=10;Db(0,1)=20;Db(1,0)=30;Db(1,1)=40;
    auto Dc = Da + Db;                             // -> heap-owned (host)
    static_assert(decltype(Dc)::ownership == own::heap, "dynamic out-of-place -> heap");
    if (Dc(0,0)!=11 || Dc(1,1)!=44) return 13;

    return 0;
}
