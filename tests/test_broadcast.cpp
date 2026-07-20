#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    // out-of-place broadcast: (3,1) + (3,4) -> (3,4)
    auto col = local<double, extents<long,3,1>>();
    auto M   = local<double, extents<long,3,4>>();
    for (long i=0;i<3;++i) col(i,0) = (i+1)*100;      // 100,200,300
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) M(i,j) = i*4+j;
    auto R = col + M;
    static_assert(decltype(R)::rank()==2 && decltype(R)::is_static, "static bcast");
    static_assert(decltype(R)::extents_type::static_extent(0)==3 && decltype(R)::extents_type::static_extent(1)==4, "result 3x4");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j)
        if (R(i,j) != col(i,0) + M(i,j)) return 1;

    // (1,4) + (3,4) -> (3,4)
    auto row = local<double, extents<long,1,4>>();
    for (long j=0;j<4;++j) row(0,j) = j*10;
    auto R2 = M + row;
    for (long i=0;i<3;++i) for (long j=0;j<4;++j)
        if (R2(i,j) != M(i,j) + row(0,j)) return 2;

    // in-place broadcast: M += col  (col broadcasts across axis 1)
    auto M2 = M;
    M2.add_(col);
    for (long i=0;i<3;++i) for (long j=0;j<4;++j)
        if (M2(i,j) != M(i,j) + col(i,0)) return 3;

    // non-broadcast (equal shapes) still works
    auto S = M + M;
    if (S(2,3) != 2*M(2,3)) return 4;

    // dynamic broadcast -> heap (host)
    using D2 = extents<long,dynamic_extent,dynamic_extent>;
    auto Dc = owned<double,D2>(D2{3,1}); for(long i=0;i<3;++i) Dc(i,0)=i+1;
    auto Dm = owned<double,D2>(D2{3,4}); for(long i=0;i<3;++i)for(long j=0;j<4;++j) Dm(i,j)=i+j;
    auto Dr = Dc + Dm;
    static_assert(decltype(Dr)::ownership==own::heap, "dynamic bcast -> heap");
    if (Dr.extent(0)!=3 || Dr.extent(1)!=4) return 5;
    if (Dr(2,3) != Dc(2,0)+Dm(2,3)) return 6;

    return 0;
}
