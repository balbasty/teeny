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

    // ---- #32: numpy LEFT-PAD broadcasting (lower-rank operand right-aligned) ----
    double vb[4] = {10,20,30,40}; auto v = wrap(vb, shape<4>{});   // rank 1
    // (3,4) + (4,) -> (3,4): rhs left-padded to (1,4)
    auto Rlp = M + v;
    static_assert(decltype(Rlp)::rank()==2, "left-pad result rank 2");
    static_assert(decltype(Rlp)::extents_type::static_extent(0)==3, "result 3x4");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (Rlp(i,j) != M(i,j)+v(j)) return 10;
    // commutes: (4,) + (3,4) -> (3,4), lhs left-padded
    auto Rlp2 = v + M;
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (Rlp2(i,j) != v(j)+M(i,j)) return 11;
    // in-place: higher-rank lhs, lower-rank rhs broadcasts in
    auto M3 = M; M3.add_(v);
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (M3(i,j) != M(i,j)+v(j)) return 12;
    // rank gap > 1: (2,1,4) + (4,) -> (2,1,4)
    double tb[8]; for(int i=0;i<8;++i) tb[i]=i; auto t3 = wrap(tb, shape<2,1,4>{});
    auto Rlp3 = t3 + v;
    static_assert(decltype(Rlp3)::rank()==3, "rank-3 left-pad");
    if (Rlp3(0,0,0)!=0+10 || Rlp3(1,0,3)!=7+40) return 13;
    // comparison broadcasts with left-pad too
    auto cmp = (M >= v);   // (3,4) vs (4,)
    static_assert(decltype(cmp)::rank()==2, "cmp left-pad rank 2");
    if (cmp(0,0) != (M(0,0) >= v(0))) return 14;
    // allclose broadcasts with left-pad
    double eb[4]; for(int j=0;j<4;++j) eb[j]=M(1,j); auto erow = wrap(eb, shape<4>{});
    if (!allclose(wrap(&M(1,0), shape<4>{}), erow)) return 15;   // (4,) vs (4,)
    // dynamic + lower static rank: (n,4) dyn + (4,) -> heap
    auto Dv = owned<double, extents<long,dynamic_extent,4>>(extents<long,dynamic_extent,4>{3});
    for(long i=0;i<3;++i) for(long j=0;j<4;++j) Dv(i,j)=i*4+j;
    auto Rdyn = Dv + v;
    static_assert(decltype(Rdyn)::ownership==own::heap, "dyn left-pad -> heap");
    if (Rdyn.extent(0)!=3 || Rdyn.extent(1)!=4 || Rdyn(2,3)!=Dv(2,3)+v(3)) return 16;

    // #32: a rank-0 (0-d) operand broadcasts as a scalar (numpy 0-d), and must
    // COMPILE (bc_ext/bc_str guard rank-0 so CCCL's rank>0 stride isn't touched).
    auto s0 = M.at(1,1);                       // rank-0 view (== M(1,1))
    static_assert(decltype(s0)::rank()==0, "at() -> rank-0");
    auto Rz = M + s0;
    static_assert(decltype(Rz)::rank()==2, "0-d operand -> result rank 2");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (Rz(i,j) != M(i,j)+M(1,1)) return 17;
    auto Mz = M; Mz.add_(s0);                  // in-place with a 0-d rhs
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (Mz(i,j) != M(i,j)+M(1,1)) return 18;

    return 0;
}
