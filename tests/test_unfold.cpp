// unfold<Axis>(size, step) (#256): pytorch-style sliding/strided window view —
// appends a NEW trailing axis of width `size` at a `step` stride along `Axis`.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    auto t = local<double, shape<10>>();
    for (long i=0;i<10;++i) t(i) = static_cast<double>(i);

    // template form: runtime size/step -> dynamic window count
    auto u = t.unfold<0>(3, 1);          // windows [0,1,2],[1,2,3],...,[7,8,9]
    if (u.rank() != 2) return 1;
    if (u.shape(0) != 8 || u.shape(1) != 3) return 2;
    for (long i=0;i<8;++i) for (long j=0;j<3;++j) if (u(i,j) != t(i+j)) return 3;

    // step > 1
    auto u2 = t.unfold<0>(3, 2);         // windows starting 0,2,4,6 -> count 4
    if (u2.shape(0) != 4 || u2.shape(1) != 3) return 4;
    for (long i=0;i<4;++i) for (long j=0;j<3;++j) if (u2(i,j) != t(2*i+j)) return 5;

    // default step == 1
    auto u3 = t.unfold<0>(4);
    if (u3.shape(0) != 7 || u3.shape(1) != 4) return 6;
    for (long i=0;i<7;++i) for (long j=0;j<4;++j) if (u3(i,j) != t(i+j)) return 7;

    // static Int<> size/step fold a fully-static output extent
    auto u4 = t.unfold<0>(Int<3>(), Int<2>());
    static_assert(decltype(u4)::extents_type::static_extent(0) == 4, "unfold: static count folds");
    static_assert(decltype(u4)::extents_type::static_extent(1) == 3, "unfold: static size folds");
    for (long i=0;i<4;++i) for (long j=0;j<3;++j) if (u4(i,j) != u2(i,j)) return 8;

    // value form: t.unfold(Int<0>(), size, step) == t.unfold<0>(size, step)
    auto u5 = t.unfold(Int<0>(), 3, 2);
    for (long i=0;i<4;++i) for (long j=0;j<3;++j) if (u5(i,j) != u2(i,j)) return 9;

    // negative axis (last axis)
    auto m = local<double, shape<2,6>>();
    for (long i=0;i<2;++i) for (long j=0;j<6;++j) m(i,j) = i*10.0 + j;
    auto um = m.unfold<-1>(2, 2);         // windows along axis 1: count (6-2)/2+1 = 3
    if (um.rank() != 3 || um.shape(1) != 3 || um.shape(2) != 2) return 10;
    for (long i=0;i<2;++i) for (long w=0;w<3;++w) for (long j=0;j<2;++j)
        if (um(i,w,j) != m(i, 2*w+j)) return 11;

    // ND unfold via chaining: unfold axis 0 then (now-shifted) axis 1 -> two
    // trailing window axes, matching nitorch's compose-by-axis nd-unfold.
    auto g = local<double, shape<4,4>>();
    for (long i=0;i<4;++i) for (long j=0;j<4;++j) g(i,j) = i*10.0 + j;
    auto und = g.unfold<0>(2,1).unfold<1>(2,1);   // (3,3,2,2): 3x3 grid of 2x2 windows
    if (und.rank() != 4) return 12;
    if (und.shape(0) != 3 || und.shape(1) != 3 || und.shape(2) != 2 || und.shape(3) != 2) return 13;
    for (long i=0;i<3;++i) for (long j=0;j<3;++j) for (long a=0;a<2;++a) for (long b=0;b<2;++b)
        if (und(i,j,a,b) != g(i+a, j+b)) return 14;

    // const overload
    const auto & ct = t;
    auto uc = ct.unfold<0>(3, 1);
    for (long i=0;i<8;++i) for (long j=0;j<3;++j) if (uc(i,j) != u(i,j)) return 15;

    // write-through: unfold is a VIEW; mutating an overlapping window mutates
    // the shared source elements (step < size -> windows alias, as in pytorch).
    auto w = local<double, shape<5>>(); w.zero_();
    auto uw = w.unfold<0>(2, 1);   // windows: [0,1],[1,2],[2,3],[3,4]
    uw.at(0,1).atomic_add_(1.0);   // bump element 1 via window 0's second tap
    if (w(1) != 1.0) return 16;

    // F-contiguous source: the new trailing axis's stride must be axis Axis's
    // ORIGINAL (un-stepped) source stride, not assumed contiguous (#339 review).
    auto f = local<double, shape<4,3>, fcontiguous>();
    for (long i=0;i<4;++i) for (long j=0;j<3;++j) f(i,j) = i*10.0 + j;
    auto uf = f.unfold<0>(2, 1);   // axis 0 (stride 1 in F-order) -> windows of 2 rows
    if (uf.shape(0) != 3 || uf.shape(1) != 3 || uf.shape(2) != 2) return 17;
    for (long i=0;i<3;++i) for (long j=0;j<3;++j) for (long a=0;a<2;++a)
        if (uf(i,j,a) != f(i+a,j)) return 18;

    // dynamic-shape source with RUNTIME size/step (the non-folding path).
    auto dyn = make_heap<double>(shape<-1>{7});
    for (long i=0;i<7;++i) dyn(i) = static_cast<double>(i);
    long rsize = 3, rstep = 2;
    auto ud = dyn.unfold<0>(rsize, rstep);   // count (7-3)/2+1 = 3
    if (ud.shape(0) != 3 || ud.shape(1) != 3) return 19;
    for (long i=0;i<3;++i) for (long j=0;j<3;++j) if (ud(i,j) != dyn(2*i+j)) return 20;

    return 0;
}
