// subsample (#258): coloured/strided sub-lattice sugar on top of slice_along +
// slice — bind named axes to a shared step k, each with its own start.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    auto t = local<double, shape<10,10>>();
    for (long i=0;i<10;++i) for (long j=0;j<10;++j) t(i,j) = i*10.0 + j;

    // template form: subsample<Axes...>(k, starts...) -- runtime args, dynamic extent
    auto s = t.subsample<0,1>(2, 1, 0);   // step 2 both axes, start (1,0)
    for (long i=0;i<5;++i) for (long j=0;j<5;++j) if (s(i,j) != t(1+2*i, 0+2*j)) return 1;

    // value form: t.subsample(axis<...>{}, k, starts...) == t.subsample<...>(k, starts...)
    auto s2 = t.subsample(axis<0,1>{}, 2, 1, 0);
    for (long i=0;i<5;++i) for (long j=0;j<5;++j) if (s2(i,j) != s(i,j)) return 2;

    // static k/starts (Int<>) fold a fully-static output extent, same as slice()
    auto s3 = t.subsample<0,1>(Int<2>(), Int<1>(), Int<0>());
    static_assert(decltype(s3)::extents_type::static_extent(0) == 5, "static Int<> path folds extent(0)");
    static_assert(decltype(s3)::extents_type::static_extent(1) == 5, "static Int<> path folds extent(1)");
    for (long i=0;i<5;++i) for (long j=0;j<5;++j) if (s3(i,j) != s(i,j)) return 3;

    // single named axis, other axis kept whole
    auto s4 = t.subsample<1>(3, 0);   // axis 1: step 3, start 0; axis 0 untouched
    for (long i=0;i<10;++i) for (long j=0;j<4;++j) if (s4(i,j) != t(i, j*3)) return 4;

    // negative start wraps (built on slice(), which already wraps) — a RUNTIME
    // start, so only when the build wraps at all: -DTNY_NO_NEGATIVE_INDEX drops
    // the wrap, and a negative runtime start is then undefined behaviour.
#ifndef TNY_NO_NEGATIVE_INDEX
    auto s5 = t.subsample<0>(2, -2);   // start -2 == 8 (last-but-one row)
    for (long j=0;j<10;++j) if (s5(0,j) != t(8,j)) return 5;
#endif

    // const overload
    const auto & ct = t;
    auto s6 = ct.subsample<0,1>(2, 1, 0);
    for (long i=0;i<5;++i) for (long j=0;j<5;++j) if (s6(i,j) != s(i,j)) return 6;

    // write-through: subsample is a VIEW, mutating it mutates the source
    auto w = local<double, shape<6>>(); w.zero_();
    auto sw = w.subsample<0>(2, 0);   // elements 0,2,4
    sw.fill_(9.0);
    if (w(0)!=9.0 || w(1)!=0.0 || w(2)!=9.0 || w(3)!=0.0 || w(4)!=9.0 || w(5)!=0.0) return 7;

    return 0;
}
