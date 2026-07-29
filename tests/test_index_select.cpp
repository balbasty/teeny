// index_select (#326): gather along one axis using an arbitrary integer index
// TENSOR (runtime data, not compile-time indices/ranges like take_along).
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    // a small "vertex buffer": 5 vertices, 3 coords each.
    auto verts = local<double, shape<5,3>>();
    for (long i=0;i<5;++i) for (long j=0;j<3;++j) verts(i,j) = i*10.0 + j;

    // static idx shape -> static (stack) result.
    auto idx = local<long, shape<3>>(); idx(0)=2; idx(1)=0; idx(2)=4;
    auto sel = verts.index_select<0>(idx);
    static_assert(decltype(sel)::extents_type::static_extent(0) == 3, "static idx -> static result axis");
    static_assert(decltype(sel)::extents_type::static_extent(1) == 3, "other axis unchanged");
    for (long j=0;j<3;++j) {
        if (sel(0,j) != verts(2,j)) return 1;
        if (sel(1,j) != verts(0,j)) return 2;
        if (sel(2,j) != verts(4,j)) return 3;
    }

    // negative idx values wrap (built on take_along, which already wraps).
    auto idxn = local<long, shape<2>>(); idxn(0)=-1; idxn(1)=-5;
    auto seln = verts.index_select<0>(idxn);
    if (seln(0,0) != verts(4,0)) return 4;   // -1 -> last row (4)
    if (seln(1,0) != verts(0,0)) return 5;   // -5 -> row 0

    // dynamic idx shape -> heap result.
    auto idxd = owned<long, shape<-1>>(shape<-1>{2}); idxd(0)=1; idxd(1)=3;
    auto seld = verts.index_select<0>(idxd);
    static_assert(decltype(seld)::ownership == storage::heap, "dynamic idx -> heap result");
    for (long j=0;j<3;++j) { if (seld(0,j)!=verts(1,j)) return 6; if (seld(1,j)!=verts(3,j)) return 7; }

    // gather along a non-leading axis.
    auto idx1 = local<long, shape<2>>(); idx1(0)=2; idx1(1)=0;
    auto sel1 = verts.index_select<1>(idx1);          // (5,2): columns 2,0
    for (long i=0;i<5;++i) { if (sel1(i,0)!=verts(i,2)) return 8; if (sel1(i,1)!=verts(i,0)) return 9; }

    // into(dest): no allocation, writes straight into a preallocated buffer.
    auto dest = local<double, shape<3,3>>();
    verts.index_select<0>(idx, into(dest));
    for (long j=0;j<3;++j) {
        if (dest(0,j) != verts(2,j)) return 10;
        if (dest(1,j) != verts(0,j)) return 11;
        if (dest(2,j) != verts(4,j)) return 12;
    }

    // repeated index (gather, not a permutation) is allowed.
    auto idxr = local<long, shape<2>>(); idxr(0)=1; idxr(1)=1;
    auto selr = verts.index_select<0>(idxr);
    if (selr(0,0) != selr(1,0) || selr(0,0) != verts(1,0)) return 13;

    // negative AXIS wraps too (the library-wide signed-axis convention).
    auto selm1 = verts.index_select<-1>(idx1);   // axis -1 == axis 1
    for (long i=0;i<5;++i) { if (selm1(i,0)!=verts(i,2)) return 14; if (selm1(i,1)!=verts(i,0)) return 15; }

    // empty idx -> extent-0 result, no crash.
    auto idxe = local<long, shape<0>>();
    auto sele = verts.index_select<0>(idxe);
    static_assert(decltype(sele)::extents_type::static_extent(0) == 0, "empty idx -> extent-0 result");
    if (sele.numel() != 0) return 16;

    // value form: t.index_select(idx, axis<Axis>{}) == t.index_select<Axis>(idx),
    // and needs no `.template` disambiguator on a type-dependent receiver (#332 review).
    auto selv = verts.index_select(idx, axis<0>{});
    for (long j=0;j<3;++j) if (selv(0,j) != verts(2,j)) return 17;
    auto destv = local<double, shape<3,3>>();
    verts.index_select(idx, axis<0>{}, into(destv));
    for (long j=0;j<3;++j) if (destv(0,j) != verts(2,j)) return 18;

    return 0;
}
