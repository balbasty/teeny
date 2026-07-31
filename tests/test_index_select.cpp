// index_select (#326): gather along one axis using an arbitrary integer index
// TENSOR (runtime data, not compile-time indices/ranges like slice_along).
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

namespace tny_test {
// #375 regression guard for the whole POINT of the axis<> value form: on a
// TYPE-DEPENDENT receiver it must deduce `Axis` from the tag and so need NO
// `.template` disambiguator. Splitting the forwarder in two (an _TNY_API static
// arm + a _TNY_HOST dynamic arm) must not change that -- both arms are still
// found by ordinary argument-dependent overload resolution, with nothing for the
// caller to spell differently. `Src`/`Idx`/`Ref` are all template parameters, so
// every call below is genuinely dependent: if the split broke deduction this
// function would not compile at all.
template <class Src, class Idx, class Ref>
bool dependent_index_select(const Src & src, const Idx & idx, const Ref & ref) {
    auto got = src.index_select(idx, axis<0>{});          // NO `.template` -- deduced
    auto exp = src.template index_select<0>(idx);         // the <Axis> twin DOES need it
    static_assert(cs::is_same<decltype(got), decltype(exp)>::value,
                  "#375: value form and <Axis> form must agree on the result type");
    if (got.rank() != ref.rank()) return false;
    for (long d = 0; d < (long) got.rank(); ++d)
        if ((long) got.shape(d) != (long) ref.shape(d)) return false;
    const long n = (long) ref.shape(0), m = (long) ref.shape(1);
    for (long i = 0; i < n; ++i)
        for (long j = 0; j < m; ++j)
            if (got(i,j) != ref(i,j) || exp(i,j) != ref(i,j)) return false;
    return true;
}
} // namespace tny_test

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

    // negative idx values wrap (built on slice_along, which already wraps).
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

    // negative idx values must still wrap when *this's own index_type is UNSIGNED
    // (#332 review: a naive cast of idx(j) straight to index_type reinterprets a
    // negative value as a huge unsigned one on such a tensor -- regression guard).
    double ubuf[15]; for (long i=0;i<15;++i) ubuf[i] = i;
    auto uverts = wrap(ubuf, cs::extents<unsigned,5,3>{});
    auto idxun = local<long, shape<2>>(); idxun(0)=-1; idxun(1)=-5;
    auto seluoop = uverts.index_select<0>(idxun);
    if (seluoop(0,0) != uverts(4,0)) return 19;   // -1 -> last row (4)
    if (seluoop(1,0) != uverts(0,0)) return 20;   // -5 -> row 0
    auto destu = local<double, shape<2,3>>();
    uverts.index_select<0>(idxun, into(destu));
    if (destu(0,0) != uverts(4,0) || destu(1,0) != uverts(0,0)) return 21;

    // #375: the axis<> value form is SPLIT into an _TNY_API (static result ->
    // stack) and a _TNY_HOST (dynamic result -> heap) forwarder, so nvcc's
    // device pass never sees a __host__ __device__ forwarder call a __host__
    // allocator. Host-side, the split must be INVISIBLE: each arm has to pick
    // the same underlying overload -- and hence the same ownership and the same
    // values -- as the equivalent <Axis> template-form spelling.
    // (a) static idx -> _TNY_API arm -> stack result, == index_select<0>(idx).
    auto vs = verts.index_select(idx, axis<0>{});
    static_assert(decltype(vs)::ownership == storage::stack, "#375: static value form -> stack (_TNY_API arm)");
    static_assert(cs::is_same<decltype(vs), decltype(verts.index_select<0>(idx))>::value,
                  "#375: static value form must yield the <Axis> form's exact type");
    for (long i=0;i<3;++i) for (long j=0;j<3;++j) if (vs(i,j) != sel(i,j)) return 22;
    // (b) dynamic idx -> _TNY_HOST arm -> heap result, == index_select<0>(idxd).
    auto vd = verts.index_select(idxd, axis<0>{});
    static_assert(decltype(vd)::ownership == storage::heap, "#375: dynamic value form -> heap (_TNY_HOST arm)");
    static_assert(cs::is_same<decltype(vd), decltype(verts.index_select<0>(idxd))>::value,
                  "#375: dynamic value form must yield the <Axis> form's exact type");
    for (long j=0;j<3;++j) { if (vd(0,j) != seld(0,j)) return 23; if (vd(1,j) != seld(1,j)) return 24; }
    // (c) a negative axis still resolves through the split forwarders.
    auto vn = verts.index_select(idx1, axis<-1>{});
    for (long i=0;i<5;++i) { if (vn(i,0) != sel1(i,0)) return 25; if (vn(i,1) != sel1(i,1)) return 26; }
    // (d) THE POINT of the value form: it must still deduce with NO `.template`
    // on a TYPE-DEPENDENT receiver -- the property a naive two-overload split is
    // most likely to break. Exercised for real inside a template, on both arms.
    if (!tny_test::dependent_index_select(verts, idx,  sel))  return 27;   // static  -> _TNY_API arm
    if (!tny_test::dependent_index_select(verts, idxd, seld)) return 28;   // dynamic -> _TNY_HOST arm

    // #366 regression guard: every receiver above is `local<double, shape<5,3>>`,
    // i.e. `ccontiguous` -- so the whole suite never instantiated index_select on
    // a `strides<...>`-layout receiver (the layout every slice/permute/flip/peel
    // result carries), which is exactly the combination #366 identified as
    // untested and where #315's private-inheritance misattribution (a raw
    // `Ei::rank()`/`Ei::static_extent(0)`/`DstE::static_extent(A)` inside
    // `tensor`'s body, instead of routing through `_shape_rank`/
    // `_shape_static_extent`) would fire on MSVC. `verts(slice(1,4), all)` is a
    // view of rows 1..3 with a folded `strides<...>` layout.
    auto vsub = verts(slice(1,4), all);
    static_assert(_is_strides<decltype(vsub)::layout_type>::value,
                  "#366: slice must produce a strides<...> layout to exercise the guard");
    auto idxs = local<long, shape<3>>(); idxs(0)=2; idxs(1)=0; idxs(2)=1;
    auto sels = vsub.index_select<0>(idxs);           // static idx -> _TNY_API/stack arm
    static_assert(decltype(sels)::ownership == storage::stack, "#366: static idx -> stack result");
    for (long j=0;j<3;++j) {
        if (sels(0,j) != vsub(2,j)) return 29;
        if (sels(1,j) != vsub(0,j)) return 30;
        if (sels(2,j) != vsub(1,j)) return 31;
    }
    auto destsub = local<double, shape<3,3>>();
    vsub.index_select<0>(idxs, into(destsub));        // the into(dest) form, same strides<...> receiver
    for (long j=0;j<3;++j) if (destsub(0,j) != vsub(2,j) || destsub(1,j) != vsub(0,j) || destsub(2,j) != vsub(1,j))
        return 32;
    auto idxsd = owned<long, shape<-1>>(shape<-1>{2}); idxsd(0)=1; idxsd(1)=0;
    auto selsd = vsub.index_select<0>(idxsd);         // dynamic idx -> _TNY_HOST/heap arm
    static_assert(decltype(selsd)::ownership == storage::heap, "#366: dynamic idx -> heap result");
    for (long j=0;j<3;++j) { if (selsd(0,j) != vsub(1,j)) return 33; if (selsd(1,j) != vsub(0,j)) return 34; }
    auto selsv = vsub.index_select(idxs, axis<0>{});  // value form on the strides<...> receiver
    for (long j=0;j<3;++j) if (selsv(0,j) != vsub(2,j)) return 35;

    return 0;
}
