#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;

int main()
{
    double buf[24];
    for (long i = 0; i < 24; ++i) buf[i] = i;
    auto t = wrap(buf, extents<long,2,3,4>{});          // strides (12,4,1)

    // ---- sub<D>(i): bind an axis, drop it ------------------------------
    auto s1 = t.take_along<1>(2);                              // fix axis 1 -> (2,4)
    static_assert(decltype(s1)::rank() == 2, "sub drops an axis");
    if (s1(1,3) != t(1,2,3)) return 1;
    if (s1(0,0) != t(0,2,0)) return 2;

    auto s0 = t.take_along<0>(1);                              // fix axis 0 -> (3,4)
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

    // ---- unsqueeze / squeeze -------------------------------------------
    auto u = t.unsqueeze<3>();                          // (2,3,4) -> (2,3,4,1)
    static_assert(decltype(u)::rank() == 4, "unsqueeze adds an axis");
    static_assert(decltype(u)::extents_type::static_extent(3) == 1, "new axis is 1");
    if (u(1,2,3,0) != t(1,2,3)) return 9;
    auto u0 = t.unsqueeze<0>();                         // (2,3,4) -> (1,2,3,4)
    static_assert(decltype(u0)::extents_type::static_extent(0) == 1, "front axis is 1");
    if (u0(0,1,2,3) != t(1,2,3)) return 10;
    auto sq = u.squeeze<3>();                           // (2,3,4,1) -> (2,3,4)
    static_assert(decltype(sq)::rank() == 3, "squeeze drops an axis");
    if (sq(1,2,3) != t(1,2,3)) return 11;
    // writes propagate through the inserted axis
    u(0,0,0,0) = 55.0;
    if (t(0,0,0) != 55.0) return 12;

    // ---- negative axis indices (python-style) --------------------------
    static_assert(t.extent(Int<-1>()) == 4, "extent(-1) = last axis");
    static_assert(t.stride(Int<-1>()) == 1, "stride(-1) = last axis (unit)");
    auto un = t.unsqueeze<-1>();                        // append trailing axis
    static_assert(decltype(un)::rank() == 4 && decltype(un)::extents_type::static_extent(3) == 1, "unsqueeze<-1> appends");
    if (un(1,2,3,0) != t(1,2,3)) return 13;
    auto sq2 = un.squeeze<-1>();                        // drop it again
    if (sq2(1,2,3) != t(1,2,3)) return 14;
    auto pr = t.permute<-1,0,1>();                      // (2,3,4) -> (4,2,3)
    if (pr(3,1,2) != t(1,2,3)) return 15;
    auto s2 = t.take_along<-2>(1);                      // bind axis 1 -> (2,4)
    if (s2(1,3) != t(1,1,3)) return 16;
    long np = 0; for (auto line : peel<0,-2>(t)) { (void)line; ++np; }  // peel axes 0,1
    if (np != 6) return 17;

    // ---- flip<Ax> (reversed-axis view; uses a negative stride) ----------
    auto fl = t.flip<2>();                             // reverse last axis
    for (long i=0;i<2;++i) for (long j=0;j<3;++j) for (long k=0;k<4;++k)
        if (fl(i,j,k) != t(i,j,3-k)) return 18;
    auto fl0 = t.flip<-3>();                            // reverse axis 0 (negative axis arg)
    if (fl0(0,0,0) != t(1,0,0)) return 19;
    fl(0,0,0) = 42.0;                                  // mutable view
    if (t(0,0,3) != 42.0) return 20;
    t(0,0,3) = 3;                                      // restore

    // ---- reshape / flatten (contiguous views) --------------------------
    auto rs = t.reshape<6,4>();                         // (2,3,4) -> (6,4)
    static_assert(decltype(rs)::rank() == 2 && decltype(rs)::extents_type::static_extent(0) == 6, "reshape");
    if (rs(0,0) != t(0,0,0) || rs(5,3) != t(1,2,3)) return 21;
    auto fla = t.flatten();
    if (fla.extent(0) != 24 || fla(23) != t(1,2,3)) return 22;
    auto rin = t.reshape<6,-1>();                       // -1 = infer (4)
    // smart reshape (#129): for a fully-static source the inferred dim FOLDS to a
    // compile-time extent (was dynamic before) — like every other static view op.
    static_assert(decltype(rin)::extents_type::static_extent(1) == 4, "inferred dim folds to static 4");
    if (rin.extent(1) != 4 || rin(5,3) != t(1,2,3)) return 24;
    rs(0,0) = 88.0;                                    // reshape is a view
    if (t(0,0,0) != 88.0) return 23;

    // ---- #68: permute/flip/unsqueeze/squeeze FOLD to static strides<...> ----
    // (a static source keeps compile-time strides, like the slice gather).
    t(0,0,0) = 0.0;                                    // restore for value checks below
    auto pf = t.permute<2,0,1>();                      // strides 12,4,1 -> 1,12,4 (static)
    static_assert(_is_strides<decltype(pf)::layout_type>::value, "permute -> strides<>");
    static_assert(_is_ic<decltype(pf.stride(Int<0>()))>::value, "permuted stride folds (static)");
    if (pf.stride(Int<0>()) != 1 || pf.stride(Int<2>()) != 4) return 25;
    auto ff = t.flip<0>();                             // axis0 stride -> -12 (static)
    static_assert(_is_strides<decltype(ff)::layout_type>::value, "flip -> strides<>");
    if (ff.stride(Int<0>()) != -12) return 26;
    auto uf = t.unsqueeze<0>();
    static_assert(_is_strides<decltype(uf)::layout_type>::value, "unsqueeze -> strides<>");
    static_assert(_is_strides<decltype(t.unsqueeze<1>().squeeze<1>())::layout_type>::value, "squeeze -> strides<>");
    if (uf(0,1,2,3) != t(1,2,3)) return 27;
    // permute of a strides<...> source stays folded
    auto ps2 = tensor<double, shape<2,3,4>, strides<12,4,1>>(buf).permute<1,0,2>();
    static_assert(_is_strides<decltype(ps2)::layout_type>::value, "permute of strides<> folds");
    if (ps2.stride(Int<0>()) != 4 || ps2.stride(Int<2>()) != 1) return 28;

    // ---- #71: unsqueeze a RANK-0 (layout_right) view -> rank-1 ----------------
    // t.at(...) is a rank-0 layout_right view; unsqueezing it used to hit CCCL's
    // rank>0 constraint on mdspan::stride() inside unsqueeze_md.
    auto r0 = t.at(1, 2, 3);                            // rank-0 layout_right view
    static_assert(decltype(r0)::rank() == 0, "at() -> rank-0");
    auto r1 = r0.unsqueeze<0>();                        // rank-0 -> rank-1
    static_assert(decltype(r1)::rank() == 1, "unsqueeze rank-0 -> rank-1");
    static_assert(decltype(r1)::extents_type::static_extent(0) == 1, "new axis is size 1");
    if (r1(0) != t(1,2,3)) return 29;
    r1(0) = 111.0;                                      // mutable view aliases the element
    if (t(1,2,3) != 111.0) return 30;

    // ---- #242: MULTI-AXIS unsqueeze<Ax...> / squeeze<Ax...> ------------------
    // unsqueeze positions are relative to the FINAL rank (numpy expand_dims), so
    // (H,W).unsqueeze<1,3>() -> (H,1,W,1): insert 1 into rank-2 -> (H,1,W), then
    // insert 3 into rank-3 -> appends.
    auto hw = wrap(buf, extents<long,3,4>{});           // (H,W) = (3,4), strides (4,1)
    auto m2 = hw.unsqueeze<1,3>();
    static_assert(decltype(m2)::rank() == 4, "unsqueeze<1,3> -> rank 4");
    static_assert(decltype(m2)::extents_type::static_extent(0) == 3
               && decltype(m2)::extents_type::static_extent(1) == 1
               && decltype(m2)::extents_type::static_extent(2) == 4
               && decltype(m2)::extents_type::static_extent(3) == 1, "unsqueeze<1,3> -> (3,1,4,1)");
    auto m2ref = hw.unsqueeze<1>().unsqueeze<3>();      // the manual single-axis fold
    static_assert(cs::is_same<decltype(m2), decltype(m2ref)>::value, "multi-axis unsqueeze == manual fold (same type)");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j)
        if (m2(i,0,j,0) != hw(i,j) || m2(i,0,j,0) != m2ref(i,0,j,0)) return 31;
    // ...and it is a mutable view of the same buffer
    m2(1,0,2,0) = 77.0;
    if (hw(1,2) != 77.0) return 32;

    // three axes: (3,4) -> (1,3,1,4,1)
    auto m3 = hw.unsqueeze<0,2,4>();
    static_assert(decltype(m3)::rank() == 5, "unsqueeze<0,2,4> -> rank 5");
    static_assert(decltype(m3)::extents_type::static_extent(0) == 1
               && decltype(m3)::extents_type::static_extent(1) == 3
               && decltype(m3)::extents_type::static_extent(2) == 1
               && decltype(m3)::extents_type::static_extent(3) == 4
               && decltype(m3)::extents_type::static_extent(4) == 1, "unsqueeze<0,2,4> -> (1,3,1,4,1)");
    auto m3ref = hw.unsqueeze<0>().unsqueeze<2>().unsqueeze<4>();
    for (long i=0;i<3;++i) for (long j=0;j<4;++j)
        if (m3(0,i,0,j,0) != hw(i,j) || m3(0,i,0,j,0) != m3ref(0,i,0,j,0)) return 33;

    // negative positions wrap against the FINAL rank: <0,-1> on rank-2 -> (1,3,4,1)
    auto mn = hw.unsqueeze<0,-1>();
    static_assert(decltype(mn)::rank() == 4, "unsqueeze<0,-1> -> rank 4");
    static_assert(decltype(mn)::extents_type::static_extent(0) == 1
               && decltype(mn)::extents_type::static_extent(1) == 3
               && decltype(mn)::extents_type::static_extent(2) == 4
               && decltype(mn)::extents_type::static_extent(3) == 1, "unsqueeze<0,-1> -> (1,3,4,1)");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j)
        if (mn(0,i,j,0) != hw(i,j)) return 34;

    // squeeze positions are relative to the SOURCE rank, dropped LARGEST-first:
    // (1,3,1,4).squeeze<0,2>(): drop 2 -> (1,3,4), drop 0 -> (3,4).
    auto q4 = hw.unsqueeze<0,2>();                      // (1,3,1,4)
    static_assert(decltype(q4)::rank() == 4, "source for squeeze<0,2>");
    auto sqm2 = q4.squeeze<0,2>();
    static_assert(decltype(sqm2)::rank() == 2, "squeeze<0,2> -> rank 2");
    static_assert(decltype(sqm2)::extents_type::static_extent(0) == 3
               && decltype(sqm2)::extents_type::static_extent(1) == 4, "squeeze<0,2> -> (3,4)");
    auto sq2ref = q4.squeeze<2>().squeeze<0>();          // the manual fold, in REVERSE order
    static_assert(cs::is_same<decltype(sqm2), decltype(sq2ref)>::value, "multi-axis squeeze == manual reverse fold (same type)");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j)
        if (sqm2(i,j) != hw(i,j) || sqm2(i,j) != sq2ref(i,j)) return 35;
    sqm2(2,1) = 66.0;                                     // still a mutable view
    if (hw(2,1) != 66.0) return 36;

    // negative squeeze axes + a longer list: (1,3,1,4,1).squeeze<0,2,-1>() -> (3,4)
    auto sqm3 = m3.squeeze<0,2,-1>();
    static_assert(decltype(sqm3)::rank() == 2, "squeeze<0,2,-1> -> rank 2");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j)
        if (sqm3(i,j) != hw(i,j)) return 37;

    // a DYNAMIC axis that is 1 only at run time squeezes too (the per-axis
    // _TNY_CHECK runs at each fold step)
    auto dsrc = wrap(buf, shape<-1,-1,-1,-1>{1,3,1,4}, {12,4,4,1});   // (1,3,1,4), every dim dynamic
    auto ds = dsrc.squeeze<0,2>();
    static_assert(decltype(ds)::rank() == 2, "dynamic squeeze<0,2> -> rank 2");
    if (ds.extent(0) != 3 || ds.extent(1) != 4) return 38;
    for (long i=0;i<3;++i) for (long j=0;j<4;++j)
        if (ds(i,j) != hw(i,j)) return 39;

    // ---- axis<...> value form on squeeze / unsqueeze / permute (#243) --
    auto pv  = t.permute(axis<2,0,1>{});                 // == t.permute<2,0,1>()
    auto pv2 = t.permute<2,0,1>();
    static_assert(cs::is_same<decltype(pv), decltype(pv2)>::value, "permute(axis<...>) == permute<...>() (same type)");
    for (long i=0;i<4;++i) for (long j=0;j<2;++j) for (long k=0;k<3;++k)
        if (pv(i,j,k) != pv2(i,j,k)) return 40;

    auto unv  = hw.unsqueeze(axis<1,3>{});               // == hw.unsqueeze<1,3>()
    auto unv2 = hw.unsqueeze<1,3>();
    static_assert(cs::is_same<decltype(unv), decltype(unv2)>::value, "unsqueeze(axis<...>) == unsqueeze<...>() (same type)");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j)
        if (unv(i,0,j,0) != unv2(i,0,j,0)) return 41;

    auto sqv  = q4.squeeze(axis<0,2>{});                 // == q4.squeeze<0,2>()
    auto sqv2 = q4.squeeze<0,2>();
    static_assert(cs::is_same<decltype(sqv), decltype(sqv2)>::value, "squeeze(axis<...>) == squeeze<...>() (same type)");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j)
        if (sqv(i,j) != sqv2(i,j)) return 42;

    // single-axis axis<> form still works (arity, not a special case)
    auto un1  = hw.unsqueeze(axis<0>{});
    auto un1r = hw.unsqueeze<0>();
    static_assert(cs::is_same<decltype(un1), decltype(un1r)>::value, "unsqueeze(axis<0>{}) == unsqueeze<0>() (same type)");

    return 0;
}
