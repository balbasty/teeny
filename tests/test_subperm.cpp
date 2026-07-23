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
    static_assert(decltype(rin)::extents_type::static_extent(1) == cs::dynamic_extent, "inferred dim dynamic");
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

    return 0;
}
