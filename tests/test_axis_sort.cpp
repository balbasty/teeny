// squeeze<Ax...>/unsqueeze<Ax...> used to require the axis pack to be listed
// strictly ascending (a compile error otherwise). #275: sort the (distinct) axes
// at compile time instead, so any order is accepted and folds to the same result.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    // ---- unsqueeze: descending order == ascending order -----------------------
    auto a = local<double, shape<3,4>>{};                 // (H,W)
    auto ua = a.unsqueeze<1,3>();                         // ascending: (H,1,W,1)
    auto ud = a.unsqueeze<3,1>();                         // descending -- now accepted
    static_assert(cs::is_same<decltype(ua), decltype(ud)>::value, "");
    if (ua.rank() != 4 || ua.shape(0)!=3 || ua.shape(1)!=1 || ua.shape(2)!=4 || ua.shape(3)!=1) return 1;
    if (ud.shape(0)!=ua.shape(0) || ud.shape(1)!=ua.shape(1) ||
        ud.shape(2)!=ua.shape(2) || ud.shape(3)!=ua.shape(3))                                  return 2;

    // negative axes, out of order
    auto un = a.unsqueeze<-1,0>();                        // -> (1,H,W,1)
    if (un.rank() != 4 || un.shape(0)!=1 || un.shape(1)!=3 || un.shape(2)!=4 || un.shape(3)!=1) return 3;

    // ---- squeeze: descending order == ascending order --------------------------
    auto b = local<double, shape<1,3,1,4>>{};             // (1,H,1,W)
    auto sa = b.squeeze<0,2>();                           // ascending: (H,W)
    auto sd = b.squeeze<2,0>();                           // descending -- now accepted
    static_assert(cs::is_same<decltype(sa), decltype(sd)>::value, "");
    if (sa.rank() != 2 || sa.shape(0)!=3 || sa.shape(1)!=4) return 4;
    if (sd.shape(0)!=sa.shape(0) || sd.shape(1)!=sa.shape(1)) return 5;

    // negative axes, out of order (-2 normalises to axis 2, the OTHER size-1 axis)
    auto sn = b.squeeze<-2,0>();                          // drop axes 2 and 0 -> (H,W) == (3,4)
    if (sn.rank() != 2 || sn.shape(0)!=3 || sn.shape(1)!=4) return 6;

    // values survive the fold correctly (not just shape)
    double buf[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    auto v  = wrap(buf, shape<1,3,1,4>{});
    auto vd = v.squeeze<2,0>();
    if (vd(0,0)!=1 || vd(2,3)!=12) return 7;

    // ---- the axis<...> value form forwards through the same path ---------------
    auto sv = b.squeeze(axis<2,0>{});
    static_assert(cs::is_same<decltype(sv), decltype(sa)>::value, "");
    if (sv.shape(0)!=sa.shape(0) || sv.shape(1)!=sa.shape(1)) return 8;
    auto uv = a.unsqueeze(axis<3,1>{});
    static_assert(cs::is_same<decltype(uv), decltype(ua)>::value, "");
    if (uv.shape(1)!=ua.shape(1) || uv.shape(3)!=ua.shape(3)) return 9;

    // ---- 3+ axes, negative and positive MIXED, in scrambled order --------------
    // (review request on #292: confirm this isn't just a happy-path 2-axis check)
    auto c  = local<double, shape<2,3>>{};                 // rank 2 -> final rank 5
    auto ur = c.unsqueeze<0,2,4>();                        // reference: ascending, all positive
    static_assert(ur.rank() == 5, "");
    if (ur.shape(0)!=1 || ur.shape(1)!=2 || ur.shape(2)!=1 || ur.shape(3)!=3 || ur.shape(4)!=1) return 10;
    auto un3 = c.unsqueeze<-1,-5,-3>();                    // all-negative, scrambled (-1,-5,-3 -> 4,0,2)
    static_assert(cs::is_same<decltype(un3), decltype(ur)>::value, "");
    if (un3.shape(0)!=ur.shape(0) || un3.shape(1)!=ur.shape(1) || un3.shape(2)!=ur.shape(2) ||
        un3.shape(3)!=ur.shape(3) || un3.shape(4)!=ur.shape(4))                                return 11;
    auto um3 = c.unsqueeze<2,-1,0>();                      // mixed sign, scrambled (2,-1,0 -> 2,4,0)
    static_assert(cs::is_same<decltype(um3), decltype(ur)>::value, "");
    if (um3.shape(0)!=ur.shape(0) || um3.shape(1)!=ur.shape(1) || um3.shape(2)!=ur.shape(2) ||
        um3.shape(3)!=ur.shape(3) || um3.shape(4)!=ur.shape(4))                                return 12;

    auto d  = local<double, shape<1,2,1,3,1>>{};           // axes 0,2,4 are the size-1 ones
    auto sr = d.squeeze<0,2,4>();                          // reference: ascending, all positive
    static_assert(sr.rank() == 2, "");
    if (sr.shape(0)!=2 || sr.shape(1)!=3) return 13;
    auto sn3 = d.squeeze<-1,-5,-3>();                      // all-negative, scrambled (-1,-5,-3 -> 4,0,2)
    static_assert(cs::is_same<decltype(sn3), decltype(sr)>::value, "");
    if (sn3.shape(0)!=sr.shape(0) || sn3.shape(1)!=sr.shape(1)) return 14;
    auto sm3 = d.squeeze<2,-1,0>();                        // mixed sign, scrambled (2,-1,0 -> 2,4,0)
    static_assert(cs::is_same<decltype(sm3), decltype(sr)>::value, "");
    if (sm3.shape(0)!=sr.shape(0) || sm3.shape(1)!=sr.shape(1)) return 15;

    // ---- an EMPTY axis list names no axis, so it is a NO-OP (#369) --------------
    // numpy's own rule: np.squeeze(a, axis=()) / np.expand_dims(a, axis=()) both
    // leave the shape alone. It used to fall through to the DEFAULTED single-axis
    // form instead -- squeeze(axis<>{}) dropped every singleton, unsqueeze(axis<>{})
    // inserted an axis at 0 -- silently, which is what generic code computing an
    // axis list that comes out empty would hit.
    double e6[6] = {1,2,3,4,5,6};
    auto e  = wrap(e6, shape<1,2,1,3>{});                  // TWO real singleton axes
    auto e0 = e.squeeze(axis<>{});                         // must keep them
    static_assert(decltype(e0)::rank() == 4, "");
    static_assert(cs::is_same<typename decltype(e0)::shape_type, shape<1,2,1,3>>::value, "");
    if (e0.shape(0)!=1 || e0.shape(1)!=2 || e0.shape(2)!=1 || e0.shape(3)!=3) return 16;
    auto e1 = e.unsqueeze(axis<>{});                       // must insert nothing
    static_assert(decltype(e1)::rank() == 4, "");
    static_assert(cs::is_same<typename decltype(e1)::shape_type, shape<1,2,1,3>>::value, "");
    if (e1.shape(0)!=1 || e1.shape(1)!=2 || e1.shape(2)!=1 || e1.shape(3)!=3) return 17;
    // same elements, same storage (a view of e, not a copy)
    if (e0(0,1,0,2)!=6 || e1(0,1,0,2)!=6 || e0.data()!=e.data() || e1.data()!=e.data()) return 18;
    // strides survive too (a no-op must not re-layout)
    for (cs::size_t r = 0; r < 4; ++r)
        if (e0.stride(r)!=e.stride(r) || e1.stride(r)!=e.stride(r)) return 19;

    // dynamic shape: same no-op, checked at run time (nothing folds it away)
    auto f  = wrap(e6, shape<-1,2,-1,3>{1,2,1,3});
    auto f0 = f.squeeze(axis<>{});
    auto f1 = f.unsqueeze(axis<>{});
    static_assert(decltype(f0)::rank() == 4 && decltype(f1)::rank() == 4, "");
    if (f0.shape(0)!=1 || f0.shape(1)!=2 || f0.shape(2)!=1 || f0.shape(3)!=3) return 20;
    if (f1.shape(0)!=1 || f1.shape(1)!=2 || f1.shape(2)!=1 || f1.shape(3)!=3) return 21;
    if (f0(0,1,0,2)!=6 || f1(0,1,0,2)!=6) return 22;

    // a const receiver takes the same path (const element type, same shape)
    const auto & ce = e;
    auto c0 = ce.squeeze(axis<>{});
    auto c1 = ce.unsqueeze(axis<>{});
    static_assert(decltype(c0)::rank() == 4 && decltype(c1)::rank() == 4, "");
    static_assert(cs::is_same<typename decltype(c0)::element_type, const double>::value, "");
    if (c0(0,1,0,2)!=6 || c1(0,1,0,2)!=6) return 23;

    // ...and an owning (stack) tensor: a no-op still hands back a VIEW of it
    auto g  = local<double, shape<1,3>>{};
    g(0,1) = 7;
    auto g0 = g.squeeze(axis<>{});
    if (g0.rank()!=2 || g0.shape(0)!=1 || g0.shape(1)!=3 || g0(0,1)!=7 || g0.data()!=g.data()) return 24;

    // CONTROL: the no-empty-pack path is untouched -- a real one-axis list still
    // squeezes exactly that axis (and is NOT the drop-all form)
    auto e_1 = e.squeeze(axis<2>{});
    static_assert(decltype(e_1)::rank() == 3, "");
    if (e_1.shape(0)!=1 || e_1.shape(1)!=2 || e_1.shape(2)!=3) return 25;
    auto e_u = e.unsqueeze(axis<0>{});
    static_assert(decltype(e_u)::rank() == 5, "");
    if (e_u.shape(0)!=1 || e_u.shape(1)!=1 || e_u.shape(4)!=3) return 26;
    // ...and the no-ARGUMENT forms keep their own (different) meanings: squeeze()
    // drops EVERY static singleton, unsqueeze() inserts at axis 0.
    auto e_all = e.squeeze();
    static_assert(decltype(e_all)::rank() == 2, "");
    if (e_all.shape(0)!=2 || e_all.shape(1)!=3) return 27;
    auto e_u0 = e.unsqueeze();
    static_assert(decltype(e_u0)::rank() == 5, "");
    if (e_u0.shape(0)!=1 || e_u0.shape(1)!=1) return 28;

    // permute needs a FULL permutation, so axis<>{} is only valid at rank 0 (the one
    // permutation of no axes) and a compile error otherwise -- never silent. No
    // no-op branch is added there; this pins the rank-0 behaviour.
    auto s0 = wrap(e6, shape<>{});                         // rank-0 view of e6[0]
    auto p0 = s0.permute(axis<>{});
    static_assert(decltype(p0)::rank() == 0, "");
    if (p0.item()!=1) return 29;

    return 0;
}
