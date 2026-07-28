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

    return 0;
}
