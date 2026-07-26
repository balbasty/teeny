// #203: dispatch_layout — runtime-classify a dynamic-strided view's contiguity and
// hand `f` a view whose LAYOUT is in the type (ccontiguous / fcontiguous / else the
// original dynamic_strides), so a later recast folds the inner strides SAFELY. Opt-in
// sibling of dispatch_index.
#include <teeny/teeny.h>
#include <teeny/dynamic.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    double buf[8]; for (int i = 0; i < 8; ++i) buf[i] = i;

    // (1) a C-contiguous DYNAMIC-strided view -> the ccontiguous arm. The payoff:
    //     recast to static inner extents folds the inner strides to immediates —
    //     C-order inner strides depend only on the (static) inner extents.
    auto vc = wrap(buf, shape<-1,-1,-1>{2,2,2}, {4,2,1});     // dynamic_strides
    static_assert(cs::is_same<decltype(vc)::layout_type, dynamic_strides>::value, "input is layout_stride");
    int hit = 0;
    dispatch_layout(vc, [&](auto w) {
        using L = typename decltype(w)::layout_type;
        if constexpr (cs::is_same<L, ccontiguous>::value) {
            hit = 1;
            auto r = w.recast(shape<-1,2,2>{});
            static_assert(_is_ic<decltype(r.stride(Int<2>()))>::value, "C-order inner stride folds to a constant");
            static_assert(decltype(r.stride(Int<2>()))::value == 1, "innermost contiguous stride == 1");
            static_assert(decltype(r.stride(Int<1>()))::value == 2, "next stride == inner extent");
            if (w(1,1,1) != 7) hit = -1;                     // values still correct through the retype
        }
    });
    if (hit != 1) return 1;

    // (2) an F-contiguous dynamic view -> the fcontiguous arm (a typed view; extent-
    //     derived strides). Values must round-trip.
    auto vf = wrap(buf, shape<-1,-1,-1>{2,2,2}, {1,2,4});
    int fhit = 0;
    dispatch_layout(vf, [&](auto w) {
        if constexpr (cs::is_same<typename decltype(w)::layout_type, fcontiguous>::value) {
            fhit = 1;
            if (w(1,1,1) != 7) fhit = -1;
        }
    });
    if (fhit != 1) return 2;

    // (3) a genuinely strided (padded) view -> stays dynamic_strides (no false promise).
    double big[12]; for (int i = 0; i < 12; ++i) big[i] = i;
    auto vs = wrap(big, shape<-1,-1,-1>{2,2,2}, {6,2,1});     // row stride 6 != 4 -> a gap
    int shit = 0;
    dispatch_layout(vs, [&](auto w) {
        if constexpr (cs::is_same<typename decltype(w)::layout_type, dynamic_strides>::value) {
            shit = 1;
            if (w(1,1,1) != 9) shit = -1;                    // 6 + 2 + 1
        }
    });
    if (shit != 1) return 3;

    // (4) composes with the anyrank boundary: a fixed<R>() cell (layout_stride) routed
    //     through dispatch_layout; a C-contiguous anyrank -> ccontiguous cell.
    long shp[3] = {2,2,2}, strd[3] = {4,2,1};
    auto at = as_anyrank(buf, shp, strd, 3);
    int ahit = 0;
    dispatch_layout(at.fixed<3>(), [&](auto w) {
        if constexpr (cs::is_same<typename decltype(w)::layout_type, ccontiguous>::value) {
            ahit = 1;
            if (w(0,1,1) != 3) ahit = -1;                    // 0 + 2 + 1
        }
    });
    if (ahit != 1) return 4;

    return 0;
}
