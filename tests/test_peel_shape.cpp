// #206: anyrank::peel_front_at<NewE[,NewL]>(i) — peel the batch DIRECTLY to a target
// trailing shape, fusing peel_front_at<-Sr>(i).recast<NewE,NewL>() into one call. NewE's
// rank = the number of kept trailing dims; static extents fold, -1 stays dynamic; the
// default keeps the carrier's runtime strides (layout_stride), a layout arg folds them.
#include <teeny/teeny.h>
#include <teeny/dynamic.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    // (*batch = 2x3, C×C = 2x2), C-contiguous
    double buf[2*3*2*2]; for (int i = 0; i < 24; ++i) buf[i] = i;
    long shp[4] = {2,3,2,2}, strd[4] = {12,4,2,1};
    auto at = as_anyrank(buf, shp, strd, 4);

    // fused peel: cell is shape<2,2> directly (no recast in the caller)
    auto A = at.peel_front_at<shape<2,2>>(5);
    static_assert(decltype(A)::rank() == 2, "target rank = kept trailing dims");
    static_assert(decltype(A)::shape_type::static_extent(0) == 2, "static inner extent folds");
    static_assert(decltype(A)::shape_type::static_extent(1) == 2, "static inner extent folds");
    static_assert(cs::is_same<decltype(A)::layout_type, dynamic_strides>::value,
                  "default keep_strides -> runtime strides (an anyrank has no static stride info)");

    // must equal the two-step it replaces, byte-for-byte
    auto A2 = at.peel_front_at<-2>(5).recast(shape<2,2>{});
    if (A.data() != A2.data()) return 1;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) if (A(i,j) != A2(i,j)) return 2;

    // mixed static/dynamic target (the fastfields (*batch,*spatial,C) shape)
    auto B = at.peel_front_at<shape<-1,2,2>>(1);
    static_assert(decltype(B)::rank() == 3, "rank 3");
    static_assert(decltype(B)::shape_type::static_extent(0) == cs::dynamic_extent, "leading kept dim stays dynamic");
    static_assert(decltype(B)::shape_type::static_extent(2) == 2, "trailing kept dim folds");
    auto B2 = at.peel_front_at<-3>(1).recast(shape<-1,2,2>{});
    if (B(0,1,1) != B2(0,1,1)) return 3;

    // value-form twin (no `.template` on a dependent receiver)
    auto C = at.peel_front_at(5, shape<2,2>{});
    if (C(1,1) != A(1,1)) return 4;

    // layout arg -> fold the inner strides (debug-checked "I promise it's contiguous")
    auto D = at.peel_front_at<shape<2,2>, ccontiguous>(5);
    static_assert(_is_ic<decltype(D.stride(Int<1>()))>::value, "inner stride folds under ccontiguous");
    static_assert(decltype(D.stride(Int<1>()))::value == 1, "innermost contiguous stride == 1");
    static_assert(decltype(D.stride(Int<0>()))::value == 2, "next stride == inner extent");
    if (D(1,1) != A(1,1)) return 5;
    auto E = at.peel_front_at(5, shape<2,2>{}, ccontiguous{});   // value-form with layout tag
    if (E(0,1) != A(0,1)) return 6;

    // composes with dispatch_layout (proven contiguity, not a promise)
    int hit = 0;
    dispatch_layout(at.peel_front_at<shape<-1,2,2>>(0), [&](auto v) {
        if constexpr (cs::is_same<typename decltype(v)::layout_type, ccontiguous>::value) {
            hit = 1;
            auto r = v.recast(shape<-1,2,2>{});
            static_assert(_is_ic<decltype(r.stride(Int<2>()))>::value, "inner stride folds (proven)");
        }
    });
    if (hit != 1) return 7;

    // memory space is preserved (a device carrier peels to gpu_view cells)
    auto dev = as_anyrank<storage::gpu_view>(buf, shp, strd, 4);
    static_assert(decltype(dev.peel_front_at<shape<2,2>>(0))::ownership == storage::gpu_view, "gpu_view preserved");

    return 0;
}
