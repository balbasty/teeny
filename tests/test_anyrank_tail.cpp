// #209/#210: anyrank carries a STATIC TRAILING shape (and, with a layout tag, static
// trailing STRIDES) in its type — spelled `anyshape<etc, ...>` at the boundary
// (etc = the erased batch) — so fixed()/peel_front() hand out cells whose inner
// extents (and strides) are already folded, no per-call recast. Back-compat: an EMPTY
// tail / keep_strides (the default) yields exactly today's dyn_tensor / layout_stride.
// A ccontiguous inner block folds its strides (fully-static tail -> EBO cell).
#include <teeny/teeny.h>
#include <teeny/dynamic.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    // ---- the anyshape<etc,...> spec: etc marks the erased region, the rest is the tail
    static_assert(cs::is_same<anyshape<etc,-1,2,2>::tail, shape<-1,2,2>>::value, "tail extracted after etc");
    static_assert(anyshape<etc,-1,2,2>::tail::rank() == 3, "tail rank");
    static_assert(anyshape<etc>::tail::rank() == 0, "bare etc == fully dynamic (empty tail)");
    static_assert(_is_anyshape<anyshape<etc,3>>::value && !_is_anyshape<shape<3>>::value, "anyshape trait");

    // (*batch = 2x3, C×C = 2x2), C-contiguous
    double buf[2*3*2*2]; for (int i = 0; i < 24; ++i) buf[i] = i;
    long shp[4] = {2,3,2,2}, strd[4] = {12,4,2,1};

    // ---- back-compat: an EMPTY-tail carrier produces EXACTLY today's types ----
    auto plain = as_anyrank(buf, shp, strd, 4);
    static_assert(cs::is_same<decltype(plain.fixed<4>()), dyn_tensor<double,long,4>>::value,
                  "empty tail: fixed<R> is byte-identical to dyn_tensor");
    static_assert(cs::is_same<decltype(plain.peel_front_at<-2>(0)), dyn_tensor<double,long,2>>::value,
                  "empty tail: peel_front_at is byte-identical to dyn_tensor");
    static_assert(cs::is_same<typename decltype(plain.peel_front<-2>())::Cell,
                              dyn_tensor<double,long,2>>::value,
                  "empty tail: peel_front iterator cell is dyn_tensor");
    static_assert(decltype(plain)::tail_rank == 0, "empty tail rank 0");

    // ---- static tail: inner extents fold in every produced view ----
    auto at = as_anyrank(buf, shp, strd, 4, anyshape<etc,-1,2,2>{});
    static_assert(decltype(at)::tail_rank == 3, "tail rank = 3");

    // fixed<R>: the last tail_rank dims fold, the leading ones stay dynamic
    auto A = at.fixed<4>();
    static_assert(decltype(A)::rank() == 4, "fixed rank 4");
    static_assert(decltype(A)::shape_type::static_extent(0) == cs::dynamic_extent, "batch stays dynamic");
    static_assert(decltype(A)::shape_type::static_extent(1) == cs::dynamic_extent, "leading tail dim (-1) dynamic");
    static_assert(decltype(A)::shape_type::static_extent(2) == 2, "inner extent folds");
    static_assert(decltype(A)::shape_type::static_extent(3) == 2, "inner extent folds");
    static_assert(cs::is_same<decltype(A)::layout_type, cs::layout_stride>::value,
                  "PR1: strides stay runtime (layout_stride)");

    // ---- Sr == K: fully-folded cell off a rank-2 static tail ----
    auto t2 = as_anyrank(buf, shp, strd, 4, anyshape<etc,2,2>{});
    auto B = t2.peel_front_at<-2>(1);
    static_assert(decltype(B)::rank() == 2, "rank 2");
    static_assert(decltype(B)::shape_type::static_extent(0) == 2 &&
                  decltype(B)::shape_type::static_extent(1) == 2, "both dims fold (Sr==K)");
    // value equals the two-step it replaces, byte-for-byte
    auto Bref = plain.peel_front_at<-2>(1).recast(shape<2,2>{});
    if (B.data() != Bref.data()) return 1;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) if (B(i,j) != Bref(i,j)) return 2;

    // ---- Sr > K: partial application (leading kept dim dynamic, tail folds) ----
    auto C = t2.peel_front_at<-3>(0);   // keep 3 dims, only last 2 static in tail
    static_assert(decltype(C)::rank() == 3, "rank 3");
    static_assert(decltype(C)::shape_type::static_extent(0) == cs::dynamic_extent, "extra leading dim dynamic");
    static_assert(decltype(C)::shape_type::static_extent(1) == 2 &&
                  decltype(C)::shape_type::static_extent(2) == 2, "tail still folds under Sr>K");

    // ---- Sr < K: peel into the suffix (keep its LAST Sr dims) ----
    auto D = t2.peel_front_at<-1>(0);
    static_assert(decltype(D)::rank() == 1, "rank 1");
    static_assert(decltype(D)::shape_type::static_extent(0) == 2, "keeps the last tail dim");

    // ---- peel_front<-Sr>() range: every cell is born folded ----
    long seen = 0;
    for (auto cell : t2.peel_front<-2>()) {
        static_assert(decltype(cell)::shape_type::static_extent(0) == 2, "iterator cell folds dim 0");
        static_assert(decltype(cell)::shape_type::static_extent(1) == 2, "iterator cell folds dim 1");
        seen += static_cast<long>(cell(0,0));
    }
    if (t2.size_front<-2>() != 6) return 3;                  // 2*3 batch cells
    // cell(0,0) is buf at each batch base: offsets 0,4,8,12,16,20 -> sum 60
    if (seen != 0+4+8+12+16+20) return 4;

    // subrange chunk (threaded sweep) folds identically
    long chunk = 0; for (auto cell : t2.peel_front<-2>().subrange(2,5)) chunk += static_cast<long>(cell(1,1));
    if (chunk != (8+3)+(12+3)+(16+3)) return 5;              // cell(1,1) = base + 3

    // ---- dispatch_rank threads the tail; the dispatched view is folded ----
    int rr = 0;
    dispatch_rank(at, [&](auto v){
        if constexpr (decltype(v)::rank() == 4) {   // constexpr: only the rank-4 arm sees static_extent(3)
            static_assert(decltype(v)::shape_type::static_extent(3) == 2, "dispatched cell folds inner");
            rr = 1;
        }
    });
    if (!rr) return 6;

    // ---- device carrier: tail preserved, cells are gpu_view ----
    auto dev = as_anyrank<storage::gpu_view>(buf, shp, strd, 4, anyshape<etc,-1,2,2>{});
    static_assert(decltype(dev)::tail_rank == 3, "device carrier keeps the tail");
    static_assert(decltype(dev.fixed<4>())::ownership == storage::gpu_view, "gpu_view preserved");
    static_assert(decltype(dev.fixed<4>())::shape_type::static_extent(3) == 2, "device cell folds inner");

    // ---- copy_meta (device-passable) carrier carries the tail too ----
    auto cp = as_anyrank(buf, shp, strd, 4, copy_meta, anyshape<etc,-1,2,2>{});
    static_assert(decltype(cp)::tail_rank == 3, "copy_meta carrier keeps the tail");
    static_assert(cs::is_trivially_copyable<decltype(cp)>::value, "copy_meta tail carrier is trivially copyable");
    if (cp.peel_front_at<-3>(1)(0,1,1) != plain.peel_front_at<-3>(1)(0,1,1)) return 7;

    // ---- #210: static trailing STRIDES via a layout tag -----------------------
    // (a) default keep_strides == #209: strides stay runtime (layout_stride cell)
    static_assert(cs::is_same<decltype(at.fixed<4>())::layout_type, cs::layout_stride>::value,
                  "keep_strides (default) -> layout_stride, byte-identical to #209");
    // (b) ccontiguous -> the inner block's strides fold to immediates
    auto cc = as_anyrank(buf, shp, strd, 4, anyshape<etc,-1,2,2>{}, ccontiguous{});
    auto F = cc.fixed<4>();
    static_assert(_is_strides<decltype(F)::layout_type>::value, "ccontiguous tail -> strides<...> layout");
    static_assert(_is_ic<decltype(F.stride(Int<3>()))>::value, "inner stride folds to immediate");
    static_assert(decltype(F.stride(Int<3>()))::value == 1, "innermost contiguous stride == 1");
    static_assert(decltype(F.stride(Int<2>()))::value == 2, "next == C");
    static_assert(decltype(F.stride(Int<1>()))::value == 4, "next == W*C (static, though its extent is dynamic)");
    static_assert(!_is_ic<decltype(F.stride(Int<0>()))>::value, "the erased batch dim's stride stays runtime");
    if (F(1,2,1,1) != buf[1*12 + 2*4 + 1*2 + 1]) return 8;              // values still correct

    // (c) fully-static contiguous tail -> EBO cell (no stride words), folded in the peel range
    auto t2c = as_anyrank(buf, shp, strd, 4, anyshape<etc,2,2>{}, ccontiguous{});
    for (auto cell : t2c.peel_front<-2>()) {
        static_assert(sizeof(cell) == sizeof(double *), "fully-static contiguous cell is pointer-sized (EBO mapping)");
        static_assert(decltype(cell.stride(Int<0>()))::value == 2 &&
                      decltype(cell.stride(Int<1>()))::value == 1, "cell strides fully fold");
        if (cell(1,1) != buf[/*base*/0 + 1*2 + 1] && cell.data() == buf) return 9;
    }

    // (d) explicit strides<...> tag imposes those strides
    auto es = as_anyrank(buf, shp, strd, 4, anyshape<etc,2,2>{}, strides<2,1>{});
    static_assert(decltype(es.fixed<4>().stride(Int<3>()))::value == 1, "explicit inner stride imposed");

    // (e) dispatch_layout still classifies a keep_strides (layout_stride) tail carrier's cell
    int lay = 0;
    dispatch_layout(at.fixed<4>(), [&](auto v){
        if constexpr (cs::is_same<typename decltype(v)::layout_type, ccontiguous>::value) lay = 1;
    });
    if (!lay) return 10;

    return 0;
}
