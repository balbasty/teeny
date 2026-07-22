#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main()
{
    double buf[24];
    for (long i = 0; i < 24; ++i) buf[i] = i;

    // anyrank is a trivially-copyable rank-erased carrier (inline store).
    static_assert(cs::is_trivially_copyable<anyrank<double,long>>::value, "trivially copyable");
    static_assert(anyrank<double,long>::max_rank == TNY_MAX_RANK, "default max_rank = TNY_MAX_RANK");

    long shape[3]  = {2,3,4};
    long stride[3] = {12,4,1};
    auto at = as_anyrank(buf, shape, stride, 3);  // runtime rank 3 (WRAPS arrays, no copy)
    if (at.ndim != 3) return 1;

    // fixed<R>() -> a concrete rank-R md::tensor view.
    auto v = at.fixed<3>();
    static_assert(decltype(v)::rank() == 3, "fixed rank");
    if (v(1,2,3) != buf[1*12+2*4+3]) return 2;
    if (v.numel() != 24) return 3;

    // dispatch_rank: pick the fixed rank by runtime ndim, run a generic kernel.
    double total = 0;
    bool ok = dispatch_rank(at, [&](auto view){ total = sum(view); });
    if (!ok) return 4;
    double expect = 0; for (int i=0;i<24;++i) expect += buf[i];
    if (total != expect) return 5;

    // works for other ranks through the SAME call site.
    long sh2[2] = {4,6}, st2[2] = {6,1};
    auto at2 = as_anyrank(buf, sh2, st2, 2);
    double t2 = 0;
    dispatch_rank(at2, [&](auto view){ t2 = sum(view); });
    if (t2 != expect) return 6;                  // same 24 elements, contiguous

    // ndim 0 (a scalar ndarray) dispatches to a rank-0 view.
    auto zero = as_anyrank(buf, shape, stride, 0);
    long r0 = -1;
    bool ok0 = dispatch_rank(zero, [&](auto v){ r0 = decltype(v)::rank(); });
    if (!ok0 || r0 != 0) return 7;

    // ---- peel_front<Sr>: peel the runtime batch dims, keep Sr static ---------
    // shape (2,3,4): treat the last Sr=2 as "interesting", the first as batch.
    long acc = 0;
    long cells = 0;
    for (auto cell : at.peel_front<-2>()) {       // 2 batch cells, each a (3,4) view
        static_assert(decltype(cell)::rank() == 2, "peel_front keeps Sr=2 static");
        acc += (long)sum(cell); ++cells;
    }
    if (cells != 2 || acc != (long)expect) return 8;

    // grid-stride form: the i-th cell directly, and its offset is baked in.
    auto c1 = at.peel_front_at<-2>(1);            // batch index 1 -> buf + 12
    if (c1(2,3) != buf[12 + 2*4 + 3]) return 9;

    // recast the dynamic inner dims back to static so they fold.
    auto cs2 = c1.recast<tny::shape<3,4>>();     // (local `shape` array shadows tny::shape)
    static_assert(decltype(cs2.stride(Int<1>()))::value == 1, "inner stride folds after recast");
    if (cs2(2,3) != buf[12 + 2*4 + 3]) return 10;

    // ---- default as_anyrank WRAPS the arrays (no copy) -----------------------
    // `at` (above) points AT `shape`/`stride`; mutating the source shows through.
    if (at.size(1) != 3 || at.step(0) != 12) return 11;
    long vacc = 0; for (auto cell : at.peel_front<-2>()) vacc += (long)sum(cell);
    if (vacc != (long)expect) return 12;
    stride[0] = 0;                                       // collapse axis 0 onto row 0
    if (at.step(0) != 0) return 13;
    stride[0] = 12;                                      // restore

    // ---- as_anyrank(..., copy_meta): inline, device-passable, INDEPENDENT copy -
    auto ac = as_anyrank(buf, shape, stride, 3, copy_meta);   // copies shape/stride inline
    static_assert(cs::is_trivially_copyable<decltype(ac)>::value, "copy store -> trivially copyable");
    static_assert(decltype(ac)::device_passable, "copy_meta carrier is device-passable");
    static_assert(!decltype(at)::device_passable, "wrap carrier is host-only");
    if (ac.ndim != 3 || ac.size(1) != 3 || ac.step(0) != 12) return 14;
    stride[0] = 0;                                       // mutate the source...
    if (ac.step(0) != 12) return 15;                     // ...the copy is unaffected
    stride[0] = 12;
    auto vc = ac.fixed<3>();
    if (vc(1,2,3) != buf[1*12+2*4+3]) return 16;

    return 0;
}
