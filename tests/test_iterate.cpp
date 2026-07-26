#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;

int main()
{
    double buf[24];
    for (long i = 0; i < 24; ++i) buf[i] = i;
    auto t = wrap(buf, extents<long,2,3,4>{});         // C-contiguous (strides 12,4,1)

    // ---- peel axis 0: 2 sub-views of shape (3,4) -----------------------
    auto r0 = peel<0>(t);
    if (r0.size() != 2) return 1;
    static_assert(decltype(r0[0])::rank() == 2, "peel one axis -> rank 2");
    if (r0[1](2,3) != t(1,2,3)) return 2;              // slice 1 == t(1,.,.)

    // range-for yields the sub-views
    long seen = 0, checkacc = 0;
    for (auto s : peel<0>(t)) { checkacc += (long)s(0,0); ++seen; }
    if (seen != 2) return 3;
    if (checkacc != (long)t(0,0,0) + (long)t(1,0,0)) return 4;   // 0 + 12

    // ---- peel axes 0 AND 1: 6 sub-views of shape (4,) ------------------
    auto r01 = peel<0,1>(t);
    if (r01.size() != 6) return 5;
    static_assert(decltype(r01[0])::rank() == 1, "peel two axes -> rank 1");
    // row-major over (axis0=2, axis1=3): linear 5 -> (i0=1, i1=2)
    if (r01[5](0) != t(1,2,0)) return 6;
    if (r01[5](3) != t(1,2,3)) return 7;

    // ---- grid-stride style: peel_at + write through the slice ---------
    // (this is the index2offset replacement: linear i -> strided sub-view)
    for (long i = 0; i < r01.size(); ++i) {
        auto s = peel_at<0,1>(t, i);                  // 1-D view over the last axis
        s.add_(1000.0);                                // mutate the ORIGINAL buffer
    }
    for (long i = 0; i < 24; ++i) if (buf[i] != i + 1000) return 8;

    // ---- peel a non-leading axis (axis 1) ------------------------------
    auto r1 = peel<1>(t);                            // 3 sub-views of shape (2,4)
    if (r1.size() != 3) return 9;
    static_assert(decltype(r1[0])::rank() == 2, "peel middle axis -> rank 2");
    if (r1[2](1,3) != t(1,2,3)) return 10;             // axis1 fixed to 2

    // ---- md math works on a peeled slice -------------------------------
    auto sub = peel_at<0>(t, 0);                      // (3,4) view of t(0,.,.)
    double before = sub(1,1);
    sub.mul_(2.0);
    if (sub(1,1) != before * 2) return 11;

    // ---- #110: the range-for is INCREMENTAL (odometer) but must visit exactly the
    //      same cells, in the same order, as the random-access peel_at. Check on a
    //      PERMUTED source so the peeled axes have non-trivial (non-contiguous) strides
    //      and the odometer's carries are exercised against real strides.
    double b2[24]; for (long i = 0; i < 24; ++i) b2[i] = i;
    auto tp = wrap(b2, extents<long,2,3,4>{}).permute<2,0,1>();   // (4,2,3), strides (1,12,4)
    {
        auto rng = peel<0,1>(tp);                    // peel first two axes -> 8 cells, each (3,)
        long idx = 0;
        for (auto cell : rng) {                      // incremental cursor
            auto ref = peel_at<0,1>(tp, idx);        // random-access decode
            if (cell.data() != ref.data()) return 12;   // same base pointer each step
            for (long k = 0; k < 3; ++k) if (cell(k) != ref(k)) return 13;
            ++idx;
        }
        if (idx != rng.size()) return 14;
    }

    // ---- #110: subrange(lo,hi) for chunked/threaded sweeps visits exactly [lo,hi) ---
    {
        auto rng = peel<0,1>(t);                     // 6 cells
        long lo = 2, hi = 5, seen = 0;
        for (auto cell : rng.subrange(lo, hi)) {
            if (cell.data() != peel_at<0,1>(t, lo + seen).data()) return 15;
            ++seen;
        }
        if (seen != hi - lo) return 16;
        // two disjoint chunks must exactly tile the whole range (the threading pattern)
        long total = 0;
        for (auto c : rng.subrange(0, 4))          { (void)c; ++total; }
        for (auto c : rng.subrange(4, rng.size())) { (void)c; ++total; }
        if (total != rng.size()) return 17;
    }

    // ---- #110: mutation through an incremental cell lands in the source buffer ------
    {
        double b3[24]; for (long i = 0; i < 24; ++i) b3[i] = 0;
        auto tt = wrap(b3, extents<long,2,3,4>{});
        double v = 1.0;
        for (auto line : peel<0,1>(tt)) { line.fill_(v); v += 1.0; }   // 6 lines, each set to 1..6
        for (long i0 = 0; i0 < 2; ++i0) for (long i1 = 0; i1 < 3; ++i1)
            for (long i2 = 0; i2 < 4; ++i2)
                if (tt(i0,i1,i2) != double(i0*3 + i1 + 1)) return 18;
    }

    return 0;
}
