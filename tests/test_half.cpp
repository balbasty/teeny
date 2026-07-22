// half-precision element types: float16 + bfloat16 as tensor elements.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
#include <cmath>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;

template <class H> static bool approx(H h, double want, double tol) {
    return std::fabs(static_cast<double>(static_cast<float>(h)) - want) <= tol;
}

int main() {
    // ---- conversions round-trip (exactly representable values) ---------
    static_assert(sizeof(half) == 2 && sizeof(bfloat16) == 2, "16-bit");
    static_assert(cs::is_trivially_copyable<half>::value, "kernel-passable");
    if (!approx(half(1.5), 1.5, 0.0)) return 1;
    if (!approx(half(-0.25), -0.25, 0.0)) return 2;
    if (!approx(bfloat16(1.5), 1.5, 0.0)) return 3;
    // representative rounding: 0.1 is inexact in both, but close
    if (!approx(half(0.1), 0.1, 1e-3)) return 4;
    if (!approx(bfloat16(0.1), 0.1, 1e-2)) return 5;

    // ---- half tensor: element access, fill_, in-place broadcast math ---
    auto a = local<half, extents<long,2,3>>();
    a.fill_(half(2.0));
    for (long i=0;i<2;++i) for (long j=0;j<3;++j) if (!approx(a(i,j), 2.0, 0.0)) return 6;

    auto col = local<half, extents<long,2,1>>();
    col(0,0) = half(10.0); col(1,0) = half(20.0);
    a.mul_(col);                                    // broadcasts across axis 1
    if (!approx(a(0,2), 20.0, 0.0) || !approx(a(1,0), 40.0, 0.0)) return 7;

    a.add_(half(1.0));                              // scalar in-place
    if (!approx(a(1,1), 41.0, 0.0)) return 8;

    // ---- out-of-place half+half -> half stack tensor -------------------
    auto b = local<half, extents<long,2,3>>(); b.fill_(half(0.5));
    auto c = a.add(b);
    static_assert(cs::is_same<decltype(c)::element_type, half>::value, "half + half -> half");
    if (!approx(c(0,0), 21.5, 0.0)) return 9;       // a(0,0)=20+1=21, +0.5

    // ---- reduction accumulates in float (precision) -------------------
    // 2049 fp16 values of 1.0: fp16 can't represent 2049 exactly (gap > 1 above
    // 2048), but the float accumulator sums correctly, then rounds to fp16 2048.
    auto big = local<half, extents<long,2049>>(); big.fill_(half(1.0));
    double s = static_cast<float>(sum(big));
    if (std::fabs(s - 2048.0) > 1.0) return 10;      // NOT stuck at ~1024 (half-acc bug)

    // ---- bfloat16 tensor basics ---------------------------------------
    auto bf = local<bfloat16, extents<long,4>>();
    for (long i=0;i<4;++i) bf(i) = bfloat16(i + 1);
    bf.mul_(bfloat16(2.0));
    if (!approx(bf(3), 8.0, 0.0)) return 11;
    if (!approx(sum(bf), 2+4+6+8, 0.0)) return 12;

    // ---- #47: explicit half/bfloat16 accumulator for max/min ----------
    // numeric_limits isn't specialized for the software half, so the max/min
    // SEED used to be 0 -> a max over ALL-NEGATIVE values wrongly returned 0.
    auto neg = local<half, extents<long,3>>();
    neg(0) = half(-5.0); neg(1) = half(-2.0); neg(2) = half(-9.0);
    if (!approx(max<half>(neg), -2.0, 0.0)) return 13;    // NOT 0 (the old bug)
    if (!approx(min<half>(neg), -9.0, 0.0)) return 14;
    auto pos = local<bfloat16, extents<long,3>>();
    pos(0) = bfloat16(3.0); pos(1) = bfloat16(7.0); pos(2) = bfloat16(4.0);
    if (!approx(min<bfloat16>(pos), 3.0, 0.0)) return 15; // NOT 0
    if (!approx(max<bfloat16>(pos), 7.0, 0.0)) return 16;
    // default accumulator (double) already worked; confirm it still does.
    if (!approx(max(neg), -2.0, 0.0)) return 17;

    return 0;
}
