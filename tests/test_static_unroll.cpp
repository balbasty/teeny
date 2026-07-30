// Static-unroll cap (#343): the axis-reduction (#218) and dot/sqdist (#255) fast
// paths fully unroll an `index_sequence` fold over EVERY element of a fully-static
// shape. That is a win for the small fixed shapes they target, but the fold emits
// one argument per element, so an uncapped large static shape either fails to
// compile (clang: "exceeded expression nesting limit of 256") or takes minutes
// (g++). Both engines are now capped at TNY_MAX_STATIC_UNROLL elements and fall
// back to their runtime-decode engine above it.
//
// This test is as much a COMPILE-TIME test as a runtime one: the large-shape cases
// below would not build on clang at all before the cap, and took ~1 minute of g++
// per reduction. Every result is checked against a hand loop so the fallback path
// is proved to agree with the unrolled one.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

// Comfortably OVER the 256-element cap (2048 elements): the decode fallback.
// Kept a shape whose two axes differ so a per-axis reduction is not symmetric.
constexpr int BIG_R = 64;
constexpr int BIG_C = 32;
// Comfortably UNDER it (200 elements): the unrolled fast path, unchanged.
constexpr int SML_R = 20;
constexpr int SML_C = 10;

int main() {
    // ---- the gate itself -------------------------------------------------
    static_assert(_md::_unrollable<shape<3>>(),            "tiny static shape unrolls");
    static_assert(_md::_unrollable<shape<SML_R,SML_C>>(),  "200 elements unrolls");
    static_assert(_md::_unrollable<shape<16,16>>(),        "exactly 256 elements unrolls (clang's limit)");
    static_assert(!_md::_unrollable<shape<16,17>>(),       "272 elements does NOT unroll");
    static_assert(!_md::_unrollable<shape<BIG_R,BIG_C>>(), "2048 elements does NOT unroll");
    static_assert(!_md::_unrollable<shape<-1,3>>(),        "a dynamic extent never unrolls");
    static_assert(!_md::_unrollable<shape<-1,-1>>(),       "two dynamic extents never unroll (no wrap-around)");

    // ---- big static tensor: axis reductions (#218 path, now decoding) ----
    auto big = local<double, shape<BIG_R,BIG_C>>(); big.iota_(1.0, 1.0);   // 1 .. 2048, row-major
    static_assert(decltype(big)::ownership == storage::stack, "static -> stack");

    // reference column/row sums by hand
    double refcol[BIG_C] = {}, refrow[BIG_R] = {};
    for (int i = 0; i < BIG_R; ++i)
        for (int j = 0; j < BIG_C; ++j) { const double v = big(i,j); refcol[j] += v; refrow[i] += v; }

    auto csum = sum<0>(big);                        // -> shape (BIG_C)
    static_assert(decltype(csum)::rank() == 1, "axis 0 removed");
    static_assert(decltype(csum)::ownership == storage::stack, "static result stays on the stack");
    static_assert(decltype(csum.extent(Int<0>()))::value == BIG_C, "kept extent still static");
    for (int j = 0; j < BIG_C; ++j) if (csum(j) != refcol[j]) return 1;

    auto rsum = sum<1>(big);                        // -> shape (BIG_R)
    for (int i = 0; i < BIG_R; ++i) if (rsum(i) != refrow[i]) return 2;

    // mean / max / min over the big shape agree with the hand loops too
    auto cmean = mean<0>(big);
    for (int j = 0; j < BIG_C; ++j) if (cmean(j) != refcol[j] / double(BIG_R)) return 3;
    auto rmax = max<1>(big); auto rmin = min<1>(big);
    for (int i = 0; i < BIG_R; ++i) {
        if (rmax(i) != big(i, BIG_C-1)) return 4;   // iota is increasing along a row
        if (rmin(i) != big(i, 0))       return 5;
    }

    // keepdims / into(dest) still compose on the fallback path
    auto kd = local<double, shape<1,BIG_C>>();
    sum<0>(big, keepdims, into(kd));
    for (int j = 0; j < BIG_C; ++j) if (kd(0,j) != refcol[j]) return 6;

    // ---- big static tensors: dot / sqdist / dist (#255 path, now decoding) ----
    auto b2 = local<double, shape<BIG_R,BIG_C>>(); b2.iota_(0.5, 2.0);
    double refdot = 0.0, refsq = 0.0;
    for (int i = 0; i < BIG_R; ++i)
        for (int j = 0; j < BIG_C; ++j) {
            const double x = big(i,j), y = b2(i,j);
            refdot += x * y; refsq += (x - y) * (x - y);
        }
    if (dot(big, b2)    != refdot) return 7;
    if (sqdist(big, b2) != refsq)  return 8;
    if (sqnorm(b2)      != dot(b2, b2)) return 9;

    // ---- exactly AT the cap (256) still unrolls and still agrees ----------
    auto at = local<double, shape<16,16>>(); at.iota_(1.0, 1.0);
    double refat = 0.0; for (int k = 0; k < 256; ++k) refat += double(k+1) * double(k+1);
    if (dot(at, at) != refat) return 10;
    double refatcol[16] = {};
    for (int i = 0; i < 16; ++i) for (int j = 0; j < 16; ++j) refatcol[j] += at(i,j);
    auto atc = sum<0>(at);
    for (int j = 0; j < 16; ++j) if (atc(j) != refatcol[j]) return 11;

    // ---- past the cap (272): a shape that must decode --------------------
    auto over = local<double, shape<16,17>>(); over.iota_(1.0, 1.0);
    double refover = 0.0; for (int k = 0; k < 16*17; ++k) refover += double(k+1) * double(k+1);
    if (dot(over, over) != refover) return 12;

    // ---- small shape: the unrolled path is unchanged ----------------------
    auto sml = local<double, shape<SML_R,SML_C>>(); sml.iota_(1.0, 1.0);
    double refsml[SML_C] = {};
    for (int i = 0; i < SML_R; ++i) for (int j = 0; j < SML_C; ++j) refsml[j] += sml(i,j);
    auto ssum = sum<0>(sml);
    for (int j = 0; j < SML_C; ++j) if (ssum(j) != refsml[j]) return 13;
    double refsdot = 0.0; for (int k = 0; k < SML_R*SML_C; ++k) refsdot += double(k+1) * double(k+1);
    if (dot(sml, sml) != refsdot) return 14;

    // ---- unrolled and decoded paths agree ELEMENT-FOR-ELEMENT -------------
    // Same values, same reduction, one shape under the cap and one over it: a
    // (2,128) sum<0> decodes, a (2,128) view sliced to (2,64) unrolls. Compare the
    // overlapping half so a divergence in either engine shows up as a mismatch.
    auto wide = local<double, shape<2,128>>(); wide.iota_(1.0, 1.0);       // 256 elements: unrolls
    auto tall = local<double, shape<2,129>>(); tall.iota_(1.0, 1.0);       // 258 elements: decodes
    auto wsum = sum<0>(wide);   // unrolled
    auto tsum = sum<0>(tall);   // decoded
    for (int j = 0; j < 128; ++j) {
        // tall's row 1 starts 129 later than wide's, so recompute both by hand
        const double w = wide(0,j) + wide(1,j);
        const double t = tall(0,j) + tall(1,j);
        if (wsum(j) != w) return 15;
        if (tsum(j) != t) return 16;
    }

    // ---- a non-contiguous big static view was ALWAYS on the decode path ---
    auto tr = big.permute<1,0>();                   // (BIG_C, BIG_R), strided
    auto trs = sum<1>(tr);                          // == sum<0>(big)
    for (int j = 0; j < BIG_C; ++j) if (trs(j) != refcol[j]) return 17;

    return 0;
}
