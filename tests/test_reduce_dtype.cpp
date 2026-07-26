#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main()
{
    // ---- default: accumulate in the reduce type, RETURN the element type -----
    float fbuf[4] = {1.f, 2.f, 3.f, 4.f};
    auto f = wrap(fbuf, shape<4>{});
    static_assert(cs::is_same<decltype(sum(f)), float>::value, "sum(float) -> float (element type)");
    static_assert(cs::is_same<decltype(mean(f)), float>::value, "mean(float) -> float");
    static_assert(cs::is_same<decltype(max(f)), float>::value, "max(float) -> float");
    if (sum(f) != 10.f) return 1;
    if (mean(f) != 2.5f) return 2;

    // internal accumulation is still WIDE (double): a sum float-accumulation would
    // get wrong (1e8 + 1 rounds away in float) is right, then cast back to float.
    float pv[3] = {1e8f, 1.f, -1e8f};
    auto P = wrap(pv, shape<3>{});
    static_assert(cs::is_same<decltype(sum(P)), float>::value, "sum -> float");
    if (sum(P) != 1.f) return 10;                 // double accumulation -> 1, not 0

    // reduce_type_t trait: floats keep the double/wide-float rule; narrow INTEGERS
    // widen to 64-bit (signed->int64_t, unsigned->uint64_t) so accumulation can't
    // overflow, while integers already >=8 bytes keep themselves. RESULT stays T.
    static_assert(cs::is_same<reduce_type_t<float>,  double>::value, "float -> double");
    static_assert(cs::is_same<reduce_type_t<double>, double>::value, "double -> double");
    static_assert(cs::is_same<reduce_type_t<half>,   double>::value, "half -> double");
    static_assert(cs::is_same<reduce_type_t<long double>, long double>::value, "wide float kept");
    static_assert(cs::is_same<reduce_type_t<signed char>,   cs::int64_t>::value,  "int8  -> int64 accumulator");
    static_assert(cs::is_same<reduce_type_t<short>,         cs::int64_t>::value,  "int16 -> int64 accumulator");
    static_assert(cs::is_same<reduce_type_t<int>,           cs::int64_t>::value,  "int32 -> int64 accumulator");
    static_assert(cs::is_same<reduce_type_t<unsigned char>, cs::uint64_t>::value, "uint8 -> uint64 accumulator");
    static_assert(cs::is_same<reduce_type_t<unsigned>,      cs::uint64_t>::value, "uint32 -> uint64 accumulator");
    static_assert(cs::is_same<reduce_type_t<bool>,          cs::uint64_t>::value, "bool -> uint64 accumulator");
    static_assert(cs::is_same<reduce_type_t<cs::int64_t>,   cs::int64_t>::value,  "int64 kept (already wide)");
    static_assert(cs::is_same<reduce_type_t<cs::uint64_t>,  cs::uint64_t>::value, "uint64 kept (already wide)");

    // ---- integers: accumulate WIDE (64-bit) but RETURN the item type ---------
    int ibuf[3] = {10, 20, 30};
    auto ii = wrap(ibuf, shape<3>{});
    static_assert(cs::is_same<decltype(sum(ii)), int>::value, "sum(int) -> int (result = item type)");
    if (sum(ii) != 60) return 3;

    // ---- explicit accumulator override --------------------------------------
    static_assert(cs::is_same<decltype(sum<float>(f)), float>::value, "sum<float> -> float");
    static_assert(cs::is_same<decltype(sum<long>(ii)), long>::value, "sum<long>(int) -> long");
    // small-int overflow: the accumulator is now wide, so accumulation is DEFINED.
    // The default result is still the item type, so the sum is cast back to int8 --
    // a DEFINED truncation (int8(400)), NOT the signed-overflow UB of an int8 accumulator.
    signed char cbuf[4] = {100, 100, 100, 100};   // true sum 400 exceeds int8 range
    auto cc = wrap(cbuf, shape<4>{});
    static_assert(cs::is_same<decltype(sum(cc)), signed char>::value, "sum(int8) -> int8 (item, default)");
    if (sum(cc) != (signed char)400) return 4;     // defined wraparound, not garbage
    if (sum<int>(cc) != 400) return 40;            // wide accumulator recovers the exact sum
    if (sum<cs::int64_t>(cc) != 400) return 41;

    // ---- mean of an integer tensor returns double (numpy: integer mean -> f64) --
    signed char mbuf[3] = {1, 2, 2};               // true mean 5/3 = 1.666...
    auto mc = wrap(mbuf, shape<3>{});
    static_assert(cs::is_same<decltype(mean(mc)), double>::value, "mean(int8) -> double");
    static_assert(cs::is_same<decltype(mean(ii)), double>::value, "mean(int32) -> double");
    if (mean(mc) < 1.6666 || mean(mc) > 1.6667) return 42;   // fractional, not integer-truncated
    static_assert(cs::is_same<decltype(mean<float>(mc)), float>::value, "mean<float>(int8) -> float (escape hatch)");
    if (mean<float>(mc) < 1.666f || mean<float>(mc) > 1.667f) return 43;

    // ---- axis reductions: default result element type = T, explicit = Acc ----
    float m[6] = {1,2,3,4,5,6};
    auto M = wrap(m, shape<2,3>{});               // rows [1,2,3],[4,5,6]
    auto r0 = sum<0>(M);                          // default -> float result (accum in double)
    static_assert(cs::is_same<typename decltype(r0)::element_type, float>::value, "sum<0>(float) -> float result");
    if (r0(0) != 5.f || r0(1) != 7.f || r0(2) != 9.f) return 5;
    auto r0d = sum<double, 0>(M);                 // explicit double accumulator + result
    static_assert(cs::is_same<typename decltype(r0d)::element_type, double>::value, "sum<double,0> -> double result");
    if (r0d(2) != 9.0) return 6;
    auto mn = mean<1>(M);                         // default -> float result
    static_assert(cs::is_same<typename decltype(mn)::element_type, float>::value, "mean<1> -> float result");
    if (mn(0) != 2.f || mn(1) != 5.f) return 7;

    // axis sum over integers keeps the item type
    int im[4] = {1,2,3,4};
    auto IM = wrap(im, shape<2,2>{});
    auto is0 = sum<0>(IM);
    static_assert(cs::is_same<typename decltype(is0)::element_type, int>::value, "sum<0>(int) -> int result");
    if (is0(0) != 4 || is0(1) != 6) return 8;

    // axis mean over integers -> a double-typed tensor (numpy), divided in double
    auto imn = mean<0>(IM);                       // columns: (1+3)/2=2, (2+4)/2=3
    static_assert(cs::is_same<typename decltype(imn)::element_type, double>::value, "mean<0>(int) -> double result");
    if (imn(0) != 2.0 || imn(1) != 3.0) return 80;
    int om[6] = {1,2,3,4,5,6};                    // rows [1,2,3],[4,5,6]; row means 2, 5
    auto OM = wrap(om, shape<2,3>{});
    auto imn1 = mean<1>(OM);
    static_assert(cs::is_same<typename decltype(imn1)::element_type, double>::value, "mean<1>(int) -> double result");
    if (imn1(0) != 2.0 || imn1(1) != 5.0) return 81;
    auto imnf = mean<float, 1>(OM);               // escape hatch keeps working
    static_assert(cs::is_same<typename decltype(imnf)::element_type, float>::value, "mean<float,1>(int) -> float result");
    if (imnf(0) != 2.f || imnf(1) != 5.f) return 82;

    // ---- dot: default result = promote(Ta,Tb), explicit = Acc ----------------
    static_assert(cs::is_same<decltype(dot(f, f)), float>::value, "dot(float,float) -> float");
    if (dot(f, f) != 30.f) return 9;              // 1+4+9+16, accumulated in double
    static_assert(cs::is_same<decltype(dot<double>(f, f)), double>::value, "dot<double> -> double");

    // ---- #218: the static + C-contiguous fast path == the reference, every axis ----
    // (a 4x4x4 iota, the case that was 3-7x slow; the static unroll must match a hand loop)
    auto P3 = zeros<double>(shape<4,4,4>{}); P3.iota_(1, 1);   // 1..64, row-major
    auto s1 = sum<double,1>(P3);                              // reduce the MIDDLE axis -> (4,4)
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 4; ++k) {
        double ref = 0; for (int j = 0; j < 4; ++j) ref += P3(i,j,k);
        if (s1(i,k) != ref) return 90;
    }
    auto s0 = sum<double,0>(P3);                              // reduce the OUTER axis
    for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) {
        double ref = 0; for (int i = 0; i < 4; ++i) ref += P3(i,j,k);
        if (s0(j,k) != ref) return 91;
    }
    auto s02 = sum<double,0,2>(P3);                           // multi-axis (0 and 2) -> (4,)
    for (int j = 0; j < 4; ++j) {
        double ref = 0; for (int i = 0; i < 4; ++i) for (int k = 0; k < 4; ++k) ref += P3(i,j,k);
        if (s02(j) != ref) return 92;
    }
    if (max<1>(P3)(3,3) != P3(3,3,3)) return 93;               // max over the middle axis (fast path too)
    // a NON-contiguous static source falls to the runtime path — must still be correct
    auto P3t = P3.permute<2,1,0>();                            // static extents, strided (not ccontiguous)
    auto st = sum<double,1>(P3t);
    for (int a2 = 0; a2 < 4; ++a2) for (int c = 0; c < 4; ++c) {
        double ref = 0; for (int b = 0; b < 4; ++b) ref += P3t(a2,b,c);
        if (st(a2,c) != ref) return 94;
    }

    return 0;
}
