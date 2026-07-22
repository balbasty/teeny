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

    // reduce_type_t trait
    static_assert(cs::is_same<reduce_type_t<float>,  double>::value, "float -> double");
    static_assert(cs::is_same<reduce_type_t<double>, double>::value, "double -> double");
    static_assert(cs::is_same<reduce_type_t<half>,   double>::value, "half -> double");
    static_assert(cs::is_same<reduce_type_t<long double>, long double>::value, "wide float kept");
    static_assert(cs::is_same<reduce_type_t<int>,    int>::value,    "int -> int (item)");
    static_assert(cs::is_same<reduce_type_t<long>,   long>::value,   "long -> long (item)");

    // ---- integers accumulate in the item type by default --------------------
    int ibuf[3] = {10, 20, 30};
    auto ii = wrap(ibuf, shape<3>{});
    static_assert(cs::is_same<decltype(sum(ii)), int>::value, "sum(int) -> int");
    if (sum(ii) != 60) return 3;

    // ---- explicit accumulator override --------------------------------------
    static_assert(cs::is_same<decltype(sum<float>(f)), float>::value, "sum<float> -> float");
    static_assert(cs::is_same<decltype(sum<long>(ii)), long>::value, "sum<long>(int) -> long");
    // small-int overflow avoided by a wider accumulator
    signed char cbuf[4] = {100, 100, 100, 100};   // sum 400 overflows int8
    auto cc = wrap(cbuf, shape<4>{});
    if (sum<int>(cc) != 400) return 4;
    static_assert(cs::is_same<decltype(sum(cc)), signed char>::value, "sum(int8) -> int8 (item, default)");

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

    // ---- dot: default result = promote(Ta,Tb), explicit = Acc ----------------
    static_assert(cs::is_same<decltype(dot(f, f)), float>::value, "dot(float,float) -> float");
    if (dot(f, f) != 30.f) return 9;              // 1+4+9+16, accumulated in double
    static_assert(cs::is_same<decltype(dot<double>(f, f)), double>::value, "dot<double> -> double");

    return 0;
}
