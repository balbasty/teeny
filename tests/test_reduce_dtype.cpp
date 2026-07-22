#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main()
{
    // ---- default accumulator: double for small floats -----------------------
    float fbuf[4] = {1.f, 2.f, 3.f, 4.f};
    auto f = wrap(fbuf, shape<4>{});
    static_assert(cs::is_same<decltype(sum(f)), double>::value, "sum(float) -> double");
    static_assert(cs::is_same<decltype(mean(f)), double>::value, "mean(float) -> double");
    static_assert(cs::is_same<decltype(max(f)), double>::value, "max(float) -> double");
    if (sum(f) != 10.0) return 1;
    if (mean(f) != 2.5) return 2;

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

    // ---- axis reductions carry the accumulator into the result element type -
    float m[6] = {1,2,3,4,5,6};
    auto M = wrap(m, shape<2,3>{});               // rows [1,2,3],[4,5,6]
    auto r0 = sum<0>(M);                          // default acc -> double result
    static_assert(cs::is_same<typename decltype(r0)::element_type, double>::value, "sum<0>(float) -> double result");
    if (r0(0) != 5.0 || r0(1) != 7.0 || r0(2) != 9.0) return 5;
    auto r0f = sum<float, 0>(M);                  // explicit float accumulator + result
    static_assert(cs::is_same<typename decltype(r0f)::element_type, float>::value, "sum<float,0> -> float result");
    if (r0f(2) != 9.0f) return 6;
    auto mn = mean<1>(M);                         // default acc double
    static_assert(cs::is_same<typename decltype(mn)::element_type, double>::value, "mean<1> -> double result");
    if (mn(0) != 2.0 || mn(1) != 5.0) return 7;

    // axis sum over integers keeps the item type
    int im[4] = {1,2,3,4};
    auto IM = wrap(im, shape<2,2>{});
    auto is0 = sum<0>(IM);
    static_assert(cs::is_same<typename decltype(is0)::element_type, int>::value, "sum<0>(int) -> int result");
    if (is0(0) != 4 || is0(1) != 6) return 8;

    // ---- dot accumulates in the reduce type by default ----------------------
    static_assert(cs::is_same<decltype(dot(f, f)), double>::value, "dot(float,float) -> double");
    if (dot(f, f) != (1.+4.+9.+16.)) return 9;
    static_assert(cs::is_same<decltype(dot<float>(f, f)), float>::value, "dot<float> -> float");

    return 0;
}
