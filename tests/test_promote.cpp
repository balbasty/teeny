// Type promotion for binary math: C++ rules, except lower-precision float wins
// (half > float > double, pytorch-style). Plus the `shape<...>` type alias.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

template <class T> using S = local<T, shape<2,2>>;   // uses the shape<> alias

template <class A, class B, class Want>
constexpr bool promotes() {
    return cs::is_same<typename decltype(cs::declval<S<A>>() + cs::declval<S<B>>())::element_type, Want>::value;
}

int main() {
    // shape<...> == extents<int64_t, ...>  (matches DLPack's shape type)
    static_assert(cs::is_same<shape<2,3>, extents<cs::int64_t,2,3>>::value, "shape alias");
    static_assert(cs::is_same<shape<dynamic_extent,3>, extents<cs::int64_t,dynamic_extent,3>>::value, "shape dyn");
    // numpy-style -1 folds to dynamic_extent (and mixes with dynamic_extent)
    static_assert(cs::is_same<shape<-1,2,3>, shape<dynamic_extent,2,3>>::value, "shape<-1> == dynamic");
    static_assert(cs::is_same<shape<2,-1>, extents<cs::int64_t,2,dynamic_extent>>::value, "shape<...,-1>");

    // --- floats: LOWER precision wins ---------------------------------
    static_assert(promotes<half,   float,  half>(),   "half + float -> half");
    static_assert(promotes<half,   double, half>(),   "half + double -> half");
    static_assert(promotes<float,  double, float>(),  "float + double -> float");
    static_assert(promotes<double, half,   half>(),   "double + half -> half (order-independent)");
    static_assert(promotes<float,  float,  float>(),  "float + float -> float");
    static_assert(promotes<bfloat16, float, bfloat16>(), "bfloat16 + float -> bfloat16");

    // --- mixing int and float: the float wins (only one float rank) ----
    static_assert(promotes<int,    double, double>(), "int + double -> double");
    static_assert(promotes<half,   int,    half>(),   "half + int -> half");
    static_assert(promotes<int,    float,  float>(),  "int + float -> float");

    // --- both integral: ordinary C++ conversions -----------------------
    static_assert(promotes<int,    long,   long>(),   "int + long -> long");
    static_assert(promotes<int,    int,    int>(),    "int + int -> int");

    // --- scalar rhs promotes the same way ------------------------------
    auto h = local<half, shape<2,2>>(); h.fill_(half(3.0));
    auto hs = h.add(2.0);                                // half tensor + double scalar
    static_assert(cs::is_same<decltype(hs)::element_type, half>::value, "half + scalar double -> half");

    // --- runtime sanity: half+float value is correct -------------------
    auto a = local<half,  shape<2,2>>(); a.fill_(half(1.5));
    auto b = local<float, shape<2,2>>(); b.fill_(0.25f);
    auto c = a + b;
    if (static_cast<float>(c(0,0)) != 1.75f) return 1;

    return 0;
}
