// dtype<...>: a value-tag carrier for the element type, the sibling of axis<...>
// for type-parameterised calls — deduces T from the argument instead of an
// explicit <T> template parameter. (#244)
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

static bool close(double a, double b) { return a - b < 1e-9 && b - a < 1e-9; }

int main() {
    // ---- factories: element type by value instead of by template arg -------
    auto e1 = empty(shape<3,3>{}, dtype<double>{});
    static_assert(cs::is_same<decltype(e1)::element_type, double>::value, "");
    static_assert(decltype(e1)::ownership == storage::stack, "");
    if (sizeof(e1) != 9 * sizeof(double)) return 1;

    // dtype<T> composes with an explicit leading O template arg (T stays deduced)
    auto e2 = empty<storage::heap>(shape<3,3>{}, dtype<double>{});
    static_assert(decltype(e2)::ownership == storage::heap, "");   // heap despite a static shape

    auto z1 = zeros(shape<3,3>{}, dtype<float>{});
    static_assert(cs::is_same<decltype(z1)::element_type, float>::value, "");
    if (z1(0,0) != 0.0f || z1(2,2) != 0.0f)  return 2;

    // dtype<T> composes with an explicit leading O template arg (not storage_c —
    // that value tag already carries O; dtype<T> lets O stay a template arg instead)
    auto o1 = ones<storage::heap>(shape<-1,3>{}, dtype<double>{});
    static_assert(decltype(o1)::ownership == storage::heap, "");
    auto o3 = ones(shape<3>{}, dtype<int>{});
    if (o3(0) != 1 || o3(2) != 1)             return 3;

    auto f1 = full(shape<4>{}, 7, dtype<double>{});   // dtype overrides the value's own (int) type
    static_assert(cs::is_same<decltype(f1)::element_type, double>::value, "");
    if (f1(0) != 7.0 || f1(3) != 7.0)          return 4;

    auto a1 = arange(5, dtype<double>{});
    static_assert(cs::is_same<decltype(a1)::element_type, double>::value, "");
    if (a1(0) != 0.0 || a1(4) != 4.0)          return 5;
    auto a2 = arange<storage::heap>(5, dtype<float>{});
    static_assert(decltype(a2)::ownership == storage::heap, "");
    if (a2(4) != 4.0f)                          return 6;

    // ---- .to(): dtype<T2> deduced sibling of .to<T2>() ----------------------
    auto t1 = local<int, shape<3>>{}; t1(0)=1; t1(1)=2; t1(2)=3;
    auto t2 = t1.to(dtype<double>{});
    static_assert(cs::is_same<decltype(t2)::element_type, double>::value, "");
    if (t2(1) != 2.0)                           return 7;

    // rvalue path (temporary receiver)
    auto t3 = local<int, shape<3>>{}.to(dtype<float>{});
    static_assert(cs::is_same<decltype(t3)::element_type, float>::value, "");
    if (t3(0) != 0.0f)                          return 8;

    // matching dtype -> no-copy borrow (same as .to<>())
    auto t4 = t1.to(dtype<int>{});
    if (t4.data() != t1.data())                 return 9;   // borrowed, same buffer

    // dynamic (heap) source -> host allocating path
    auto dh = zeros<int>(shape<-1>{3}); dh(0)=1; dh(1)=2; dh(2)=3;
    auto dh2 = dh.to(dtype<double>{});
    static_assert(cs::is_same<decltype(dh2)::element_type, double>::value, "");
    if (dh2(2) != 3.0)                          return 10;

    // ---- reductions: dtype<Acc> deduced sibling of NAME<Acc>(a) -------------
    auto v = local<double, shape<3>>{}; v(0)=1; v(1)=2; v(2)=3;
    auto s = sum(v, dtype<float>{});
    static_assert(cs::is_same<decltype(s), float>::value, "");
    if (s != 6.0f)                              return 11;
    if (prod(v, dtype<double>{}) != 6.0)        return 12;
    if (max(v, dtype<double>{}) != 3.0)         return 13;
    if (min(v, dtype<double>{}) != 1.0)         return 14;
    if (mean(v, dtype<double>{}) != 2.0)        return 15;
    if (dot(v, v, dtype<double>{}) != 14.0)     return 16;
    if (sqnorm(v, dtype<double>{}) != 14.0)     return 17;
    if (!close(norm(v, dtype<double>{}), 3.7416573867739413)) return 18;

    // dtype<Acc> == the explicit <Acc> template form (byte-for-byte the same call)
    if (sum(v, dtype<float>{}) != sum<float>(v)) return 19;

    // ---- reductions AS METHODS: dtype<Acc> parity (#251 method surface) -----
    if (v.sum(dtype<float>{}) != v.sum<float>())     return 20;
    if (v.prod(dtype<double>{}) != 6.0)               return 21;
    if (v.max(dtype<double>{}) != 3.0)                return 22;
    if (v.min(dtype<double>{}) != 1.0)                return 23;
    if (v.mean(dtype<double>{}) != 2.0)               return 24;
    if (v.dot(v, dtype<double>{}) != 14.0)            return 25;
    if (v.sqnorm(dtype<double>{}) != 14.0)            return 26;
    if (!close(v.norm(dtype<double>{}), 3.7416573867739413)) return 27;

    return 0;
}
