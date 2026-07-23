// operator[] — the C++23 multidimensional-subscript alias of operator() (mdspan's
// own spelling). Under C++17/20 (no __cpp_multidimensional_subscript) operator[]
// is absent, so this test is a no-op pass; the real checks run under -std=c++23
// (`make cxx23`), where t[i,j] must behave exactly like t(i,j).
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main()
{
#if defined(__cpp_multidimensional_subscript)
    double b[12];
    auto t = wrap(b, shape<3, 4>{}); t.iota_(0.0);        // t(i,j) = 4*i + j

    // element access: t[i,j] == t(i,j), same lvalue
    for (long i = 0; i < 3; ++i)
        for (long j = 0; j < 4; ++j) {
            if (t[i, j] != t(i, j))   return 1;
            if (&t[i, j] != &t(i, j)) return 2;
        }
    t[1, 2] = 99.0; if (t(1, 2) != 99.0) return 3;         // writes through

    // slice: t[0, all] is the same VIEW type/result as t(0, all)
    auto r0 = t[0, all];
    auto r1 = t(0, all);
    static_assert(cs::is_same<decltype(r0), decltype(r1)>::value, "subscript slice type == call slice type");
    if (r0.extent(0) != 4) return 4;
    for (long j = 0; j < 4; ++j) if (r0(j) != r1(j)) return 5;

    // range + ellipsis args forward too
    auto s = t[all, slice(1, 3)];
    if (s.extent(0) != 3 || s.extent(1) != 2) return 6;
    auto e = t[ellipsis];
    if (e(2, 3) != t(2, 3)) return 7;

    // negative index wraps like operator() (checked path)
    if (t[-1, -1] != t(2, 3)) return 8;
#endif
    return 0;
}
