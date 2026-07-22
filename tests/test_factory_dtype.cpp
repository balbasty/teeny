#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

template <class Expect, class Got> constexpr bool is = cs::is_same<Expect, typename Got::element_type>::value;

int main()
{
    // ---- value-less factories default to float ------------------------------
    auto z = zeros(shape<2,3>{});
    static_assert(is<float, decltype(z)>, "zeros defaults to float");
    auto o = ones(shape<4>{});
    static_assert(is<float, decltype(o)>, "ones defaults to float");
    auto l = make_local(shape<2,2>{});
    static_assert(is<float, decltype(l)>, "make_local defaults to float");
    // explicit T still honoured
    static_assert(is<double, decltype(zeros<double>(shape<2>{}))>, "zeros<double>");
    static_assert(is<int, decltype(make_local<int>(shape<2>{}))>, "make_local<int>");

    // ---- full infers the element type from the VALUE (numpy) ---------------
    auto fi = full(shape<3>{}, 3);        // int value -> int tensor
    static_assert(is<int, decltype(fi)>, "full(_, 3) is int");
    if (fi(0) != 3) return 1;
    auto fd = full(shape<3>{}, 2.5);      // double value -> double tensor
    static_assert(is<double, decltype(fd)>, "full(_, 2.5) is double");
    if (fd(1) != 2.5) return 2;
    auto ff = full<float>(shape<3>{}, 3); // explicit override
    static_assert(is<float, decltype(ff)>, "full<float>(_, 3)");
    if (ff(2) != 3.0f) return 3;

    // ---- arange defaults to an integer range -------------------------------
    auto r = arange(4);
    static_assert(is<cs::int64_t, decltype(r)>, "arange defaults to int64");
    if (r(0) != 0 || r(3) != 3) return 4;
    auto rf = arange<float>(3);
    static_assert(is<float, decltype(rf)>, "arange<float>");
    if (rf(2) != 2.0f) return 5;

    // ---- static arange: value form and Int<> form --------------------------
    auto sr = arange<double, 5>();
    static_assert(is<double, decltype(sr)>, "static arange element type");
    static_assert(decltype(sr)::rank() == 1, "static arange rank");
    static_assert(decltype(sr)::extents_type::static_extent(0) == 5, "static arange extent");
    if (sr(4) != 4.0) return 6;
    auto si = arange(Int<3>());            // T defaults to int64
    static_assert(is<cs::int64_t, decltype(si)>, "arange(Int<3>()) default int64");
    static_assert(decltype(si)::extents_type::static_extent(0) == 3, "arange(Int<3>()) extent");
    if (si(0) != 0 || si(2) != 2) return 7;
    auto si2 = arange<float>(Int<4>());
    static_assert(is<float, decltype(si2)>, "arange<float>(Int<4>())");
    if (si2(3) != 3.0f) return 8;

    return 0;
}
