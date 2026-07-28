// #268 MSVC diagnostic -- ORDER B: the SAME four cases as order_a.cpp, REVERSED.
// If a case's pass/fail flips depending on which one comes first in the
// translation unit, that confirms the defect is TU-context-dependent (an
// instantiation-order artifact) rather than keyed on the specific
// (T,Layout,Storage,Args) combination alone.
#include <teeny/teeny.h>
using namespace tny;

int main() {
    // CASE 4 (now FIRST): double, view, fcontiguous, 2 int args -- was reported FAILS
    double buf4[6] = {1,2,3,4,5,6};
    auto v4 = wrap(buf4, shape<2,3>{}, fcontiguous{});
    v4(0,0) = 20.0;

    // CASE 3: double, view, ccontiguous, 2 int args -- was reported WORKS
    double buf3[6] = {1,2,3,4,5,6};
    auto v3 = wrap(buf3, shape<2,3>{});
    v3(0,0) = 10.0;

    // CASE 2: int, stack, ccontiguous, 1 int arg -- was reported FAILS
    auto v2 = local<int, shape<3>>{};
    v2(0) = 1; v2(1) = 2; v2(2) = 3;

    // CASE 1 (now LAST): double, stack, ccontiguous, 1 int arg -- was reported WORKS
    auto v1 = local<double, shape<3>>{};
    v1(0) = 1.0; v1(1) = 2.0; v1(2) = 3.0;

    return 0;
}
