// #268 MSVC diagnostic -- ORDER A (matches the order these combos first appeared
// historically: known-pass, known-fail, known-pass, known-fail). Compiled standalone
// via cl.exe (no /fmax-errors-equivalent stop-early flag), so every failing line
// should show its own error in one compile -- compare against order_b.cpp, which
// has the SAME four cases in REVERSED order, to test whether success/failure
// tracks the (T,Layout,Storage,Args) combination itself or the file/TU context
// (what was instantiated earlier in the same translation unit).
#include <teeny/teeny.h>
using namespace tny;

int main() {
    // CASE 1: double, stack, ccontiguous, 1 int arg -- test_vecalg.cpp reported this WORKS
    auto v1 = local<double, shape<3>>{};
    v1(0) = 1.0; v1(1) = 2.0; v1(2) = 3.0;

    // CASE 2: int, stack, ccontiguous, 1 int arg -- original #268 report: this FAILS
    auto v2 = local<int, shape<3>>{};
    v2(0) = 1; v2(1) = 2; v2(2) = 3;

    // CASE 3: double, view, ccontiguous, 2 int args -- test_wrap_layout_tag.cpp: WORKS
    double buf3[6] = {1,2,3,4,5,6};
    auto v3 = wrap(buf3, shape<2,3>{});
    v3(0,0) = 10.0;

    // CASE 4: double, view, fcontiguous, 2 int args -- test_wrap_layout_tag.cpp: FAILS
    double buf4[6] = {1,2,3,4,5,6};
    auto v4 = wrap(buf4, shape<2,3>{}, fcontiguous{});
    v4(0,0) = 20.0;

    return 0;
}
