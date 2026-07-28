// #268 MSVC diagnostic -- ISOLATED: the exact case test_vecalg.cpp reported as
// WORKING (double, stack, ccontiguous, 1 int arg), alone in an otherwise-empty
// file with nothing else instantiated first. Compare against preceded.cpp, which
// wraps the SAME call with the unrelated preceding template code that appears
// before it in test_unroll.cpp (where the identical case FAILED).
#include <teeny/teeny.h>
using namespace tny;

int main() {
    auto v = local<double, shape<3>>{};
    v(0) = 1.0; v(1) = 2.0; v(2) = 3.0;
    return (v(0)==1.0 && v(1)==2.0 && v(2)==3.0) ? 0 : 1;
}
