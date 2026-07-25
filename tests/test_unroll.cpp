// #184: TNY_UNROLL — a portable full-unroll hint for a small STATIC-trip-count loop.
// clang/nvcc honour `#pragma unroll`; gcc SILENTLY IGNORES it and needs
// `#pragma GCC unroll N`, which the macro emits. This test pins that the macro
// EXPANDS and COMPILES cleanly on both compilers (the `_Pragma` is the fragile part),
// works with a TEMPLATE-parameter trip count (the static-C kernel case gcc rejects
// for a per-count pragma), and stays correctness-neutral.
#include <teeny/teeny.h>
using namespace tny;

template <int N>                                   // N is a template param — the case
static long dot_static(const long (&a)[N], const long (&b)[N]) {   // gcc's count pragma can't take
    long s = 0;
    TNY_UNROLL                                     // ...but the fixed-count full unroll can
    for (int i = 0; i < N; ++i) s += a[i] * b[i];
    return s;
}

int main() {
    long a[4] = {1, 2, 3, 4}, b[4] = {5, 6, 7, 8};
    if (dot_static(a, b) != 1*5 + 2*6 + 3*7 + 4*8) return 1;   // 70

    // composes with teeny — a static-C workspace loop (the fastfields kernel shape).
    // Correctness must be identical with/without the hint.
    local<double, shape<3>> v; v.iota_(1.0, 1.0);              // 1,2,3
    double acc = 0;
    TNY_UNROLL
    for (int i = 0; i < 3; ++i) acc += v(i) * v(i);            // 1+4+9
    if (acc != 14.0) return 2;

    return 0;
}
