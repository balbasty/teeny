// #268 MSVC diagnostic -- PRECEDED: the identical tensor call from isolated.cpp,
// but preceded by the unrelated template function that appears earlier in
// tests/test_unroll.cpp (where this exact tensor case FAILED). If this file
// fails to compile the tensor lines while isolated.cpp compiles them fine, that
// is direct evidence that unrelated PRECEDING template instantiations in the
// same translation unit affect whether operator() resolves on MSVC.
#include <teeny/teeny.h>
using namespace tny;

template <int N>
static long dot_static(const long (&a)[N], const long (&b)[N]) {
    long s = 0;
    for (int i = 0; i < N; ++i) s += a[i] * b[i];
    return s;
}

int main() {
    long a[4] = {1, 2, 3, 4}, b[4] = {5, 6, 7, 8};
    if (dot_static(a, b) != 1*5 + 2*6 + 3*7 + 4*8) return 1;

    auto v = local<double, shape<3>>{};
    v(0) = 1.0; v(1) = 2.0; v(2) = 3.0;
    return (v(0)==1.0 && v(1)==2.0 && v(2)==3.0) ? 0 : 2;
}
