// peel_zip (#327): walk 2 or 3 broadcast-compatible tensors in lock-step, yielding
// a cs::tuple of views per step. A distinct name from peel (not an overload).
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

template <class TA, class TB>
double const_zip_sum(const TA & a, const TB & b) {
    // dependent-receiver context: exercises peel_zip's own template-arg deduction,
    // and the const overload, from inside a function template.
    double out = 0;
    for (auto [va, vb] : peel_zip<0>(a, b)) out += va(0) + vb(0);
    return out;
}

int main() {
    // ---- 2-tensor same-shape zip ---------------------------------------
    auto a = local<double, shape<3,4>>(); a.iota_(0.0, 1.0);
    auto b = local<double, shape<3,4>>(); b.iota_(100.0, 1.0);
    long n = 0;
    for (auto [va, vb] : peel_zip<0>(a, b)) {
        for (long j = 0; j < 4; ++j) if (vb(j) - va(j) != 100.0) return 1;
        ++n;
    }
    if (n != 3) return 2;

    // ---- 3-tensor same-shape zip: the triangle-vertex idiom -------------
    auto va3 = local<double, shape<4,3>>(); va3.iota_(0.0, 1.0);
    auto vb3 = local<double, shape<4,3>>(); vb3.iota_(100.0, 1.0);
    auto vc3 = local<double, shape<4,3>>(); vc3.iota_(200.0, 1.0);
    n = 0;
    for (auto [ta, tb, tc] : peel_zip<0>(va3, vb3, vc3)) {
        for (long j = 0; j < 3; ++j) { if (tb(j)-ta(j) != 100.0) return 3; if (tc(j)-ta(j) != 200.0) return 4; }
        ++n;
    }
    if (n != 4) return 5;

    // ---- broadcasting: an extent-1 axis stretches (numpy right-align) --
    auto A = local<double, shape<3,4>>(); A.iota_(0.0,1.0);
    auto B = local<double, shape<1,4>>(); B.iota_(1000.0,1.0);
    n = 0;
    for (auto [x, y] : peel_zip<0>(A, B)) { for (long j=0;j<4;++j) if (y(j) != B(0,j)) return 6; ++n; }
    if (n != 3) return 7;

    // ---- broadcasting: different RANKS (right-aligned) ------------------
    auto C = local<double, shape<4>>(); C.iota_(5000.0, 1.0);
    n = 0;
    for (auto [x, y] : peel_zip<0>(A, C)) { for (long j=0;j<4;++j) if (y(j) != C(j)) return 8; ++n; }
    if (n != 3) return 9;

    // ---- value form: peel_zip(a, b, axis<Axis>{}) == peel_zip<Axis>(a, b) ---
    n = 0;
    for (auto [x, y] : peel_zip(A, B, axis<0>{})) ++n;
    if (n != 3) return 10;
    n = 0;
    for (auto [x, y, z] : peel_zip(va3, vb3, vc3, axis<0>{})) ++n;
    if (n != 4) return 11;

    // ---- enumerate: (multi_index, tuple) per step, same shape as peel's ----
    n = 0;
    for (auto [m, cell] : peel_zip<0>(A, B).enumerate()) {
        if (m[0] != n) return 12;
        ++n;
    }
    if (n != 3) return 13;

    // ---- subrange: chunked sweep, incl. straight off a temporary -----------
    n = 0;
    for (auto cell : peel_zip<0>(A, B).subrange(1, 3)) { (void)cell; ++n; }
    if (n != 2) return 14;
    n = 0;
    for (auto item : peel_zip<0>(A, B).enumerate().subrange(1, 3)) { (void)item; ++n; }
    if (n != 2) return 15;

    // ---- mutable write-through -------------------------------------------
    auto dest = local<double, shape<3,4>>(); dest.zero_();
    for (auto [x, d] : peel_zip<0>(A, dest)) d.copy_(x);
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (dest(i,j) != A(i,j)) return 16;

    // ---- const overload, called from a dependent (template) receiver ------
    double s = 0; for (long i=0;i<3;++i) s += a(i,0) + b(i,0);
    if (const_zip_sum(a, b) != s) return 17;

    // ---- dynamic-shape operand mixed with static ---------------------------
    auto d = owned<double, shape<-1,4>>(shape<-1,4>{3}); d.iota_(0.0, 1.0);
    n = 0;
    for (auto [x, y] : peel_zip<0>(a, d)) { (void)x; (void)y; ++n; }
    if (n != 3) return 18;

    // ---- negative axis wraps (library-wide signed-axis convention) --------
    n = 0;
    for (auto [x, y] : peel_zip<-2>(a, b)) { (void)x; (void)y; ++n; }   // axis -2 == axis 0 (rank 2)
    if (n != 3) return 19;

    return 0;
}
