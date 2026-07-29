// scan_/scan (#254): in-place/out-of-place sequential fold along one axis,
// batched (peeled) over every other axis.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

// running-min-plus fold (the L1 distance-transform recurrence, one direction):
// carry = min(carry + w, x); matches examples/distance_transform.cpp's l1_line.
struct min_plus {
    double w;
    _TNY_API double operator()(double carry, double x) const { return (carry + w) < x ? (carry + w) : x; }
};
struct sum_op { _TNY_API double operator()(double carry, double x) const { return carry + x; } };

int main() {
    // rank-1: cumulative sum via scan_
    auto t = local<double, shape<5>>();
    for (long i=0;i<5;++i) t(i) = static_cast<double>(i+1);   // 1,2,3,4,5
    scan_<0>(t, 0.0, sum_op{});
    double cum[5] = {1,3,6,10,15};
    for (long i=0;i<5;++i) if (t(i) != cum[i]) return 1;

    // rank-2: scan along axis 1, batched over axis 0 (each row independently)
    auto m = local<double, shape<3,4>>();
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) m(i,j) = static_cast<double>(j+1);   // rows: 1,2,3,4
    scan_<1>(m, 0.0, sum_op{});
    for (long i=0;i<3;++i) { double c[4]={1,3,6,10}; for (long j=0;j<4;++j) if (m(i,j) != c[j]) return 2; }

    // scan along axis 0, batched over axis 1 (each column independently)
    auto m2 = local<double, shape<4,3>>();
    for (long i=0;i<4;++i) for (long j=0;j<3;++j) m2(i,j) = static_cast<double>(i+1);   // cols: 1,2,3,4
    scan_<0>(m2, 0.0, sum_op{});
    for (long j=0;j<3;++j) { double c[4]={1,3,6,10}; for (long i=0;i<4;++i) if (m2(i,j) != c[i]) return 3; }

    // negative axis
    auto m3 = local<double, shape<2,4>>();
    for (long i=0;i<2;++i) for (long j=0;j<4;++j) m3(i,j) = static_cast<double>(j+1);
    scan_<-1>(m3, 0.0, sum_op{});
    for (long i=0;i<2;++i) { double c[4]={1,3,6,10}; for (long j=0;j<4;++j) if (m3(i,j) != c[j]) return 4; }

    // value form: scan_(t, axis<Axis>{}, init, f) == scan_<Axis>(t, init, f)
    auto m4 = local<double, shape<3,4>>();
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) m4(i,j) = static_cast<double>(j+1);
    scan_(m4, axis<1>{}, 0.0, sum_op{});
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (m4(i,j) != m(i,j)) return 5;

    // reverse sweep composes with flip: scan the reversed view (named lvalue --
    // scan_/peel take a non-const lvalue ref, like peel<Axes...> itself, so a
    // temporary view must be bound to a name first)
    auto r = local<double, shape<5>>();
    for (long i=0;i<5;++i) r(i) = static_cast<double>(i+1);   // 1,2,3,4,5
    auto rf = r.flip<0>();
    scan_<0>(rf, 0.0, sum_op{});
    // flip<0>() reverses -> scan sees 5,4,3,2,1 -> cumsum 5,9,12,14,15 -> written
    // back through the flipped view into r as 15,14,12,9,5
    double rref[5] = {15,14,12,9,5};
    for (long i=0;i<5;++i) if (r(i) != rref[i]) return 6;

    // real distance-transform recurrence: min_plus forward sweep matches the
    // hand-written l1_line forward half (examples/distance_transform.cpp).
    auto d = local<double, shape<7>>();
    double init[7] = {0.0, 1e9, 1e9, 1e9, 1e9, 1e9, 0.0};
    for (long i=0;i<7;++i) d(i) = init[i];
    scan_<0>(d, 1e9, min_plus{1.0});
    // forward sweep of Felzenszwalb-style min-plus: carry starts huge, so the
    // first element is just itself; hand-compute the reference forward pass.
    double fwd[7]; double carry = 1e9;
    for (long i=0;i<7;++i) { double cand = carry + 1.0; carry = cand < init[i] ? cand : init[i]; fwd[i] = carry; }
    for (long i=0;i<7;++i) if (d(i) != fwd[i]) return 7;

    // out-of-place: scan<Axis>(t, init, f) -> fresh tensor, source untouched
    auto s = local<double, shape<4>>();
    for (long i=0;i<4;++i) s(i) = static_cast<double>(i+1);
    auto so = scan<0>(s, 0.0, sum_op{});
    double sref[4] = {1,3,6,10};
    for (long i=0;i<4;++i) { if (so(i) != sref[i]) return 8; if (s(i) != static_cast<double>(i+1)) return 9; }

    // out-of-place value form
    auto so2 = scan(s, axis<0>{}, 0.0, sum_op{});
    for (long i=0;i<4;++i) if (so2(i) != sref[i]) return 10;

    // into(dest): no allocation, writes into a preallocated buffer
    auto dest = local<double, shape<4>>(); dest.zero_();
    auto & destref = scan<0>(s, 0.0, sum_op{}, into(dest));
    for (long i=0;i<4;++i) if (dest(i) != sref[i]) return 11;
    if (&destref != &dest) return 12;

    // dynamic-shape source (heap): out-of-place static/dynamic dispatch
    auto dyn = make_heap<double>(shape<-1>{4});
    for (long i=0;i<4;++i) dyn(i) = static_cast<double>(i+1);
    auto dyno = scan<0>(dyn, 0.0, sum_op{});
    for (long i=0;i<4;++i) if (dyno(i) != sref[i]) return 13;

    return 0;
}
