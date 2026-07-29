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

namespace tny_test {
// #375 regression guard: `scan`'s axis<> value form is SPLIT into an _TNY_API
// (static shape -> stack result) and a _TNY_HOST (dynamic shape -> heap result)
// forwarder, matching the `<Axis>` pair it forwards to, so nvcc's device pass
// never sees a __host__ __device__ forwarder call a __host__ allocator. The whole
// POINT of the value form is that it deduces `Axis` and so needs no `.template`
// on a type-dependent receiver -- the property a naive two-overload split is most
// likely to break. `Tn`/`F` are template parameters, so both calls below are
// genuinely dependent.
template <class Tn, class F>
bool dependent_scan(const Tn & t, F f) {
    auto got = scan(t, axis<0>{}, 0.0, f);   // value form, deduced -- no `.template`
    auto exp = scan<0>(t, 0.0, f);           // the <Axis> twin (a free fn: no `.template` either)
    static_assert(cs::is_same<decltype(got), decltype(exp)>::value,
                  "#375: value form and <Axis> form must agree on the result type");
    const long n = (long) t.shape(0);
    for (long i = 0; i < n; ++i) if (got(i) != exp(i)) return false;
    return true;
}
} // namespace tny_test

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

    // reverse sweep composes with flip: scan the reversed view DIRECTLY --
    // scan_ has both lvalue and rvalue overloads, so a temporary flip() view
    // binds fine (mutates the same underlying storage as any named view would).
    auto r = local<double, shape<5>>();
    for (long i=0;i<5;++i) r(i) = static_cast<double>(i+1);   // 1,2,3,4,5
    scan_<0>(r.flip<0>(), 0.0, sum_op{});
    // flip<0>() reverses -> scan sees 5,4,3,2,1 -> cumsum 5,9,12,14,15 -> written
    // back through the flipped view into r as 15,14,12,9,5
    double rref[5] = {15,14,12,9,5};
    for (long i=0;i<5;++i) if (r(i) != rref[i]) return 6;

    // a NAMED lvalue view works identically (both spellings are equivalent)
    auto r2 = local<double, shape<5>>();
    for (long i=0;i<5;++i) r2(i) = static_cast<double>(i+1);
    auto rf2 = r2.flip<0>();
    scan_<0>(rf2, 0.0, sum_op{});
    for (long i=0;i<5;++i) if (r2(i) != rref[i]) return 14;

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

    // rvalue + value form together: scan_(rvalue_view, axis<Axis>{}, init, f)
    auto r3 = local<double, shape<5>>();
    for (long i=0;i<5;++i) r3(i) = static_cast<double>(i+1);
    scan_(r3.flip<0>(), axis<0>{}, 0.0, sum_op{});
    for (long i=0;i<5;++i) if (r3(i) != rref[i]) return 15;

    // into(dest) with a DIFFERENT element type: copy_ casts FIRST, then scan_
    // walks dest in DEST's own precision (unlike index_select/reductions'
    // into(dest), which only cast the FINAL result) -- with values that round
    // exactly in float this still matches the double computation elementwise.
    auto fdest = local<float, shape<4>>(); fdest.zero_();
    scan<0>(s, 0.0, sum_op{}, into(fdest));
    for (long i=0;i<4;++i) if (fdest(i) != static_cast<float>(sref[i])) return 16;

    // #375: the axis<> value form is SPLIT into an _TNY_API (static -> stack) and
    // a _TNY_HOST (dynamic -> heap) forwarder so nvcc's device pass never sees a
    // __host__ __device__ forwarder call a __host__ allocator. Host-side the split
    // must be INVISIBLE: each arm picks the same underlying overload -- same
    // ownership, same values -- as the equivalent <Axis> template-form spelling.
    // (a) static shape -> _TNY_API arm -> stack result (== scan<0>(s, ...)).
    static_assert(decltype(so2)::ownership == storage::stack, "#375: static value form -> stack (_TNY_API arm)");
    static_assert(cs::is_same<decltype(so2), decltype(scan<0>(s, 0.0, sum_op{}))>::value,
                  "#375: static value form must yield the <Axis> form's exact type");
    // (b) dynamic shape -> _TNY_HOST arm -> heap result (== scan<0>(dyn, ...)).
    auto dynv = scan(dyn, axis<0>{}, 0.0, sum_op{});
    static_assert(decltype(dynv)::ownership == storage::heap, "#375: dynamic value form -> heap (_TNY_HOST arm)");
    static_assert(cs::is_same<decltype(dynv), decltype(scan<0>(dyn, 0.0, sum_op{}))>::value,
                  "#375: dynamic value form must yield the <Axis> form's exact type");
    for (long i=0;i<4;++i) { if (dynv(i) != sref[i]) return 17; if (dynv(i) != dyno(i)) return 18; }
    // (c) the source is still untouched by the out-of-place value form.
    for (long i=0;i<4;++i) if (dyn(i) != static_cast<double>(i+1)) return 19;
    // (d) both arms must still deduce with NO `.template` on a type-dependent
    // receiver -- exercised for real inside a template, once per arm.
    if (!tny_test::dependent_scan(s,   sum_op{})) return 20;   // static  -> _TNY_API arm
    if (!tny_test::dependent_scan(dyn, sum_op{})) return 21;   // dynamic -> _TNY_HOST arm

    return 0;
}
