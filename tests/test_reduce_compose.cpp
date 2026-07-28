// Reduction trailing keyword bag (#284): dtype<Acc>{} x axis<...>{} x keepdims x
// into(dest), any SUBSET and any ORDER, via the generic _kw mechanism (kwargs.h).
// Complements test_axred.cpp (explicit-template Axes + keepdims/into) and
// test_reduce_dtype.cpp (bare dtype<Acc>{}) by exercising the compositions that
// were previously "not yet wired up" (see CLAUDE.md's old cheat-sheet caveat).
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main() {
    // (2,3) matrix, row-major values 0..5
    auto m = local<double, shape<2,3>>(); m.iota_(0.0, 1.0);   // [[0,1,2],[3,4,5]]

    // --- dtype<Acc>{} composed with an axis<...>{} value tag (previously bare-only) ---
    auto c1 = sum(m, axis<0>{}, dtype<float>{});
    static_assert(cs::is_same<typename decltype(c1)::element_type, float>::value, "dtype tag sets Acc");
    if (c1(0)!=3.0f || c1(1)!=5.0f || c1(2)!=7.0f) return 1;
    // order shouldn't matter
    auto c2 = sum(m, dtype<float>{}, axis<0>{});
    static_assert(cs::is_same<decltype(c1), decltype(c2)>::value, "order-independent");
    if (c2(0)!=3.0f || c2(1)!=5.0f || c2(2)!=7.0f) return 2;

    // --- dtype x axis x keepdims, any order ---
    auto k1 = sum(m, dtype<float>{}, axis<0>{}, keepdims);
    static_assert(decltype(k1)::rank() == 2, "keepdims keeps rank");
    static_assert(cs::is_same<typename decltype(k1)::element_type, float>::value, "dtype tag sets Acc (keepdims)");
    if (k1(0,1)!=5.0f) return 3;
    auto k2 = sum(m, keepdims, axis<0>{}, dtype<float>{});
    static_assert(cs::is_same<decltype(k1), decltype(k2)>::value, "keepdims order-independent");
    if (k2(0,1)!=5.0f) return 4;

    // --- dtype x axis x into, any order ---
    auto d1 = local<float, shape<3>>();
    sum(m, dtype<float>{}, axis<0>{}, into(d1));
    if (d1(0)!=3.0f || d1(1)!=5.0f || d1(2)!=7.0f) return 5;
    auto d2 = local<float, shape<3>>();
    auto & rd2 = sum(m, into(d2), dtype<float>{}, axis<0>{});
    static_assert(cs::is_same<decltype(rd2), decltype(d2)&>::value, "into returns dest&");
    if (&rd2 != &d2) return 6;
    if (d2(0)!=3.0f || d2(1)!=5.0f || d2(2)!=7.0f) return 7;

    // --- full composition: dtype x axis x keepdims x into, any order ---
    auto kd1 = local<float, shape<1,3>>();
    sum(m, dtype<float>{}, axis<0>{}, keepdims, into(kd1));
    if (kd1(0,0)!=3.0f || kd1(0,1)!=5.0f || kd1(0,2)!=7.0f) return 8;
    auto kd2 = local<float, shape<1,3>>();
    sum(m, into(kd2), keepdims, dtype<float>{}, axis<0>{});
    if (kd2(0,0)!=3.0f || kd2(0,1)!=5.0f || kd2(0,2)!=7.0f) return 9;

    // --- same composition on other reductions sharing the shared axis-core (prod/max/min/sqnorm) ---
    if (prod(m, axis<1>{}, dtype<float>{})(1) != 60.0f) return 10;
    if (max(m, dtype<float>{}, axis<1>{})(1) != 5.0f) return 11;
    if (min(m, axis<1>{}, dtype<float>{}, keepdims)(1,0) != 3.0f) return 12;
    auto v = local<double, shape<3>>(); v(0)=3; v(1)=0; v(2)=4;
    if (sqnorm(v, dtype<float>{}) != 25.0f) return 13;   // bare dtype (no axis) still works

    // --- mean: int->double rule still holds through the generic path ---
    auto mi = local<int, shape<2,3>>(); mi.iota_(0, 1);
    auto mm = mean(mi, axis<0>{});
    static_assert(cs::is_same<typename decltype(mm)::element_type, double>::value, "integer mean -> double");
    if (mm(0) != 1.5 || mm(1) != 2.5 || mm(2) != 3.5) return 14;
    // explicit dtype overrides the int->double default
    auto mm2 = mean(mi, dtype<float>{}, axis<0>{});
    static_assert(cs::is_same<typename decltype(mm2)::element_type, float>::value, "dtype tag overrides mean's int rule");
    if (mm2(0) != 1.5f) return 15;

    // --- norm: dtype x axis x keepdims x into (layers sqrt + floating result on top) ---
    auto vv = local<double, shape<2,3>>(); vv.iota_(1.0, 1.0);
    auto n1 = norm(vv, axis<1>{}, dtype<float>{}, keepdims);
    static_assert(decltype(n1)::rank() == 2, "norm keepdims keeps rank");
    static_assert(cs::is_same<typename decltype(n1)::element_type, float>::value, "norm dtype tag");
    // row 0: sqrt(1+4+9) = sqrt(14)
    if (!(n1(0,0) > 3.7416f && n1(0,0) < 3.7417f)) return 16;

    // --- dot: dtype x into composed (bare, no axis concept) ---
    auto a3 = local<double, shape<3>>(); a3(0)=1; a3(1)=2; a3(2)=3;
    auto b3 = local<double, shape<3>>(); b3(0)=4; b3(1)=5; b3(2)=6;
    auto cell = local<float, shape<>>();
    dot(a3, b3, dtype<float>{}, into(cell));           // 1*4+2*5+3*6 = 32
    if (cell.item() != 32.0f) return 17;
    auto cell2 = local<float, shape<>>();
    dot(a3, b3, into(cell2), dtype<float>{});
    if (cell2.item() != 32.0f) return 18;

    // --- methods: same composition, via m.sum(...) etc. ---
    auto mc1 = m.sum(dtype<float>{}, axis<0>{});
    static_assert(cs::is_same<decltype(mc1), decltype(c1)>::value, "method matches free function");
    if (mc1(0)!=3.0f || mc1(1)!=5.0f || mc1(2)!=7.0f) return 19;
    auto mkd = local<float, shape<1,3>>();
    m.sum(dtype<float>{}, axis<0>{}, keepdims, into(mkd));
    if (mkd(0,0)!=3.0f || mkd(0,1)!=5.0f || mkd(0,2)!=7.0f) return 20;
    auto mcell = local<float, shape<>>();
    a3.dot(b3, dtype<float>{}, into(mcell));
    if (mcell.item() != 32.0f) return 21;

    // --- dynamic shape: same composition allocates on the heap ---
    auto dm = owned<double, rank<2>>(shape<-1,-1>{2,3}); dm.copy_(m);
    auto dc = sum(dm, dtype<float>{}, axis<0>{});
    static_assert(decltype(dc)::ownership == storage::heap, "dynamic result -> heap");
    if (dc(0)!=3.0f || dc(1)!=5.0f || dc(2)!=7.0f) return 22;
    auto ddest = local<float, shape<3>>();
    sum(dm, dtype<float>{}, axis<0>{}, into(ddest));
    if (ddest(0)!=3.0f || ddest(1)!=5.0f || ddest(2)!=7.0f) return 23;

    return 0;
}
