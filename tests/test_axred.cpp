// Axis reductions: sum/prod/max/min/mean<Axes...> -> a lower-rank tensor.
// Static shape -> stack result; dynamic -> heap. Reduced axes are removed.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main() {
    // (2,3) matrix, row-major values 0..5
    auto m = local<double, shape<2,3>>(); m.iota_(0.0, 1.0);   // [[0,1,2],[3,4,5]]

    // sum over axis 0 -> shape (3): column sums [3,5,7]
    auto c = sum<0>(m);
    static_assert(decltype(c)::rank() == 1, "axis 0 removed");
    static_assert(decltype(c)::ownership == storage::stack, "static -> stack");
    static_assert(decltype(c.extent(Int<0>()))::value == 3, "kept extent static");
    if (c(0)!=3.0 || c(1)!=5.0 || c(2)!=7.0) return 1;

    // sum over axis 1 -> shape (2): row sums [3,12]
    auto r = sum<1>(m);
    if (r(0)!=3.0 || r(1)!=12.0) return 2;

    // sum over a negative axis (== last)
    auto rn = sum<-1>(m);
    if (rn(0)!=3.0 || rn(1)!=12.0) return 3;

    // mean over axis 0 -> [1.5, 2.5, 3.5]
    auto mn = mean<0>(m);
    if (mn(0)!=1.5 || mn(1)!=2.5 || mn(2)!=3.5) return 4;

    // max / min over axis 1 -> [2,5] / [0,3]
    auto mx = max<1>(m); if (mx(0)!=2.0 || mx(1)!=5.0) return 5;
    auto mi = min<1>(m); if (mi(0)!=0.0 || mi(1)!=3.0) return 6;

    // prod over axis 1 -> [0*1*2, 3*4*5] = [0,60]
    auto pr = prod<1>(m); if (pr(0)!=0.0 || pr(1)!=60.0) return 7;

    // reduce multiple axes of a 3-D tensor -> scalar-shaped rank-1... here keep one
    auto t = local<double, shape<2,2,2>>(); t.iota_(1.0, 1.0);   // 1..8
    auto s02 = sum<0,2>(t);    // reduce axes 0 and 2 -> shape (2)
    static_assert(decltype(s02)::rank() == 1, "two axes removed");
    // axis1==0: t(:, 0, :) = 1,2,5,6 -> 14 ; axis1==1: 3,4,7,8 -> 22
    if (s02(0)!=14.0 || s02(1)!=22.0) return 8;

    // reducing the DYNAMIC axis leaves a fully-static result -> stack
    auto d = owned<double, shape<-1,3>>(shape<-1,3>{2,3}); d.iota_(0.0,1.0);   // (2,3), 0..5
    auto dc = sum<0>(d);                       // axis0 (dynamic) removed -> shape<3> static
    static_assert(decltype(dc)::ownership == storage::stack, "reduced the dynamic axis -> static");
    if (dc(0)!=3.0 || dc(2)!=7.0) return 9;

    // reducing a STATIC axis and keeping a dynamic one -> heap result
    auto dr = sum<1>(d);                        // axis1 (static 3) removed -> shape<-1> dynamic
    static_assert(decltype(dr)::ownership == storage::heap, "dynamic result -> heap");
    if (dr(0)!=3.0 || dr(1)!=12.0) return 10;

    // ---- axis<...> value form (numpy `axis=`): NAME(a, axis<...>{}) == NAME<...>(a) ----
    // Each value form must be TYPE-identical to its template form, and behave the same.
    static_assert(cs::is_same<decltype(sum(t, axis<0,2>{})), decltype(sum<0,2>(t))>::value, "sum axis value form");
    if (sum(t, axis<0,2>{})(0)!=14.0 || sum(t, axis<0,2>{})(1)!=22.0) return 11;
    // Acc form: leading TYPE arg + deduced axes
    static_assert(cs::is_same<decltype(sum<double>(t, axis<1>{})), decltype(sum<double,1>(t))>::value, "sum<Acc> axis value form");
    // mean / max / min / prod
    static_assert(cs::is_same<decltype(mean(m, axis<0>{})), decltype(mean<0>(m))>::value, "mean axis value form");
    static_assert(cs::is_same<decltype(max(t,  axis<2>{})), decltype(max<2>(t))>::value,  "max axis value form");
    static_assert(cs::is_same<decltype(min(t,  axis<0,1>{})), decltype(min<0,1>(t))>::value, "min axis value form");
    static_assert(cs::is_same<decltype(prod(t, axis<1>{})), decltype(prod<1>(t))>::value, "prod axis value form");
    if (max(t, axis<2>{})(0,0) != max<2>(t)(0,0)) return 12;
    // integer mean -> double, preserved through the value form
    auto im = local<int, shape<2,4>>(); im.iota_(1,1);
    static_assert(cs::is_same<decltype(mean(im, axis<1>{})), decltype(mean<1>(im))>::value, "int mean axis value form");
    // dynamic source: value form matches, stays on the same (heap) path
    static_assert(cs::is_same<decltype(sum(d, axis<0>{})), decltype(sum<0>(d))>::value, "dynamic sum axis value form");
    if (sum(d, axis<0>{})(0) != 3.0 || sum(d, axis<0>{})(2) != 7.0) return 13;

    // ---- keepdims: keep the reduced axes as size-1 (numpy `keepdims=True`) -----
    auto ks = sum<0>(m, keepdims);                    // (2,3) -> (1,3), same values as sum<0>(m)
    static_assert(decltype(ks)::rank() == 2, "keepdims keeps rank");
    static_assert(decltype(ks)::extents_type::static_extent(0) == 1, "reduced axis folds to static 1");
    static_assert(decltype(ks)::extents_type::static_extent(1) == 3, "kept axis extent unchanged");
    if (ks(0,0)!=3.0 || ks(0,1)!=5.0 || ks(0,2)!=7.0) return 14;

    auto ks1 = sum<1>(m, keepdims);                   // (2,3) -> (2,1)
    if (ks1(0,0)!=3.0 || ks1(1,0)!=12.0) return 15;

    // value form: NAME(a, axis<...>{}, keepdims) == NAME<...>(a, keepdims)
    static_assert(cs::is_same<decltype(sum(m, axis<0>{}, keepdims)), decltype(sum<0>(m, keepdims))>::value,
                  "keepdims value form matches template form");
    if (sum(m, axis<0>{}, keepdims)(0,1) != 5.0) return 16;

    // leading TYPE (Acc) + keepdims, both spellings
    static_assert(cs::is_same<decltype(sum<double,0>(m, keepdims)), decltype(sum<double>(m, axis<0>{}, keepdims))>::value,
                  "Acc keepdims value form matches template form");
    if (sum<float,0>(m, keepdims)(0,2) != 7.0f) return 17;

    // every reduction gets it: prod/max/min/mean/sqnorm/norm
    if (prod<0>(m, keepdims)(0,1) != 4.0) return 18;      // 1*4
    if (max<0>(m, keepdims)(0,2)  != 5.0) return 19;
    if (min<1>(m, keepdims)(1,0)  != 3.0) return 20;
    if (mean<1>(m, keepdims)(0,0) != 1.0) return 21;      // (0+1+2)/3
    auto v3 = local<double, shape<3>>(); v3(0)=3; v3(1)=0; v3(2)=4;
    auto v3k = v3.unsqueeze<0>();                          // (1,3): a rank-2 "batch of vectors" case
    if (sqnorm<1>(v3k, keepdims)(0,0) != 25.0) return 22;   // 9+0+16
    if (!(norm<1>(v3k, keepdims)(0,0) == 5.0))  return 23;  // sqrt(25)

    // composes with into(dest) — dest matches the KEPT-DIMS shape
    auto kd = local<double, shape<1,3>>();
    sum<0>(m, keepdims, into(kd));
    if (kd(0,0)!=3.0 || kd(0,2)!=7.0) return 24;
    auto kd2 = local<double, shape<1,3>>();
    sum(m, axis<0>{}, keepdims, into(kd2));
    if (kd2(0,1)!=5.0) return 25;

    // dynamic-shape source stays on the heap path
    auto dk = sum<0>(d, keepdims);                    // (2,3) dynamic -> (1,3): axis0 removed was dynamic
    static_assert(decltype(dk)::rank() == 2, "dynamic keepdims keeps rank");
    if (dk(0,0)!=3.0 || dk(0,2)!=7.0) return 26;

    // ---- mean/norm result-type rules, by VALUE (the type side is asserted above) ----
    // integer mean divides in double and RETURNS double (numpy) -- not truncated:
    // im = [[1,2,3,4],[5,6,7,8]]  ->  mean<1> = [2.5, 6.5], mean<0> = [3,4,5,6]
    static_assert(cs::is_same<decltype(mean<1>(im))::element_type, double>::value, "integer mean -> double");
    if (mean<1>(im)(0) != 2.5 || mean<1>(im)(1) != 6.5) return 27;
    if (mean<0>(im)(0) != 3.0 || mean<0>(im)(3) != 6.0) return 28;
    if (mean<1>(im, keepdims)(0,0) != 2.5) return 29;
    // an explicit integer accumulator opts back into integer division (truncating)
    static_assert(cs::is_same<decltype(mean<int,1>(im))::element_type, int>::value, "mean<Acc> result is Acc");
    if (mean<int,1>(im)(0) != 2) return 30;
    // integer norm is floating too (the same rule), and folds the same as mean
    static_assert(cs::is_same<decltype(norm<1>(im))::element_type, double>::value, "integer norm -> double");
    auto i34 = local<int, shape<1,2>>(); i34(0,0)=3; i34(0,1)=4;
    if (norm<1>(i34)(0) != 5.0) return 31;
    if (sqnorm<1>(i34)(0) != 25) return 32;
    // floating mean/norm keep the element type, accumulating wide
    static_assert(cs::is_same<decltype(mean<1>(m))::element_type, double>::value, "double mean stays double");
    auto fm = local<float, shape<2,2>>(); fm.iota_(1.0f, 1.0f);   // [[1,2],[3,4]]
    static_assert(cs::is_same<decltype(mean<1>(fm))::element_type, float>::value, "float mean stays float");
    if (mean<1>(fm)(0) != 1.5f || mean<1>(fm)(1) != 3.5f) return 33;
    static_assert(cs::is_same<decltype(norm<0>(fm))::element_type, float>::value, "float norm stays float");
    // dynamic (heap) source takes the same branches
    auto di = owned<int, shape<-1,-1>>(shape<-1,-1>{2,3}); di.iota_(1, 1);   // [[1,2,3],[4,5,6]]
    static_assert(cs::is_same<decltype(mean<1>(di))::element_type, double>::value, "dynamic integer mean -> double");
    if (mean<1>(di)(0) != 2.0 || mean<1>(di)(1) != 5.0) return 34;
    if (mean<0>(di)(0) != 2.5) return 35;
    if (norm<0>(di)(0) != cs::sqrt(17.0)) return 36;   // sqrt(1+16)

    // ---- #433: a DUPLICATE axis is a compile error, keepdims or not -------------
    // A repeated axis has no useful meaning (the engine just marks that axis
    // reduced twice), so it is rejected up front. Before #433 only the `keepdims`
    // path checked -- `sum<0,0>(m, keepdims)` failed, but `sum<0,0>(m)` compiled
    // and silently computed `sum<0>(m)`. Both spellings now hit the SAME shared
    // guard in the axis-reduction core ("<name>: axes must be distinct -- each
    // axis may be reduced only once").
    // These lines cannot be exercised by the runtime suite (a static_assert is a
    // hard error and there is no compile-fail harness here, same as test_into.cpp
    // / test_to.cpp); enabling any of them is a compile error:
    //   sum<0,0>(m);                 // plain form -- the #433 regression
    //   sum<0,0>(m, keepdims);       // keepdims form (already rejected pre-#433)
    //   sum(m, axis<0,0>{});         // the axis<...> value-tag entry point
    //   m.sum<0,0>();                // ...and the method forwarder
    //   sum<double,0,0>(m);          // the <Acc, Axes...> form
    //   mean<1,-2>(t);               // a duplicate spelled two ways (-2 == 1 at rank 3)
    //   prod<0,0>(t); max<0,0>(t); min<0,0>(t); sqnorm<0,0>(t); norm<0,0>(t);
    //   sum<0,0>(d);                 // the dynamic (heap) overload too
    //   normalize<0,0>(t);           // and the keepdim fold's own direct caller
    // What must KEEP compiling is a distinct list in any order (#371) -- pinned
    // here by value so the guard can never over-reject:
    static_assert(cs::is_same<decltype(sum<2,0>(t)), decltype(sum<0,2>(t))>::value, "distinct, descending: still fine");
    if (!allclose(sum<2,0>(t), sum<0,2>(t)))   return 37;
    if (!allclose(sum<-1,0>(t), sum<0,2>(t)))  return 38;   // mixed sign, distinct
    if (!allclose(sum(t, axis<2,0>{}), sum<0,2>(t))) return 39;
    auto d3 = zeros<double>(shape<-1,-1,-1>{2,3,4}); d3.iota_(1.0);   // dynamic source: heap overload
    if (!allclose(sum<2,0>(d3), sum<0,2>(d3))) return 40;

    return 0;
}
