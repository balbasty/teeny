// Value-form axis args (x.squeeze(Int<1>()) == x.squeeze<1>()) and squeeze()-all.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main() {
    auto t = local<double, shape<2,3,4>>(); t.iota_(0.0, 1.0);

    // permute: template form vs value form must produce the same view type + values
    auto p1 = t.permute<2,0,1>();
    auto p2 = t.permute(Int<2>(), Int<0>(), Int<1>());
    static_assert(cs::is_same<decltype(p1), decltype(p2)>::value, "permute value form == template form");
    if (p1(3,1,2) != p2(3,1,2) || p2(3,1,2) != t(1,2,3)) return 1;

    // flip
    auto f1 = t.flip<2>();  auto f2 = t.flip(Int<2>());
    static_assert(cs::is_same<decltype(f1), decltype(f2)>::value, "flip forms match");
    if (f2(0,0,0) != t(0,0,3)) return 2;

    // unsqueeze / squeeze round-trip via value forms
    auto u = t.unsqueeze(Int<1>());              // (2,1,3,4)
    static_assert(decltype(u)::rank() == 4, "unsqueeze value form");
    auto s = u.squeeze(Int<1>());                // back to (2,3,4)
    static_assert(decltype(s)::rank() == 3, "squeeze value form");
    if (s(1,2,3) != t(1,2,3)) return 3;

    // reshape value form matches the template form for the same (static) args
    auto r1 = t.reshape<6,4>();
    auto r2 = t.reshape(Int<6>(), Int<4>());
    static_assert(cs::is_same<decltype(r1), decltype(r2)>::value, "reshape forms match");
    if (r2(5,3) != t(1,2,3)) return 4;
    // and the -1 inference works through the value form (that extent is dynamic)
    auto r3 = t.reshape(Int<6>(), Int<-1>());
    if (r3(5,3) != t(1,2,3)) return 8;

    // recast value form: deduce the target extents from a shape<> value
    double buf[9]; for (int i=0;i<9;++i) buf[i]=i;
    auto dyn = wrap(buf, shape<-1,-1>{3,3});
    auto st  = dyn.recast(shape<3,3>{});
    static_assert(decltype(st.extent(Int<0>()))::value == 3, "recast value form folds");
    if (st(2,2) != buf[8]) return 5;

    // ---- squeeze() with no arg: drop every statically-size-1 axis ----------
    auto q = local<double, shape<1,3,1,4>>(); q.iota_(0.0, 1.0);
    auto qs = q.squeeze();                        // -> (3,4)
    static_assert(decltype(qs)::rank() == 2, "squeeze() drops all singletons");
    static_assert(decltype(qs.extent(Int<0>()))::value == 3, "kept extent 3");
    static_assert(decltype(qs.extent(Int<1>()))::value == 4, "kept extent 4");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (qs(i,j) != q(0,i,0,j)) return 6;

    // squeeze() on a tensor with no singletons is a no-op (same rank)
    auto ns = t.squeeze();
    static_assert(decltype(ns)::rank() == 3, "no singletons -> unchanged rank");
    if (ns(1,2,3) != t(1,2,3)) return 7;

    // ---- axis<...> value selectors (numpy-like `axis=`): the value form must yield
    //      the SAME type as the template form (so a kernel avoids `.template`). ----
    // peel_at<Axes...>(t, i) vs peel_at(t, i, axis<Axes...>{})
    auto pa1 = peel_at<0,1>(t, 3);
    auto pa2 = peel_at(t, 3, axis<0,1>{});
    static_assert(cs::is_same<decltype(pa1), decltype(pa2)>::value, "peel_at value form == template form");
    if (pa1(2) != pa2(2)) return 9;

    // peel<Axes...>(t) range vs peel(t, axis<Axes...>{})
    auto pr1 = peel<0,1>(t);
    auto pr2 = peel(t, axis<0,1>{});
    static_assert(cs::is_same<decltype(pr1), decltype(pr2)>::value, "peel value form == template form");
    if (pr1.size() != pr2.size() || pr1[5](1) != pr2[5](1)) return 10;

    // single-axis (numpy scalar `axis`): peel<1>(t) vs peel(t, axis<1>{})
    auto ps1 = peel<1>(t);
    auto ps2 = peel(t, axis<1>{});
    static_assert(cs::is_same<decltype(ps1), decltype(ps2)>::value, "peel single-axis value form");
    if (ps1.size() != ps2.size()) return 11;

    // member take_along<Axes...>(args...) vs take_along(axis<Axes...>{}, args...) — the
    // genuine `.template` gap on a dependent receiver; axis<...> also disambiguates it.
    auto ta1 = t.take_along<0,2>(1, 2);
    auto ta2 = t.take_along(axis<0,2>{}, 1, 2);
    static_assert(cs::is_same<decltype(ta1), decltype(ta2)>::value, "take_along value form == template form");
    for (long k = 0; k < (long)ta1.extent(0); ++k) if (ta1(k) != ta2(k) || ta1(k) != t(1,k,2)) return 12;

    // both forms coexist: the explicit-template form still selects correctly
    static_assert(cs::is_same<decltype(t.take_along<1>(2)), decltype(t.take_along(axis<1>{}, 2))>::value,
                  "take_along explicit-template and value forms agree");

    return 0;
}
