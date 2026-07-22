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
    auto dyn = view(buf, shape<-1,-1>{3,3});
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

    return 0;
}
