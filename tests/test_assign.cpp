// Assign-into-a-view semantics: `a(slice) = b` / `a(ellipsis) = b` copy CONTENTS
// (numpy a[:] = b), while `a = b` on a named lvalue view REBINDS (shallow). The
// dangerous case is a SAME-TYPE rhs (view = view): the compiler-generated
// assignment must NOT out-rank the deep-copy for an rvalue `*this` (that made
// `a(slice) = x(slice)` a silent no-op). Views must also stay trivially copyable.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main()
{
    // ---- view <- view, SAME TYPE, must COPY (the regressed case) -----------
    double xb[6], yb[6];
    auto x = wrap(xb, shape<6>{}); x.iota_(10.0);          // 10..15
    auto y = wrap(yb, shape<6>{}); y.zero_();
    y(slice(1, 5)) = x(slice(1, 5));                       // both are the SAME view type
    // rows 1..4 copied, 0 and 5 untouched
    if (y(0) != 0 || y(5) != 0)                     return 1;
    for (long i = 1; i < 5; ++i) if (y(i) != x(i))  return 2;

    // ellipsis / all, same-type rhs
    double zb[6]; auto z = wrap(zb, shape<6>{}); z.zero_();
    z(ellipsis) = x(ellipsis);
    for (long i = 0; i < 6; ++i) if (z(i) != x(i))  return 3;
    double wb[6]; auto w = wrap(wb, shape<6>{}); w.zero_();
    w(all) = x(all);
    for (long i = 0; i < 6; ++i) if (w(i) != x(i))  return 4;

    // 2-D slice, same-type rhs
    double ab[12], bb[12];
    auto a = wrap(ab, shape<3,4>{}); a.iota_(0.0);
    auto b = wrap(bb, shape<3,4>{}); b.zero_();
    b(slice(0,2), all) = a(slice(0,2), all);
    for (long i=0;i<2;++i) for (long j=0;j<4;++j) if (b(i,j) != a(i,j)) return 5;
    if (b(2,0) != 0)                                return 6;              // row 2 untouched

    // rank-0 view <- rank-0 view, same type: copies the ELEMENT
    a.at(2,3) = b.at(0,0);                                 // b(0,0) == a(0,0) == 0
    if (a(2,3) != 0.0)                              return 7;

    // ---- broadcast + owning rhs (different type) still work ----------------
    y(all) = 3.0;                    if (y(2) != 3.0)      return 8;       // scalar fill
    auto oc = x.clone();                                   // owning, different type
    z(ellipsis) = oc;                                      // owning rhs -> deep-copy
    for (long i = 0; i < 6; ++i) if (z(i) != x(i))  return 9;
    double rb[3]; auto row = wrap(rb, shape<3>{});
    row(all) = a(all, 1);            // rhs a column (broadcast/copy) -> still works
    for (long i=0;i<3;++i) if (row(i) != a(i,1))    return 10;

    // ---- lvalue REBIND still shallow (a named view = another view) ---------
    double p[4] = {1,2,3,4}, q[4] = {5,6,7,8};
    auto va = wrap(p, shape<4>{});
    auto vb = wrap(q, shape<4>{});
    va = vb;                                               // lvalue rebind (shallow)
    if (va.data() != q)                             return 11;             // now aliases q
    if (p[0] != 1.0)                                return 12;             // p untouched (no copy)
    va(0) = 99.0; if (q[0] != 99.0)                 return 13;             // writes through to q

    // ---- views remain trivially copyable (kernel-passable) -----------------
    static_assert(cs::is_trivially_copyable<tensor<float, shape<2,3>, ccontiguous, own::view>>::value,
                  "view still trivially copyable");
    static_assert(cs::is_trivially_copyable<decltype(x(slice(1,5)))>::value,
                  "slice view still trivially copyable");

    return 0;
}
