// Comparison operators -> bool tensors (broadcast) + all()/any() reductions +
// compile-time slice<...>() forms.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main() {
    auto a = local<double, shape<4>>(); a(0)=1; a(1)=2; a(2)=3; a(3)=4;
    auto b = local<double, shape<4>>(); b(0)=4; b(1)=2; b(2)=1; b(3)=4;

    // tensor cmp tensor -> bool tensor
    auto lt = a < b;
    static_assert(cs::is_same<decltype(lt)::element_type, bool>::value, "compare -> bool");
    if (lt(0)!=true || lt(1)!=false || lt(2)!=false || lt(3)!=false) return 1;
    auto eq = a == b; if (eq(1)!=true || eq(3)!=true || eq(0)!=false) return 2;

    // tensor cmp scalar, and scalar cmp tensor (reversed)
    auto ge = a >= 3.0;  if (ge(0)||ge(1)|| !ge(2) || !ge(3)) return 3;
    auto sl = 3.0 < a;   // == a > 3.0 -> only a(3)=4
    if (sl(2) || !sl(3)) return 4;

    // all() / any() (members; chain after a comparison)
    if (!(a > 0.0).all()) return 5;
    if ( (a > 3.0).all()) return 6;
    if (!(a > 3.0).any()) return 7;
    if ( (a > 9.0).any()) return 8;
    // (note: sum() preserves dtype, so sum(bool_mask) is a saturating bool, not
    // a count — use .all()/.any(), or cast the mask, to reduce a comparison.)

    // broadcast: (C,1) vs (1,W)
    auto col = local<int, shape<3,1>>(); col(0,0)=0; col(1,0)=1; col(2,0)=2;
    auto row = local<int, shape<1,3>>(); row(0,0)=0; row(0,1)=1; row(0,2)=2;
    auto mask = col == row;             // (3,3) identity-ish
    static_assert(decltype(mask)::rank()==2, "broadcast compare");
    if (!mask(0,0) || !mask(1,1) || mask(0,1)) return 10;

    // ---- compile-time slice forms ------------------------------------
    auto t = local<double, shape<2,8>>(); t.iota_(0.0, 1.0);
    auto s1 = t(0, slice(1,5));           // runtime form
    auto s2 = t(0, slice<1,5>());         // value template form
    auto s3 = t(0, slice<Int<1>, Int<5>>());  // type template form
    if (s1(0)!=s2(0) || s2(0)!=s3(0) || s3(0)!=t(0,1)) return 11;
    if (s2(3)!=t(0,4)) return 12;
    // static step folds
    auto ss = t(0, slice<0,8,2>());
    static_assert(decltype(ss.stride(Int<0>()))::value == 2, "static step folds to stride 2");
    if (ss(1)!=t(0,2)) return 13;

    // ---- allclose ----------------------------------------------------
    auto u = local<double, shape<3>>(); u(0)=1; u(1)=2; u(2)=3;
    auto w = local<double, shape<3>>(); w(0)=1; w(1)=2; w(2)=3 + 1e-9;   // within tol
    if (!allclose(u, w)) return 14;
    auto far = local<double, shape<3>>(); far(0)=1; far(1)=2; far(2)=3.1;
    if (allclose(u, far)) return 15;
    if (!allclose(u, far, /*rtol*/0.1)) return 16;                       // loose tol -> close
    // broadcasts a scalar-shaped operand
    auto ones3 = local<double, shape<3>>(); ones3.fill_(1.0);
    auto one1  = local<double, shape<1>>(); one1(0)=1.0;
    if (!allclose(ones3, one1)) return 17;

    return 0;
}
