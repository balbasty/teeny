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
    auto pt3 = local<double, shape<3>>(); pt3(0)=1; pt3(1)=2; pt3(2)=3.1;
    if (allclose(u, pt3)) return 15;
    if (!allclose(u, pt3, /*rtol*/0.1)) return 16;                       // loose tol -> close
    // broadcasts a scalar-shaped operand
    auto ones3 = local<double, shape<3>>(); ones3.fill_(1.0);
    auto one1  = local<double, shape<1>>(); one1(0)=1.0;
    if (!allclose(ones3, one1)) return 17;

    // ---- allclose: method form + dtype/into composition (#350) --------
    // the method mirrors the free function one-for-one, at every tolerance arity
    if (!u.allclose(w))                return 18;
    if ( u.allclose(pt3))              return 19;
    if (!u.allclose(pt3, /*rtol*/0.1)) return 20;
    if (!u.allclose(pt3, 0.0, 0.2))    return 21;   // rtol 0, atol loose enough
    if ( u.allclose(pt3, 0.0, 0.05))   return 22;

    // dtype<Acc>{} / <Acc> pick the COMPARISON's compute type. 1e10 and 1e10+1 are
    // one double apart but the SAME float, so a tolerance below 1 splits the two.
    auto big  = local<double, shape<1>>(); big(0)  = 1e10;
    auto big1 = local<double, shape<1>>(); big1(0) = 1e10 + 1.0;
    if ( allclose(big, big1, 0.0, 0.5))                    return 23;   // double: |diff| = 1 > 0.5
    if (!allclose(big, big1, 0.0, 0.5, dtype<float>{}))    return 24;   // float: both round equal
    if (!allclose<float>(big, big1, 0.0, 0.5))             return 25;   // == the <Acc> template form
    if (!big.allclose(big1, 0.0, 0.5, dtype<float>{}))     return 26;   // ... as a method
    if (!big.allclose<float>(big1, 0.0, 0.5))              return 27;
    static_assert(cs::is_same<decltype(allclose(u, w, dtype<float>{})), bool>::value,
                  "allclose(dtype) still answers bool");

    // into(dest): the answer lands in a rank-0 cell, no allocation, dest& returned
    auto cell = local<bool, shape<>>{};
    auto & ret = allclose(u, w, into(cell));
    if (!cell.item())                  return 28;
    if (&ret != &cell)                 return 29;
    allclose(u, pt3, into(cell));
    if (cell.item())                   return 30;
    allclose(u, pt3, /*rtol*/0.1, into(cell));           // one tolerance + a keyword
    if (!cell.item())                  return 31;
    allclose(u, pt3, 0.0, 0.05, into(cell));             // both tolerances + a keyword
    if (cell.item())                   return 32;
    // a non-bool cell takes the cast (numpy's 0/1)
    auto icell = local<int, shape<>>{};
    allclose(u, w, into(icell));
    if (icell.item() != 1)             return 33;
    // dtype and into compose, in EITHER order, at any tolerance arity
    allclose(big, big1, 0.0, 0.5, dtype<float>{}, into(cell));
    if (!cell.item())                  return 34;
    allclose(big, big1, 0.0, 0.5, into(cell), dtype<double>{});
    if (cell.item())                   return 35;
    allclose(u, w, into(cell), dtype<float>{});
    if (!cell.item())                  return 36;
    // ... and the same through the method
    cell.fill_(false);
    u.allclose(w, into(cell));
    if (!cell.item())                  return 37;
    big.allclose(big1, 0.0, 0.5, dtype<float>{}, into(cell));
    if (!cell.item())                  return 38;
    big.allclose(big1, 0.1, into(icell), dtype<double>{});   // 1 <= 0.1*1e10 -> close
    if (icell.item() != 1)             return 39;

    return 0;
}
