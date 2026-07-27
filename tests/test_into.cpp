// into(dest): optional output destination on the out-of-place math producers,
// plus fused out-of-place add/sub (a + alpha*b). (#228)
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
#include <cmath>
using namespace tny;
namespace cs = cuda::std;

static bool close(double a, double b) { return std::fabs(a - b) < 1e-9; }

int main() {
    auto a = local<double, shape<3>>(); a(0)=1; a(1)=2; a(2)=3;
    auto b = local<double, shape<3>>(); b(0)=10; b(1)=20; b(2)=30;

    // ---- elementwise into(dest) : add/sub/mul/div ----------------------
    auto y = local<double, shape<3>>();
    auto & r = a.add(b, into(y));                    // y = a + b
    if (y(0)!=11 || y(1)!=22 || y(2)!=33)   return 1;
    if (&r != &y)                            return 2;   // returns the dest by reference
    a.sub(b, into(y)); if (y(0)!=-9 || y(2)!=-27)  return 3;
    a.mul(b, into(y)); if (y(1)!=40)               return 4;
    b.div(a, into(y)); if (y(0)!=10 || y(2)!=10)   return 5;
    a.pow(local<double,shape<3>>{}.fill_(2.0), into(y)); if (y(1)!=4 || y(2)!=9) return 6;   // a^2

    // operands are untouched (out is a distinct buffer, one pass)
    if (a(0)!=1 || b(2)!=30)                 return 7;

    // ---- scalar rhs into(dest) -----------------------------------------
    a.add(100.0, into(y)); if (y(0)!=101 || y(2)!=103) return 8;
    a.mul(0.5, into(y));   if (y(1)!=1.0)              return 9;

    // ---- fused out-of-place axpy: a + alpha*b (new) and into ------------
    auto f = a.add(b, 2.0);                          // new = a + 2b -> {21,42,63}
    if (f(0)!=21 || f(1)!=42 || f(2)!=63)   return 10;
    static_assert(decltype(f)::is_static, "fused add -> stack (static)");
    a.sub(b, 0.5, into(y));                          // y = a - 0.5b -> {-4,-8,-12}
    if (y(0)!=-4 || y(2)!=-12)               return 11;
    a.add(b, 2.0, into(y));                          // y = a + 2b
    if (y(0)!=21 || y(2)!=63)                return 12;

    // ---- unary into(dest) ----------------------------------------------
    auto e = local<double, shape<3>>(); e(0)=0; e(1)=1; e(2)=2;
    exp(e, into(y));  if (!close(y(1), std::exp(1.0))) return 13;
    neg(a, into(y));  if (y(0)!=-1 || y(2)!=-3)        return 14;
    auto sq = local<double, shape<3>>(); sq(0)=4; sq(1)=9; sq(2)=16;
    sqrt(sq, into(y)); if (y(0)!=2 || y(2)!=4)         return 15;

    // ---- minimum/maximum/clamp into(dest) ------------------------------
    minimum(a, b, into(y)); if (y(0)!=1 || y(2)!=3)   return 16;   // min(a,b)=a
    maximum(a, b, into(y)); if (y(0)!=10 || y(2)!=30) return 17;   // max=b
    minimum(b, 15.0, into(y)); if (y(0)!=10 || y(2)!=15) return 18; // scalar rhs
    clamp(b, 12.0, 25.0, into(y)); if (y(0)!=12 || y(1)!=20 || y(2)!=25) return 19;

    // ---- normalize into(dest) ------------------------------------------
    auto v = local<double, shape<3>>(); v(0)=3; v(1)=0; v(2)=4;
    normalize(v, into(y));
    if (!close(y(0),0.6) || !close(y(2),0.8)) return 20;
    if (v(0)!=3)                              return 21;   // source untouched

    // ---- cross into(dest) = ff's crossto -------------------------------
    auto p = local<double, shape<3>>(); p(0)=1; p(1)=2; p(2)=3;
    auto q = local<double, shape<3>>(); q(0)=4; q(1)=5; q(2)=6;
    auto n = local<double, shape<3>>();
    cross(p, q, into(n));                             // {-3,6,-3}
    if (!close(n(0),-3) || !close(n(1),6) || !close(n(2),-3)) return 22;
    // aliasing: out == an operand is safe (components buffered first)
    auto pa = local<double, shape<3>>(); pa(0)=1; pa(1)=2; pa(2)=3;
    cross(pa, q, into(pa));
    if (!close(pa(0),-3) || !close(pa(2),-3)) return 23;

    // ---- broadcast into a (2,3) destination ----------------------------
    auto M = local<double, shape<2,3>>(); M.fill_(1.0);
    auto Y = local<double, shape<2,3>>();
    M.add(a, into(Y));                               // row vector a broadcasts
    if (Y(0,0)!=2 || Y(1,2)!=4)              return 24;

    // ---- into a STRIDED (non-contiguous) destination view --------------
    double buf[6] = {0,0,0,0,0,0};
    auto strided = wrap(buf, shape<3>{}, strides<2>{});   // writes buf[0],buf[2],buf[4]
    a.add(b, into(strided));
    if (buf[0]!=11 || buf[2]!=22 || buf[4]!=33) return 25;
    if (buf[1]!=0 || buf[3]!=0)              return 26;    // gaps untouched

    // ---- into a different dtype (result casts to dest element type) -----
    auto fi = local<float, shape<3>>();
    a.add(b, into(fi));                              // double result -> float dest
    if (!close(fi(0),11.0) || !close(fi(2),33.0)) return 27;

    // ---- out-of-place ops AS METHODS (parity with a.add(b)) ------------
    auto sq2 = local<double, shape<3>>(); sq2(0)=4; sq2(1)=9; sq2(2)=16;
    if (!close(sq2.sqrt()(0), 2.0) || !close(sq2.sqrt()(2), 4.0)) return 28;   // a.sqrt()
    if (a.neg()(0) != -1)                    return 29;                        // a.neg()
    if (!close(e.exp()(1), std::exp(1.0)))   return 30;                        // a.exp()
    sq2.sqrt(into(y)); if (y(0)!=2 || y(2)!=4) return 31;                      // a.sqrt(into(y))
    if (a.minimum(b)(2) != 3)                return 32;                        // a.minimum(tensor)
    if (a.maximum(b)(0) != 10)               return 33;
    if (b.minimum(15.0)(2) != 15)            return 34;                        // a.minimum(scalar)
    if (b.clamp(12.0, 25.0)(1) != 20)        return 35;                        // a.clamp
    b.maximum(a, into(y)); if (y(0)!=10)     return 36;                        // a.maximum(t, into)
    auto vv = local<double, shape<3>>(); vv(0)=3; vv(1)=0; vv(2)=4;
    if (!close(vv.normalize()(2), 0.8))      return 37;                        // a.normalize()
    auto uu = local<double, shape<3>>(); vv.normalize(into(uu));
    if (!close(uu(0), 0.6))                  return 38;                        // a.normalize(into)
    auto pp = local<double, shape<3>>(); pp(0)=1; pp(1)=2; pp(2)=3;
    auto qq = local<double, shape<3>>(); qq(0)=4; qq(1)=5; qq(2)=6;
    if (!close(pp.cross(qq)(0), -3.0))       return 39;                        // a.cross(b)
    auto nn = local<double, shape<3>>(); pp.cross(qq, into(nn));
    if (!close(nn(2), -3.0))                 return 40;                        // a.cross(b, into)

    // ================= into(dest) on REDUCTIONS (#233) =================
    // ---- FULL reduction -> a rank-0 destination ------------------------
    auto s0 = local<double, shape<>>();              // rank-0 scalar cell
    auto & rs = sum(a, into(s0));
    if (!close(s0.item(), 6.0))              return 41;   // 1+2+3
    if (&rs != &s0)                          return 42;   // returns dest by ref
    prod(a, into(s0)); if (!close(s0.item(), 6.0))  return 43;   // 1*2*3
    max(b, into(s0));  if (!close(s0.item(), 30.0)) return 44;
    min(b, into(s0));  if (!close(s0.item(), 10.0)) return 45;
    mean(a, into(s0)); if (!close(s0.item(), 2.0))  return 46;

    // ---- FULL reduction into a bare ADDRESS via a rank-0 view ----------
    double scal = 0;
    auto scell = wrap(&scal, shape<>{});             // rank-0 view over an address
    sum(b, into(scell));
    if (!close(scal, 60.0))                  return 47;

    // ---- norm / sqnorm / dot into rank-0 -------------------------------
    auto vg = local<double, shape<3>>(); vg(0)=3; vg(1)=0; vg(2)=4;
    sqnorm(vg, into(s0)); if (!close(s0.item(), 25.0)) return 48;   // 9+0+16
    norm(vg,   into(s0)); if (!close(s0.item(),  5.0)) return 49;   // √25
    dot(a, b, into(s0)); if (!close(s0.item(), 140.0)) return 50;  // 10+40+90

    // ---- dtype cast into rank-0 (double result -> int dest) ------------
    auto si = local<int, shape<>>();
    sum(a, into(si)); if (si.item() != 6)    return 51;

    // ---- leading TYPE = accumulator+result on a full reduction ---------
    auto sf = local<float, shape<>>();
    sum<float>(a, into(sf)); if (!close(sf.item(), 6.0)) return 52;

    // ---- AXIS reduction -> a lower-rank destination --------------------
    auto m = local<double, shape<2,3>>();
    m(0,0)=1; m(0,1)=2; m(0,2)=3; m(1,0)=4; m(1,1)=5; m(1,2)=6;
    auto col = local<double, shape<3>>();
    sum<0>(m, into(col));                            // reduce axis 0 -> length-3
    if (col(0)!=5 || col(1)!=7 || col(2)!=9) return 53;
    auto rowv = local<double, shape<2>>();
    sum<1>(m, into(rowv));                           // reduce axis 1 -> length-2
    if (rowv(0)!=6 || rowv(1)!=15)           return 54;

    // ---- axis reduction: value form + leading TYPE + mean --------------
    col.zero_(); sum(m, axis<0>{}, into(col));       // value-form into
    if (col(0)!=5 || col(2)!=9)              return 55;
    auto colf = local<float, shape<3>>();
    sum<float, 0>(m, into(colf));                    // leading-type acc + axis
    if (!close(colf(1), 7.0))                return 56;
    mean<1>(m, into(rowv));
    if (!close(rowv(0), 2.0) || !close(rowv(1), 5.0)) return 57;

    // ---- axis reduction into a DYNAMIC-shape destination (host path) ---
    auto md = zeros<double>(shape<-1,3>{2}); md.copy_(m);
    auto rowd = zeros<double>(shape<-1>{2});
    sum<1>(md, into(rowd)); if (rowd(0)!=6 || rowd(1)!=15) return 58;

    return 0;
}
