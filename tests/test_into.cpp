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

    return 0;
}
