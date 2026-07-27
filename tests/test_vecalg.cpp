// Vector algebra & geometry helpers (#225): sqnorm / norm, normalize_ / normalize,
// fused axpy add_(b,alpha)/sub_(b,alpha), and the 3D cross product.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
#include <cmath>
using namespace tny;
namespace cs = cuda::std;

static bool close(double a, double b) { return std::fabs(a - b) < 1e-9; }
static bool closeh(double a, double b) { return std::fabs(a - b) < 5e-3; }   // half precision

int main() {
    // ---- sqnorm / norm : scalar reductions over all axes -----------------
    auto v = local<double, shape<3>>();  v(0) = 3; v(1) = 0; v(2) = 4;   // |v| = 5
    if (sqnorm(v) != 25.0)         return 1;
    if (!close(norm(v), 5.0))      return 2;

    // result element type: double vector -> double
    static_assert(cs::is_same<decltype(sqnorm(v)), double>::value, "sqnorm(double)->double");
    static_assert(cs::is_same<decltype(norm(v)),   double>::value, "norm(double)->double");

    // 2-D (matrix) sqnorm/norm reduce over ALL axes (Frobenius)
    auto M = local<double, shape<2,2>>(); M(0,0)=1; M(0,1)=2; M(1,0)=2; M(1,1)=4;
    if (sqnorm(M) != 1.0+4.0+4.0+16.0)      return 3;
    if (!close(norm(M), std::sqrt(25.0)))   return 4;

    // ---- integer input: sqnorm casts down (like sum), norm -> double -----
    auto iv = local<int, shape<3>>(); iv(0)=1; iv(1)=2; iv(2)=2;   // sq = 9
    if (sqnorm(iv) != 9)            return 5;
    static_assert(cs::is_same<decltype(sqnorm(iv)), int>::value,    "sqnorm(int)->int");
    static_assert(cs::is_same<decltype(norm(iv)),   double>::value, "norm(int)->double (mean rule)");
    if (!close(norm(iv), 3.0))      return 6;
    // explicit accumulator/result type
    static_assert(cs::is_same<decltype(sqnorm<double>(iv)), double>::value, "sqnorm<double> result");
    if (!close(sqnorm<double>(iv), 9.0))    return 7;

    // ---- norm over a STRIDED (non-contiguous) view -----------------------
    double buf[6] = {3, 99, 0, 99, 4, 99};
    auto s = wrap(buf, shape<3>{}, strides<2>{});   // stride 2: picks 3,0,4
    if (!close(norm(s), 5.0))       return 8;

    // ---- normalize (out-of-place) ---------------------------------------
    auto u = normalize(v);
    if (!close(u(0), 0.6) || !close(u(1), 0.0) || !close(u(2), 0.8)) return 9;
    static_assert(decltype(u)::is_static, "normalize(static) -> stack (static)");
    static_assert(cs::is_same<decltype(u)::element_type, double>::value, "normalize(double)->double");
    if (!close(norm(u), 1.0))       return 10;
    // v itself is untouched (out-of-place)
    if (v(0) != 3 || v(2) != 4)     return 11;
    // integer input normalizes into a double tensor
    auto ui = normalize(iv);
    static_assert(cs::is_same<decltype(ui)::element_type, double>::value, "normalize(int)->double");
    if (!close(norm(ui), 1.0))      return 12;

    // ---- normalize_ (in-place) ------------------------------------------
    auto w = local<double, shape<2>>(); w(0) = 3; w(1) = 4;
    w.normalize_();
    if (!close(w(0), 0.6) || !close(w(1), 0.8)) return 13;

    // ---- fused axpy: add_(b, alpha) / sub_(b, alpha) --------------------
    auto y = local<double, shape<3>>(); y(0)=1; y(1)=1; y(2)=1;
    auto x = local<double, shape<3>>(); x(0)=10; x(1)=20; x(2)=30;
    y.add_(x, 2.0);                          // y += 2*x -> {21,41,61}
    if (y(0)!=21 || y(1)!=41 || y(2)!=61)   return 14;
    y.sub_(x, 1.0);                          // y -= 1*x -> {11,21,31}
    if (y(0)!=11 || y(1)!=21 || y(2)!=31)   return 15;

    // scaled copy y = a*x via zero_().add_(x,a)
    auto z = local<double, shape<3>>();
    z.zero_().add_(x, 0.5);                   // {5,10,15}
    if (z(0)!=5 || z(1)!=10 || z(2)!=15)    return 16;

    // fused axpy broadcasts the rhs (row vector into a matrix)
    auto Y = local<double, shape<2,3>>(); Y.fill_(1.0);
    Y.add_(x, 3.0);                          // each row += 3*x
    if (Y(0,0)!=31 || Y(1,2)!=91)           return 17;

    // ---- cross product ---------------------------------------------------
    auto e1 = local<double, shape<3>>(); e1(0)=1; e1(1)=0; e1(2)=0;
    auto e2 = local<double, shape<3>>(); e2(0)=0; e2(1)=1; e2(2)=0;
    auto e3 = cross(e1, e2);                 // x cross y = z = {0,0,1}
    static_assert(decltype(e3)::rank() == 1, "cross -> rank 1");
    static_assert(decltype(e3)::extents_type::static_extent(0) == 3, "cross -> length 3");
    static_assert(decltype(e3)::is_static, "cross -> static stack result");
    if (!close(e3(0),0) || !close(e3(1),0) || !close(e3(2),1)) return 18;

    // anti-commutativity: b x a = -(a x b)
    auto e3n = cross(e2, e1);
    if (!close(e3n(2), -1.0))       return 19;

    // a general pair, checked against the determinant formula
    auto p = local<double, shape<3>>(); p(0)=1; p(1)=2; p(2)=3;
    auto q = local<double, shape<3>>(); q(0)=4; q(1)=5; q(2)=6;
    auto pq = cross(p, q);                    // {2*6-3*5, 3*4-1*6, 1*5-2*4} = {-3,6,-3}
    if (!close(pq(0),-3) || !close(pq(1),6) || !close(pq(2),-3)) return 20;
    // cross is orthogonal to both operands
    if (!close(dot(pq,p),0.0) || !close(dot(pq,q),0.0)) return 21;

    // crossto_ into an existing (possibly aliasing) output
    auto out = local<double, shape<3>>();
    crossto_(out, p, q);
    if (!close(out(0),-3) || !close(out(2),-3)) return 22;
    // aliasing: out == a is safe (components buffered first)
    auto pa = local<double, shape<3>>(); pa(0)=1; pa(1)=2; pa(2)=3;
    crossto_(pa, pa, q);
    if (!close(pa(0),-3) || !close(pa(1),6) || !close(pa(2),-3)) return 23;

    // ---- half coverage (computes in float) -------------------------------
    auto hv = local<half, shape<3>>(); hv(0)=half(3); hv(1)=half(0); hv(2)=half(4);
    if (!closeh((double)(float)norm(hv), 5.0))  return 24;
    static_assert(cs::is_same<decltype(norm(hv)), half>::value, "norm(half)->half");
    auto hn = normalize(hv);
    static_assert(cs::is_same<typename decltype(hn)::element_type, half>::value, "normalize(half)->half");
    if (!closeh((double)(float)hn(2), 0.8))     return 25;

    // ---- dynamic-shape norm (heap-owned, host) ---------------------------
    auto d = zeros<double>(shape<-1>{3}); d(0)=3; d(1)=0; d(2)=4;
    if (!close(norm(d), 5.0))        return 26;
    auto dn = normalize(d);          // heap result
    if (!close(norm(dn), 1.0))       return 27;

    return 0;
}
