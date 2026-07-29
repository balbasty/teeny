// Vector algebra & geometry helpers (#225): sqnorm / norm, normalize_ / normalize,
// fused axpy add_(b,alpha)/sub_(b,alpha), and the 3D cross product.
//
// Checks 58-64 cover mixed index WIDTHS across dot/sqdist's two operands (#342);
// checks 65+ cover mixed SIGNEDNESS on the same engine (#355) — a flipped
// (negative-stride) signed-indexed operand next to an unsigned-indexed one.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
#include <cmath>
using namespace tny;
namespace cs = cuda::std;

// A buffer big enough for a stride that overflows a 16-bit index (#342, below);
// at file scope so the 320 KB never lands on the stack.
static double wide_buf[40002];

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

    // ---- sqdist / dist : fused point-to-point distance (#324) ------------
    auto pa1 = local<double, shape<3>>(); pa1(0)=1; pa1(1)=2; pa1(2)=3;
    auto pb1 = local<double, shape<3>>(); pb1(0)=4; pb1(1)=6; pb1(2)=8;   // deltas 3,4,5
    if (sqdist(pa1, pb1) != 9.0+16.0+25.0)          return 38;
    if (!close(dist(pa1, pb1), std::sqrt(50.0)))    return 39;
    // matches the un-fused sqnorm(a-b)/norm(a-b) spelling exactly
    if (sqdist(pa1, pb1) != sqnorm(pa1 - pb1))       return 40;
    if (!close(dist(pa1, pb1), norm(pa1 - pb1)))     return 41;
    // symmetry: sqdist(a,b) == sqdist(b,a)
    if (sqdist(pa1, pb1) != sqdist(pb1, pa1))        return 42;
    // method forms
    if (pa1.sqdist(pb1) != sqdist(pa1, pb1))         return 43;
    if (!close(pa1.dist(pb1), dist(pa1, pb1)))        return 44;
    // integer input: sqdist stays int (sum rule), dist -> double (mean rule)
    auto ia1 = local<int, shape<2>>(); ia1(0)=1; ia1(1)=2;
    auto ib1 = local<int, shape<2>>(); ib1(0)=4; ib1(1)=6;   // deltas 3,4 -> sq=25
    if (sqdist(ia1, ib1) != 25)                      return 45;
    static_assert(cs::is_same<decltype(sqdist(ia1, ib1)), int>::value,    "sqdist(int)->int");
    static_assert(cs::is_same<decltype(dist(ia1, ib1)),   double>::value, "dist(int)->double (mean rule)");
    if (!close(dist(ia1, ib1), 5.0))                 return 46;
    // explicit accumulator: sqdist<Acc>/dist<Acc> makes Acc BOTH accumulator and result
    static_assert(cs::is_same<decltype(sqdist<double>(ia1, ib1)), double>::value, "sqdist<double> result");
    if (!close(sqdist<double>(ia1, ib1), 25.0))      return 47;
    // dtype<T>{} value-tag form == the <Acc> template form
    if (sqdist(ia1, ib1, dtype<double>{}) != sqdist<double>(ia1, ib1)) return 48;
    // into(dest): full reduction writes its scalar into a rank-0 dest
    auto cell = local<double, shape<>>{};
    sqdist(pa1, pb1, into(cell));
    if (cell.item() != 50.0)                         return 49;
    dist(pa1, pb1, into(cell));
    if (!close(cell.item(), std::sqrt(50.0)))        return 50;

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

    // write into a preallocated slot with no temporary: slot.copy_(cross(a,b))
    auto out = local<double, shape<3>>();
    out.copy_(cross(p, q));
    if (!close(out(0),-3) || !close(out(2),-3)) return 22;
    // in-place member: a becomes a × b (aliasing-safe — components buffered first)
    auto pa = local<double, shape<3>>(); pa(0)=1; pa(1)=2; pa(2)=3;
    pa.cross_(q);
    if (!close(pa(0),-3) || !close(pa(1),6) || !close(pa(2),-3)) return 23;
    // cross_ chains and returns *this
    auto r = local<double, shape<3>>(); r(0)=1; r(1)=0; r(2)=0;
    r.cross_(e2);                             // x × y = z
    if (!close(r(2), 1.0))          return 24;

    // ---- half coverage (computes in float) -------------------------------
    auto hv = local<half, shape<3>>(); hv(0)=half(3); hv(1)=half(0); hv(2)=half(4);
    if (!closeh((double)(float)norm(hv), 5.0))  return 25;
    static_assert(cs::is_same<decltype(norm(hv)), half>::value, "norm(half)->half");
    auto hn = normalize(hv);
    static_assert(cs::is_same<typename decltype(hn)::element_type, half>::value, "normalize(half)->half");
    if (!closeh((double)(float)hn(2), 0.8))     return 26;

    // ---- dynamic-shape norm (heap-owned, host) ---------------------------
    auto d = zeros<double>(shape<-1>{3}); d(0)=3; d(1)=0; d(2)=4;
    if (!close(norm(d), 5.0))        return 27;
    auto dn = normalize(d);          // heap result
    if (!close(norm(dn), 1.0))       return 28;

    // ---- axis sqnorm / norm (reduction API: <Axes...>, axis<...>, <Acc>) ----
    auto M2 = local<double, shape<2,3>>();
    M2(0,0)=3; M2(0,1)=0; M2(0,2)=4;   // row0: sq=25, |.|=5
    M2(1,0)=0; M2(1,1)=6; M2(1,2)=8;   // row1: sq=100, |.|=10
    auto sqr = sqnorm<1>(M2);          // over axis 1 -> (2,)
    static_assert(decltype(sqr)::rank() == 1, "axis sqnorm -> lower rank");
    if (sqr(0) != 25 || sqr(1) != 100) return 29;
    auto nr = norm<1>(M2);
    if (!close(nr(0), 5.0) || !close(nr(1), 10.0)) return 30;
    if (!close(norm(M2, axis<1>{})(1), 10.0))      return 31;   // value form
    if (sqnorm<double,0>(M2)(1) != 36.0)           return 32;   // over axis 0, <Acc>
    // integer axis norm -> double (mean rule)
    auto Mi = local<int, shape<2,2>>(); Mi(0,0)=3; Mi(0,1)=4; Mi(1,0)=6; Mi(1,1)=8;
    auto nri = norm<1>(Mi);
    static_assert(cs::is_same<typename decltype(nri)::element_type, double>::value, "int axis norm -> double");
    if (!close(nri(0), 5.0) || !close(nri(1), 10.0)) return 33;

    // ---- axis normalize / normalize_ (keepdim broadcast) -----------------
    auto un = normalize<1>(M2);        // unit rows
    if (!close(un(0,0),0.6) || !close(un(0,2),0.8) || !close(un(1,1),0.6)) return 34;
    if (!close(norm<1>(un)(0), 1.0) || !close(norm<1>(un)(1), 1.0))        return 35;
    auto Mn = M2; Mn.normalize_<1>();  // in place
    if (!close(Mn(0,2),0.8) || !close(Mn(1,2),0.8)) return 36;
    // normalize along axis 0 (columns)
    auto uc = normalize<0>(M2);
    if (!close(norm<0>(uc)(0), 1.0))   return 37;

    // ---- dot/sqdist STATIC fast path (#255): both static + C-contiguous
    // operands unroll (no per-step decode); a non-contiguous static operand
    // must still fall back to the general decode and agree exactly ----------
    auto sa = local<double, shape<4>>(); sa(0)=1; sa(1)=2; sa(2)=3; sa(3)=4;
    auto sb = local<double, shape<4>>(); sb(0)=5; sb(1)=6; sb(2)=7; sb(3)=8;
    if (dot(sa, sb) != 1*5+2*6+3*7+4*8)             return 51;   // both contiguous -> fast path
    // one operand STRIDED (not ccontiguous) -> falls back to the decode path,
    // must still match the same value computed by hand
    double sbuf[8] = {1,99,2,99,3,99,4,99};
    auto sas = wrap(sbuf, shape<4>{}, strides<2>{});   // picks 1,2,3,4 at stride 2
    if (dot(sas, sb) != 1*5+2*6+3*7+4*8)             return 52;
    if (sqdist(sas, sb) != (1-5)*(1-5)+(2-6)*(2-6)+(3-7)*(3-7)+(4-8)*(4-8)) return 53;
    // both operands non-contiguous (a permuted 2x2 view) -> also decode path
    auto MA = local<double, shape<2,2>>(); MA(0,0)=1; MA(0,1)=2; MA(1,0)=3; MA(1,1)=4;
    auto MB = local<double, shape<2,2>>(); MB(0,0)=5; MB(0,1)=6; MB(1,0)=7; MB(1,1)=8;
    auto MAt = MA.permute<1,0>();   // {{1,3},{2,4}}
    auto MBt = MB.permute<1,0>();   // {{5,7},{6,8}}
    if (dot(MAt, MBt) != 1*5+3*7+2*6+4*8)           return 54;   // same as dot(MA,MB) (dot is order-free)
    if (dot(MAt, MBt) != dot(MA, MB))                return 55;
    // rank-0 (scalar) operands: unrolls to exactly one step
    auto r0a = local<double, shape<>>{}; r0a() = 6.0;
    auto r0b = local<double, shape<>>{}; r0b() = 7.0;
    if (dot(r0a, r0b) != 42.0)                       return 56;
    // rank-3 static-contiguous: exercises the unroll over more than one axis
    auto ca = local<double, shape<2,3,2>>();
    auto cb = local<double, shape<2,3,2>>();
    double expect = 0.0;
    for (long i=0;i<2;++i) for (long j=0;j<3;++j) for (long k=0;k<2;++k) {
        double av = i*6.0+j*2.0+k+1, bv = 12.0-(i*6.0+j*2.0+k);
        ca(i,j,k) = av; cb(i,j,k) = bv; expect += av*bv;
    }
    if (dot(ca, cb) != expect)                       return 57;

    // ---- dot/sqdist across MIXED index widths (#342) ----------------------
    // The two operands may carry different offset index types (an int32-indexed
    // `shape32`/`reindex` view dotted with an int64-indexed one). The general
    // decode engine runs its offset math in the WIDER of the two — the same rule
    // a mixed-width broadcast follows (#167) — so neither operand's extents or
    // strides are narrowed into the other's width.
    double mbuf[8]  = {1,2,3,4,5,6,7,8};
    double mbuf2[8] = {2,4,6,8,10,12,14,16};
    auto m64 = wrap(mbuf,  shape<2,4>{});
    auto n64 = wrap(mbuf2, shape<2,4>{});
    auto m32 = m64.reindex<cs::int32_t>();                 // int32-indexed twin
    // both static + C-contiguous -> the #255 unroll (no index math at all)
    if (dot(m32, n64) != 2.0+8.0+18.0+32.0+50.0+72.0+98.0+128.0)   return 58;
    // sliced -> NOT ccontiguous -> the general decode path, where the widths meet
    auto m32s = m32(all, slice(0,2));                      // (2,2), stride (4,1)
    auto n64s = n64(all, slice(0,2));
    if (dot(m32s, n64s) != 1*2.0+2*4.0+5*10.0+6*12.0)      return 59;
    if (sqdist(m32s, n64s) != 1.0+4.0+25.0+36.0)           return 60;
    if (dot(n64s, m32s) != dot(m32s, n64s))                return 61;   // order-independent
    // dynamic shape (also decode): int32-indexed vs int64-indexed
    auto d32 = wrap(mbuf,  shape_as<cs::int32_t,-1,-1>{2,4});
    auto d64 = wrap(mbuf2, shape<-1,-1>{2,4});
    if (dot(d32, d64) != 2.0+8.0+18.0+32.0+50.0+72.0+98.0+128.0)   return 62;

    // ...and the WIDE operand's strides survive: a stride that does not fit the
    // NARROW operand's index type must not be truncated. Shown with an int16 x
    // int64 pair, which needs only a 40k-element buffer — the int32 x int64
    // analogue would need a >2^31-element one to exhibit the same truncation.
    for (long i = 0; i < 40002; ++i) wide_buf[i] = 0.0;
    wide_buf[0] = 1; wide_buf[1] = 2; wide_buf[40000] = 3; wide_buf[40001] = 4;   // b's four elements
    double small4[4] = {10, 20, 30, 40};
    auto a16 = wrap(small4, shape_as<cs::int16_t,2,2>{});          // int16-indexed, contiguous
    auto bw  = wrap(wide_buf, shape<2,2>{}, {40000, 1});               // int64-indexed, stride > int16
    if (dot(a16, bw) != 10*1.0 + 20*2.0 + 30*3.0 + 40*4.0)         return 63;
    if (sqdist(a16, bw) != 81.0+324.0+729.0+1296.0)                return 64;

    // ---- dot/sqdist/dist across mixed SIGNEDNESS (#355) -------------------
    // Width alone does not settle the engine's offset type. When the WIDER operand
    // is UNSIGNED and the narrower one is SIGNED with a NEGATIVE stride (any
    // flipped/reversed view), casting that -1 stride into the unsigned type makes it
    // 4294967295, which zero-extends into the pointer offset instead of stepping
    // backwards — `dot(flipped_int16_view, uint32_view)` read far off the front of
    // the buffer and crashed. So a MIXED-signedness pair decodes in a signed type
    // wide enough for both sides; all-signed and all-unsigned pairs keep the plain
    // width pick (checks 71-72 below pin that they did not move).
    // Every expectation is hand-computed, so a "no longer crashes but reads the
    // wrong element" fix fails these just as loudly as the crash did.
    double sv[4] = {1, 2, 3, 4};
    double bvv[4] = {10, 20, 30, 40};

    // (65) the reported repro: flipped int16-indexed operand x uint32-indexed one.
    {
        auto fa = wrap(sv + 3, shape_as<cs::int16_t,4>{}, strides<-1>{});   // 4,3,2,1
        auto ub = wrap(bvv, shape_as<unsigned int,4>{});                    // 10,20,30,40
        static_assert(cs::is_same<typename decltype(fa)::index_type, cs::int16_t>::value, "flipped operand is int16-indexed");
        static_assert(cs::is_same<typename decltype(ub)::index_type, unsigned int>::value, "other operand is uint32-indexed");
        if (dot(fa, ub) != 4*10.0 + 3*20.0 + 2*30.0 + 1*40.0)          return 65;   // 200
        if (dot(ub, fa) != 200.0)                                      return 66;   // the mirror order
        if (sqdist(fa, ub) != 36.0 + 289.0 + 784.0 + 1521.0)           return 67;   // 2630
        if (sqdist(ub, fa) != 2630.0)                                  return 68;
        if (!close(dist(fa, ub), std::sqrt(2630.0)))                   return 69;
    }

    // (70) EQUAL width, disagreeing signedness (int32 vs uint32) — the case a
    //      width-only pick cannot express at all: it must step up to a signed 64-bit
    //      decode. Note the old tie rule kept the FIRST operand's type, so only the
    //      unsigned-first order actually crashed; both are pinned here.
    {
        auto f32 = wrap(sv + 3, shape32<4>{}, strides<-1>{});               // 4,3,2,1
        auto u32 = wrap(bvv, shape_as<unsigned int,4>{});
        if (dot(f32, u32) != 200.0)                                    return 70;
        if (dot(u32, f32) != 200.0)                                    return 71;
    }

    // (72) rank-2, flipped along axis 0, against an unsigned-indexed operand.
    {
        double pv[4] = {1, 2, 3, 4}, qv[4] = {10, 20, 30, 40};
        auto fd = wrap(pv + 2, shape_as<cs::int16_t,2,2>{}, strides<-2,1>{});  // (3,4),(1,2)
        auto uq = wrap(qv, shape_as<unsigned int,2,2>{});
        if (dot(fd, uq) != 3*10.0 + 4*20.0 + 1*30.0 + 2*40.0)          return 72;   // 220
        if (dot(uq, fd) != 220.0)                                      return 73;
    }

    // (74) controls that must NOT move — the two same-signedness families, which
    //      keep the identical width pick they had before. An all-UNSIGNED pair
    //      cannot carry a negative stride at all (teeny's flipped views require a
    //      signed index type), and an all-SIGNED flipped pair was never at risk.
    {
        auto us = wrap(sv,  shape_as<unsigned short,4>{});
        auto ui = wrap(bvv, shape_as<unsigned int,4>{});
        if (dot(us, ui) != 1*10.0 + 2*20.0 + 3*30.0 + 4*40.0)          return 74;   // 300
        if (dot(ui, us) != 300.0)                                      return 75;
        auto fs  = wrap(sv + 3, shape<4>{}, strides<-1>{});             // int64, 4,3,2,1
        auto p32 = wrap(bvv, shape32<4>{});                             // int32-indexed
        if (dot(fs, p32) != 200.0)                                     return 76;
        if (dot(p32, fs) != 200.0)                                     return 77;
        if (sqdist(fs, p32) != 2630.0)                                 return 78;
    }

    return 0;
}
