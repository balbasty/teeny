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

    // ---- axis-into for the remaining reductions + dot<Acc> into --------
    prod<0>(m, into(col)); if (col(0)!=4  || col(2)!=18) return 59;   // [1·4, 2·5, 3·6]
    max<0>(m,  into(col)); if (col(0)!=4  || col(2)!=6)  return 60;
    min<0>(m,  into(col)); if (col(0)!=1  || col(2)!=3)  return 61;
    sqnorm<1>(m, into(rowv));
    if (!close(rowv(0), 14.0) || !close(rowv(1), 77.0))  return 62;   // 1+4+9 / 16+25+36
    norm<1>(m, into(rowv));
    if (!close(rowv(0), std::sqrt(14.0)) || !close(rowv(1), std::sqrt(77.0))) return 63;
    dot<double>(vg, vg, into(s0)); if (!close(s0.item(), 25.0)) return 64;  // dot<Acc> into

    // ============ into(dest) SHAPE VALIDATION (#357) ====================
    // A scalar-rhs or unary producer takes its loop bounds from the SOURCE and
    // writes through the DEST's strides, so the two shapes must agree exactly —
    // there is nothing to broadcast on either side. A mis-shaped dest used to be
    // written past its end; it is now rejected (see the two commented-out repros
    // at the end of this block). What must keep working is every CORRECTLY-shaped
    // call, on the dynamic path as much as the static one.

    // ---- scalar-rhs / unary into a matching DYNAMIC dest ----------------
    auto dsrc = zeros<double>(shape<-1,-1>{2,3}); dsrc.iota_(1.0, 1.0);   // 1..6
    auto ddst = zeros<double>(shape<-1,-1>{2,3});
    dsrc.mul(2.0, into(ddst));                        // scalar rhs, dynamic shapes
    if (ddst(0,0)!=2 || ddst(1,2)!=12)       return 65;
    neg(dsrc, into(ddst));                            // unary, dynamic shapes
    if (ddst(0,0)!=-1 || ddst(1,2)!=-6)      return 66;
    if (dsrc(0,0)!=1 || dsrc(1,2)!=6)        return 67;   // source untouched

    // ---- dest with a NARROWER index type than the source ----------------
    // (the extent check compares across index types; #353 made the offsets
    //  decode in a type that covers both)
    auto own32 = zeros<double>(shape<-1,-1>{2,3});         // owner outlives the view
    auto n32 = own32.reindex<int32_t>();
    dsrc.mul(3.0, into(n32)); if (n32(0,0)!=3 || n32(1,2)!=18) return 68;
    sqrt(dsrc, into(n32));    if (!close(n32(1,2), std::sqrt(6.0))) return 69;

    // ---- a size-1 axis is a real extent here, NOT a stretch -------------
    // (1,3) source -> (1,3) dest is fine; the contrast is that (1,3) into (2,3)
    // is a mismatch for these producers, while the TENSOR-rhs producer below
    // still broadcasts it — the two rules are deliberately different.
    auto row = local<double, shape<1,3>>(); row(0,0)=1; row(0,1)=2; row(0,2)=3;
    auto rowout = local<double, shape<1,3>>();
    row.mul(10.0, into(rowout)); if (rowout(0,2)!=30) return 70;
    exp(row, into(rowout)); if (!close(rowout(0,0), std::exp(1.0))) return 71;

    // ---- the BROADCASTING (tensor-rhs) producer is unchanged ------------
    auto B23 = local<double, shape<2,3>>(); B23.fill_(1.0);
    auto O23 = local<double, shape<2,3>>();
    B23.add(row, into(O23));                          // (1,3) rhs stretches over (2,3)
    if (O23(0,0)!=2 || O23(1,2)!=4)          return 72;
    a.add(b, into(y)); if (y(0)!=11 || y(2)!=33) return 73;   // and the plain same-shape case

    // ---- unary/scalar into a dest of a DIFFERENT dtype ------------------
    auto fi2 = local<float, shape<3>>();
    a.mul(2.0, into(fi2));  if (!close(fi2(0), 2.0) || !close(fi2(2), 6.0)) return 74;
    neg(a, into(fi2));      if (!close(fi2(2), -3.0))                       return 75;

    // A MIS-SHAPED dest is now rejected: a compile error when both shapes are
    // fully static (the issue's own repro), a debug-time check otherwise. Left
    // commented out because neither a static_assert nor an assert() failure can
    // be exercised from the runtime suite (no compile-fail harness exists here —
    // same convention as test_math.cpp's dot note); both were verified manually:
    //   auto a8 = zeros<double>(shape<8,8>{}); auto y8 = zeros<double>(shape<2,2>{});
    //   a8.mul(2.0, into(y8));   // static: compile error
    //   exp(a8, into(y8));       // static: compile error
    //   auto ad = zeros<double>(shape<-1,-1>{8,8}); auto y2 = zeros<double>(shape<-1,-1>{2,2});
    //   ad.mul(2.0, into(y2));   // dynamic: _TNY_CHECK fires (was a 64-element write into a 4-element buffer)
    //   exp(ad, into(y2));       // dynamic: same

    // ========== into() on a TEMPORARY VIEW destination (#380) ============
    // Every view-producing op returns its view BY VALUE, so a slot of a bigger
    // output is a temporary — `into()` takes it directly, no named intermediate.
    // The write goes through to the storage the temporary view aliases.
    static_assert(cs::is_same<decltype(into(local<double,shape<2,3>>{}(0, all))),
                             into_t<view<double, shape<3>, strides<1>>>>::value,
                  "into() binds a temporary VIEW (rvalue), same into_t as an lvalue");

    // ---- the documented "crossto": cross straight into row i of a matrix ---
    auto Nrows = local<double, shape<2,3>>();
    cross(p, q, into(Nrows(0, all)));                 // p x q = {-3,6,-3}
    if (!close(Nrows(0,0),-3) || !close(Nrows(0,1),6) || !close(Nrows(0,2),-3)) return 76;
    if (Nrows(1,0)!=0 || Nrows(1,2)!=0)      return 77;   // the other row untouched
    p.cross(q, into(Nrows(1, all)));                  // method form, same slot spelling
    if (!close(Nrows(1,1), 6.0))             return 78;

    // ---- a temporary COLUMN view (a strided slot, stride 2) ----------------
    auto Ncols = local<double, shape<3,2>>();
    cross(q, p, into(Ncols(all, 1)));                 // q x p = {3,-6,3} down column 1
    if (Ncols(0,1)!=3 || Ncols(1,1)!=-6 || Ncols(2,1)!=3) return 79;
    if (Ncols(0,0)!=0 || Ncols(2,0)!=0)      return 92;   // column 0 untouched

    // ---- elementwise / unary / scalar into a temporary view ----------------
    auto Mt = local<double, shape<2,3>>(); Mt.zero_();
    a.add(b, into(Mt(0, all)));              if (Mt(0,0)!=11 || Mt(0,2)!=33) return 80;
    a.mul(2.0, into(Mt(1, all)));            if (Mt(1,0)!=2  || Mt(1,2)!=6)  return 81;
    exp(local<double,shape<3>>{}.zero_(), into(Mt(1, all)));
    if (!close(Mt(1,0), 1.0) || !close(Mt(1,2), 1.0)) return 82;
    // a temporary PERMUTED view is a destination too (same storage, transposed)
    auto Tt = local<double, shape<3,2>>(); Tt.zero_();
    a.add(b, into(Tt.permute<1,0>()(0, all)));
    if (Tt(0,0)!=11 || Tt(1,0)!=22 || Tt(2,0)!=33) return 83;

    // ---- reductions into a temporary rank-0 CELL and a temporary slot ------
    auto cells = local<double, shape<2,2>>(); cells.zero_();
    sum(a, into(cells.at(0,0)));             if (!close(cells(0,0), 6.0))  return 84;
    dot(a, b, into(cells.at(0,1)));          if (!close(cells(0,1), 140.0)) return 85;
    norm(vg, into(cells.at(1,0)));           if (!close(cells(1,0), 5.0))  return 86;
    sum<double>(a, into(cells.at(1,1)));     if (!close(cells(1,1), 6.0))  return 87;
    // axis reduction -> a temporary lower-rank slot of a bigger buffer
    auto rows = local<double, shape<2,3>>(); rows.zero_();
    sum<0>(m, into(rows(0, all)));            // m is (2,3) -> length-3
    if (rows(0,0)!=5 || rows(0,2)!=9)        return 88;
    mean(m, axis<0>{}, into(rows(1, all)));   // value form + a temporary dest
    if (!close(rows(1,0), 2.5) || !close(rows(1,2), 4.5)) return 89;

    // ---- gather / scan into a temporary view -------------------------------
    auto isrc = local<double, shape<4>>(); isrc.iota_(1.0, 1.0);   // 1,2,3,4
    auto iidx = local<int, shape<3>>(); iidx(0)=3; iidx(1)=0; iidx(2)=2;
    auto gdst = local<double, shape<2,3>>(); gdst.zero_();
    isrc.index_select<0>(iidx, into(gdst(1, all)));
    if (gdst(1,0)!=4 || gdst(1,1)!=1 || gdst(1,2)!=3) return 90;
    if (gdst(0,0)!=0)                        return 91;   // the other row untouched

    // A temporary OWNING destination stays a compile error (its storage dies with
    // the expression, so the result would be discarded) — same commented-out
    // convention as the shape checks above; verified manually:
    //   sum(a, into(local<double, shape<>>{}));         // static_assert: temporary must be a VIEW
    //   a.add(b, into(zeros<double>(shape<-1>{3})));    // same, for a heap owner

    return 0;
}
