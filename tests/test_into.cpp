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

    // ====== TENSOR-RHS into(dest) SHAPE VALIDATION, at COMPILE TIME (#361) ======
    // The broadcasting producer's own compile-time gate only ever compared the two
    // OPERANDS with each other, so a static, provably-mismatched dest compiled and
    // was left to the debug-only runtime check — which `-DNDEBUG` removes, at which
    // point a dest LARGER than an operand reads that operand past its end (the loop
    // bounds come from the dest, the operand offsets from its own strides).
    //
    // The rule is ASYMMETRIC, and that is the whole subtlety: an OPERAND axis of
    // extent 1 stretches (stride 0), a DEST axis of extent 1 does not. So (1,3) into
    // (2,3) is legal as an operand and illegal as a dest. `bc_static_ok_dest` is the
    // predicate the two static_asserts in the engine ask; exercising it directly is
    // the only compile-time check this runtime suite can actually run — a firing
    // static_assert is a hard error, and there is no compile-fail harness here (same
    // convention as the #357 repros above and the ones commented out below).
    {
        using E22 = shape<2,2>; using E88 = shape<8,8>; using E23 = shape<2,3>;
        using E13 = shape<1,3>; using E3  = shape<3>;   using Edd = shape<-1,-1>;
        constexpr auto ix2 = cs::make_index_sequence<2>{};
        // the issue's own repro: an (8,8) operand into a (2,2) dest -> ill-formed
        static_assert(!_md::bc_static_ok_dest<E88, E22, 2>(ix2),
                      "#361: a static (8,8) operand into a static (2,2) dest must be rejected");
        // ...and the other direction, the one that reads OOB under -DNDEBUG
        static_assert(!_md::bc_static_ok_dest<E22, E88, 2>(ix2),
                      "#361: a dest LARGER than the operand must be rejected too");
        // a size-1 OPERAND axis stretches to the dest -> fine
        static_assert(_md::bc_static_ok_dest<E13, E23, 2>(ix2),
                      "#361: an extent-1 operand axis broadcasts into the dest");
        // ...but a size-1 DEST axis does NOT stretch to the operand
        static_assert(!_md::bc_static_ok_dest<E23, E13, 2>(ix2),
                      "#361: an extent-1 DEST axis is a real extent, not a stretch");
        // a SHORTER operand right-aligns; its missing leading axes are extent 1 -> fine
        static_assert(_md::bc_static_ok_dest<E3, E23, 2>(ix2),
                      "#361: a shorter operand's padded leading axes are extent 1");
        // matching shapes, and anything dynamic on either side, stay for the runtime check
        static_assert(_md::bc_static_ok_dest<E23, E23, 2>(ix2),
                      "#361: matching static shapes are accepted");
        static_assert(_md::bc_static_ok_dest<Edd, E22, 2>(ix2) &&
                      _md::bc_static_ok_dest<E88, Edd, 2>(ix2),
                      "#361: a dynamic extent on either side is unknowable here -> runtime check");
    }

    // What must keep working: every CORRECTLY-shaped tensor-rhs into(dest) call.
    auto bd = local<double, shape<2,3>>(); bd.iota_(1.0, 1.0);      // 1..6
    auto bcol = local<double, shape<2,1>>(); bcol(0,0)=10; bcol(1,0)=20;
    auto out23 = local<double, shape<2,3>>();
    bd.add(bcol, into(out23));                     // (2,1) rhs stretches over (2,3)
    if (out23(0,0)!=11 || out23(1,2)!=26)    return 116;
    auto v3 = local<double, shape<3>>(); v3(0)=100; v3(1)=200; v3(2)=300;
    bd.add(v3, into(out23));                      // SHORTER rhs, right-aligned
    if (out23(0,0)!=101 || out23(1,2)!=306)  return 117;
    // ...and a comparison, which drives the same engine directly (bypassing the
    // wrapper the old operand-vs-operand gate lived in)
    auto cmask = bd > v3;                         // all false: 1..6 < 100..300
    if (cmask(0,0) || cmask(1,2))            return 118;
    // dynamic shapes still go to the runtime check and must pass when they match
    auto ddyn = zeros<double>(shape<-1,-1>{2,3}); ddyn.iota_(1.0, 1.0);
    auto odyn = zeros<double>(shape<-1,-1>{2,3});
    ddyn.add(ddyn, into(odyn));
    if (odyn(0,0)!=2 || odyn(1,2)!=12)       return 119;
    // a MIXED static/dynamic pair: the static axis folds, the dynamic one defers
    auto dmix = zeros<double>(shape<-1,3>{2,3}); dmix.iota_(1.0, 1.0);
    auto omix = zeros<double>(shape<-1,3>{2,3});
    dmix.add(dmix, into(omix));
    if (omix(0,0)!=2 || omix(1,2)!=12)       return 120;

    // The mis-shaped calls are now COMPILE errors, so they cannot live in a running
    // test — same commented-out convention as the #357 repros above; verified by hand:
    //   auto a8 = zeros<double>(shape<8,8>{}); auto y2 = zeros<double>(shape<2,2>{});
    //   a8.add(a8, into(y2));     // static: compile error (was: debug-only _TNY_CHECK)
    //   y2.add(y2, into(a8));     // static: compile error (was: OOB read under -DNDEBUG)
    //   auto r23 = zeros<double>(shape<2,3>{}); auto r13 = zeros<double>(shape<1,3>{});
    //   r23.add(r23, into(r13));  // static: compile error (a dest axis does not stretch)
    //   auto ad = zeros<double>(shape<-1,-1>{8,8}); auto yd = zeros<double>(shape<-1,-1>{2,2});
    //   ad.add(ad, into(yd));     // dynamic: _TNY_CHECK fires (unchanged)
    //   auto a13 = zeros<double>(shape<1,3>{}); auto b23 = zeros<double>(shape<2,3>{});
    //   a13.add_(b23);            // in-place static rhs mismatch: compile error too (was: debug-only _TNY_CHECK)

    // ========= dest's dtype casts the RESULT, not the arithmetic (#379) =========
    // `into(dest)` is the one path where the caller picks the destination's element
    // type, and what that type does is receive the CAST result: the computation
    // itself runs in the operands' own precision, so every call below is numerically
    // identical to `dest.copy_(a.op(b))` — the `into` form only skips the temporary.
    // It used to take the compute type from the DEST, which re-ran the whole
    // computation in that type: the operands, a scalar rhs and an axpy coefficient
    // all truncated BEFORE the op. An `int` dest makes that visible in whole
    // numbers (the buggy value is quoted on each line).
    auto d3 = local<double, shape<3>>(); d3(0)=1.5; d3(1)=2.5; d3(2)=3.5;
    auto o3 = local<double, shape<3>>(); o3.fill_(1.5);
    auto yi = local<int, shape<3>>();

    // scalar rhs: 1.5*0.5=0.75, 2.5*0.5=1.25, 3.5*0.5=1.75 -> {0,1,1}
    yi.zero_(); d3.mul(0.5, into(yi));
    if (yi(0)!=0 || yi(1)!=1 || yi(2)!=1)    return 76;   // was {0,0,0}: 0.5 became int(0.5)==0
    // unary: exp(1.5)=4.4817, exp(2.5)=12.182, exp(3.5)=33.115 -> {4,12,33}
    yi.zero_(); exp(d3, into(yi));
    if (yi(0)!=4 || yi(1)!=12 || yi(2)!=33)  return 77;   // was {2,7,20}: exp of the TRUNCATED input
    // tensor rhs: 1.5+1.5=3, 2.5+1.5=4, 3.5+1.5=5
    yi.zero_(); d3.add(o3, into(yi));
    if (yi(0)!=3 || yi(1)!=4 || yi(2)!=5)    return 78;   // was {2,3,4}: 1+1, 2+1, 3+1
    // fused axpy: 1.5+0.5*1.5=2.25, 2.5+0.75=3.25, 3.5+0.75=4.25 -> {2,3,4}
    yi.zero_(); d3.add(o3, 0.5, into(yi));
    if (yi(0)!=2 || yi(1)!=3 || yi(2)!=4)    return 79;   // was {1,2,3}, i.e. d3 unchanged
                                                          // (alpha became int(0.5)==0)
    // ...and the same for the rest of the family, each against its allocating twin
    // copied into the same dest (the invariant, checked rather than restated).
    auto ri = local<int, shape<3>>();
    yi.zero_(); d3.div(o3, into(yi));       ri.copy_(d3.div(o3));
    if (yi(0)!=ri(0) || yi(1)!=ri(1) || yi(2)!=ri(2)) return 80;   // {1,1,2}
    if (yi(0)!=1 || yi(1)!=1 || yi(2)!=2)   return 81;             // was {1,2,3}
    yi.zero_(); d3.sub(o3, 0.5, into(yi));  ri.copy_(d3.sub(o3, 0.5));
    if (yi(0)!=ri(0) || yi(2)!=ri(2))       return 82;             // 1.5-0.75=0.75 -> 0
    if (yi(0)!=0 || yi(2)!=2)               return 83;             // 3.5-0.75=2.75 -> 2
    yi.zero_(); minimum(d3, 2.2, into(yi)); ri.copy_(minimum(d3, 2.2));
    if (yi(0)!=ri(0) || yi(1)!=ri(1))       return 84;             // {1,2,2}
    if (yi(0)!=1 || yi(1)!=2 || yi(2)!=2)   return 85;
    yi.zero_(); clamp(d3, 2.0, 3.0, into(yi)); ri.copy_(clamp(d3, 2.0, 3.0));
    if (yi(0)!=ri(0) || yi(2)!=ri(2))       return 86;             // {2,2,3}
    if (yi(0)!=2 || yi(1)!=2 || yi(2)!=3)   return 87;
    yi.zero_(); sqrt(o3, into(yi));         ri.copy_(sqrt(o3));
    if (yi(0)!=ri(0))                       return 88;             // √1.5=1.2247 -> 1
    if (yi(0)!=1 || yi(2)!=1)               return 89;             // was 1 too (√1==1)

    // The dest's STRIDES don't change the rule: a non-contiguous int dest takes the
    // per-element decode path instead of the linear fast path, and must agree.
    auto pad = local<int, shape<3,2>>(); pad.zero_();
    auto icol = pad.slice_along<1>(0);
    d3.mul(0.5, into(icol));
    if (pad(0,0)!=0 || pad(1,0)!=1 || pad(2,0)!=1) return 90;
    if (pad(0,1)!=0 || pad(1,1)!=0 || pad(2,1)!=0) return 91;   // gaps untouched

    // Not int-specific — a narrower FLOAT dest formed the value in its own precision
    // too. Double sources, float dest, fused axpy: 1 + (1/3)*7 = 3.3333333333333335
    // in double, whose nearest float is 3.33333325f; formed in float it is
    // 3.33333349f (one ulp up). The reference is built in separate statements so no
    // contraction can fuse it differently from the engine's own expression.
    auto p1 = local<double, shape<1>>(); p1(0) = 1.0;
    auto q1 = local<double, shape<1>>(); q1(0) = 7.0;
    const double alpha3 = 1.0 / 3.0;
    const double prod3  = alpha3 * 7.0;
    const float  want3  = static_cast<float>(1.0 + prod3);
    auto yf1 = local<float, shape<1>>();
    yf1.zero_(); p1.add(q1, alpha3, into(yf1));
    if (yf1(0) != want3)                    return 92;

    // The compute type is the OPERANDS' promoted type — teeny's own `promote_t`, not
    // "the widest type in sight". Two int sources divide as INTS even into a double
    // dest, exactly as the allocating twin does (dest's dtype never promotes the op).
    auto n7 = local<int, shape<2>>(); n7(0)=7; n7(1)=9;
    auto n2 = local<int, shape<2>>(); n2.fill_(2);
    auto rd2 = local<double, shape<2>>(); auto yd2 = local<double, shape<2>>();
    yd2.zero_(); n7.div(n2, into(yd2)); rd2.copy_(n7.div(n2));
    if (yd2(0)!=rd2(0) || yd2(1)!=rd2(1))   return 93;
    if (yd2(0)!=3.0 || yd2(1)!=4.0)         return 94;   // 7/2=3, 9/2=4 (integer division)

    // CONTROL: a dest whose dtype already matches the promoted operand type is
    // unaffected — same values before and after the fix, on all four forms.
    auto yd3 = local<double, shape<3>>();
    yd3.zero_(); d3.mul(0.5, into(yd3));
    if (!close(yd3(0),0.75) || !close(yd3(1),1.25) || !close(yd3(2),1.75)) return 95;
    yd3.zero_(); exp(d3, into(yd3));
    if (!close(yd3(0), std::exp(1.5)) || !close(yd3(2), std::exp(3.5)))   return 96;
    yd3.zero_(); d3.add(o3, into(yd3));
    if (!close(yd3(0),3.0) || !close(yd3(2),5.0))                         return 97;
    yd3.zero_(); d3.add(o3, 0.5, into(yd3));
    if (!close(yd3(0),2.25) || !close(yd3(2),4.25))                       return 98;
    // ...and an int dest fed by int operands and an int scalar: nothing to cast at
    // all, the whole computation was already in `int` both before and after.
    auto ni = local<int, shape<3>>(); ni(0)=1; ni(1)=2; ni(2)=3;
    yi.zero_(); ni.mul(3, into(yi));
    if (yi(0)!=3 || yi(1)!=6 || yi(2)!=9)   return 99;

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

    // ---- half/bfloat16 SOURCE into a wider dest (#379 follow-up) --------
    // The engine computes in `compute_type_t<promote_t<Ta,Tb>>` (float, for a
    // half/bfloat16 source), but the twin `y.copy_(a.op(b))` first materialises
    // its result as a `promote_t<Ta,Tb>` tensor — a 16-bit float — which rounds
    // the float value down to ~11 bits of mantissa BEFORE it is ever widened to
    // `y`'s type. `into(dest)` must round through that same 16-bit stop, not
    // straight from float to `y`'s type, or the two disagree (this exact repro
    // was a real, reviewer-found gap in the original #379 fix): with
    // `half a = 1.3`, `a.add(1.5, into(double_y))` must equal
    // `double_y.copy_(a.add(1.5))` bit for bit, not the "one rounding, more
    // precise" value a straight float->double cast would give.
    auto ha = local<half, shape<1>>(); ha(0) = static_cast<half>(1.3);
    auto hy_into = local<double, shape<1>>(), hy_twin = local<double, shape<1>>();
    ha.add(1.5, into(hy_into));
    hy_twin.copy_(ha.add(1.5));
    if (hy_into(0) != hy_twin(0))            return 100;
    if (hy_into(0) != 2.80078125)            return 101;   // pinned: NOT 2.7998046875

    auto bfa = local<bfloat16, shape<1>>(); bfa(0) = static_cast<bfloat16>(1.3);
    auto by_into = local<double, shape<1>>(), by_twin = local<double, shape<1>>();
    bfa.mul(1.3, into(by_into));
    by_twin.copy_(bfa.mul(1.3));
    if (by_into(0) != by_twin(0))            return 102;
    if (by_into(0) != 1.6875)                return 103;   // pinned: NOT 1.6859374046325684...

    // Same rule for a tensor rhs and for `cross` (which rounds through its own
    // internal `_cross3`, not one of the shared engines).
    auto hb = local<half, shape<1>>(); hb(0) = static_cast<half>(2.7);
    auto hz_into = local<double, shape<1>>(), hz_twin = local<double, shape<1>>();
    ha.add(hb, into(hz_into));
    hz_twin.copy_(ha.add(hb));
    if (hz_into(0) != hz_twin(0))             return 104;

    auto ha3 = local<half, shape<3>>();
    ha3(0) = static_cast<half>(1.3); ha3(1) = static_cast<half>(2.7); ha3(2) = static_cast<half>(3.1);
    auto hb3 = local<half, shape<3>>();
    hb3(0) = static_cast<half>(0.4); hb3(1) = static_cast<half>(1.9); hb3(2) = static_cast<half>(2.2);
    auto hc_into = local<double, shape<3>>(), hc_twin = local<double, shape<3>>();
    cross(ha3, hb3, into(hc_into));
    hc_twin.copy_(cross(ha3, hb3));
    if (hc_into(0) != hc_twin(0) || hc_into(1) != hc_twin(1) || hc_into(2) != hc_twin(2)) return 105;

    // ---- the two 16-bit floats INTO EACH OTHER (#379 follow-up, round 2) ----
    // `half` and `bfloat16` convert only FROM arithmetic types and only TO `float`,
    // so there is no direct `half` -> `bfloat16` conversion. The rounding stop above
    // must therefore hand the engine back a value in the COMPUTE type (`float`) and
    // let the store make the final cast — exactly the two steps the twin's `copy_`
    // takes. Rounding to the 16-bit type and stopping there would make these four
    // calls fail to COMPILE while their twins compile fine, which is how this was
    // first shipped and caught. Least-travelled corner of the whole `into` family:
    // pin both directions, on all three engines plus `cross`.
    auto hh = local<half, shape<3>>();
    hh(0) = static_cast<half>(1.3); hh(1) = static_cast<half>(2.7); hh(2) = static_cast<half>(3.1);
    auto hh2 = local<half, shape<3>>();
    hh2(0) = static_cast<half>(0.4); hh2(1) = static_cast<half>(1.9); hh2(2) = static_cast<half>(2.2);

    auto hb_into = local<bfloat16, shape<3>>(), hb_twin = local<bfloat16, shape<3>>();
    hh.add(hh2, into(hb_into));                 // bzip_ engine
    hb_twin.copy_(hh.add(hh2));
    for (long i = 0; i < 3; ++i) if (!(hb_into(i) == hb_twin(i))) return 106;

    hh.mul(0.5, into(hb_into));                 // scalo_ engine
    hb_twin.copy_(hh.mul(0.5));
    for (long i = 0; i < 3; ++i) if (!(hb_into(i) == hb_twin(i))) return 107;

    exp(hh, into(hb_into));                     // unaryo_ engine
    hb_twin.copy_(exp(hh));
    for (long i = 0; i < 3; ++i) if (!(hb_into(i) == hb_twin(i))) return 108;

    cross(hh, hh2, into(hb_into));              // _cross3
    hb_twin.copy_(cross(hh, hh2));
    for (long i = 0; i < 3; ++i) if (!(hb_into(i) == hb_twin(i))) return 109;

    // ...and the reverse direction: bfloat16 sources into a `half` dest.
    auto bb = local<bfloat16, shape<3>>();
    bb(0) = static_cast<bfloat16>(1.3); bb(1) = static_cast<bfloat16>(2.7); bb(2) = static_cast<bfloat16>(3.1);
    auto bb2 = local<bfloat16, shape<3>>();
    bb2(0) = static_cast<bfloat16>(0.4); bb2(1) = static_cast<bfloat16>(1.9); bb2(2) = static_cast<bfloat16>(2.2);

    auto bh_into = local<half, shape<3>>(), bh_twin = local<half, shape<3>>();
    bb.add(bb2, into(bh_into));
    bh_twin.copy_(bb.add(bb2));
    for (long i = 0; i < 3; ++i) if (!(bh_into(i) == bh_twin(i))) return 110;

    bb.mul(0.5, into(bh_into));
    bh_twin.copy_(bb.mul(0.5));
    for (long i = 0; i < 3; ++i) if (!(bh_into(i) == bh_twin(i))) return 111;

    exp(bb, into(bh_into));
    bh_twin.copy_(exp(bb));
    for (long i = 0; i < 3; ++i) if (!(bh_into(i) == bh_twin(i))) return 112;

    cross(bb, bb2, into(bh_into));
    bh_twin.copy_(cross(bb, bb2));
    for (long i = 0; i < 3; ++i) if (!(bh_into(i) == bh_twin(i))) return 113;

    // A MIXED half/bfloat16 operand pair, into each of the two (promote_t picks one
    // of them, so one direction rounds through the OTHER 16-bit type on its way out).
    auto mx_into = local<bfloat16, shape<3>>(), mx_twin = local<bfloat16, shape<3>>();
    hh.add(bb, into(mx_into));
    mx_twin.copy_(hh.add(bb));
    for (long i = 0; i < 3; ++i) if (!(mx_into(i) == mx_twin(i))) return 114;

    auto mh_into = local<half, shape<3>>(), mh_twin = local<half, shape<3>>();
    bb.add(hh, into(mh_into));
    mh_twin.copy_(bb.add(hh));
    for (long i = 0; i < 3; ++i) if (!(mh_into(i) == mh_twin(i))) return 115;

    return 0;
}
