// Slicing folds static strides into teeny's strides<...> layout (item 1), and
// every view op works on a strides<...> source without submdspan (item 2).
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main() {
    // ---- item 1: a slice of a static contiguous tensor keeps FOLDED strides --
    auto t = local<double, shape<2,3,4>>(); t.iota_(0.0, 1.0);
    auto s = t(1, all, all);                       // rank-2 view over the (3,4) plane
    // the kept strides are compile-time constants (integral_constant), not runtime
    static_assert(_is_ic<decltype(s.stride(Int<0>()))>::value, "outer stride should fold");
    static_assert(_is_ic<decltype(s.stride(Int<1>()))>::value, "inner stride should fold");
    static_assert(decltype(s.stride(Int<0>()))::value == 4, "row stride 4");
    static_assert(decltype(s.stride(Int<1>()))::value == 1, "col stride 1");
    static_assert(decltype(s.extent(Int<0>()))::value == 3, "extent kept static");
    if (s(2,3) != t(1,2,3)) return 1;

    // a static-step range folds too: stride = source_stride * step
    auto r = t(all, slice(0, 3, Int<2>()), all);   // axis1 step 2 -> stride 4*2 = 8
    static_assert(decltype(r.stride(Int<1>()))::value == 8, "range stride folds to 8");
    static_assert(decltype(r.stride(Int<0>()))::value == 12, "outer stride 12");
    if (r(1,0,0) != t(1,0,0) || r(1,1,0) != t(1,2,0)) return 2;

    // ---- item 2: a strides<...> source supports every view op ----------------
    double buf[24]; for (int i=0;i<24;++i) buf[i] = i;
    auto st = tensor<double, shape<2,3,4>, strides<12,4,1>>(buf);   // fully static strides
    if (st(1,2,3) != buf[12+8+3]) return 3;

    auto sv = st(1, all, all);                     // slice a strides source (was submdspan)
    if (sv(2,3) != st(1,2,3)) return 4;
    auto ta = st.slice_along<2>(1);                 // drop axis 2 at index 1
    if (ta(0,0) != st(0,0,1) || ta(1,2) != st(1,2,1)) return 5;
    auto rv = st(all, slice(1,3), all);            // range on a strides source
    if (rv(0,0,0) != st(0,1,0) || rv(1,1,3) != st(1,2,3)) return 6;

    auto pv = st.permute<2,1,0>();                 // permute a strides source
    if (pv(3,2,1) != st(1,2,3)) return 7;
    auto fv = st.flip<2>();                        // flip a strides source
    if (fv(0,0,0) != st(0,0,3)) return 8;
    auto uq = st.unsqueeze<1>();                    // (2,1,3,4)
    if (uq(1,0,2,3) != st(1,2,3)) return 9;

    // ---- a strides<...> source also feeds peel (batch iteration) -------------
    double acc = 0;
    for (auto plane : peel<0>(st)) acc += plane(2,3);   // sum st(0,2,3)+st(1,2,3)
    if (acc != st(0,2,3) + st(1,2,3)) return 10;

    // ---- slice bounds clamp python-style (no out-of-bounds views) ------------
    auto row = local<double, shape<5>>(); row.iota_(0.0, 1.0);     // 0..4
    auto over = row(slice(1, 100));                                 // clamps to [1,5)
    if (over.extent(0) != 4 || over(0) != 1.0 || over(3) != 4.0) return 11;
    auto rev = row(slice(100, none, -1));                           // clamps start to 4
    if (rev.extent(0) != 5 || rev(0) != 4.0 || rev(4) != 0.0) return 12;
    auto revstop = row(slice(100, 1, -1));                          // 4,3,2
    if (revstop.extent(0) != 3 || revstop(0) != 4.0 || revstop(2) != 2.0) return 13;
    auto emptyish = row(slice(3, 1));                               // start>stop -> empty
    if (emptyish.extent(0) != 0) return 14;

    // ---- recast: re-type extents, validate, fold inner dims (contiguous src) --
    auto vv = wrap(buf, shape<-1,3,4>{2});     // (2,3,4) contiguous, dynamic outer
    auto rc = vv.recast<shape<2,3,4>>();        // all static now
    static_assert(decltype(rc.stride(Int<1>()))::value == 4, "recast folds inner stride");
    // keep_strides PRESERVES the source layout TYPE (a contiguous source stays
    // ccontiguous; the strides derive/fold in the accessor — no strides<> synthesis).
    static_assert(cs::is_same<decltype(rc)::layout_type, ccontiguous>::value, "recast keeps ccontiguous layout");
    if (rc(1,2,3) != buf[12+8+3]) return 15;

    // ---- recast PRESERVES strides for a non-contiguous source (#116) ----------
    // (used to silently mis-address: it forced row-major, ignoring the real strides.)
    auto tp  = wrap(buf, shape<2,3>{}).permute<1,0>();   // 3x2, strides (1,3) — NOT row-major
    auto tpr = tp.recast<shape<3,2>>();
    static_assert(cs::is_same<decltype(tpr)::layout_type, decltype(tp)::layout_type>::value, "recast keeps the strides<> layout");
    static_assert(decltype(tpr.stride(Int<0>()))::value == 1, "recast preserves+folds transposed stride 0");
    static_assert(decltype(tpr.stride(Int<1>()))::value == 3, "recast preserves+folds transposed stride 1");
    for (long i = 0; i < 3; ++i) for (long j = 0; j < 2; ++j)
        if (tpr(i, j) != tp(i, j)) return 34;            // values match the source, not a row-major walk

    // ---- recast<Shape, Layout>: explicit layout override -----------------------
    // A runtime-strided but contiguous view: default keeps runtime strides; the
    // explicit ccontiguous form folds them to immediates (the "I promise" form).
    double eb[18]; for (int i = 0; i < 18; ++i) eb[i] = i;
    auto ds = wrap(eb, shape<-1,-1,-1>{2,3,3}, {9,3,1});             // dynamic_strides (contiguous values)
    auto ccf = ds.recast<shape<-1,3,3>, ccontiguous>();             // fold inner strides
    static_assert(decltype(ccf.stride(Int<2>()))::value == 1, "ccontiguous unit stride folds");
    static_assert(decltype(ccf.stride(Int<1>()))::value == 3, "ccontiguous inner stride folds");
    static_assert(decltype(ccf.stride(Int<0>()))::value == 9, "ccontiguous outer stride folds");
    if (ccf(1,2,2) != eb[9+6+2]) return 35;
    auto ccv = ds.recast(shape<-1,3,3>{}, ccontiguous{});           // functional form
    static_assert(cs::is_same<decltype(ccv), decltype(ccf)>::value, "functional recast == type form");
    auto imp = ds.recast<shape<2,3,3>, strides<9,3,1>>();           // impose exact static strides
    static_assert(cs::is_same<decltype(imp)::layout_type, strides<9,3,1>>::value, "imposed strides<>");
    if (imp(1,2,2) != eb[9+6+2]) return 36;

    // ---- is_dense (any/exact order) vs is_contiguous (C-order default) -------
    auto cc = wrap(buf, shape<2,3,4>{});               // C-contiguous
    // is_dense(): dense block in SOME axis order; is_dense<L>(): exactly that layout.
    if (!cc.is_dense()) return 16;                     // dense
    if (!cc.is_dense<layout_right>()) return 17;       // exactly C
    if (cc.is_dense<layout_left>())  return 18;        // not F
    auto perm = cc.permute<2,0,1>();                   // permuted: still dense in memory
    if (!perm.is_dense()) return 19;                   // order-agnostic -> true
    if (perm.is_dense<layout_right>()) return 20;      // but not exact C
    auto ff = wrap<layout_left>(buf, shape<2,3,4>{});  // F-contiguous
    if (!ff.is_dense()) return 21;                     // dense
    if (!ff.is_dense<layout_left>())  return 22;       // exactly F
    if (ff.is_dense<layout_right>())  return 23;       // not C
    auto gap = cc(all, slice(0,2), all);               // a hole along axis 1
    if (gap.is_dense()) return 24;                     // not dense
    auto flp = cc.flip<0>();                           // negative stride
    if (flp.is_dense()) return 25;                     // flips are not dense here
    auto sq  = wrap(buf, shape<3,1,4>{});              // size-1 axis is ignored
    if (!sq.is_dense()) return 26;
    if (!cc.is_dense(layout_right{})) return 27;       // value form == is_dense<layout_right>()

    // is_contiguous(): C-order by DEFAULT (numpy/pytorch), <fcontiguous> for F —
    // a thin alias of is_dense<Layout>(). The redefault: a permuted/F view that is
    // DENSE is NOT is_contiguous() (it isn't C-order).
    if (!cc.is_contiguous()) return 28;                // C source IS C-contiguous
    if (cc.is_contiguous<fcontiguous>()) return 29;    // ...but not F
    if (perm.is_contiguous()) return 30;               // a PERMUTED (dense) view is NOT C-contiguous
    if (ff.is_contiguous()) return 31;                 // F-order (dense) is NOT C-contiguous
    if (!ff.is_contiguous<fcontiguous>()) return 37;   // ...but it IS F-contiguous
    if (!cc.is_contiguous(layout_right{})) return 38;  // value form
    if (cc.is_contiguous(layout_left{}))   return 39;

    // rank-0 (an at() result): one element -> dense and contiguous in every layout.
    auto r0 = cc.at(0,0,0);                             // cc is rank-3 -> bind all axes
    static_assert(decltype(r0)::rank() == 0, "at() -> rank-0");
    if (!r0.is_dense())                    return 32;
    if (!r0.is_contiguous())               return 40;
    if (!r0.is_contiguous<layout_left>())  return 41;
    // and a rank-0 result flows through clone() (it gates on is_contiguous())
    auto r0c = r0.clone();
    if (r0c.item() != cc(0,0,0))           return 33;

    return 0;
}
