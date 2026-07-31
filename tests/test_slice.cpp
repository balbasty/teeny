#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    double buf[24];
    for (long i=0;i<24;++i) buf[i]=i;
    auto t = wrap(buf, shape<2,3,4>{});        // strides (12,4,1)

    // all-integer -> element access (T&)
    if (t(1,2,3) != 1*12+2*4+3) return 1;
    static_assert(cs::is_same<decltype(t(0,0,0)), double&>(), "int... -> element");

    // negative indices wrap python-style (only meaningful when the build wraps at
    // all — -DTNY_NO_NEGATIVE_INDEX drops the wrap, so a negative RUNTIME index is
    // undefined behaviour there; the static/`none` folds below are unaffected)
#ifndef TNY_NO_NEGATIVE_INDEX
    if (t(-1,-1,-1) != t(1,2,3)) return 2;
    if (t(-2,0,0)  != t(0,0,0)) return 3;
#endif

    // static index (Int<>) as an element index
    if (t(Int<1>(), 2, 3) != t(1,2,3)) return 4;

    // slice: any `all` / `slice` argument -> a sub-view (tensor)
    auto row = t(1, all, all);                        // fix axis 0 -> (3,4) view
    static_assert(decltype(row)::rank() == 2, "one integer + all,all -> rank 2");
    if (row(2,3) != t(1,2,3)) return 5;

    auto col = t(all, 2, all);                        // fix axis 1 -> (2,4) view
    static_assert(decltype(col)::rank() == 2, "middle fixed");
    if (col(1,3) != t(1,2,3)) return 6;

    // slice: half-open range keeps the axis
    auto sub = t(1, slice(1,3), all);                   // axis0 fixed, axis1 [1,3) -> (2,4)
    static_assert(decltype(sub)::rank() == 2, "slice keeps axis");
    if (sub(0,0) != t(1,1,0)) return 7;               // first row of the range
    if (sub(1,3) != t(1,2,3)) return 8;

    // peel are mutable views (write-through)
    row(0,0) = 999.0;
    if (t(1,0,0) != 999.0) return 9;
    row(0,0) = 0.0;                                   // restore

    // ---- python-like slice: none / negative / step --------------------
    // `none` open ends: slice(none, k) starts at 0; slice(k, none) runs to end.
    auto a = t(0, slice(none, 2), all);               // axis1 [0,2)
    if (a.extent(0) != 2 || a(0,0) != t(0,0,0) || a(1,3) != t(0,1,3)) return 10;
    auto b = t(0, slice(1, none), all);               // axis1 [1,3)
    if (b.extent(0) != 2 || b(0,0) != t(0,1,0)) return 11;
    // slice(none,none) folds to full_extent (== all): keeps the axis AND its
    // static extent.
    auto c = t(0, slice(none,none), all);
    static_assert(decltype(c)::extents_type::static_extent(0) == 3, "slice(none,none)==all folds");
    if (c.extent(0) != 3 || c(1,2) != t(0,1,2)) return 12;

    // negative bounds wrap (count from the back) — a RUNTIME bound, so likewise
    // only when the build wraps (-DTNY_NO_NEGATIVE_INDEX takes -2 literally)
#ifndef TNY_NO_NEGATIVE_INDEX
    auto d = t(0, slice(-2, none), all);              // last two of axis1 -> [1,3)
    if (d.extent(0) != 2 || d(0,0) != t(0,1,0) || d(1,0) != t(0,2,0)) return 13;
#endif

    // step: every other element along the last axis (0,2) of 4 -> length 2
    auto e = t(0, 0, slice(0, 4, Int<2>()));
    if (e.extent(0) != 2 || e(0) != t(0,0,0) || e(1) != t(0,0,2)) return 14;
    auto f = t(0, 0, slice(none, none, Int<2>()));    // whole axis, stride 2
    if (f.extent(0) != 2 || f(1) != t(0,0,2)) return 15;

    // #46: a COMPILE-TIME range folds its output extent to a static value (was
    // dynamic). Source static + static start/stop/step -> length known now.
    auto cs0 = t(0, slice<1,3>(), 0);                 // axis1 [1,3) -> static 2
    static_assert(decltype(cs0)::extents_type::static_extent(0) == 2, "slice<1,3> folds extent");
    if (cs0.extent(0) != 2 || cs0(0) != t(0,1,0) || cs0(1) != t(0,2,0)) return 30;
    auto cs1 = t(0, slice<0,4,2>(), 0);               // step 2 -> static 2
    static_assert(decltype(cs1)::extents_type::static_extent(0) == 2, "slice<0,4,2> folds extent");
    if (cs1.extent(0) != 2) return 31;
    // reversed compile-time slice folds too (matches the runtime length exactly)
    auto csr = t(0, 0, slice<none_t, none_t, cs::integral_constant<cs::int64_t,-1>>());
    static_assert(decltype(csr)::extents_type::static_extent(0) == 4, "reversed folds to 4");
    if (csr.extent(0) != 4 || csr(0) != t(0,0,3) || csr(3) != t(0,0,0)) return 32;
    // a runtime range stays dynamic (only the compile-time form folds)
    auto csd = t(0, slice(1,3), 0);
    static_assert(decltype(csd)::extents_type::static_extent(0) == cs::dynamic_extent, "runtime range stays dynamic");

    // #46 safety: on an UNSIGNED index_type a negative step is not foldable — the
    // runtime casts step to unsigned (forward branch, empty) while a signed fold
    // would reverse. The fold must fall back to dynamic there (else static!=runtime
    // -> UB); the runtime value then fills it. (Fable-review edge case.)
    float ub[8] = {}; auto ut = wrap(ub, cs::extents<unsigned,8>{});
    auto uneg = ut(slice<none_t, none_t, cs::integral_constant<cs::int64_t,-1>>());
    static_assert(decltype(uneg)::extents_type::static_extent(0) == cs::dynamic_extent,
                  "unsigned index + negative step -> not folded (stays dynamic)");
    if ((long)uneg.extent(0) != 0) return 34;   // runtime: unsigned step-cast -> empty

    // slice also works through slice_along (same resolution)
    auto g = t.slice_along<2>(slice(1, none));         // keep axes 0,1; axis2 [1,4)
    static_assert(decltype(g)::rank() == 3, "slice_along keeps unnamed axes");
    if (g.extent(2) != 3 || g(1,2,0) != t(1,2,1)) return 16;

    // negative indices/bounds must work even with an UNSIGNED index_type
    // (the wrap is done in a signed domain, not after casting to index_type).
    auto u = wrap(buf, cs::extents<cs::size_t,2,3,4>{});
    static_assert(cs::is_unsigned<decltype(u)::index_type>::value, "unsigned index_type");
#ifndef TNY_NO_NEGATIVE_INDEX
    if (u(-1,-1,-1) != u(1,2,3)) return 17;           // negative element index wraps
    auto us = u(0, slice(-2, none), all);             // negative slice bound wraps
    if (us.extent(0) != 2 || us(0,0) != u(0,1,0)) return 18;
#endif

    // ---- negative step (python a[::-1], a[5:0:-1], strided reverse) ------
    auto line = wrap(buf, shape<4>{});                // signed index for reverse
    auto rev = line(slice(none,none,-1));             // full reverse
    if (rev.extent(0) != 4) return 19;
    for (long i=0;i<4;++i) if (rev(i) != line(3-i)) return 20;
    auto rev2 = line(slice(3,0,-1));                  // 3,2,1 (stop 0 excluded)
    if (rev2.extent(0) != 3 || rev2(0)!=line(3) || rev2(2)!=line(1)) return 21;
    auto rstep = line(slice(none,none,-2));           // 3,1
    if (rstep.extent(0) != 2 || rstep(0)!=line(3) || rstep(1)!=line(1)) return 22;
    // reverse is a mutable view
    rev(0) = 77.0; if (line(3) != 77.0) return 23; line(3) = 3;

    // 2-D: reverse rows via slicing, forward columns
    auto M2 = wrap(buf, shape<3,4>{});
    auto Mr = M2(slice(none,none,-1), all);
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (Mr(i,j) != M2(2-i,j)) return 24;

    // ---- #67: negative-step start clamps to [-1,n-1] like numpy (empty vs 1) --
    // A negative-step start that wraps below -1 makes an EMPTY slice (numpy), not
    // a spurious 1-element one. `line` has extent 4 here (restored above).
    auto e0 = line(slice(-100, none, -1));            // start wraps below -1 -> empty
    if (e0.extent(0) != 0) return 25;
    auto e1 = line(slice(-100, -100, -1));            // both below -1 -> empty
    if (e1.extent(0) != 0) return 26;
    auto e2 = line(slice(0, 0, -1));                  // start==stop -> empty
    if (e2.extent(0) != 0) return 27;
    // unaffected cases still behave (start >= 0, or a real reverse):
    auto k1 = line(slice(1, -100, -1));               // stop wraps below -1 -> [1,0]
    if (k1.extent(0) != 2 || k1(0) != line(1) || k1(1) != line(0)) return 28;
    auto k2 = line(slice(none, none, -1));            // full reverse still 4 elts
    if (k2.extent(0) != 4) return 29;

    // the compile-time fold agrees (static extent == runtime), over a static n=5:
    double b5[5]; for (long i=0;i<5;++i) b5[i]=i;
    auto s5 = wrap(b5, shape<5>{});
    static_assert(decltype(s5(slice<-8,-8,-1>()))::extents_type::static_extent(0) == 0, "static: empty");
    static_assert(decltype(s5(slice<-100,0,-1>()))::extents_type::static_extent(0) == 0, "static: start below -1 -> empty");
    static_assert(decltype(s5(slice<1,-100,-1>()))::extents_type::static_extent(0) == 2, "static: [1,0]");
    if (s5(slice<-8,-8,-1>()).extent(0) != 0) return 30;      // runtime matches the fold
    if (s5(slice<1,-100,-1>()).extent(0) != 2) return 31;

    // ---- #80: a multi-axis EMPTY slice keeps its base pointer in bounds -------
    // Each empty axis's offset is zeroed, so the summed base never runs past the
    // buffer (buf+3*4 + buf+4*1 = +16 on a 12-elt buffer would be UB to form).
    auto M3 = wrap(buf, shape<3,4>{});
    auto em2 = M3(slice(3,3), slice(4,4));            // both axes empty (positive step)
    if (em2.numel() != 0) return 32;
    if (em2.data() < &buf[0] || em2.data() > &buf[12]) return 33;   // base pointer in [buf, buf+numel]
    auto emn = M3(slice(-100,none,-1), all);          // empty NEGATIVE axis + full axis
    if (emn.extent(0) != 0 || emn.extent(1) != 4) return 34;
    if (emn.data() < &buf[0] || emn.data() > &buf[12]) return 35;

    return 0;
}
