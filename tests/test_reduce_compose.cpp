// Reduction trailing keyword bag (#284): dtype<Acc>{} x axis<...>{} x keepdims x
// into(dest), any SUBSET and any ORDER, via the generic _kw mechanism (kwargs.h).
// Complements test_axred.cpp (explicit-template Axes + keepdims/into) and
// test_reduce_dtype.cpp (bare dtype<Acc>{}) by exercising the compositions that
// were previously "not yet wired up" (see CLAUDE.md's old cheat-sheet caveat).
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main() {
    // (2,3) matrix, row-major values 0..5
    auto m = local<double, shape<2,3>>(); m.iota_(0.0, 1.0);   // [[0,1,2],[3,4,5]]

    // --- dtype<Acc>{} composed with an axis<...>{} value tag (previously bare-only) ---
    auto c1 = sum(m, axis<0>{}, dtype<float>{});
    static_assert(cs::is_same<typename decltype(c1)::element_type, float>::value, "dtype tag sets Acc");
    if (c1(0)!=3.0f || c1(1)!=5.0f || c1(2)!=7.0f) return 1;
    // order shouldn't matter
    auto c2 = sum(m, dtype<float>{}, axis<0>{});
    static_assert(cs::is_same<decltype(c1), decltype(c2)>::value, "order-independent");
    if (c2(0)!=3.0f || c2(1)!=5.0f || c2(2)!=7.0f) return 2;

    // --- dtype x axis x keepdims, any order ---
    auto k1 = sum(m, dtype<float>{}, axis<0>{}, keepdims);
    static_assert(decltype(k1)::rank() == 2, "keepdims keeps rank");
    static_assert(cs::is_same<typename decltype(k1)::element_type, float>::value, "dtype tag sets Acc (keepdims)");
    if (k1(0,1)!=5.0f) return 3;
    auto k2 = sum(m, keepdims, axis<0>{}, dtype<float>{});
    static_assert(cs::is_same<decltype(k1), decltype(k2)>::value, "keepdims order-independent");
    if (k2(0,1)!=5.0f) return 4;

    // --- dtype x axis x into, any order ---
    auto d1 = local<float, shape<3>>();
    sum(m, dtype<float>{}, axis<0>{}, into(d1));
    if (d1(0)!=3.0f || d1(1)!=5.0f || d1(2)!=7.0f) return 5;
    auto d2 = local<float, shape<3>>();
    auto & rd2 = sum(m, into(d2), dtype<float>{}, axis<0>{});
    static_assert(cs::is_same<decltype(rd2), decltype(d2)&>::value, "into returns dest&");
    if (&rd2 != &d2) return 6;
    if (d2(0)!=3.0f || d2(1)!=5.0f || d2(2)!=7.0f) return 7;

    // --- full composition: dtype x axis x keepdims x into, any order ---
    auto kd1 = local<float, shape<1,3>>();
    sum(m, dtype<float>{}, axis<0>{}, keepdims, into(kd1));
    if (kd1(0,0)!=3.0f || kd1(0,1)!=5.0f || kd1(0,2)!=7.0f) return 8;
    auto kd2 = local<float, shape<1,3>>();
    sum(m, into(kd2), keepdims, dtype<float>{}, axis<0>{});
    if (kd2(0,0)!=3.0f || kd2(0,1)!=5.0f || kd2(0,2)!=7.0f) return 9;

    // --- same composition on other reductions sharing the shared axis-core (prod/max/min/sqnorm) ---
    if (prod(m, axis<1>{}, dtype<float>{})(1) != 60.0f) return 10;
    if (max(m, dtype<float>{}, axis<1>{})(1) != 5.0f) return 11;
    if (min(m, axis<1>{}, dtype<float>{}, keepdims)(1,0) != 3.0f) return 12;
    auto v = local<double, shape<3>>(); v(0)=3; v(1)=0; v(2)=4;
    if (sqnorm(v, dtype<float>{}) != 25.0f) return 13;   // bare dtype (no axis) still works

    // --- mean: int->double rule still holds through the generic path ---
    auto mi = local<int, shape<2,3>>(); mi.iota_(0, 1);
    auto mm = mean(mi, axis<0>{});
    static_assert(cs::is_same<typename decltype(mm)::element_type, double>::value, "integer mean -> double");
    if (mm(0) != 1.5 || mm(1) != 2.5 || mm(2) != 3.5) return 14;
    // explicit dtype overrides the int->double default
    auto mm2 = mean(mi, dtype<float>{}, axis<0>{});
    static_assert(cs::is_same<typename decltype(mm2)::element_type, float>::value, "dtype tag overrides mean's int rule");
    if (mm2(0) != 1.5f) return 15;

    // --- norm: dtype x axis x keepdims x into (layers sqrt + floating result on top) ---
    auto vv = local<double, shape<2,3>>(); vv.iota_(1.0, 1.0);
    auto n1 = norm(vv, axis<1>{}, dtype<float>{}, keepdims);
    static_assert(decltype(n1)::rank() == 2, "norm keepdims keeps rank");
    static_assert(cs::is_same<typename decltype(n1)::element_type, float>::value, "norm dtype tag");
    // row 0: sqrt(1+4+9) = sqrt(14)
    if (!(n1(0,0) > 3.7416f && n1(0,0) < 3.7417f)) return 16;

    // --- dot: dtype x into composed (bare, no axis concept) ---
    auto a3 = local<double, shape<3>>(); a3(0)=1; a3(1)=2; a3(2)=3;
    auto b3 = local<double, shape<3>>(); b3(0)=4; b3(1)=5; b3(2)=6;
    auto cell = local<float, shape<>>();
    dot(a3, b3, dtype<float>{}, into(cell));           // 1*4+2*5+3*6 = 32
    if (cell.item() != 32.0f) return 17;
    auto cell2 = local<float, shape<>>();
    dot(a3, b3, into(cell2), dtype<float>{});
    if (cell2.item() != 32.0f) return 18;

    // --- methods: same composition, via m.sum(...) etc. ---
    auto mc1 = m.sum(dtype<float>{}, axis<0>{});
    static_assert(cs::is_same<decltype(mc1), decltype(c1)>::value, "method matches free function");
    if (mc1(0)!=3.0f || mc1(1)!=5.0f || mc1(2)!=7.0f) return 19;
    auto mkd = local<float, shape<1,3>>();
    m.sum(dtype<float>{}, axis<0>{}, keepdims, into(mkd));
    if (mkd(0,0)!=3.0f || mkd(0,1)!=5.0f || mkd(0,2)!=7.0f) return 20;
    auto mcell = local<float, shape<>>();
    a3.dot(b3, dtype<float>{}, into(mcell));
    if (mcell.item() != 32.0f) return 21;

    // --- dynamic shape: same composition allocates on the heap ---
    auto dm = owned<double, rank<2>>(shape<-1,-1>{2,3}); dm.copy_(m);
    auto dc = sum(dm, dtype<float>{}, axis<0>{});
    static_assert(decltype(dc)::ownership == storage::heap, "dynamic result -> heap");
    if (dc(0)!=3.0f || dc(1)!=5.0f || dc(2)!=7.0f) return 22;
    auto ddest = local<float, shape<3>>();
    sum(dm, dtype<float>{}, axis<0>{}, into(ddest));
    if (ddest(0)!=3.0f || ddest(1)!=5.0f || ddest(2)!=7.0f) return 23;

    // ---- an EXPLICITLY EMPTY axis list reduces over NO axis (#398) --------------
    // `axis<>{}` names no axis, so each output cell aggregates the ONE element at
    // its own index and the result keeps the SOURCE's shape -- numpy's rule for
    // `np.sum(a, axis=())`, and the same "empty axis list = identity" the rest of
    // the family already has (squeeze/unsqueeze/take_along/peel, #369).
    // It used to be indistinguishable, inside the keyword bag, from "no axis
    // keyword at all" (both landed on an `axis<>` sentinel), so it silently did
    // the OPPOSITE extreme: a full, all-axes reduction (sum -> 15, not [[0..5]]).
    auto z = sum(m, axis<>{});                       // m is [[0,1,2],[3,4,5]]
    static_assert(decltype(z)::rank() == 2, "empty axis list reduces over NO axis");
    static_assert(cs::is_same<typename decltype(z)::shape_type, shape<2,3>>::value, "...so the shape is the source's");
    static_assert(cs::is_same<typename decltype(z)::element_type, double>::value, "...and the element type is the source's");
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) if (z(i,j) != m(i,j)) return 24;
    if (z.data() == m.data()) return 25;             // an owning result, not a view
    // prod/max/min/mean are the same identity over an empty axis list
    if (prod(m, axis<>{})(1,2) != 5.0 || max(m, axis<>{})(1,2) != 5.0) return 26;
    if (min(m, axis<>{})(0,1) != 1.0 || mean(m, axis<>{})(1,2) != 5.0) return 27;
    // sqnorm/norm are Σaᵢ²/√Σaᵢ² OVER THE NAMED AXES, so over none of them they are
    // the elementwise a² and |a| (the same limit, not a plain identity) -- checked
    // with a negative element, where the two differ from a copy.
    auto sgn = local<double, shape<2>>(); sgn(0) = -3; sgn(1) = 4;
    auto sq = sqnorm(sgn, axis<>{});
    if (sq(0) != 9.0 || sq(1) != 16.0) return 28;
    auto nr = norm(sgn, axis<>{});
    if (nr(0) != 3.0 || nr(1) != 4.0) return 29;
    // CONTROL: NO axis keyword at all still means the documented FULL reduction
    if (sum(m) != 15.0 || sum(m, dtype<double>{}) != 15.0) return 30;
    if (sqnorm(v, dtype<float>{}) != 25.0f) return 31;
    // ...and a NON-empty axis list is untouched
    if (sum(m, axis<0>{})(1) != 5.0) return 32;

    // the other keywords compose with an empty axis list exactly as with a real one
    auto ez = sum(m, axis<>{}, dtype<float>{});
    static_assert(cs::is_same<typename decltype(ez)::element_type, float>::value, "dtype tag sets the result type");
    if (ez(1,2) != 5.0f) return 33;
    auto ek = sum(m, dtype<float>{}, axis<>{}, keepdims);   // no axis was reduced -> nothing to keep
    static_assert(decltype(ek)::rank() == 2, "keepdims over an empty axis list is the identity");
    if (ek(1,2) != 5.0f) return 34;
    auto ed = local<float, shape<2,3>>();
    auto & red = sum(m, axis<>{}, into(ed));
    static_assert(cs::is_same<decltype(red), decltype(ed)&>::value, "into returns dest&");
    if (&red != &ed || ed(0,1) != 1.0f || ed(1,2) != 5.0f) return 35;
    // integer mean keeps its int->double rule here too
    auto emi = mean(mi, axis<>{});
    static_assert(cs::is_same<typename decltype(emi)::element_type, double>::value, "integer mean -> double");
    if (emi(0,1) != 1.0 || emi(1,2) != 5.0) return 36;
    // methods: same rule (and the same result type) as the free functions
    auto em = m.sum(axis<>{});
    static_assert(cs::is_same<decltype(em), decltype(z)>::value, "method matches free function");
    if (em(1,2) != 5.0) return 37;
    if (m.norm(axis<>{})(1,2) != 5.0 || m.mean(axis<>{})(0,1) != 1.0) return 38;
    // dynamic shape: the result keeps the SOURCE's (dynamic) extents, so it is
    // heap-owned -- the host-only overload, keyed off the same empty tag.
    auto edm = sum(dm, axis<>{});
    static_assert(decltype(edm)::ownership == storage::heap, "dynamic source -> heap result");
    static_assert(decltype(edm)::rank() == 2, "...with the source's rank");
    if (edm.shape(0)!=2 || edm.shape(1)!=3 || edm(1,2)!=5.0 || edm(0,1)!=1.0) return 39;
    auto edd = local<double, shape<2,3>>();
    sum(dm, axis<>{}, into(edd));
    if (edd(1,2) != 5.0) return 40;
    // CROSSED: a DYNAMIC source with `axis<>{}` AND `keepdims` together. No axis was
    // reduced, so keepdims has nothing to re-insert and the result -- which on the
    // dynamic path is a MOVE-ONLY heap tensor -- must be handed back as-is. This used
    // to be a hard compile error ("use of deleted function tensor(const tensor&)"):
    // the keepdims arm ran even for an empty axis pack and fed that heap tensor to
    // `_keepdims<>`'s BY-VALUE base overload. (On the static path the same arm merely
    // paid two silent whole-tensor copies -- also gone.)
    auto edk = sum(dm, axis<>{}, keepdims);
    static_assert(decltype(edk)::ownership == storage::heap, "dynamic source -> heap result");
    static_assert(decltype(edk)::rank() == 2, "keepdims over an empty axis list is the identity");
    if (edk.shape(0)!=2 || edk.shape(1)!=3) return 41;
    if (edk(0,1) != 1.0 || edk(1,2) != 5.0) return 42;
    // ...and with the other keywords crossed in as well (dtype, into, method form)
    auto edk2 = dm.sum(dtype<float>{}, axis<>{}, keepdims);
    static_assert(decltype(edk2)::rank() == 2, "keepdims over an empty axis list is the identity (method)");
    static_assert(cs::is_same<typename decltype(edk2)::element_type, float>::value, "dtype tag sets the result type");
    if (edk2(0,1) != 1.0f || edk2(1,2) != 5.0f) return 43;
    auto edki = local<double, shape<2,3>>();
    auto & redk = sum(dm, axis<>{}, keepdims, into(edki));
    static_assert(cs::is_same<decltype(redk), decltype(edki)&>::value, "into returns dest&");
    if (&redk != &edki || edki(0,1) != 1.0 || edki(1,2) != 5.0) return 44;
    // the rest of the family takes the same crossed path (dynamic x empty axis x keepdims)
    if (mean(dm, axis<>{}, keepdims)(1,2) != 5.0 || min(dm, axis<>{}, keepdims)(0,1) != 1.0) return 45;
    if (norm(dm, axis<>{}, keepdims)(1,2) != 5.0 || prod(dm, axis<>{}, keepdims)(0,1) != 1.0) return 46;

    // an EXPLICIT leading accumulator template arg composes with the empty axis tag
    // just like the `dtype<Acc>{}` value tag does (both fold into the same `RAcc`)
    auto ea = sum<float>(m, axis<>{});
    static_assert(decltype(ea)::rank() == 2, "empty axis list reduces over NO axis");
    static_assert(cs::is_same<typename decltype(ea)::element_type, float>::value, "<Acc> sets the result type");
    if (ea(0,1) != 1.0f || ea(1,2) != 5.0f) return 47;
    auto en = norm<float>(sgn, axis<>{});                // sgn is [-3, 4]
    static_assert(cs::is_same<typename decltype(en)::element_type, float>::value, "<Acc> sets the result type (norm)");
    if (en(0) != 3.0f || en(1) != 4.0f) return 48;

    // a NON-CONTIGUOUS source: still elementwise over the VIEW's own geometry
    // (columns 0 and 2 of m), materialised into a dense owning result.
    auto ns = sum(m(all, slice(0,3,2)), axis<>{});       // [[0,2],[3,5]]
    static_assert(decltype(ns)::rank() == 2, "empty axis list keeps the source's rank");
    if (ns.shape(0) != 2 || ns.shape(1) != 2) return 49;
    if (ns(0,0)!=0.0 || ns(0,1)!=2.0 || ns(1,0)!=3.0 || ns(1,1)!=5.0) return 50;
    if (!ns.is_contiguous()) return 51;                  // gathered into a dense block

    return 0;
}
