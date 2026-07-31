// #349: flip gained a MULTI-AXIS form (numpy `np.flip(a, axis=(0,2))`), matching
// squeeze/unsqueeze's `<Ax...>` + `axis<...>{}` pair. Flips commute, so the axis
// list is a SET: any order gives the very same view type and elements, and the
// whole thing is built in ONE pass (no chain of intermediate views).
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    // (2,3,4) row-major, values 0..23 so an element is its own C-order offset.
    double buf[24];
    for (int i = 0; i < 24; ++i) buf[i] = i;
    auto t = wrap(buf, shape<2,3,4>{});                     // strides 12,4,1

    // ---- multi-axis == chained single-axis (same TYPE, same elements) ----------
    auto ch = t.flip<0>().flip<2>();
    auto m  = t.flip<0,2>();
    static_assert(cs::is_same<decltype(m), decltype(ch)>::value, "flip<0,2> == flip<0>().flip<2>()");
    if (m.rank()!=3 || m.shape(0)!=2 || m.shape(1)!=3 || m.shape(2)!=4) return 1;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) for (int k = 0; k < 4; ++k)
        if (m(i,j,k) != t(1-i, j, 3-k)) return 2;
    if (m.data() != ch.data()) return 3;

    // ---- ORDER-INDEPENDENCE: flips commute, so flip<2,0> is flip<0,2> ----------
    auto mr = t.flip<2,0>();
    static_assert(cs::is_same<decltype(mr), decltype(m)>::value, "flip axis list is a set");
    if (mr.data() != m.data()) return 4;
    for (int i = 0; i < 2; ++i) for (int k = 0; k < 4; ++k)
        if (mr(i,0,k) != m(i,0,k)) return 5;

    // ---- all three axes, and every ordering of them ---------------------------
    auto all3 = t.flip<0,1,2>();
    if (all3.data() != &buf[23]) return 6;                  // origin = the last element
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) for (int k = 0; k < 4; ++k)
        if (all3(i,j,k) != t(1-i, 2-j, 3-k)) return 7;
    auto p1 = t.flip<2,0,1>(); auto p2 = t.flip<1,2,0>();
    static_assert(cs::is_same<decltype(p1), decltype(all3)>::value, "");
    static_assert(cs::is_same<decltype(p2), decltype(all3)>::value, "");
    if (p1.data()!=all3.data() || p2.data()!=all3.data()) return 8;

    // ---- NEGATIVE axes wrap (numpy), mixed with positive ones ------------------
    auto neg = t.flip<-3,-1>();                             // == flip<0,2>
    static_assert(cs::is_same<decltype(neg), decltype(m)>::value, "");
    if (neg.data() != m.data()) return 9;
    auto mix = t.flip<-1,0>();                              // mixed sign, scrambled
    static_assert(cs::is_same<decltype(mix), decltype(m)>::value, "");
    if (mix.data() != m.data()) return 10;

    // ---- STATIC FOLDING: a static-shape multi-flip stays a compile-time strides<>
    using ML = decltype(m)::layout_type;
    static_assert(_is_strides<ML>::value, "multi-flip -> strides<...>");
    static_assert(ML::all_static(), "multi-flip over a static source folds every stride");
    static_assert(ML::static_stride(0) == -12, "axis 0 stride negated at compile time");
    static_assert(ML::static_stride(1) ==   4, "untouched axis keeps its stride");
    static_assert(ML::static_stride(2) ==  -1, "axis 2 stride negated at compile time");
    static_assert(cs::is_same<decltype(m)::shape_type, shape<2,3,4>>::value, "extents unchanged");
    using AL = decltype(all3)::layout_type;
    static_assert(AL::all_static() && AL::static_stride(0)==-12 && AL::static_stride(1)==-4
                  && AL::static_stride(2)==-1, "");

    // an EBO'd view of a fully static shape stays exactly a pointer
    static_assert(sizeof(m) == sizeof(double *), "folded multi-flip view is just its pointer");

    // ---- the axis<...>{} VALUE form (no `.template` on a dependent receiver) ----
    auto av = t.flip(axis<0,2>{});
    static_assert(cs::is_same<decltype(av), decltype(m)>::value, "flip(axis<0,2>{}) == flip<0,2>()");
    if (av.data() != m.data()) return 11;
    auto avr = t.flip(axis<2,0>{});                         // scrambled, same view
    if (avr.data() != m.data()) return 12;
    auto av1 = t.flip(axis<1>{});                           // single-axis value form
    static_assert(cs::is_same<decltype(av1), decltype(t.flip<1>())>::value, "");
    for (int j = 0; j < 3; ++j) if (av1(0,j,0) != t(0,2-j,0)) return 13;
    auto av0 = t.flip(axis<>{});                            // empty list = no-op (#369)
    static_assert(decltype(av0)::rank() == 3, "");
    if (av0.data()!=t.data() || av0(1,2,3)!=t(1,2,3)) return 14;
    for (cs::size_t r = 0; r < 3; ++r) if (av0.stride(r) != t.stride(r)) return 15;

    // ---- BACKWARD COMPATIBILITY: the single-axis forms are untouched -----------
    auto s2 = t.flip<2>();
    if (s2.data() != &buf[3]) return 16;
    for (int k = 0; k < 4; ++k) if (s2(0,0,k) != t(0,0,3-k)) return 17;
    auto sd = t.flip<>();                                   // defaulted axis == flip<0>()
    static_assert(cs::is_same<decltype(sd), decltype(t.flip<0>())>::value, "flip<>() == flip<0>()");
    auto si = t.flip(Int<2>());                             // Int<k>() value form
    static_assert(cs::is_same<decltype(si), decltype(s2)>::value, "");
    if (si.data() != s2.data()) return 18;

    // ---- an involution: flipping the same axes twice is the identity ------------
    auto back = t.flip<0,2>().flip<0,2>();
    if (back.data() != t.data()) return 19;
    for (int i = 0; i < 2; ++i) for (int k = 0; k < 4; ++k)
        if (back(i,1,k) != t(i,1,k)) return 20;

    // ---- DYNAMIC extents: the offsets are computed at run time ------------------
    auto d = wrap(buf, shape<-1,-1,4>{2,3});
    auto dm = d.flip<0,1>();
    if (dm.data() != &buf[20]) return 21;                   // (2-1)*12 + (3-1)*4 = 20
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) for (int k = 0; k < 4; ++k)
        if (dm(i,j,k) != d(1-i, 2-j, k)) return 22;
    // a stride the source can't derive statically stays dynamic under the flip;
    // a derivable one still folds — NEGATED where the axis is flipped.
    using DL = decltype(dm)::layout_type;
    static_assert(_is_strides<DL>::value, "");
    static_assert(DL::static_stride(0) == dynamic_stride, "axis 0 spans a dynamic extent");
    static_assert(DL::static_stride(1) == -4, "axis 1's derivable stride folds, negated");
    static_assert(DL::static_stride(2) ==  1, "the untouched trailing stride folds");

    // ---- a NON-contiguous source: flip composes with a slice/permute ------------
    auto sl = t(all, slice(0,2), all).permute<1,0,2>();      // (2,2,4), strides 4,12,1
    auto fs = sl.flip<0,2>();
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) for (int k = 0; k < 4; ++k)
        if (fs(i,j,k) != sl(1-i, j, 3-k)) return 23;
    // ...and re-flipping an ALREADY-flipped axis restores it (strides re-negate)
    auto again = fs.flip<2>();
    for (int i = 0; i < 2; ++i) for (int k = 0; k < 4; ++k)
        if (again(i,0,k) != sl(1-i,0,k)) return 24;

    // ---- an owning tensor flips into a view of its own storage -----------------
    auto o = local<double, shape<2,3>>{};
    o(0,0)=1; o(0,2)=3; o(1,0)=4; o(1,2)=6;
    auto of = o.flip<0,1>();
    if (of.data() != &o(1,2) || of(0,0)!=6 || of(1,2)!=1) return 25;

    // ---- a CONST receiver keeps the const element type -------------------------
    const auto & ct = t;
    auto cf = ct.flip<0,2>();
    static_assert(cs::is_same<decltype(cf)::element_type, const double>::value, "");
    if (cf(0,0,0) != t(1,0,3)) return 26;

    // ---- math over a multi-flipped view (the engines see the negative strides) --
    auto sum_all = sum(m);
    if (sum_all != sum(t)) return 27;                        // same elements, reordered
    auto cl = m.clone();                                     // materialise the reversal
    static_assert(cs::is_same<decltype(cl)::shape_type, shape<2,3,4>>::value, "");
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) for (int k = 0; k < 4; ++k)
        if (cl(i,j,k) != t(1-i,j,3-k)) return 28;

    return 0;
}
