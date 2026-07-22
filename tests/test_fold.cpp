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
    auto ta = st.take_along<2>(1);                 // drop axis 2 at index 1
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

    // ---- recast keeps only when contiguous; validates + folds inner dims -----
    auto vv = wrap(buf, shape<-1,3,4>{2});     // (2,3,4) contiguous, dynamic outer
    auto rc = vv.recast<shape<2,3,4>>();        // all static now
    static_assert(decltype(rc.stride(Int<1>()))::value == 4, "recast folds inner stride");
    if (rc(1,2,3) != buf[12+8+3]) return 15;

    return 0;
}
