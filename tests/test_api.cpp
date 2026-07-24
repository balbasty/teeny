// Coverage for the kernel-support primitives added after the API review:
// fill_/zero_/copy_, at().add_<true>() (atomic-on-device scatter), dispatch_value,
// peel_front, and static-preserving slice.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;

int main() {
    // ---- fill_ / zero_ / copy_ ----------------------------------------
    auto a = local<double, extents<long,2,3>>();
    a.fill_(7.0);
    for (long i=0;i<2;++i) for (long j=0;j<3;++j) if (a(i,j) != 7.0) return 1;
    a.zero_();
    if (sum(a) != 0.0) return 2;

    auto b = local<double, extents<long,2,3>>();
    for (long i=0;i<2;++i) for (long j=0;j<3;++j) b(i,j) = i*3+j;
    a.copy_(b);
    for (long i=0;i<2;++i) for (long j=0;j<3;++j) if (a(i,j) != b(i,j)) return 3;

    // copy_ broadcasts a (2,1) column across the row axis
    auto col = local<double, extents<long,2,1>>(); col(0,0)=10; col(1,0)=20;
    a.copy_(col);
    for (long j=0;j<3;++j) if (a(0,j)!=10 || a(1,j)!=20) return 4;

    // ---- at().add_<true>(): scatter-accumulate -------------------------
    auto acc = local<double, extents<long,4>>(); acc.zero_();
    acc.at(2).add_<true>(1.5);
    acc.at(2).add_<true>(2.5);          // accumulates
    acc.at(-1).add_<true>(9.0);         // negative index wraps -> index 3
    if (acc(2) != 4.0 || acc(3) != 9.0 || acc(0) != 0.0) return 5;

    // ---- dispatch_value: runtime value -> static ----------------------
    int seen = -1;
    bool ok = dispatch_value<1,2,3>(2, [&](auto d){ seen = (int)d.value; });
    if (!ok || seen != 2) return 6;
    if (dispatch_value<1,2,3>(7, [&](auto){})) return 7;   // no match -> false

    // dispatched value is usable as a template argument
    int rank_seen = 0;
    dispatch_value<1,2,3>(3, [&](auto d){
        auto t = local<double, extents<long, d.value>>();   // static extent from runtime D
        rank_seen = (int)t.extent(0);
    });
    if (rank_seen != 3) return 8;

    // ---- peel_front<N>: peel the first N (batch) axes ---------------
    double buf[2*3*4];
    for (int i=0;i<2*3*4;++i) buf[i]=i;
    auto t = wrap(buf, extents<long,2,3,4>{});
    long count = 0, checksum = 0;
    for (auto line : peel_front<2>(t)) {                  // peel axes 0,1 -> (4,) lines
        ++count;
        for (long k=0;k<4;++k) checksum += (long)line(k);
    }
    if (count != 6) return 9;                               // 2*3 lines
    if (checksum != (2*3*4-1)*(2*3*4)/2) return 10;          // sum 0..23

    // ---- size_front<N>: peel count without building the range ----------
    if (size_front<2>(t)  != 6)  return 30;                 // == peel_front<2>(t).size()
    if (size_front<-1>(t) != 6)  return 31;                 // keep last 1 -> product of first 2
    if (size_front<-2>(t) != 2)  return 32;                 // flattened batch (all but last 2)
    if (size_front<0>(t)  != 1)  return 33;                 // peel 0 -> one cell
    if (size_front<2>(t)  != (long)peel_front<2>(t).size()) return 34;

    // ---- slicing: kept axes stay static, ranged axis resolves at runtime -
    // (a range goes through a layout_stride view — see the CCCL note in
    //  tensor.h — so the ranged axis is dynamic, but `all`-kept axes stay static)
    auto M = local<double, extents<long,5,6>>();
    auto sv = M(all, slice(1, 4));                            // [1,4) on axis 1
    static_assert(decltype(sv)::extents_type::static_extent(0) == 5, "kept axis stays static");
    if (sv.extent(0) != 5 || sv.extent(1) != 3) return 11;
    if (sv(4,2) != M(4,3)) return 12;                        // value is correct (right strides)

    // ---- creation factories + iota_ -----------------------------------
    auto z = zeros<double>(shape<2,3>{});                    // static -> stack
    static_assert(decltype(z)::ownership == storage::stack, "static zeros -> stack");
    if (sum(z) != 0.0) return 13;
    auto o = ones<double>(shape<2,2>{});   if (sum(o) != 4.0) return 14;
    auto fu = full<int>(shape<3>{}, 7);    if (fu(0) != 7 || fu(2) != 7) return 15;
    auto dz = zeros<double>(shape<-1,3>{2});                 // dynamic -> heap
    static_assert(decltype(dz)::ownership == storage::heap, "dynamic zeros -> heap");
    if (dz.extent(0) != 2 || sum(dz) != 0.0) return 16;
    auto ar = arange<long>(5);             if (ar(0) != 0 || ar(4) != 4) return 17;
    auto it = local<double, shape<2,3>>(); it.iota_(10.0, 2.0);   // 10,12,...,20 row-major
    if (it(0,0) != 10.0 || it(0,1) != 12.0 || it(1,2) != 20.0) return 18;

    // ---- generic map_ / zip_with_ + is_contiguous / clone -------------
    struct sq  { double operator()(double x) const { return x*x; } };
    struct add { double operator()(double a, double b) const { return a+b; } };
    auto q = local<double, shape<3>>(); q.iota_(1.0); q.map_(sq{});   // 1,4,9
    if (q(2) != 9.0) return 19;
    auto r = local<double, shape<3>>(); r.fill_(1.0); r.zip_with_(add{}, q);
    if (r(2) != 10.0) return 20;

    double cb[6]; for (int i=0;i<6;++i) cb[i]=i;
    auto vp = wrap(cb, shape<2,3>{}).permute<1,0>();                  // dense, but NOT row-major
    if (!vp.is_dense()) return 21;                                   // dense in SOME order (permuted) -> true
    if (vp.is_contiguous()) return 24;                              // ...but NOT C-contiguous (the redefault)
    auto cl = vp.clone();                                            // dense copy
    static_assert(cs::is_same<decltype(cl)::layout_type, cs::layout_right>::value, "clone row-major");
    if (!cl.is_contiguous() || cl(2,1) != vp(2,1)) return 22;       // clone IS C-contiguous

    // ---- recast: recover static inner dims at the dynamic boundary -----
    double rb[18]; for (int i=0;i<18;++i) rb[i]=i;
    auto dyn = wrap(rb, shape<-1,-1,-1>{2,3,3});                     // fully dynamic (dlpack-like)
    auto stc = dyn.recast<shape<-1,3,3>>();                          // recover static 3x3
    static_assert(decltype(stc)::extents_type::static_extent(1) == 3, "recast recovers static inner");
    if (stc.extent(0) != 2 || stc(1,2,2) != 17) return 23;

    return 0;
}
