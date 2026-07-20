// Coverage for the kernel-support primitives added after the API review:
// fill_/zero_/copy_, add_at (atomic-on-device scatter), dispatch_value,
// slices_front, and static-preserving rng.
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

    // ---- add_at: scatter-accumulate ------------------------------------
    auto acc = local<double, extents<long,4>>(); acc.zero_();
    acc.add_at(1.5, 2);
    acc.add_at(2.5, 2);          // accumulates
    acc.add_at(9.0, -1);         // negative index wraps -> index 3
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

    // ---- slices_front<N>: peel the first N (batch) axes ---------------
    double buf[2*3*4];
    for (int i=0;i<2*3*4;++i) buf[i]=i;
    auto t = view(buf, extents<long,2,3,4>{});
    long count = 0, checksum = 0;
    for (auto line : slices_front<2>(t)) {                  // peel axes 0,1 -> (4,) lines
        ++count;
        for (long k=0;k<4;++k) checksum += (long)line(k);
    }
    if (count != 6) return 9;                               // 2*3 lines
    if (checksum != (2*3*4-1)*(2*3*4)/2) return 10;          // sum 0..23

    // ---- static-preserving rng ----------------------------------------
    auto M = local<double, extents<long,5,6>>();
    auto sv = M(all, rng(Int<1>(), Int<4>()));              // static [1,4) on axis 1
    static_assert(decltype(sv)::extents_type::static_extent(1) == 3,
                  "static rng -> static subextent (folds)");
    static_assert(decltype(sv)::extents_type::static_extent(0) == 5, "axis 0 kept static");

    return 0;
}
