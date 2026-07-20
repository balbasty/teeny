#include <teeny/md.h>
#include <cuda/std/type_traits>

using namespace tny::md;
namespace cs = cuda::std;
using cs::extents;

int main()
{
    double buf[24];
    for (long i = 0; i < 24; ++i) buf[i] = i;
    auto t = view(buf, extents<long,2,3,4>{});         // C-contiguous (strides 12,4,1)

    // ---- peel axis 0: 2 sub-views of shape (3,4) -----------------------
    auto r0 = slices<0>(t);
    if (r0.size() != 2) return 1;
    static_assert(decltype(r0[0])::rank() == 2, "peel one axis -> rank 2");
    if (r0[1](2,3) != t(1,2,3)) return 2;              // slice 1 == t(1,.,.)

    // range-for yields the sub-views
    long seen = 0, checkacc = 0;
    for (auto s : slices<0>(t)) { checkacc += (long)s(0,0); ++seen; }
    if (seen != 2) return 3;
    if (checkacc != (long)t(0,0,0) + (long)t(1,0,0)) return 4;   // 0 + 12

    // ---- peel axes 0 AND 1: 6 sub-views of shape (4,) ------------------
    auto r01 = slices<0,1>(t);
    if (r01.size() != 6) return 5;
    static_assert(decltype(r01[0])::rank() == 1, "peel two axes -> rank 1");
    // row-major over (axis0=2, axis1=3): linear 5 -> (i0=1, i1=2)
    if (r01[5](0) != t(1,2,0)) return 6;
    if (r01[5](3) != t(1,2,3)) return 7;

    // ---- grid-stride style: slice_at + write through the slice ---------
    // (this is the index2offset replacement: linear i -> strided sub-view)
    for (long i = 0; i < r01.size(); ++i) {
        auto s = slice_at<0,1>(t, i);                  // 1-D view over the last axis
        s.add_(1000.0);                                // mutate the ORIGINAL buffer
    }
    for (long i = 0; i < 24; ++i) if (buf[i] != i + 1000) return 8;

    // ---- peel a non-leading axis (axis 1) ------------------------------
    auto r1 = slices<1>(t);                            // 3 sub-views of shape (2,4)
    if (r1.size() != 3) return 9;
    static_assert(decltype(r1[0])::rank() == 2, "peel middle axis -> rank 2");
    if (r1[2](1,3) != t(1,2,3)) return 10;             // axis1 fixed to 2

    // ---- md math works on a peeled slice -------------------------------
    auto sub = slice_at<0>(t, 0);                      // (3,4) view of t(0,.,.)
    double before = sub(1,1);
    sub.mul_(2.0);
    if (sub(1,1) != before * 2) return 11;

    return 0;
}
