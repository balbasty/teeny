// Sub-task 1 of smart reshape (#129): the numpy no-copy-reshape predicate
// can_reshape_without_copy<NewExt...>() — true iff reshape can be a VIEW, which is
// more than C-contiguity (splitting an axis / merging a stride-compatible run).
// This only tests the PREDICATE; reshape() itself still requires C-contiguity until
// sub-task 2 wires this in.
#include <teeny/teeny.h>
using namespace tny;
namespace cs = cuda::std;

int main() {
    double big[3*16]; for (int i = 0; i < 48; ++i) big[i] = i;

    // ---- C-contiguous source: ANY numel-matching reshape is a view -----------
    auto c = wrap(big, shape<6,4>{});                    // strides (4,1)
    if (!c.can_reshape_without_copy<2,3,4>()) return 1;  // split
    if (!c.can_reshape_without_copy<24>())    return 2;  // full merge
    if (!c.can_reshape_without_copy<8,3>())   return 3;
    if (!c.can_reshape_without_copy<-1,3>())  return 4;  // inferred 8
    if (c.can_reshape_without_copy<5,5>())    return 5;  // numel 25 != 24 -> false

    // ---- NON-contiguous source that is still reshapable -----------------------
    // a (3,8) window into a (3,16) buffer: row stride 16 (a gap), inner stride 1.
    auto w = wrap(big, shape<3,8>{}, {16,1});            // dynamic_strides, NOT C-contiguous
    if (w.is_contiguous()) return 6;                     // sanity: not C-contiguous (and not even dense — a gap)
    if (!w.can_reshape_without_copy<3,2,4>()) return 7;  // split the CONTIGUOUS inner axis (8 -> 2,4): viewable
    if (!w.can_reshape_without_copy<3,8>())   return 8;  // identity
    if (!w.can_reshape_without_copy<3,4,2>()) return 9;  // 8 -> 4,2 also a contiguous split
    if (w.can_reshape_without_copy<24>())     return 10; // merging ACROSS the row gap (3,8)->24: NOT viewable
    if (w.can_reshape_without_copy<6,4>())    return 11; // (3,8)->(6,4) crosses the gap: not viewable
    if (w.can_reshape_without_copy<2,12>())   return 12; // crosses the gap: not viewable

    // ---- a permuted (transposed) view ----------------------------------------
    auto p = wrap(big, shape<3,16>{}).permute<1,0>();    // (16,3) strides (1,16)
    if (!p.can_reshape_without_copy<16,3>()) return 13;  // identity
    if (!p.can_reshape_without_copy<16,3,1>()) return 14; // append a size-1 axis
    if (p.can_reshape_without_copy<48>())    return 15;  // (16,3)/(1,16) can't merge to 48 (not a C-run)
    if (p.can_reshape_without_copy<8,6>())   return 16;  // regroup across incompatible strides: no

    // ---- size-1 axes are transparent -----------------------------------------
    auto s1 = wrap(big, shape<1,6,1,4>{});               // C-contiguous with size-1 axes
    if (!s1.can_reshape_without_copy<24>())  return 17;
    if (!s1.can_reshape_without_copy<6,4>()) return 18;
    if (!s1.can_reshape_without_copy<2,3,4>()) return 19;

    // ---- fully static source: the predicate folds to a compile-time answer ----
    auto st = local<double, shape<6,4>>{};
    static_assert(noexcept(st.can_reshape_without_copy<2,3,4>()), "noexcept");
    if (!st.can_reshape_without_copy<2,3,4>()) return 20;

    return 0;
}
