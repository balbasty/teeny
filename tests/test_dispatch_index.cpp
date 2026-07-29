// int32 offsets, PR B (#115 / #165): dispatch_index narrows a fixed-rank view's
// offset width when the span fits (int32 arm) else keeps it (int64 arm); the compile-
// time `narrow_index` flag folds it into dispatch_rank's leaf (rank outer, width inner).
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

// records the dispatched index width (bytes) and touches an element (proves mutability)
struct Rec { int * width; template <class V> void operator()(V v) {
    *width = static_cast<int>(sizeof(typename V::index_type));
    if (v.numel() > 0) v.data()[0] += 1.0;
}};

int main() {
    double buf[24]; for (int i = 0; i < 24; ++i) buf[i] = i;
    int w = 0;

    // (1) dispatch_index: an in-range view runs the int32 arm.
    auto sm = wrap(buf, shape<-1,-1>{2,3}, {3,1});
    dispatch_index(sm, Rec{&w});
    if (w != 4) return 1;

    // (2) dispatch_index: a >2^31-span view runs the wide (int64) arm.
    auto big = wrap(buf, shape<-1>{2}, {3000000000LL});
    w = 0; dispatch_index(big, Rec{&w});
    if (w != 8) return 2;

    // (3) explicit target width other than int32 (dispatch_index<Idx2>).
    w = 0; dispatch_index<cs::int16_t>(wrap(buf, shape<-1>{4}, {2}), Rec{&w});  // reach 6 fits int16
    if (w != 2) return 3;

    // (4) dispatch_rank<narrow_index>: the fixed cell is int32 when it fits.
    // cs::int64_t (not `long`, which is only 32-bit on Windows/LLP64): the carrier's
    // offset_t is deduced from this array's element type, and check (6) below needs a
    // real 64-bit carrier to construct its >2^31 stride without truncating.
    cs::int64_t sh[2] = {2,3}, st[2] = {3,1};
    auto at = as_anyrank(buf, sh, st, 2);                  // carrier offset_t = int64
    w = 0; dispatch_rank<narrow_index>(at, Rec{&w});
    if (w != 4) return 4;

    // (5) plain dispatch_rank is unchanged — the cell keeps the carrier's offset_t.
    w = 0; dispatch_rank(at, Rec{&w});
    if (w != 8) return 5;

    // (6) dispatch_rank<narrow_index> on a >2^31 carrier falls back to the wide arm.
    cs::int64_t shb[1] = {2}, stb[1] = {3000000000LL};
    auto atb = as_anyrank(buf, shb, stb, 1);
    w = 0; dispatch_rank<narrow_index>(atb, Rec{&w});
    if (w != 8) return 6;

    // (7) element identity: the int32 arm addresses exactly the same elements.
    auto v32 = sm.reindex<cs::int32_t>();
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j) if (v32(i,j) != sm(i,j)) return 7;

    // (8) NEGATIVE strides: dispatch a flipped view -> int32 arm, addresses correctly.
    auto fl = wrap(buf, shape<3,4>{}).flip<0>();               // strides<-4,1>, negative stride
    int ok = 0;
    dispatch_index(fl, [&](auto v){
        ok = (sizeof(typename decltype(v)::index_type) == 4);
        for (long i = 0; i < 3; ++i) for (long j = 0; j < 4; ++j) if (v(i,j) != buf[(2-i)*4 + j]) ok = 0;
    });
    if (!ok) return 8;

    return 0;
}
