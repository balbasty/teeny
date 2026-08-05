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
// the same, for an `anyrank` carrier (no `index_type`: `size(d)` returns its offset type)
struct RecA { int * width; template <class A> void operator()(const A & a) {
    *width = static_cast<int>(sizeof(a.size(0)));
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

    // ---- (9)-(12) a STATIC extent too large for the target (#491) -------------
    // `shape<300,2>` cannot be narrowed to int8_t: extent 300 alone rules it out, so
    // `index_fits<int8_t>()` answers false at run time anyway and the narrow arm is
    // dead. dispatch_index must therefore still COMPILE — it drops that arm at compile
    // time instead of instantiating `reindex<int8_t>()`, whose static-extent assert
    // used to fail the build — and take the wide arm. (A direct `reindex<int8_t>()` on
    // the same view stays a compile error: there the narrowing IS the request.)
    static double bigbuf[600]; for (int i = 0; i < 600; ++i) bigbuf[i] = i;

    // (9) the repro: compiles, and dispatches to the wide (int64) arm.
    w = 0; dispatch_index<cs::int8_t>(wrap(bigbuf, shape<300,2>{}), Rec{&w});
    if (w != 8) return 9;
    // ...and the wide arm hands `f` the source view untouched: same extents, same elements.
    ok = 0;
    dispatch_index<cs::int8_t>(wrap(bigbuf, shape<300,2>{}), [&](auto v){
        ok = (sizeof(typename decltype(v)::index_type) == 8) && (v.shape(0) == 300) && (v.shape(1) == 2);
        for (long i = 0; i < 300; ++i) for (long j = 0; j < 2; ++j) if (v(i,j) != bigbuf[i*2 + j]) ok = 0;
    });
    if (!ok) return 10;

    // (10) POSITIVE CONTROLS — the compile-time gate only drops PROVABLY dead arms:
    //   (a) static extents AND reach both fit -> the narrow arm still runs;
    w = 0; dispatch_index<cs::int8_t>(wrap(buf, shape<3,4>{}), Rec{&w});             // reach 11
    if (w != 1) return 11;
    //   (b) static extents fit but the REACH does not -> still a RUNTIME decision
    //       (the gate must not fire), landing on the wide arm.
    w = 0; dispatch_index<cs::int8_t>(wrap(bigbuf, shape<3,4>{}, {100,1}), Rec{&w}); // reach 203
    if (w != 8) return 12;
    //   (c) a static shape the DEFAULT int32 target can hold is still narrowed.
    w = 0; dispatch_index(wrap(bigbuf, shape<300,2>{}), Rec{&w});
    if (w != 4) return 13;

    // (11) the same with the DEFAULT target — the shape a real caller would need for
    // this to bite: a static extent above 2^31 can never be an int32 extent, so the
    // narrow arm folds away and `f` sees the int64 view. (Nothing dereferences past
    // element 0; the point is that the TYPE compiles and picks the right arm.)
    w = 0; dispatch_index(wrap(bigbuf, shape<3000000000>{}), Rec{&w});
    if (w != 8) return 14;

    // (12) the same gate on an `anyrank`: a static `Tail` extent too large for the
    // target rules the whole carrier out, so dispatch_index compiles and hands `f` the
    // untouched (int64) carrier.
    cs::int64_t bshp[2] = {2, 300}, bstr[2] = {300, 1};
    auto tail300 = as_anyrank(bigbuf, bshp, bstr, 2, anyshape<etc,300>{});
    w = 0; dispatch_index<cs::int8_t>(tail300, RecA{&w});
    if (w != 8) return 15;
    //   ...a fitting static tail still narrows,
    cs::int64_t sshp[2] = {2, 3}, sstr[2] = {3, 1};
    w = 0; dispatch_index<cs::int8_t>(as_anyrank(buf, sshp, sstr, 2, anyshape<etc,3>{}), RecA{&w});
    if (w != 1) return 16;
    //   ...and dispatch_rank<narrow_index> (whose leaf goes through dispatch_index at the
    //   default int32 width, which extent 300 fits) is unchanged.
    w = 0; dispatch_rank<narrow_index>(tail300, Rec{&w});
    if (w != 4) return 17;

    // ---- (13) exact-boundary pin (#497) ----------------------------------------
    // #495's review manually verified the two dead-arm-selection predicates agree
    // exactly at the `Idx2::max()` boundary (e.g. int8_t/shape<127> vs shape<128>),
    // but that verification wasn't captured as a permanent test — pin it here.
    // int8_t's max is 127 (INT8_MAX): a static extent of EXACTLY 127 fits, one past
    // it (128 == INT8_MAX+1) does not. Checked BOTH directly against the compile-time
    // predicate and via `dispatch_index`'s actual arm selection.
    static_assert(_static_extents_fit<cs::int8_t, shape<127>>(),
                  "int8_t: static extent 127 (== INT8_MAX) must fit");
    static_assert(!_static_extents_fit<cs::int8_t, shape<128>>(),
                  "int8_t: static extent 128 (== INT8_MAX+1) must not fit");

    // shape<127>: fits statically (and its reach of 126 fits at run time too) -> narrow
    // (int8, width 1) arm runs.
    w = 0; dispatch_index<cs::int8_t>(wrap(bigbuf, shape<127>{}), Rec{&w});
    if (w != 1) return 18;
    // shape<128>: does NOT fit statically -> the compile-time gate drops the narrow
    // arm (which would otherwise fail to compile), so the wide (int64, width 8) arm
    // runs unconditionally.
    w = 0; dispatch_index<cs::int8_t>(wrap(bigbuf, shape<128>{}), Rec{&w});
    if (w != 8) return 19;

    return 0;
}
