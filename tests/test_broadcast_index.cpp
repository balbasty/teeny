// Cross-width broadcasting (#167 / #115 open-Q3): the broadcast RESULT inherits the
// WIDER of the two operands' offset index types. This is lossless (the engine already
// casts both operands to the result index type) and prevents a wide operand's strides
// from being truncated to a narrow result's width. Same-width mixes are unchanged.
//
// Checks 6+ cover the IN-PLACE / `into(dest)` side of the same rule (#346): there the
// destination is NOT free to broaden (its index type is baked into the caller's own
// tensor type), so the engine instead runs its offset math in the widest of the three
// participating index types and leaves every tensor's own type alone. Before the fix a
// narrow destination truncated a wide rhs's strides — silently, and straight off the
// front of the buffer.
//
// Checks 13+ cover the SIGNEDNESS half of that rule. Width alone does not decide the
// engine's offset type: when the widest participant is UNSIGNED and a narrower one is
// SIGNED with a NEGATIVE stride (any flipped/reversed view), casting that -1 stride to
// the unsigned type makes it 4294967295, which zero-extends into the pointer offset
// instead of stepping backwards. So a MIXED-signedness set decodes in a signed type
// wide enough for both sides; all-signed and all-unsigned sets keep the width pick.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

template <class T> using idx_of = typename T::index_type;

// A 40002-element buffer so a 2x2 window can carry a stride of 40000 — well past the
// int16 range (40000 folds to -25536), which is what made the truncation segfault
// rather than merely read the wrong element. The int32-vs-int64 analogue would need a
// >2^31-element buffer to exhibit it, hence int16 (same reasoning as #342's test).
static double big[40002];
static void fill_big() { for (int i = 0; i < 40002; ++i) big[i] = double(i) + 0.5; }

int main() {
    double buf[12]; for (int i = 0; i < 12; ++i) buf[i] = i;

    // (1) int32 (lhs) + int64 (rhs) -> result index type broadens to int64.
    auto a32 = wrap(buf, shape32<2,3>{});          // int32-indexed view
    auto b64 = wrap(buf, shape<2,3>{});            // int64-indexed view
    auto c1  = a32 + b64;
    static_assert(cs::is_same<idx_of<decltype(c1)>, cs::int64_t>::value, "int32+int64 -> int64 result");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j)
        if (c1(i,j) != a32(i,j) + b64(i,j)) return 1;

    // (2) order-independent: int64 (lhs) + int32 (rhs) -> int64 too.
    auto c2 = b64 + a32;
    static_assert(cs::is_same<idx_of<decltype(c2)>, cs::int64_t>::value, "int64+int32 -> int64 result");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j)
        if (c2(i,j) != b64(i,j) + a32(i,j)) return 2;

    // (3) same-width unchanged: int32+int32 -> int32, int64+int64 -> int64.
    auto s3 = wrap(buf, shape32<2,3>{}) + wrap(buf, shape32<2,3>{});
    static_assert(cs::is_same<idx_of<decltype(s3)>, cs::int32_t>::value, "int32+int32 stays int32");
    auto s4 = wrap(buf, shape<2,3>{}) + wrap(buf, shape<2,3>{});
    static_assert(cs::is_same<idx_of<decltype(s4)>, cs::int64_t>::value, "int64+int64 stays int64");

    // (4) mixed-width WITH broadcast (stretch on the narrow operand): (int32 2x1) +
    //     (int64 2x3) -> (2,3) int64. Element identity vs the int64 reference proves the
    //     wide operand's strides are NOT truncated through a narrow result.
    auto col32 = wrap(buf, shape32<2,1>{});        // int32, stretches across axis 1
    auto M64   = wrap(buf, shape<2,3>{});
    auto c4    = col32 + M64;
    static_assert(cs::is_same<idx_of<decltype(c4)>, cs::int64_t>::value, "broadcast mix -> int64");
    static_assert(decltype(c4)::extents_type::static_extent(0) == 2 &&
                  decltype(c4)::extents_type::static_extent(1) == 3, "result 2x3");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j)
        if (c4(i,j) != col32(i,0) + M64(i,j)) return 3;

    // (5) comparisons share the same result extents -> the bool result also broadens.
    auto m = a32 < b64;
    static_assert(cs::is_same<idx_of<decltype(m)>, cs::int64_t>::value, "compare result broadens too");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j)
        if (m(i,j) != (a32(i,j) < b64(i,j))) return 4;

    // (6) dynamic-shape mixed width -> heap result, still int64-indexed.
    auto da = wrap(buf, shape_as<cs::int32_t,-1,-1>{2,3});   // int32 dynamic
    auto db = wrap(buf, shape<-1,-1>{2,3});                  // int64 dynamic
    auto c6 = da + db;
    static_assert(cs::is_same<idx_of<decltype(c6)>, cs::int64_t>::value, "dynamic mix -> int64");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j)
        if (c6(i,j) != da(i,j) + db(i,j)) return 5;

    /* ---- in-place / into(dest): a NARROW destination + a WIDE rhs (#346) ------- *
     * The destination keeps its own (narrow) index type; only the engine's internal
     * offset math widens. Every expectation below is hand-computed from the buffer's
     * own `big[k] == k + 0.5` rule, so a "no longer crashes but reads the wrong
     * element" fix would fail these just as loudly as the segfault did. */
    fill_big();

    // (7) the reported repro: int16-indexed 2x2 destination, int64-indexed rhs whose
    //     stride 40000 does not fit int16. Truncation made `ob` negative -> segfault.
    {
        auto bw = wrap(big, shape<2,2>{}, strides<40000,1>{});     // (0,1,40000,40001)
        double ad[4] = {10, 20, 30, 40};
        auto a16 = wrap(ad, shape_as<short,2,2>{});
        static_assert(cs::is_same<idx_of<decltype(a16)>, short>::value, "dest stays int16");
        static_assert(cs::is_same<idx_of<decltype(bw)>, cs::int64_t>::value, "rhs is int64");
        a16.add_(bw);
        // the destination's OWN type must be untouched by the widening (unevaluated —
        // `decltype` does not run the call, so `ad` below is the result of ONE add_).
        static_assert(cs::is_same<idx_of<cs::remove_reference_t<decltype(a16.add_(bw))>>,
                                  short>::value, "add_ still returns the int16 dest");
        if (ad[0] != 10 + 0.5) return 6;                           // big[0]
        if (ad[1] != 20 + 1.5) return 7;                           // big[1]
        if (ad[2] != 30 + 40000.5) return 8;                       // big[40000]
        if (ad[3] != 40 + 40001.5) return 9;                       // big[40001]
    }

    // (8) the same pair through the other in-place writers: copy_, mul_, and the fused
    //     axpy add_(b, alpha) — all of them ride the same engine.
    {
        auto bw = wrap(big, shape<2,2>{}, strides<40000,1>{});
        double cd[4] = {0,0,0,0};
        auto c16 = wrap(cd, shape_as<short,2,2>{});
        c16.copy_(bw);
        if (cd[0] != 0.5 || cd[1] != 1.5 || cd[2] != 40000.5 || cd[3] != 40001.5) return 10;
        c16.mul_(bw);
        if (cd[0] != 0.5*0.5 || cd[2] != 40000.5*40000.5) return 11;
        double dd[4] = {1,1,1,1};
        auto d16 = wrap(dd, shape_as<short,2,2>{});
        d16.add_(bw, 2.0);                                          // d += 2*b
        if (dd[2] != 1 + 2*40000.5) return 12;
        if (dd[3] != 1 + 2*40001.5) return 13;
    }

    // (9) narrow destination + wide rhs WITH a stretched axis: the rhs is (2,1) with
    //     stride 40000, broadcast across axis 1 of the (2,2) destination.
    {
        auto col = wrap(big, shape<2,1>{}, strides<40000,1>{});      // (0, 40000)
        double ed[4] = {0,0,0,0};
        auto e16 = wrap(ed, shape_as<short,2,2>{});
        e16.add_(col);
        if (ed[0] != 0.5 || ed[1] != 0.5) return 14;
        if (ed[2] != 40000.5 || ed[3] != 40000.5) return 15;
    }

    // (10) `into(dest)`: BOTH operands wide, destination narrow (the out-of-place
    //      producers allocate a result that already carries the wider index type, so
    //      only a caller-supplied dest can be the narrow one here).
    {
        auto bw = wrap(big, shape<2,2>{}, strides<40000,1>{});
        auto ow = wrap(big, shape<2,2>{});                           // (0.5,1.5,2.5,3.5)
        double fd[4] = {0,0,0,0};
        auto f16 = wrap(fd, shape_as<short,2,2>{});
        static_assert(cs::is_same<idx_of<decltype(f16)>, short>::value, "into-dest stays int16");
        bw.add(ow, into(f16));
        if (fd[0] != 0.5 + 0.5 || fd[1] != 1.5 + 1.5) return 16;
        if (fd[2] != 40000.5 + 2.5 || fd[3] != 40001.5 + 3.5) return 17;
    }

    // (11) the mirror case (wide destination, narrow rhs) was always fine — keep it
    //      pinned so the widening never regresses it.
    {
        double gd[4] = {0,0,0,0};
        auto g64 = wrap(gd, shape<2,2>{});
        auto n16 = wrap(big, shape_as<short,2,2>{});                 // (0.5,1.5,2.5,3.5)
        static_assert(cs::is_same<idx_of<decltype(g64)>, cs::int64_t>::value, "dest stays int64");
        g64.copy_(n16);
        if (gd[0] != 0.5 || gd[1] != 1.5 || gd[2] != 2.5 || gd[3] != 3.5) return 18;
    }

    // (12) same-width in-place stays exactly as it was (both int32, both int64), and a
    //      mixed-width pair whose strides all fit the narrow side agrees element for
    //      element with the same-width spelling.
    {
        double hd[4] = {1,2,3,4};
        auto h32 = wrap(hd, shape32<2,2>{});
        auto k32 = wrap(big, shape32<2,2>{});
        h32.add_(k32);
        if (hd[0] != 1 + 0.5 || hd[3] != 4 + 3.5) return 19;
        double jd[4] = {1,2,3,4};
        auto j16 = wrap(jd, shape_as<short,2,2>{});
        auto k64 = wrap(big, shape<2,2>{});
        j16.add_(k64);
        for (int i = 0; i < 4; ++i) if (jd[i] != hd[i]) return 20;
    }

    /* ---- mixed SIGNEDNESS: an unsigned-indexed participant next to a flipped
     * (negative-stride) signed one. A width-only pick lands on the unsigned type and
     * turns the -1 stride into 4294967295 — a write/read far off the front of the
     * buffer. Both orders, since either side can be the unsigned one. */

    // (13) unsigned-indexed rhs, flipped int16-indexed destination (the reported case).
    {
        double sv[4] = {10, 20, 30, 40};
        auto ub = wrap(sv, shape_as<unsigned int,4>{});                 // uint32-indexed rhs
        double dv[4] = {0,0,0,0};
        auto fd = wrap(dv + 3, shape_as<short,4>{}, strides<-1>{});     // int16-indexed, stride -1
        static_assert(cs::is_same<idx_of<decltype(ub)>, unsigned int>::value, "rhs is uint32-indexed");
        static_assert(cs::is_same<idx_of<decltype(fd)>, short>::value, "flipped dest stays int16");
        fd.copy_(ub);                                                   // fd(k) is dv[3-k]
        if (dv[0] != 40 || dv[1] != 30 || dv[2] != 20 || dv[3] != 10) return 21;
        fd.add_(ub);                                                    // same pair, accumulating
        if (dv[0] != 80 || dv[1] != 60 || dv[2] != 40 || dv[3] != 20) return 22;
    }

    // (14) the other order: flipped int16-indexed rhs, unsigned-indexed destination.
    {
        double sv[4] = {1, 2, 3, 4};
        auto fb = wrap(sv + 3, shape_as<short,4>{}, strides<-1>{});     // int16-indexed, stride -1
        double dv[4] = {0,0,0,0};
        auto ud = wrap(dv, shape_as<unsigned int,4>{});                 // uint32-indexed dest
        ud.copy_(fb);
        if (dv[0] != 4 || dv[1] != 3 || dv[2] != 2 || dv[3] != 1) return 23;
        ud.add_(fb, 2.0);                                               // fused axpy, same pair
        if (dv[0] != 12 || dv[3] != 3) return 24;
    }

    // (15) EQUAL width, disagreeing signedness (int32 vs uint32) — the case a width-only
    //      pick cannot express at all: it must step up to a signed 64-bit decode.
    {
        double sv[4] = {5, 6, 7, 8};
        auto f32 = wrap(sv + 3, shape32<4>{}, strides<-1>{});           // int32-indexed, stride -1
        double dv[4] = {0,0,0,0};
        auto u32 = wrap(dv, shape_as<unsigned int,4>{});
        u32.copy_(f32);
        if (dv[0] != 8 || dv[1] != 7 || dv[2] != 6 || dv[3] != 5) return 25;
        double ev[4] = {0,0,0,0};
        auto fe32 = wrap(ev + 3, shape32<4>{}, strides<-1>{});          // ...and mirrored
        auto ub32 = wrap(sv, shape_as<unsigned int,4>{});
        fe32.copy_(ub32);
        if (ev[0] != 8 || ev[1] != 7 || ev[2] != 6 || ev[3] != 5) return 26;
    }

    // (16) mixed signedness + a NEGATIVE stride + a broadcast axis + `into(dest)`.
    {
        double sv[2] = {100, 200};
        auto ucol = wrap(sv, shape_as<unsigned int,2,1>{});             // (2,1) uint32, stretches
        double dv[4] = {0,0,0,0};
        // rows reversed: element (i,j) lands on dv[2*(1-i)+j]
        auto fd = wrap(dv + 2, shape_as<short,2,2>{}, strides<-2,1>{});
        fd.add_(ucol);
        if (dv[0] != 200 || dv[1] != 200 || dv[2] != 100 || dv[3] != 100) return 27;
        double gv[4] = {0,0,0,0};
        auto gd = wrap(gv + 2, shape_as<short,2,2>{}, strides<-2,1>{});  // flipped `into` dest
        auto uw = wrap(sv, shape_as<unsigned int,2,1>{});
        uw.add(ucol, into(gd));
        if (gv[0] != 400 || gv[1] != 400 || gv[2] != 200 || gv[3] != 200) return 28;
    }

    // (17) controls that must NOT move: an all-UNSIGNED mixed-width pair (no negative
    //      stride is expressible there — teeny's flipped views need a signed index) and
    //      an all-SIGNED flipped pair. Both keep the plain width pick.
    {
        double sv[4] = {1,2,3,4};
        double dv[4] = {10,20,30,40};
        auto ud = wrap(dv, shape_as<unsigned int,4>{});
        auto us = wrap(sv, shape_as<unsigned short,4>{});
        ud.add_(us);
        if (dv[0] != 11 || dv[1] != 22 || dv[2] != 33 || dv[3] != 44) return 29;
        double pv[4] = {0,0,0,0};
        auto fs = wrap(sv + 3, shape<4>{}, strides<-1>{});              // int64-indexed, stride -1
        auto fp = wrap(pv, shape32<4>{});
        fp.copy_(fs);
        if (pv[0] != 4 || pv[1] != 3 || pv[2] != 2 || pv[3] != 1) return 30;
    }

    return 0;
}
