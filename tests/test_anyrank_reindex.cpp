// #467: whole-carrier offset-width narrowing. `at.index_fits<Idx2>()` is the signed
// reach test over the WHOLE carrier; `at.reindex<Idx2>()` returns the same carrier
// (same data pointer, same memory space, same static Head/Tail geometry) with `ndim`
// and the runtime shape/strides copied into an inline `Idx2` meta store. Cells peeled
// off the narrowed carrier are then `Idx2`-indexed for free, and address exactly the
// same elements. `dispatch_index` composes with the carrier unchanged.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

// records the dispatched offset width (bytes) of whatever dispatch_index handed over
struct RecW { int * width; template <class A> void operator()(const A & a) {
    *width = static_cast<int>(sizeof(a.size(0)));      // size(d) returns the carrier's offset type
}};

int main() {
    double buf[2*3*4]; for (int i = 0; i < 24; ++i) buf[i] = i;
    cs::int64_t shp[3]  = {2,3,4};
    cs::int64_t strd[3] = {12,4,1};

    // ---- (1) index_fits over the whole carrier -------------------------------
    auto at = as_anyrank(buf, shp, strd, 3);                 // wraps the arrays (no copy)
    if (!at.index_fits<cs::int32_t>()) return 1;             // reach 23 — fits easily
    if (!at.index_fits<cs::int16_t>()) return 2;

    // boundary: reach exactly INT32_MAX fits, one past does not (wild strides — the
    // reach is only computed, never dereferenced).
    cs::int64_t bshp[1] = {2};
    cs::int64_t bfit[1] = {2147483647LL}, bover[1] = {2147483648LL};
    if (!as_anyrank(buf, bshp, bfit,  1).index_fits<cs::int32_t>()) return 3;
    if ( as_anyrank(buf, bshp, bover, 1).index_fits<cs::int32_t>()) return 4;
    // negative strides: SIGNED int32 accepts the min-side reach, unsigned rejects it.
    cs::int64_t bneg[1] = {-2147483648LL}, bneg2[1] = {-2147483649LL};
    if (!as_anyrank(buf, bshp, bneg,  1).index_fits<cs::int32_t>())  return 5;
    if ( as_anyrank(buf, bshp, bneg2, 1).index_fits<cs::int32_t>())  return 6;
    if ( as_anyrank(buf, bshp, bneg,  1).index_fits<cs::uint32_t>()) return 7;
    // a stride-0 (broadcast) axis adds no reach, whatever its extent.
    cs::int64_t hshp[1] = {1000000}, hstr[1] = {0};
    if (!as_anyrank(buf, hshp, hstr, 1).index_fits<cs::int32_t>()) return 8;

    // ---- (2) reindex: type, values, element identity -------------------------
    auto at32 = at.reindex<cs::int32_t>();
    static_assert(cs::is_same<decltype(at32.size(0)), cs::int32_t>::value, "carrier offset width narrowed");
    static_assert(decltype(at32)::space == storage::view, "space preserved");
    static_assert(decltype(at32)::device_passable, "reindex always yields an inline (copy) store");
    static_assert(cs::is_trivially_copyable<decltype(at32)>::value, "narrowed carrier is device-passable");
    if (at32.data != at.data || at32.ndim != at.ndim) return 9;
    for (int d = 0; d < 3; ++d)
        if (at32.size(d) != at.size(d) || at32.step(d) != at.step(d)) return 10;

    // cells come out int32-indexed, and address exactly the source's elements
    auto v   = at.fixed<3>();
    auto v32 = at32.fixed<3>();
    static_assert(cs::is_same<decltype(v32)::index_type, cs::int32_t>::value, "fixed cell is int32");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j) for (long k = 0; k < 4; ++k)
        if (v32(i,j,k) != v(i,j,k)) return 11;

    // the batch idiom survives narrowing: same cells, same order, int32 offsets
    long cells = 0;
    for (auto cell : at32.peel_front<-2>()) {
        static_assert(cs::is_same<typename decltype(cell)::index_type, cs::int32_t>::value,
                      "peeled cell inherits the narrowed offset width");
        auto ref = at.peel_front_at<-2>(cells);
        if (cell.data() != ref.data()) return 12;
        for (long j = 0; j < 3; ++j) for (long k = 0; k < 4; ++k)
            if (cell(j,k) != ref(j,k)) return 13;
        ++cells;
    }
    if (cells != at.size_front<-2>()) return 14;

    // ---- (3) NEGATIVE strides survive the narrowing --------------------------
    // a reversed axis 0: base at the last row, stride -12.
    cs::int64_t rstrd[3] = {-12,4,1};
    auto rev   = as_anyrank(buf + 12, shp, rstrd, 3);
    auto rev32 = rev.reindex<cs::int32_t>();
    if (rev32.step(0) != -12) return 15;
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j) for (long k = 0; k < 4; ++k)
        if (rev32.fixed<3>()(i,j,k) != rev.fixed<3>()(i,j,k)) return 16;

    // ---- (4) the copy_meta store actually SHRINKS ----------------------------
    auto cp   = as_anyrank(buf, shp, strd, 3, copy_meta);    // inline int64 store
    auto cp32 = cp.reindex<cs::int32_t>();
    static_assert(sizeof(decltype(cp32)) < sizeof(decltype(cp)),
                  "int32 meta store halves the carrier's by-value footprint");
    static_assert(decltype(cp32)::max_rank == decltype(cp)::max_rank, "capacity preserved by default");
    if (cp32.fixed<3>()(1,2,3) != cp.fixed<3>()(1,2,3)) return 17;
    // a smaller inline capacity can be requested explicitly
    auto cp8 = cp.reindex<cs::int32_t, 8>();
    static_assert(decltype(cp8)::max_rank == 8, "explicit MaxRank");
    static_assert(sizeof(decltype(cp8)) < sizeof(decltype(cp32)), "smaller capacity, smaller carrier");
    if (cp8.fixed<3>()(1,2,3) != cp.fixed<3>()(1,2,3)) return 18;

    // ---- (5) static Head/Tail geometry is TYPE info — untouched by narrowing --
    // (C_in=2, *middle, C_out=4) over the same (2,3,4) buffer.
    auto ht   = as_anyrank(buf, shp, strd, 3, anyshape<2, etc, 4>{});
    auto ht32 = ht.reindex<cs::int32_t>();
    static_assert(decltype(ht32)::head_rank == 1 && decltype(ht32)::tail_rank == 1, "ends preserved");
    static_assert(decltype(ht32)::tail_type::static_extent(0) == 4, "tail extent value untouched");
    static_assert(decltype(ht32)::head_type::static_extent(0) == 2, "head extent value untouched");
    static_assert(cs::is_same<typename decltype(ht32)::tail_type::index_type, cs::int32_t>::value,
                  "only the tail extents' INDEX TYPE follows Idx2");
    auto HF = ht32.fixed<3>();
    static_assert(decltype(HF)::shape_type::static_extent(0) == 2, "head still folds");
    static_assert(decltype(HF)::shape_type::static_extent(1) == cs::dynamic_extent, "middle still dynamic");
    static_assert(decltype(HF)::shape_type::static_extent(2) == 4, "tail still folds");
    if (HF(1,2,3) != buf[1*12 + 2*4 + 3]) return 19;

    // static trailing STRIDES (a layout tag) survive too: the folded strides<...> cell
    // layout is index-type independent, so it is carried over verbatim.
    auto cc   = as_anyrank(buf, shp, strd, 3, anyshape<etc,3,4>{}, ccontiguous{});
    auto cc32 = cc.reindex<cs::int32_t>();
    static_assert(cs::is_same<typename decltype(cc32)::tail_stride_type,
                              typename decltype(cc)::tail_stride_type>::value, "tail strides untouched");
    auto CF = cc32.fixed<3>();
    static_assert(decltype(CF.stride(Int<2>()))::value == 1, "inner stride still folds to 1");
    static_assert(decltype(CF.stride(Int<1>()))::value == 4, "next folds to C");
    if (CF(1,2,3) != buf[1*12 + 2*4 + 3]) return 20;

    // a device-tagged carrier keeps its space (metadata only — never dereferenced)
    auto dev = as_anyrank<storage::gpu_view>(buf, shp, strd, 3).reindex<cs::int32_t>();
    static_assert(decltype(dev)::is_device, "device space preserved");
    static_assert(decltype(dev.fixed<3>())::ownership == storage::gpu_view, "cells stay gpu_view");

    // ---- (6) dispatch_index composes with the carrier -------------------------
    int w = 0;
    dispatch_index(at, RecW{&w});
    if (w != 4) return 21;                                   // in range -> int32 arm
    cs::int64_t wide[1] = {3000000000LL};
    auto big = as_anyrank(buf, bshp, wide, 1);
    w = 0; dispatch_index(big, RecW{&w});
    if (w != 8) return 22;                                   // >2^31 span -> wide arm
    w = 0; dispatch_index<cs::int16_t>(at, RecW{&w});
    if (w != 2) return 23;                                   // explicit target width

    // ---- (7) re-narrowing / same-width reindex is a plain copy ---------------
    auto again = at32.reindex<cs::int32_t>();
    if (again.data != at.data || again.ndim != 3) return 24;
    for (int d = 0; d < 3; ++d) if (again.step(d) != at.step(d)) return 25;

    // ---- (8) per-axis reach fits individually, but the ACCUMULATED sum overflows --
    // two axes each with reach 1.5e9 / 1.0e9 (each well under INT32_MAX alone); their
    // SUM (2.5e9) exceeds it, so the whole-carrier test must still reject narrowing
    // (#469 — regression test; the accumulation already got this right at #468).
    cs::int64_t sshp[2] = {2, 2};
    cs::int64_t sstr[2] = {1500000000LL, 1000000000LL};
    if (as_anyrank(buf, sshp, sstr, 2).index_fits<cs::int32_t>()) return 26;

    // ---- (9) free forms (deduced, no `.template`) mirror the member calls ----
    // symmetric with the tensor free forms in test_reindex.cpp (#469).
    if (index_fits<cs::int32_t>(as_anyrank(buf, sshp, sstr, 2))) return 27;
    if (!index_fits<cs::int32_t>(at)) return 28;
    auto freeform32 = reindex<cs::int32_t>(at);
    static_assert(cs::is_same<decltype(freeform32.size(0)), cs::int32_t>::value, "free reindex narrows");
    if (freeform32.data != at32.data || freeform32.ndim != at32.ndim) return 29;

    // ---- (10) PATHOLOGICAL strides near ~1e18 (#471 regression) --------------
    // the internal `long long` accumulator must never overflow (UB) while deciding
    // this — it must bail out (return false) cleanly instead. Verified UB-free
    // under -fsanitize=undefined; mirrors the tensor-side case in test_reindex.cpp.
    //   (a) two positive-stride axes: each axis's reach (6e18) alone fits `long long`,
    //       but their SUM (1.2e19) would overflow it during accumulation.
    cs::int64_t pshp[2] = {2, 2};
    cs::int64_t pstr[2] = {6000000000000000000LL, 6000000000000000000LL};
    if (as_anyrank(buf, pshp, pstr, 2).index_fits<cs::int32_t>()) return 30;
    //   (b) a single axis whose (extent-1)*stride PRODUCT alone already overflows
    //       `long long` (3 * 4e18 = 1.2e19) — must be caught before the multiply.
    cs::int64_t qshp[1] = {4};
    cs::int64_t qstr[1] = {4000000000000000000LL};
    if (as_anyrank(buf, qshp, qstr, 1).index_fits<cs::int64_t>()) return 31;
    //   (c) the same two shapes, mirrored onto the negative-stride (`mino`) side.
    cs::int64_t nstr[2] = {-6000000000000000000LL, -6000000000000000000LL};
    if (as_anyrank(buf, pshp, nstr, 2).index_fits<cs::int32_t>()) return 32;
    //   (d) mirrored onto the NEGATIVE-stride PRODUCT guard specifically (#474): a
    //       single axis whose (extent-1)*stride PRODUCT ALONE already overflows
    //       `long long` on the negative side (3 * -4e18 = -1.2e19), as opposed to
    //       (c)'s accumulation-only overflow — must be caught before the multiply.
    cs::int64_t rqshp[1] = {4};
    cs::int64_t rqstr[1] = {-4000000000000000000LL};
    if (as_anyrank(buf, rqshp, rqstr, 1).index_fits<cs::int64_t>()) return 33;

    // ---- (11) WIDE UNSIGNED targets (#484) -----------------------------------
    // The positive and negative reach accumulate in separate 64-bit domains
    // (unsigned / signed), so the carrier's test is exact for EVERY integral Idx2 up
    // to 64 bits — uint64_t/size_t included, which used to be a compile error (#475).
    // Mirrors the tensor-side section in test_reindex.cpp.
    //   (a) a negative stride still can't fit an UNSIGNED target (min offset < 0).
    if (as_anyrank(buf, bshp, bneg, 1).index_fits<cs::uint64_t>()) return 34;
    //   (b) the false negative this fixes: a reach of 1.2e19 is in (2^63-1, 2^64-1],
    //       so it genuinely FITS a uint64_t index — unreachable for any single
    //       `long long` accumulator (the sum overflows it and the old code bailed).
    if (!as_anyrank(buf, pshp, pstr, 2).index_fits<cs::uint64_t>()) return 35;
    //       ...the same carrier still doesn't fit int64_t (1.2e19 > 2^63-1) — cases
    //       (10a) above keep the int32 answer, so only the wrong answer moved.
    if (as_anyrank(buf, pshp, pstr, 2).index_fits<cs::int64_t>()) return 36;
    //   (c) exact upper boundary: reach == UINT64_MAX fits, one past it doesn't
    //       (2*(2^63-1) + 1 == 2^64-1).
    cs::int64_t eshp[3] = {2, 2, 2};
    cs::int64_t efit[3] = {9223372036854775807LL, 9223372036854775807LL, 1LL};
    cs::int64_t eover[3] = {9223372036854775807LL, 9223372036854775807LL, 2LL};
    if (!as_anyrank(buf, eshp, efit,  3).index_fits<cs::uint64_t>()) return 37;
    if ( as_anyrank(buf, eshp, eover, 3).index_fits<cs::uint64_t>()) return 38;
    //   (d) the UNSIGNED accumulator's own overflow guards (unsigned overflow WRAPS
    //       rather than trapping, so a wrapped value would answer wrongly — both the
    //       product and the accumulation are checked before the fact).
    cs::int64_t oshp[1] = {5};
    cs::int64_t ostr[1] = {9000000000000000000LL};            // 4 * 9e18 = 3.6e19 > UINT64_MAX
    if (as_anyrank(buf, oshp, ostr, 1).index_fits<cs::uint64_t>()) return 39;
    cs::int64_t ashp[3] = {2, 2, 2};
    cs::int64_t astr[3] = {9000000000000000000LL, 9000000000000000000LL, 9000000000000000000LL};
    if (as_anyrank(buf, ashp, astr, 3).index_fits<cs::uint64_t>()) return 40;   // sum 2.7e19

    // ---- (12) reindex<uint64_t>() on the carrier: an ordinary widening retype --
    // compiles and addresses identically in a DEBUG build and under -DNDEBUG (where
    // `_TNY_CHECK` compiles out) alike — #480's build-mode asymmetry is gone at the
    // root, there being no static_assert left for the unevaluated form to skip.
    auto atu = at.reindex<cs::uint64_t>();
    static_assert(cs::is_same<decltype(atu.size(0)), cs::uint64_t>::value, "carrier retyped to uint64");
    if (atu.data != at.data || atu.ndim != at.ndim) return 41;
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j) for (long k = 0; k < 4; ++k)
        if (atu.fixed<3>()(i,j,k) != v(i,j,k)) return 42;
    if (!index_fits<cs::size_t>(at)) return 43;               // the LP64 spelling, equally ordinary

    // ---- (13) UNSIGNED carrier metadata (#486) -------------------------------
    // `as_anyrank(data, shape, stride, ndim)` deduces the carrier's offset type from
    // the CALLER's arrays, so a routine C-interop boundary that already holds its
    // metadata as `uint64_t`/`size_t` builds a carrier whose raw extents/strides can
    // exceed `LLONG_MAX`. The reach test used to `static_cast` each one to `long long`
    // first, wrapping it NEGATIVE: a wrapped extent read as "size <= 1, no reach", so
    // an axis with an enormous true reach contributed nothing and the carrier looked
    // narrowable — a false positive feeding `reindex`'s UB contract. This is the
    // reachability path for that bug; the tensor-side twin is in test_reindex.cpp.
    //   (a) THE repro: extent 2^63+2, stride 1 -> a true reach of 2^63+1.
    cs::uint64_t ushp[1] = { 9223372036854775810ULL }, ustr[1] = { 1ULL };
    if (as_anyrank(buf, ushp, ustr, 1).index_fits<cs::int32_t>()) return 44;
    //       ...and that same reach does fit a uint64_t index — now computed exactly,
    //       rather than reached via an extent that wrapped to "no reach at all".
    if (!as_anyrank(buf, ushp, ustr, 1).index_fits<cs::uint64_t>()) return 45;
    //   (b) the same hazard on a STRIDE: 2^63+2 is a large POSITIVE stride in an
    //       unsigned index type; the old cast read it as negative and accumulated it
    //       on the wrong (min) side.
    cs::uint64_t vshp[1] = { 2ULL }, vstr[1] = { 9223372036854775810ULL };
    if ( as_anyrank(buf, vshp, vstr, 1).index_fits<cs::int32_t>())  return 46;
    if ( as_anyrank(buf, vshp, vstr, 1).index_fits<cs::int64_t>())  return 47;   // > INT64_MAX
    if (!as_anyrank(buf, vshp, vstr, 1).index_fits<cs::uint64_t>()) return 48;   // ...fits uint64
    //   (c) the int64 edge, exactly: reach == e-1, so e == 2^63 fits int64 (reach
    //       INT64_MAX) and e == 2^63+1 does not — only an exact `e-1` separates them.
    cs::uint64_t eshp1[1] = { 9223372036854775808ULL }, eshp2[1] = { 9223372036854775809ULL };
    if (!as_anyrank(buf, eshp1, ustr, 1).index_fits<cs::int64_t>())  return 49;
    if ( as_anyrank(buf, eshp2, ustr, 1).index_fits<cs::int64_t>())  return 50;
    if (!as_anyrank(buf, eshp2, ustr, 1).index_fits<cs::uint64_t>()) return 51;
    //   (d) an ordinary unsigned-metadata carrier is untouched — and narrows, peels
    //       and addresses exactly as its int64 twin does.
    cs::uint64_t ushp3[3] = {2,3,4}, ustr3[3] = {12,4,1};
    auto un = as_anyrank(buf, ushp3, ustr3, 3);
    if (!un.index_fits<cs::int32_t>()) return 52;
    auto un32 = un.reindex<cs::int32_t>();
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j) for (long k = 0; k < 4; ++k)
        if (un32.fixed<3>()(i,j,k) != v(i,j,k)) return 53;
    //   (e) a stride-0 axis of huge extent still has no reach (unchanged — this test
    //       is about reachable OFFSETS, not extent values; see #487).
    cs::uint64_t bshp2[1] = { 9223372036854775810ULL }, bstr2[1] = { 0ULL };
    if (!as_anyrank(buf, bshp2, bstr2, 1).index_fits<cs::int32_t>()) return 54;

    return 0;
}
