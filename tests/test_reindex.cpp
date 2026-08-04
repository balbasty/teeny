// int32 offsets, PR A (#115 / #163): reindex<Idx2>() is a no-copy, layout-preserving
// retype of the offset index width; index_fits<Idx2>() is the signed-reach guard.
// A static strides<...> pack is unchanged; dynamic strides/extents narrow to Idx2 and
// the by-value footprint shrinks. Element identity vs the source is preserved.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    double buf[24]; for (int i = 0; i < 24; ++i) buf[i] = i;

    // (1) ccontiguous static: index narrowed, extent VALUES kept, values identical.
    auto t   = wrap(buf, shape<2,3,4>{});
    auto t32 = t.reindex<cs::int32_t>();
    static_assert(cs::is_same<decltype(t32)::index_type, cs::int32_t>::value, "index_type narrowed");
    static_assert(decltype(t32)::extents_type::static_extent(0) == 2 &&
                  decltype(t32)::extents_type::static_extent(2) == 4, "extent values kept");
    static_assert(cs::is_same<decltype(t32)::layout_type, ccontiguous>::value, "layout kind preserved");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j) for (long k = 0; k < 4; ++k)
        if (t32(i,j,k) != t(i,j,k)) return 1;

    // (2) static strides<...>: the literal pack is UNCHANGED, only the index narrows.
    auto s   = tensor<double, shape<2,3>, strides<3,1>>(buf);
    auto s32 = reindex<cs::int32_t>(s);                        // free form (no `.template`)
    static_assert(cs::is_same<decltype(s32)::layout_type, strides<3,1>>::value, "strides<> pack unchanged");
    static_assert(cs::is_same<decltype(s32)::index_type, cs::int32_t>::value, "index narrowed");
    if (s32(1,2) != s(1,2)) return 2;

    // (3) transposed (folded static strides<1,3>): strides + values preserved.
    auto tr   = wrap(buf, shape<2,3>{}).permute<1,0>();        // (3,2), strides (1,3)
    auto tr32 = tr.reindex<cs::int32_t>();
    if (tr32.stride(0) != 1 || tr32.stride(1) != 3) return 3;
    for (long i = 0; i < 3; ++i) for (long j = 0; j < 2; ++j) if (tr32(i,j) != tr(i,j)) return 4;

    // (4) genuinely DYNAMIC strides: values preserved AND footprint shrinks.
    auto dv   = wrap(buf, shape<-1,-1>{2,3}, {3,1});           // dynamic_strides, dynamic extents
    auto dv32 = dv.reindex<cs::int32_t>();
    static_assert(cs::is_same<decltype(dv32)::index_type, cs::int32_t>::value, "dyn index narrowed");
    static_assert(sizeof(decltype(dv32)) < sizeof(decltype(dv)), "int32 dynamic view is smaller");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j) if (dv32(i,j) != dv(i,j)) return 5;

    // (5) index_fits: broadcast (stride 0 adds 0) and a flipped (negative-stride) view.
    if (!wrap(buf, shape<2,3>{}, {0,1}).index_fits<cs::int32_t>()) return 6;
    if (!wrap(buf, shape<3,4>{}).flip<0>().index_fits<cs::int32_t>()) return 7;

    // (6) index_fits boundaries: reach exactly INT32_MAX/MIN fits; one past doesn't.
    //     (wild strides — never dereferenced, only the reach is computed.)
    if (!wrap(buf, shape<2>{}, { 2147483647LL}).index_fits<cs::int32_t>()) return 8;   // == INT32_MAX
    if ( wrap(buf, shape<2>{}, { 2147483648LL}).index_fits<cs::int32_t>()) return 9;   // +1
    if (!wrap(buf, shape<2>{}, {-2147483648LL}).index_fits<cs::int32_t>()) return 10;  // == INT32_MIN (min-off binds)
    if ( wrap(buf, shape<2>{}, {-2147483649LL}).index_fits<cs::int32_t>()) return 11;  // -1
    if ( index_fits<cs::int32_t>(wrap(buf, shape<2>{}, {3000000000LL}))) return 12;     // >2^31 span -> false (free form)

    // ---- NEGATIVE strides: the signed-reach path must narrow & address correctly ----
    auto m = wrap(buf, shape<3,4>{});
    // (a) static flip -> folded strides<-4,1>; reindex keeps the negative static stride.
    auto fl = m.flip<0>();
    auto fl32 = fl.reindex<cs::int32_t>();
    static_assert(cs::is_same<decltype(fl32)::index_type, cs::int32_t>::value, "flip reindex int32");
    for (long i = 0; i < 3; ++i) for (long j = 0; j < 4; ++j) if (fl32(i,j) != fl(i,j)) return 13;
    if (fl32(0,0) != m(2,0)) return 14;                          // flipped row 0 == source last row
    // (b) RUNTIME reversed-step slice -> strides<4, dynamic_stride>: the dynamic
    //     negative stride must narrow int64->int32 (the mixed-ctor fix) and address right.
    auto rev = m(all, slice(none,none,-1));
    auto rev32 = rev.reindex<cs::int32_t>();
    for (long i = 0; i < 3; ++i) for (long j = 0; j < 4; ++j) if (rev32(i,j) != m(i, 3 - j)) return 15;
    // (c) both axes reversed.
    auto both = m.flip<0>().flip<1>();
    auto both32 = both.reindex<cs::int32_t>();
    for (long i = 0; i < 3; ++i) for (long j = 0; j < 4; ++j) if (both32(i,j) != m(2 - i, 3 - j)) return 16;
    // (d) index_fits: SIGNED int32 accepts a negative stride; UNSIGNED rejects it
    //     (min offset < 0 can't fit [0, uint32_max]) — the guard blocks a bad narrow.
    if (!fl.index_fits<cs::int32_t>())  return 17;
    if ( fl.index_fits<cs::uint32_t>()) return 18;

    // (7) per-axis reach fits individually, but the ACCUMULATED sum overflows int32:
    // two axes each with reach 1.5e9 / 1.0e9 (each well under INT32_MAX alone); their
    // SUM (2.5e9) exceeds it (#469 — regression test; the accumulation already got
    // this right, mirrors the equivalent anyrank case in test_anyrank_reindex.cpp).
    if (wrap(buf, shape<2,2>{}, {1500000000LL, 1000000000LL}).index_fits<cs::int32_t>()) return 19;

    // (8) PATHOLOGICAL strides near ~1e18 (#471 regression): the internal `long long`
    // accumulator must never overflow (UB) while deciding this — it must bail out
    // (return false) cleanly instead. Verified UB-free under -fsanitize=undefined.
    //   (a) two positive-stride axes: each axis's reach (6e18) alone fits `long long`,
    //       but their SUM (1.2e19) would overflow it during accumulation.
    if (wrap(buf, shape<2,2>{}, {6000000000000000000LL, 6000000000000000000LL})
            .index_fits<cs::int32_t>()) return 20;
    //   (b) a single axis whose (extent-1)*stride PRODUCT alone already overflows
    //       `long long` (3 * 4e18 = 1.2e19) — must be caught before the multiply.
    if (wrap(buf, shape<4>{}, {4000000000000000000LL}).index_fits<cs::int64_t>()) return 21;
    //   (c) the same two shapes, mirrored onto the negative-stride (`mino`) side.
    if (wrap(buf, shape<2,2>{}, {-6000000000000000000LL, -6000000000000000000LL})
            .index_fits<cs::int32_t>()) return 22;
    //   (d) mirrored onto the NEGATIVE-stride PRODUCT guard specifically (#474): a
    //       single axis whose (extent-1)*stride PRODUCT ALONE already overflows
    //       `long long` on the negative side (3 * -4e18 = -1.2e19), as opposed to
    //       (c)'s accumulation-only overflow — must be caught before the multiply.
    if (wrap(buf, shape<4>{}, {-4000000000000000000LL}).index_fits<cs::int64_t>()) return 23;

    // (9) WIDE UNSIGNED targets (#484). The two reach sides never interact, so each
    // accumulates and compares in its OWN 64-bit domain (positive: unsigned long long,
    // negative: long long). That makes the test exact for EVERY integral Idx2 up to 64
    // bits — uint64_t/size_t included, which used to be a compile error (#475).
    //   (a) a flipped (negative-stride) view still cannot fit an UNSIGNED target: its
    //       min offset is below 0, i.e. outside [0, UINT64_MAX].
    if (wrap(buf, shape<3,4>{}).flip<0>().index_fits<cs::uint64_t>()) return 24;
    //   (b) THE false negative this split fixes: a non-negative-strided reach of 1.2e19
    //       lies in (2^63-1, 2^64-1], so it genuinely FITS a uint64_t index — and no
    //       single `long long` accumulator could ever say so (the sum overflows it, and
    //       the old code bailed out with `false` right there). Only the unsigned
    //       positive-side accumulator can answer this one.
    if (!wrap(buf, shape<2,2>{}, {6000000000000000000LL, 6000000000000000000LL})
             .index_fits<cs::uint64_t>()) return 25;
    //       ...while the SAME view still doesn't fit int64_t (1.2e19 > 2^63-1), nor
    //       int32_t (case (8a) above): the answer moved only where it was wrong.
    if (wrap(buf, shape<2,2>{}, {6000000000000000000LL, 6000000000000000000LL})
            .index_fits<cs::int64_t>()) return 26;
    //   (c) the exact upper boundary: a reach of exactly UINT64_MAX fits, one past it
    //       does not (2*(2^63-1) + 1 == 2^64-1) — pins the accumulation guard's edge.
    if (!wrap(buf, shape<2,2,2>{}, {9223372036854775807LL, 9223372036854775807LL, 1LL})
             .index_fits<cs::uint64_t>()) return 27;
    if ( wrap(buf, shape<2,2,2>{}, {9223372036854775807LL, 9223372036854775807LL, 2LL})
            .index_fits<cs::uint64_t>()) return 28;
    //   (d) the UNSIGNED accumulator's own overflow guards — the #471 pair re-derived
    //       for a domain where overflow WRAPS rather than being UB (a wrapped value
    //       answers the query WRONGLY, so both are still checked before the fact).
    //       PRODUCT: (5-1) * 9e18 = 3.6e19 > UINT64_MAX, caught before the multiply.
    if (wrap(buf, shape<5>{}, {9000000000000000000LL}).index_fits<cs::uint64_t>()) return 29;
    //       ACCUMULATION: three axes of reach 9e18 — each product fits, the sum
    //       (2.7e19) does not, caught before the third add.
    if (wrap(buf, shape<2,2,2>{}, {9000000000000000000LL, 9000000000000000000LL,
                                   9000000000000000000LL}).index_fits<cs::uint64_t>()) return 30;

    // (10) reindex<uint64_t>() is now an ordinary (widening) retype: it compiles and
    // addresses identically in a DEBUG build and under -DNDEBUG/__CUDA_ARCH__ alike —
    // #480's build-mode asymmetry is gone at the root, since there is no longer a
    // static_assert for `_TNY_CHECK`'s unevaluated form to skip inconsistently.
    auto u64 = t.reindex<cs::uint64_t>();
    static_assert(cs::is_same<decltype(u64)::index_type, cs::uint64_t>::value, "reindex to uint64");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j) for (long k = 0; k < 4; ++k)
        if (u64(i,j,k) != t(i,j,k)) return 31;
    if (!t.index_fits<cs::size_t>()) return 32;               // the LP64 spelling, equally ordinary

    return 0;
}
