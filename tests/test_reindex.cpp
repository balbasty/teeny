// int32 offsets, PR A (#115 / #163): reindex<Idx2>() is a no-copy, layout-preserving
// retype of the offset index width; index_fits<Idx2>() is the signed-reach guard.
// A static strides<...> pack is unchanged; dynamic strides/extents narrow to Idx2 and
// the by-value footprint shrinks. Element identity vs the source is preserved.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

// records the dispatched offset width (bytes) of whatever `dispatch_index` handed over
struct RecIdxW { int * width; template <class V> void operator()(const V & v) {
    *width = static_cast<int>(sizeof(typename V::index_type));
}};

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

    // (11) RAW EXTENTS/STRIDES ABOVE `long long` (#486). A view's own index type may
    // be UNSIGNED 64-bit (`shape_as<uint64_t, ...>`), so a raw extent or stride can
    // exceed `LLONG_MAX`. The reach test used to `static_cast` each one to
    // `long long` first, wrapping it NEGATIVE — an extent then read as "size <= 1,
    // no reach" and the whole view looked narrowable. Each value is now only
    // converted into a domain it is provably representable in.
    //   (a) THE false positive: extent 2^63+2, stride 1 -> a true reach of 2^63+1,
    //       which obviously does not fit int32 (it used to answer `true`).
    using UE = shape_as<cs::uint64_t, -1>;
    auto ubig = wrap(buf, UE{9223372036854775810ULL}, {cs::uint64_t(1)});
    if (ubig.index_fits<cs::int32_t>()) return 33;
    //       ...and the SAME reach genuinely fits a uint64_t index — now computed
    //       exactly (the positive reach accumulates in an unsigned 64-bit domain),
    //       not arrived at via an extent that wrapped to "no reach at all".
    if (!ubig.index_fits<cs::uint64_t>()) return 34;
    //   (b) the same hazard on a STRIDE: 2^63+2 is a large POSITIVE stride in an
    //       unsigned index type; the old cast read it as a negative one and
    //       accumulated it on the wrong (min) side.
    auto usbig = wrap(buf, UE{2}, {cs::uint64_t(9223372036854775810ULL)});
    if (usbig.index_fits<cs::int32_t>())   return 35;
    if (usbig.index_fits<cs::int64_t>())   return 36;   // 2^63+2 > INT64_MAX
    if (!usbig.index_fits<cs::uint64_t>()) return 37;   // ...but does fit uint64
    //   (c) ordinary geometries in an unsigned index type are untouched.
    auto usmall = wrap(buf, shape_as<cs::uint64_t, -1, -1>{2, 3}, {cs::uint64_t(3), cs::uint64_t(1)});
    if (!usmall.index_fits<cs::int32_t>()) return 38;
    if (!usmall.index_fits<cs::uint8_t>()) return 39;   // reach 4
    //   (d) an unsigned extent straddling the `long long` edge, with the answer
    //       hinging on an EXACT `e-1`. At stride 2 the reach is `2*(e-1)`, so
    //       e == 2^63 reaches 2^64-2 (fits uint64) while e == 2^63+1 reaches 2^64
    //       (does not) — nothing but an exact `e-1` separates them. uint64 is also
    //       the only target that can hold either extent VALUE; against int64 both are
    //       rejected on the extent half of the predicate (#489, section (12) below),
    //       which is why the same pair at unit stride now answers `false` there.
    if (!wrap(buf, UE{9223372036854775808ULL}, {cs::uint64_t(2)}).index_fits<cs::uint64_t>()) return 40;
    if ( wrap(buf, UE{9223372036854775809ULL}, {cs::uint64_t(2)}).index_fits<cs::uint64_t>()) return 41;
    if ( wrap(buf, UE{9223372036854775808ULL}, {cs::uint64_t(1)}).index_fits<cs::int64_t>())  return 42;
    //   (e) a stride-0 axis of huge extent has no REACH — but its extent value must
    //       still survive the narrowing, so this is rejected on the extent half of
    //       the predicate (see (12) below, #489).
    if (wrap(buf, UE{9223372036854775810ULL}, {cs::uint64_t(0)}).index_fits<cs::int32_t>()) return 43;
    //       ...and is accepted for a target that CAN hold the extent (uint64).
    if (!wrap(buf, UE{9223372036854775810ULL}, {cs::uint64_t(0)}).index_fits<cs::uint64_t>()) return 44;

    // (12) EXTENT VALUES, not just offsets (#489). `index_fits<Idx2>()` is the gate on
    // `reindex<Idx2>()`, which narrows the EXTENTS as well as the offsets — and a
    // narrowed extent is the loop bound everywhere downstream (`numel()`, the peel
    // odometer, every kernel loop, the hardened bounds checks). A stride-0 broadcast
    // axis has no reach at all, so the offset test alone waved a huge extent through
    // and `reindex` then truncated it silently. Both halves are now ONE predicate.
    //   (a) plain truncation: extent 300 does not fit int8_t (it read back as 44).
    if (wrap(buf, shape<-1,2>{300, 2}, {0, 1}).index_fits<cs::int8_t>()) return 45;
    //   (b) THE realistic case: a broadcast axis of extent 2^31+5 truncated to a large
    //       NEGATIVE int32_t (-2147483643), so every `for (i = 0; i < e; ++i)` ran ZERO
    //       iterations and `numel()` went negative — silently absent work, no crash and
    //       nothing out of bounds for a sanitizer to catch, at exactly the GPU-narrowing
    //       boundary this family exists to protect.
    if (wrap(buf, shape<-1,2>{2147483653LL, 2}, {0, 1}).index_fits<cs::int32_t>()) return 46;
    //   (c) the negative-stride exact-boundary window: extent `e` with `e-1 ==
    //       |int8_t::min()|` reaches exactly -128, which DOES fit int8_t, while the
    //       extent 129 itself does not — only the extent half can reject this one.
    if (wrap(buf, shape<-1>{129}, {-1}).index_fits<cs::int8_t>()) return 47;
    //   (d) POSITIVE CONTROLS — a broadcast axis whose extent DOES fit is not
    //       over-rejected; the predicate moved only where it was wrong. The boundary is
    //       exact (127 fits int8_t, 128 doesn't) and target-signedness-aware.
    if (!wrap(buf, shape<-1,2>{127, 2}, {0, 1}).index_fits<cs::int8_t>())  return 48;
    if ( wrap(buf, shape<-1,2>{128, 2}, {0, 1}).index_fits<cs::int8_t>())  return 49;
    if (!wrap(buf, shape<-1,2>{128, 2}, {0, 1}).index_fits<cs::uint8_t>()) return 50;
    if (!wrap(buf, shape<-1>{127}, {-1}).index_fits<cs::int8_t>())         return 51;  // reach -126
    //   (e) a NEGATIVE extent (only reachable off a corrupt/adversarial import, whose
    //       index type is signed) is representable in a signed target and not in an
    //       unsigned one — the extent check answers at the target's own signedness.
    if (!wrap(buf, shape<-1>{-5}, {1}).index_fits<cs::int32_t>()) return 52;
    if ( wrap(buf, shape<-1>{-5}, {1}).index_fits<cs::uint32_t>()) return 53;
    //   (f) `dispatch_index` inherits the widened gate for free — the extent-300 case
    //       takes the WIDE arm rather than narrowing into a truncated extent.
    int w = 0;
    dispatch_index<cs::int8_t>(wrap(buf, shape<-1,2>{300, 2}, {0, 1}), RecIdxW{&w});
    if (w != 8) return 54;
    w = 0;                              // ...while a fitting one still narrows
    dispatch_index<cs::int8_t>(wrap(buf, shape<-1,2>{4, 2}, {0, 1}), RecIdxW{&w});
    if (w != 1) return 55;
    //   (g) a STATIC extent too large for the target is caught at COMPILE time instead
    //       (`_reindex_extents`, alias.h): `wrap(buf, shape<300,2>{}, strides<0,1>{})
    //       .reindex<cs::int8_t>()` is now a static_assert failure — it used to compile
    //       clean and produce an `extents<int8_t, 300>` whose `extent(0)` read 44,
    //       which mdspan's own static-extent mandate forbids. A static extent that DOES
    //       fit is untouched:
    auto sfit = wrap(buf, shape<3,2>{}, strides<0,1>{}).reindex<cs::int8_t>();
    static_assert(cs::is_same<decltype(sfit)::index_type, cs::int8_t>::value, "static extent narrows");
    static_assert(decltype(sfit)::extents_type::static_extent(0) == 3, "extent value kept");
    if (sfit.extent(0) != 3) return 56;

    // (13) POSITIVE EQUALITY-BOUNDARY CONTROL (#492). Section (11)(d) above restored
    // #486's `e-1`-exactness sensitivity after #490 widened the contract and flipped
    // the original int64 case to `false` (extent 2^63 itself doesn't fit int64_t, even
    // though its reach did) — but that re-expression didn't bring back an equivalent
    // POSITIVE control: a case where the reach lands EXACTLY on Idx2::max() and the
    // extent VALUE also fits, so the whole predicate answers `true`. Here the exact
    // reach comes from the STRIDE rather than from `e-1`, so the extent (2) itself
    // trivially fits int64_t: reach = (2-1)*INT64_MAX == INT64_MAX exactly.
    if (!wrap(buf, shape<-1>{2}, {9223372036854775807LL}).index_fits<cs::int64_t>()) return 57;
    //     ...one more axis's worth of reach (extent 2, stride 1) pushes the ACCUMULATED
    //     reach to INT64_MAX+1 — one past the boundary, `false`.
    if (wrap(buf, shape<-1,-1>{2,2}, {9223372036854775807LL, 1LL}).index_fits<cs::int64_t>()) return 58;

    // (14) NEGATIVE EQUALITY-BOUNDARY CONTROL (#494). The mirror of (13) on the `mino`
    // side: a reach landing EXACTLY on `Idx2::min()`, with a fitting extent, also
    // answers `true`. The nearest existing case ((12)(c) above) hits reach -128 but is
    // rejected on the EXTENT half of the predicate (129 doesn't fit int8_t) — it never
    // exercises the reach-equality half. Here the extent (2) trivially fits int64_t, so
    // only the reach-equality half is in play: reach = (2-1)*INT64_MIN == INT64_MIN
    // exactly. (`INT64_MIN` is spelled `-9223372036854775807LL - 1` — the positive
    // magnitude 9223372036854775808 doesn't fit `long long`, so the direct literal
    // would misparse; this form is exact and portable.)
    constexpr cs::int64_t neg_llmin = -9223372036854775807LL - 1;
    if (!wrap(buf, shape<-1>{2}, {neg_llmin}).index_fits<cs::int64_t>()) return 59;
    //     ...one more axis's worth of negative reach (extent 2, stride -1) pushes the
    //     ACCUMULATED reach to INT64_MIN-1 — one past the floor, `false`.
    if (wrap(buf, shape<-1,-1>{2,2}, {neg_llmin, -1LL}).index_fits<cs::int64_t>()) return 60;

    return 0;
}
