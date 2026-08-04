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

    return 0;
}
