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

    return 0;
}
