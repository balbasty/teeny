// peel_zip (#327): walk 2 or 3 broadcast-compatible tensors in lock-step, yielding
// a cs::tuple of views per step. A distinct name from peel (not an overload).
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

template <class TA, class TB>
double const_zip_sum(const TA & a, const TB & b) {
    // dependent-receiver context: exercises peel_zip's own template-arg deduction,
    // and the const overload, from inside a function template.
    double out = 0;
    for (auto [va, vb] : peel_zip<0>(a, b)) out += va(0) + vb(0);
    return out;
}

int main() {
    // ---- 2-tensor same-shape zip ---------------------------------------
    auto a = local<double, shape<3,4>>(); a.iota_(0.0, 1.0);
    auto b = local<double, shape<3,4>>(); b.iota_(100.0, 1.0);
    long n = 0;
    for (auto [va, vb] : peel_zip<0>(a, b)) {
        for (long j = 0; j < 4; ++j) if (vb(j) - va(j) != 100.0) return 1;
        ++n;
    }
    if (n != 3) return 2;

    // ---- 3-tensor same-shape zip: the triangle-vertex idiom -------------
    auto va3 = local<double, shape<4,3>>(); va3.iota_(0.0, 1.0);
    auto vb3 = local<double, shape<4,3>>(); vb3.iota_(100.0, 1.0);
    auto vc3 = local<double, shape<4,3>>(); vc3.iota_(200.0, 1.0);
    n = 0;
    for (auto [ta, tb, tc] : peel_zip<0>(va3, vb3, vc3)) {
        for (long j = 0; j < 3; ++j) { if (tb(j)-ta(j) != 100.0) return 3; if (tc(j)-ta(j) != 200.0) return 4; }
        ++n;
    }
    if (n != 4) return 5;

    // ---- broadcasting: an extent-1 axis stretches (numpy right-align) --
    auto A = local<double, shape<3,4>>(); A.iota_(0.0,1.0);
    auto B = local<double, shape<1,4>>(); B.iota_(1000.0,1.0);
    n = 0;
    for (auto [x, y] : peel_zip<0>(A, B)) { for (long j=0;j<4;++j) if (y(j) != B(0,j)) return 6; ++n; }
    if (n != 3) return 7;

    // ---- broadcasting: different RANKS (right-aligned) ------------------
    auto C = local<double, shape<4>>(); C.iota_(5000.0, 1.0);
    n = 0;
    for (auto [x, y] : peel_zip<0>(A, C)) { for (long j=0;j<4;++j) if (y(j) != C(j)) return 8; ++n; }
    if (n != 3) return 9;

    // ---- value form: peel_zip(a, b, axis<Axis>{}) == peel_zip<Axis>(a, b) ---
    n = 0;
    for (auto [x, y] : peel_zip(A, B, axis<0>{})) ++n;
    if (n != 3) return 10;
    n = 0;
    for (auto [x, y, z] : peel_zip(va3, vb3, vc3, axis<0>{})) ++n;
    if (n != 4) return 11;

    // ---- enumerate: (multi_index, tuple) per step, same shape as peel's ----
    n = 0;
    for (auto [m, cell] : peel_zip<0>(A, B).enumerate()) {
        if (m[0] != n) return 12;
        ++n;
    }
    if (n != 3) return 13;

    // ---- subrange: chunked sweep, incl. straight off a temporary -----------
    n = 0;
    for (auto cell : peel_zip<0>(A, B).subrange(1, 3)) { (void)cell; ++n; }
    if (n != 2) return 14;
    n = 0;
    for (auto item : peel_zip<0>(A, B).enumerate().subrange(1, 3)) { (void)item; ++n; }
    if (n != 2) return 15;

    // ---- mutable write-through -------------------------------------------
    auto dest = local<double, shape<3,4>>(); dest.zero_();
    for (auto [x, d] : peel_zip<0>(A, dest)) d.copy_(x);
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (dest(i,j) != A(i,j)) return 16;

    // ---- const overload, called from a dependent (template) receiver ------
    double s = 0; for (long i=0;i<3;++i) s += a(i,0) + b(i,0);
    if (const_zip_sum(a, b) != s) return 17;

    // ---- dynamic-shape operand mixed with static ---------------------------
    auto d = owned<double, shape<-1,4>>(shape<-1,4>{3}); d.iota_(0.0, 1.0);
    n = 0;
    for (auto [x, y] : peel_zip<0>(a, d)) { (void)x; (void)y; ++n; }
    if (n != 3) return 18;

    // ---- negative axis wraps (library-wide signed-axis convention) --------
    n = 0;
    for (auto [x, y] : peel_zip<-2>(a, b)) { (void)x; (void)y; ++n; }   // axis -2 == axis 0 (rank 2)
    if (n != 3) return 19;

    // ---- peeling EVERY axis -> rank-0 cells (pure element-wise lock-step) ---
    n = 0;
    double sum = 0;
    for (auto [x, y] : peel_zip<0,1>(a, b)) { sum += static_cast<double>(x) + static_cast<double>(y); ++n; }
    if (n != 12) return 20;
    double expect = 0; for (long i=0;i<3;++i) for (long j=0;j<4;++j) expect += a(i,j) + b(i,j);
    if (sum != expect) return 21;

    /* ================================================================== *
     *  Mixed INDEX SIGNEDNESS across the zipped operands (#362).
     *
     *  `peel_zip` picked its decode type with `cs::common_type_t`, which
     *  applies the usual arithmetic conversions -- so at EQUAL width the
     *  UNSIGNED type wins (`common_type_t<int32_t,uint32_t>` is `uint32_t`).
     *  A flipped operand's stride of -1 then zero-extended to 4294967295,
     *  and both halves of that one type broke:
     *    (a) the OFFSET DECODE -- each cell's base pointer landed ~4G
     *        elements past the buffer (a hard SEGV under ASan); and
     *    (b) the CELL's OWN type -- a `peel_zip` cell is a VIEW of its
     *        operand, not a fresh allocation, so a kept axis's stride can
     *        legitimately be negative and an unsigned index type cannot
     *        represent it at all.
     *  It now decodes in math.h's signedness-aware `_offset_int_t`, the same
     *  rule every other multi-tensor engine uses (bzip_/zipreduce_decode_/
     *  scalo_/unaryo_/allclose_ -- see tests/test_broadcast_index.cpp).
     *  All offsets below are hand-computed.
     * ================================================================== */
    using zidx = cs::int64_t;                       // the shape<> default index type
    // The type a range decodes in, spelled once (the range type is internal, but
    // its `Idx` member is exactly what this issue is about).
    #define ZIP_IDX(...) typename cs::remove_reference<decltype(__VA_ARGS__)>::type::Idx

    // ---- (22) the reported repro: a FLIPPED int32-indexed operand zipped with
    //           an equal-width UNSIGNED-indexed one. This is the pair
    //           `common_type_t` resolved to `uint32_t`.
    {
        double da[6] = {0,1,2,3,4,5};
        double db[6] = {10,20,30,40,50,60};
        auto a  = wrap(da, shape_as<cs::int32_t, 6>{});
        auto af = a.flip<0>();                      // stride -1, base at da+5
        auto bu = wrap(db, shape_as<unsigned int, 6>{});
        static_assert(cs::is_signed<ZIP_IDX(peel_zip<0>(af, bu))>::value,
                      "mixed-signedness zip must decode in a SIGNED type");
        // the flipped operand walks 5,4,3,2,1,0 while the unsigned one walks 0..5
        const long long want[6] = {5,4,3,2,1,0};
        long k = 0; double s = 0;
        for (auto [x, y] : peel_zip<0>(af, bu)) {
            if (x.data() - da != want[k]) return 22;
            if (y.data() - db != k)       return 23;
            static_assert(cs::is_signed<decltype(x)::index_type>::value,
                          "the cell's own index type must be signed too");
            s += static_cast<double>(x) * static_cast<double>(y);
            ++k;
        }
        if (k != 6) return 24;
        // 5*10 + 4*20 + 3*30 + 2*40 + 1*50 + 0*60
        if (s != 350.0) return 25;

        // ---- (26) the MIRROR: same pair, operand order swapped (the decode type
        //           is a property of the SET, so neither order may crash).
        k = 0; s = 0;
        for (auto [y, x] : peel_zip<0>(bu, af)) {
            if (x.data() - da != want[k]) return 26;
            s += static_cast<double>(x) * static_cast<double>(y);
            ++k;
        }
        if (k != 6 || s != 350.0) return 27;

        // ---- (28) ...and through enumerate()/subrange(), which share the decode.
        k = 0;
        for (auto it : peel_zip<0>(af, bu).enumerate()) {
            if (it.index[0] != k) return 28;
            if (cs::get<0>(it.cell).data() - da != want[k]) return 29;
            ++k;
        }
        if (k != 6) return 30;
        k = 0;
        for (auto cell : peel_zip<0>(af, bu).subrange(2, 5)) {
            if (cs::get<0>(cell).data() - da != want[2 + k]) return 31;
            ++k;
        }
        if (k != 3) return 32;
    }

    // ---- (33) 3-tensor form: `_offset_int_t` is variadic, so the rule is stated
    //           over the whole participant SET -- one unsigned operand anywhere in
    //           the zip used to poison the decode for every other operand.
    {
        double d0[4] = {1,2,3,4};
        double d1[4] = {10,20,30,40};
        double d2[4] = {100,200,300,400};
        auto f0 = wrap(d0, shape_as<cs::int32_t, 4>{}).flip<0>();   // 4,3,2,1
        auto u1 = wrap(d1, shape_as<unsigned int, 4>{});            // 10,20,30,40
        auto s2 = wrap(d2, shape_as<zidx, 4>{});                    // 100,200,300,400
        static_assert(cs::is_signed<ZIP_IDX(peel_zip<0>(f0, u1, s2))>::value,
                      "3-tensor mixed-signedness zip must decode in a SIGNED type");
        const long long want[4] = {3,2,1,0};
        long k = 0; double s = 0;
        for (auto [x, y, z] : peel_zip<0>(f0, u1, s2)) {
            if (x.data() - d0 != want[k]) return 33;
            if (y.data() - d1 != k)       return 34;
            if (z.data() - d2 != k)       return 35;
            s += static_cast<double>(x) * static_cast<double>(y) * static_cast<double>(z);
            ++k;
        }
        if (k != 4) return 36;
        // 4*10*100 + 3*20*200 + 2*30*300 + 1*40*400
        if (s != 4000.0 + 12000.0 + 18000.0 + 16000.0) return 37;
    }

    // ---- (38) the CELL's OWN metadata, independent of the base offset: a KEPT
    //           axis whose stride is DYNAMIC and NEGATIVE. Here the PEELED axis's
    //           stride is positive, so every base pointer was already right --
    //           what broke was the cell carrying 4294967295 as its own stride.
    {
        double m[12], u[12];
        for (int i = 0; i < 12; ++i) { m[i] = i; u[i] = 100 + i; }
        // runtime strides {4,-1} over a (3,4) window based at m+3: row i reads
        // m[3+4i], m[2+4i], m[1+4i], m[0+4i] -- i.e. each row reversed.
        auto rev = wrap(m + 3, shape_as<cs::int32_t, 3, 4>{}, {4, -1});
        auto uu  = wrap(u, shape_as<unsigned int, 3, 4>{});
        long i = 0;
        for (auto [x, y] : peel_zip<0>(rev, uu)) {
            static_assert(cs::is_signed<decltype(x)::index_type>::value,
                          "a cell over a negative-strided kept axis needs a signed index type");
            if (x.data() - m != 3 + 4 * i) return 38;
            if (x.stride(0) != -1)         return 39;
            for (long j = 0; j < 4; ++j) if (x(j) != double(3 + 4 * i - j)) return 40;
            for (long j = 0; j < 4; ++j) if (y(j) != double(100 + 4 * i + j)) return 41;
            ++i;
        }
        if (i != 3) return 42;
    }

    // ---- (43) CONTROLS that must NOT move. `common_type_t` was never wrong on
    //           WIDTH, so an all-signed or all-unsigned zip decoded correctly
    //           before and must decode in the very same type now -- only the
    //           mixed case changes. (Element identity across every same-signedness
    //           configuration was probed old-vs-new alongside this test.)
    {
        double p[12], q[12];
        for (int i = 0; i < 12; ++i) { p[i] = i; q[i] = 100 + i; }
        auto s32 = wrap(p, shape_as<cs::int32_t, 3, 4>{});
        auto s64 = wrap(q, shape_as<zidx,        3, 4>{});
        auto u32 = wrap(p, shape_as<unsigned int,   3, 4>{});
        auto u64 = wrap(q, shape_as<cs::uint64_t,   3, 4>{});
        // all SIGNED -> the widest of them (equal width keeps it; mixed width widens)
        static_assert(cs::is_same<ZIP_IDX(peel_zip<0>(s32, s32)), cs::int32_t>::value, "int32+int32");
        static_assert(cs::is_same<ZIP_IDX(peel_zip<0>(s32, s64)), zidx>::value,        "int32+int64");
        static_assert(cs::is_same<ZIP_IDX(peel_zip<0>(s64, s32)), zidx>::value,        "int64+int32");
        // all UNSIGNED -> likewise the widest (a negative stride cannot arise:
        // teeny's negative-stride views require a signed index type)
        static_assert(cs::is_same<ZIP_IDX(peel_zip<0>(u32, u32)), unsigned int>::value,  "uint32+uint32");
        static_assert(cs::is_same<ZIP_IDX(peel_zip<0>(u32, u64)), cs::uint64_t>::value,  "uint32+uint64");
        // MIXED -> signed, and wide enough for the unsigned side's whole range
        static_assert(cs::is_same<ZIP_IDX(peel_zip<0>(s32, u32)), cs::int64_t>::value,   "int32+uint32");
        static_assert(cs::is_same<ZIP_IDX(peel_zip<0>(s64, u32, s32)), cs::int64_t>::value, "3-tensor mixed");
        // and the values still agree with the operands' own element order
        long k = 0;
        for (auto [x, y] : peel_zip<0>(s32, s64)) {
            for (long j = 0; j < 4; ++j) { if (x(j) != double(4*k+j)) return 43;
                                           if (y(j) != double(100+4*k+j)) return 44; }
            ++k;
        }
        if (k != 3) return 45;
        k = 0;
        for (auto [x, y] : peel_zip<0>(u32, u64)) {
            for (long j = 0; j < 4; ++j) { if (x(j) != double(4*k+j)) return 46;
                                           if (y(j) != double(100+4*k+j)) return 47; }
            ++k;
        }
        if (k != 3) return 48;
    }
    #undef ZIP_IDX

    return 0;
}
