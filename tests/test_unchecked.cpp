// Unchecked accessors (#91): uget / uat skip the negative-index wrap for runtime
// signed args (the per-call form of -DTNY_NO_NEGATIVE_INDEX). `uget` mirrors
// operator() in full — element / slice / ellipsis, one entry point.
// They must (a) agree byte-for-byte with the checked ops for NON-NEGATIVE indices,
// (b) produce the SAME result TYPE (static bounds still fold), and (c) genuinely
// drop the wrap for a runtime negative slice bound (observably, without UB).
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main()
{
    double buf[24];
    for (int i = 0; i < 24; ++i) buf[i] = (double)i;
    auto t = wrap(buf, shape<4, 6>{});        // 4x6 row-major, value = 6*i + j

    // ---- uget: non-negative index == operator(), same address --------------
    for (long i = 0; i < 4; ++i)
        for (long j = 0; j < 6; ++j) {
            if (t.uget(i, j) != t(i, j))                 return 1;
            if (&t.uget(i, j) != &t(i, j))               return 2;   // same lvalue
            if (t.uget(i, j) != (double)(6 * i + j))     return 3;
        }
    // uget writes through
    t.uget(2, 3) = -7.0; if (buf[2 * 6 + 3] != -7.0)     return 4;
    t.uget(2, 3) =  15.0;                                             // restore

    // static-index args go through the same path (integral_constant)
    if (t.uget(Int<1>(), Int<2>()) != t(1, 2))           return 5;

    // ---- uat: rank-0 view; item()/write; == at() ---------------------------
    if (t.uat(1, 4).item() != t(1, 4))                   return 6;
    static_assert(cs::is_same<decltype(t.uat(0, 0)), decltype(t.at(0, 0))>::value,
                  "uat result type == at result type");
    t.uat(0, 0) = 99.0; if (t(0, 0) != 99.0)             return 7;
    t.uat(0, 0) = 0.0;

    // ---- uat().add_<true>(): unchecked scatter-accumulate ------------------
    double before = t(3, 5);
    t.uat(3, 5).add_<true>(4.0); if (t(3, 5) != before + 4.0)   return 8;

    // ---- uget slice-form: non-negative / none bounds == operator() ---------
    {
        auto a = t(1, all);
        auto b = t.uget(1, all);
        static_assert(cs::is_same<decltype(a), decltype(b)>::value, "uget(all) type == slice type");
        if (a.extent(0) != b.extent(0)) return 9;
        for (long j = 0; j < 6; ++j) if (a(j) != b(j))   return 10;
    }
    {
        auto a = t(slice(1, 3), slice(none, 4));
        auto b = t.uget(slice(1, 3), slice(none, 4));
        static_assert(cs::is_same<decltype(a), decltype(b)>::value, "uget runtime-bound type == slice type");
        if (a.extent(0) != b.extent(0) || a.extent(1) != b.extent(1)) return 11;
        for (long i = 0; i < a.extent(0); ++i)
            for (long j = 0; j < a.extent(1); ++j) if (a(i, j) != b(i, j)) return 12;
    }
    // integer-drop arg: positive index unchanged
    {
        auto b = t.uget(2, slice(1, 5));
        if (b.rank() != 1 || b.extent(0) != 4) return 13;
        for (long j = 0; j < 4; ++j) if (b(j) != t(2, 1 + j)) return 14;
    }

    // ---- uget ellipsis form: expands like operator(), stays unchecked ------
    // (folds to element when what remains is all-integer, else a view; the type
    //  matches the checked ellipsis exactly.)
    {
        // ellipsis + integers -> element T& (same lvalue as operator())
        if (&t.uget(1, ellipsis, 2) != &t(1, 2)) return 20;
        static_assert(cs::is_same<decltype(t.uget(1, ellipsis, 2)), decltype(t(1, ellipsis, 2))>::value,
                      "uget ellipsis element type == operator() ellipsis");
        // ellipsis leaving an axis -> a view (drop the first axis, keep the rest)
        auto a = t(1, ellipsis);
        auto b = t.uget(1, ellipsis);
        static_assert(cs::is_same<decltype(a), decltype(b)>::value, "uget ellipsis view type == operator()");
        if (b.rank() != 1 || b.extent(0) != 6) return 21;
        for (long j = 0; j < 6; ++j) if (a(j) != b(j)) return 22;
    }

    // ---- static bounds still fold identically (folded extent + type) -------
    {
        auto a = t(slice<1, 4>(), all);
        auto b = t.uget(slice<1, 4>(), all);
        static_assert(cs::is_same<decltype(a), decltype(b)>::value, "compile-time slice folds the same");
        static_assert(decltype(b)::extents_type::static_extent(0) == 3, "folded static extent kept");
    }
    // a STATIC negative bound folds the SAME under uget as under operator()
    // (the _is_ic guard keeps static bounds wrapping regardless of the per-call
    // flag) — so their extents agree under any build (whether or not the global
    // -DTNY_NO_NEGATIVE_INDEX wraps static negatives).
    {
        auto a = t(slice<0, -1>(), all);
        auto b = t.uget(slice<0, -1>(), all);
        static_assert(cs::is_same<decltype(a), decltype(b)>::value, "static negative bound folds the same");
        if (a.extent(0) != b.extent(0)) return 15;
    }

#ifndef TNY_NO_NEGATIVE_INDEX
    // ---- the observable difference: a RUNTIME negative bound ---------------
    // (only meaningful when the default build wraps; -DTNY_NO_NEGATIVE_INDEX
    // makes the checked op behave like the unchecked one, erasing the contrast.)
    // checked slice wraps -1 -> n-1 (a real window); unchecked takes -1 as-is,
    // which the forward clamp collapses to an EMPTY axis. No UB either way.
    {
        long n = t.extent(1);                     // 6
        auto chk = t(0, slice(1, (long)-1));      // [1 .. 5) -> 4 elems (wrap)
        auto unw = t.uget(0, slice(1, (long)-1)); // [1 .. -1) -> empty (no wrap)
        if (chk.extent(0) != n - 2) return 16;
        if (unw.extent(0) != 0)     return 17;
    }

    // ---- regression: the CHECKED ops still wrap negatives ------------------
    if (t(-1, -1) != t(3, 5))                     return 18;   // operator() wraps
    if (t.at(-1, -1).item() != t(3, 5))           return 19;   // at() wraps
#endif

    return 0;
}
