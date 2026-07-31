// #338: tuple-unpack indexing — `t(m)` where `m` is a SINGLE tuple-like index pack
// (cs::array / cs::tuple) carrying the whole index list, numpy's `x[(a,b,c)]`.
// It is pure packing sugar: the pack is unpacked and re-dispatched through the
// ordinary variadic entry point, so element/view/ellipsis dispatch, result TYPES and
// element identity must be IDENTICAL to writing the arguments out. Also on
// at()/uget()/uat(). Closes the loop with peel(...).enumerate(), whose multi-index
// is exactly a cs::array.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
#include <cuda/std/array>
#include <cuda/std/tuple>

using namespace tny;
namespace cs = cuda::std;

int main()
{
    double buf[24];
    for (int i = 0; i < 24; ++i) buf[i] = (double)i;
    auto t = wrap(buf, shape<4, 6>{});          // 4x6 row-major, value = 6*i + j

    // ---- element access via a cs::array of ints ----------------------------
    for (long i = 0; i < 4; ++i)
        for (long j = 0; j < 6; ++j) {
            cs::array<long, 2> m{ i, j };
            if (t(m) != t(i, j))                                  return 1;
            if (&t(m) != &t(i, j))                                return 2;   // same lvalue
            if (t(m) != (double)(6 * i + j))                      return 3;
        }
    static_assert(cs::is_same<decltype(t(cs::array<long,2>{})), double &>::value,
                  "an all-integer pack yields the element reference, like t(i,j)");

    // writes through the pack
    { cs::array<long, 2> m{ 2, 3 }; t(m) = -7.0; if (buf[2 * 6 + 3] != -7.0) return 4; t(m) = 15.0; }

    // negative indices wrap, exactly as in the variadic call (only meaningful when
    // the build wraps at all — -DTNY_NO_NEGATIVE_INDEX drops the wrap everywhere)
#ifndef TNY_NO_NEGATIVE_INDEX
    { cs::array<long, 2> m{ -1, -1 }; if (&t(m) != &t(3, 5)) return 5; }
#endif

    // ---- a homogeneous cs::tuple of ints, and a static Int<> pack ----------
    if (t(cs::make_tuple(1L, 2L)) != t(1, 2))                     return 6;
    if (t(cs::make_tuple(Int<1>(), Int<2>())) != t(1, 2))         return 7;
    static_assert(cs::is_same<decltype(t(cs::make_tuple(Int<1>(), Int<2>()))), double &>::value,
                  "a static pack is still an element access");

    // ---- mixed tuple: int + slice / all / none / ellipsis -------------------
    // int + all -> rank-1 row view, identical type and elements to t(1, all)
    {
        auto r  = t(cs::make_tuple(1L, all));
        auto r2 = t(1, all);
        static_assert(cs::is_same<decltype(r), decltype(r2)>::value, "pack view type == variadic view type");
        if (r.rank() != 1 || r.shape(0) != 6)                     return 8;
        for (long j = 0; j < 6; ++j) if (&r(j) != &r2(j))         return 9;
    }
    // all + range -> rank-2 strided view
    {
        auto s  = t(cs::make_tuple(all, slice(1, 4)));
        auto s2 = t(all, slice(1, 4));
        static_assert(cs::is_same<decltype(s), decltype(s2)>::value, "pack slice type == variadic slice type");
        if (s.rank() != 2 || s.shape(0) != 4 || s.shape(1) != 3)  return 10;
        for (long i = 0; i < 4; ++i) for (long j = 0; j < 3; ++j)
            if (&s(i, j) != &s2(i, j))                            return 11;
    }
    // compile-time range in a pack still folds its extent
    {
        auto s = t(cs::make_tuple(all, slice<1, 4>()));
        static_assert(decltype(s)::shape_type::static_extent(1) == 3, "compile-time slice folds through the pack");
        if (&s(0, 0) != &t(0, 1))                                 return 12;
    }
    // bare `none` (newaxis) in a pack
    {
        auto u  = t(cs::make_tuple(none, all, all));
        auto u2 = t(none, all, all);
        static_assert(cs::is_same<decltype(u), decltype(u2)>::value, "newaxis through the pack");
        if (u.rank() != 3 || u.shape(0) != 1)                     return 13;
        if (&u(0, 2, 3) != &t(2, 3))                              return 14;
    }
    // ellipsis in a pack (expands to `all`s, then re-dispatches)
    {
        auto e  = t(cs::make_tuple(1L, ellipsis));
        auto e2 = t(1, ellipsis);
        static_assert(cs::is_same<decltype(e), decltype(e2)>::value, "ellipsis through the pack");
        for (long j = 0; j < 6; ++j) if (&e(j) != &t(1, j))       return 15;
        // an ellipsis that fills every axis leaves an all-integer call -> element
        if (t(cs::make_tuple(ellipsis, 2L, 3L)) != t(2, 3))       return 16;
    }

    // ---- at() / uget() / uat() take the same pack --------------------------
    {
        cs::array<long, 2> m{ 1, 4 };
        static_assert(cs::is_same<decltype(t.at(m)), decltype(t.at(1, 4))>::value, "at(pack) type == at(i,j) type");
        if (t.at(m).item() != t(1, 4))                            return 17;
        if (t.uget(m) != t(1, 4))                                 return 18;
        if (&t.uget(m) != &t(1, 4))                               return 19;
        static_assert(cs::is_same<decltype(t.uat(m)), decltype(t.uat(1, 4))>::value, "uat(pack) type == uat(i,j) type");
        if (t.uat(m).item() != t(1, 4))                           return 20;
        t.at(m) = 42.0; if (t(1, 4) != 42.0)                      return 21;
        t.uat(m) = (double)(6 * 1 + 4);                                        // restore
        // uget takes a slicing pack too (unchecked view)
        auto r = t.uget(cs::make_tuple(1L, all));
        static_assert(cs::is_same<decltype(r), decltype(t.uget(1, all))>::value, "uget(pack) view type");
        if (&r(2) != &t(1, 2))                                    return 22;
    }

    // ---- a const tensor / const pack ---------------------------------------
    {
        const auto ct = t.view();
        cs::array<long, 2> m{ 0, 5 };
        const cs::array<long, 2> cm{ 0, 5 };
        if (ct(m) != t(0, 5))                                     return 23;
        if (ct(cm) != t(0, 5))                                    return 24;
    }

    // ---- rank-1: the collision case (a lone pack must NOT read as one axis) -
    {
        auto v = wrap(buf, shape<24>{});
        cs::array<long, 1> m{ 7 };
        static_assert(cs::is_same<decltype(v(m)), double &>::value,
                      "a rank-1 pack is an ELEMENT, not a full view");
        if (&v(m) != &v(7))                                       return 25;
    }

    // ---- rank-3, and a pack built from a peel enumerate multi-index --------
    {
        auto a = zeros<double>(shape<2, 3, 4>{}); a.iota_(0, 1);
        auto b = zeros<double>(shape<2, 3, 4>{});
        // THE motivating loop: write by coordinate, feeding the iterator's index()
        // (a cs::array) straight back into the destination's operator().
        long n = 0;
        for (auto [m, cell] : peel(a, axis<0, 1, 2>{}).enumerate()) {
            b(m) = double(cell) * 2.0;                            // <- t(m), pack-indexed write
            ++n;
        }
        if (n != 24)                                              return 26;
        for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j) for (long k = 0; k < 4; ++k)
            if (b(i, j, k) != a(i, j, k) * 2.0)                   return 27;

        // partial peel: m has the peeled rank, and (m, all) addresses the row
        auto c = zeros<double>(shape<2, 3, 4>{});
        for (auto [m, row] : peel(a, axis<0, 1>{}).enumerate())
            c(cs::make_tuple(m[0], m[1], all)).copy_(row);
        for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j) for (long k = 0; k < 4; ++k)
            if (c(i, j, k) != a(i, j, k))                         return 28;

        // and straight off the raw iterator's index()
        auto it = peel(a, axis<0, 1, 2>{}).begin();
        ++it; ++it;
        if (a(it.index()) != double(*it))                         return 29;
    }

    // ---- C++23 operator[] inherits the pack overload for free --------------
    // (no-op under C++17/20, like tests/test_subscript.cpp; real check in `make cxx23`)
#if defined(__cpp_multidimensional_subscript)
    {
        cs::array<long, 2> m{ 2, 5 };
        if (&t[m] != &t(2, 5))                                    return 30;
        auto r = t[cs::make_tuple(1L, all)];
        static_assert(cs::is_same<decltype(r), decltype(t(1, all))>::value, "t[pack] view type == t(pack) view type");
        if (&r(3) != &t(1, 3))                                    return 31;
    }
#endif

    return 0;
}
