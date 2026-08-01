// #463: dispatch_values — the product form of dispatch_value. One candidate list per
// parameter (`candidates<1,2,3>(d)`), one call instead of a nesting pyramid; `f` gets
// one integral_constant per parameter, in list order. Same per-parameter match test and
// the same "no match -> f not called, returns false" contract as dispatch_value, and an
// enum-typed runtime value dispatches without a hand static_cast.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

enum class bnd { zero, replicate, dct1, dct2, dst1, dst2, dft, nocheck };  // fastfields' `bound`
enum plain_e { plain_lo = 1, plain_hi = 2 };                               // unscoped enum too

// A lambda-free (device-safe style) functor, to prove the engine needs no lambda.
struct Rec3 {
    int * d; int * o; int * b;
    template <class D, class O, class B>
    void operator()(D dd, O oo, B bb) const { *d = (int)dd.value; *o = (int)oo.value; *b = (int)bb.value; }
};

int main() {
    // ---- the candidate list is a compile-time-typed carrier of a runtime value ----
    constexpr auto c123 = candidates<1,2,3>(2);
    static_assert(cs::is_same<decltype(c123), const candidates_t<1,2,3>>::value,
                  "candidates<Vs...>(v) -> candidates_t<Vs...>");
    static_assert(c123.value == 2, "the runtime value rides along (constexpr here)");
    static_assert(_is_candidates<candidates_t<1,2,3>>::value, "the dispatch_values guard");
    static_assert(!_is_candidates<int>::value, "...rejects a bare value");

    // ---- (1) one parameter: exactly dispatch_value -------------------------------
    int seen = -1;
    bool ok = dispatch_values([&](auto d){ seen = (int)d.value; }, candidates<1,2,3>(2));
    if (!ok || seen != 2) return 1;

    // ---- (2) three parameters, one call ------------------------------------------
    // f receives one integral_constant per list, in list order, each usable as a
    // template argument (the static_asserts below hold in EVERY instantiated arm).
    int D = 0, O = 0, B = 0; long ext = 0;
    ok = dispatch_values([&](auto d, auto o, auto b) {
            static_assert(cs::is_same<typename decltype(d)::value_type, int>::value, "Int<> per list");
            static_assert(decltype(d)::value >= 1 && decltype(d)::value <= 3, "list 0 candidates");
            static_assert(decltype(o)::value >= 0 && decltype(o)::value <= 3, "list 1 candidates");
            static_assert(decltype(b)::value >= 0 && decltype(b)::value <= 7, "list 2 candidates");
            auto t = local<double, shape<d.value>>();     // static extent from a runtime value
            ext = (long)t.extent(0);
            D = (int)d.value; O = (int)o.value; B = (int)b.value;
         },
         candidates<1,2,3>(3), candidates<0,1,2,3>(2), candidates<0,1,2,3,4,5,6,7>(5));
    if (!ok || D != 3 || O != 2 || B != 5 || ext != 3) return 2;

    // ...and with a plain functor (no lambda anywhere)
    D = O = B = -1;
    if (!dispatch_values(Rec3{&D,&O,&B}, candidates<1,2,3>(1), candidates<0,1>(0), candidates<4,5>(5)))
        return 3;
    if (D != 1 || O != 0 || B != 5) return 4;

    // ---- (3) enum-typed runtime values: no hand static_cast ----------------------
    bnd bm = bnd::dst1;                       // == 4
    int got = -1;
    ok = dispatch_values([&](auto b){ got = (int)b.value; }, candidates<0,1,2,3,4,5,6,7>(bm));
    if (!ok || got != 4) return 5;

    plain_e pe = plain_hi;                    // == 2 (unscoped enum)
    got = -1;
    ok = dispatch_values([&](auto p){ got = (int)p.value; }, candidates<1,2>(pe));
    if (!ok || got != 2) return 6;

    // an enum mixed with plain integers in the same call
    got = -1; int gd = -1;
    ok = dispatch_values([&](auto d, auto b){ gd = (int)d.value; got = (int)b.value; },
                         candidates<1,2,3>(2), candidates<0,1,2,3,4,5,6,7>(bnd::dft /* 6 */));
    if (!ok || gd != 2 || got != 6) return 7;

    // ---- (4) out-of-list, each parameter position independently ------------------
    // dispatch_value's contract: no match simply doesn't fire — no assert, no abort.
    int calls = 0;
    auto count = [&](auto...){ ++calls; };
    if (dispatch_values(count, candidates<1,2,3>(7), candidates<0,1>(0)))                     return 8;
    if (dispatch_values(count, candidates<1,2,3>(1), candidates<0,1>(9)))                     return 9;
    if (dispatch_values(count, candidates<1,2>(1), candidates<0,1>(9), candidates<0,1>(0)))   return 10;
    if (dispatch_values(count, candidates<>(0), candidates<0,1>(0)))                          return 11;
    if (calls != 0) return 12;

    // ---- (5) identical to the nested dispatch_value pyramid, over the whole grid --
    for (int d = 0; d <= 4; ++d) for (int o = -1; o <= 3; ++o) {
        int nd = -1, no = -1; bool nested = false;
        dispatch_value<1,2,3>(d, [&](auto DD) {
            nested = dispatch_value<0,1,2>(o, [&](auto OO) { nd = (int)DD.value; no = (int)OO.value; });
        });
        int pd = -1, po = -1;
        bool prod = dispatch_values([&](auto DD, auto OO) { pd = (int)DD.value; po = (int)OO.value; },
                                    candidates<1,2,3>(d), candidates<0,1,2>(o));
        if (prod != nested || pd != nd || po != no) return 13;
    }

    // ---- (6) only the matching combination runs (all are instantiated) -----------
    calls = 0;
    int hit_d = 0, hit_o = 0;
    if (!dispatch_values([&](auto d, auto o){ ++calls; hit_d = (int)d.value; hit_o = (int)o.value; },
                         candidates<1,2,3>(2), candidates<0,1,2,3>(3))) return 14;
    if (calls != 1 || hit_d != 2 || hit_o != 3) return 15;

    return 0;
}
