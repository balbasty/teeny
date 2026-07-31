// The `into(dest)` EXACT-SHAPE guard, now one helper (#363) — and, since #434, its
// fourth caller: the axis-scoped `normalize<Axes...>(a, into(y))`.
//
// A SINGLE-SOURCE producer -- a scalar-rhs op (`a.mul(2.0, into(y))`), a unary op
// (`exp(a, into(y))`), and `scan(t, init, f, into(y))` -- takes its loop BOUNDS
// from the source and writes through the DEST's strides, so the two shapes must
// agree in every axis or the write runs off the end of the destination (#357).
// There is nothing to broadcast on either side: unlike the tensor-rhs family, an
// extent 1 against an extent n is a plain mismatch here, not a stretch.
//
// That guard used to be written out three times -- `_md::scalo_`, `_md::unaryo_`
// and `scan`'s `into(dest)` form -- and `scan`'s copy hand-rolled its runtime loop
// comparing extents as `long`. `long` is 32 bits under Windows' LLP64 while
// teeny's default index type is `int64_t`, so on that platform it TRUNCATED any
// extent above 2^31 and a source of extent 2^32+5 compared EQUAL to a dest of
// extent 5. The math engines' copy cast to `cs::size_t` and did not. All three now
// call `_md::check_into_same_shape`, whose runtime comparison type
// (`_md::ext_cmp_t`) is at least 64 bits on EVERY platform.
//
// SCOPE, honestly: that `long` truncation is a real TYPE bug, but it was not
// letting a silent out-of-bounds write through, and #363 framed this as
// maintainability for good reason ("No correctness bug in any of the three copies
// as they stand"). Two things stood behind `scan`'s guard already. (a) `scan`'s
// very next statement is `out.dest.copy_(t)`, and `copy_`'s own per-axis extent
// check inside `bzip_` runs in `_offset_int_t` (64-bit here, whatever `long` is),
// so it rejects a truncating pair independently -- with `scan`'s guard call
// deleted outright, both the 8x8-into-2x2 and the 2^32+5-into-5 cases below still
// abort, just with `copy_`'s broadcast wording instead of the `into(dest)` one.
// (b) `_TNY_CHECK` is `assert`-based, so none of this exists under NDEBUG anyway.
// The outcome of the fix is therefore "on one platform (Windows, debug builds) a
// worse diagnostic becomes the right one", not "an OOB write was prevented".
//
// The one shape pair where `scan`'s OWN guard is the only thing standing is a
// SOURCE extent of 1 against a DEST extent n > 1: `bzip_`'s check is
// broadcast-aware (`ae[r] == ce[r] || ae[r] == 1`), so it happily STRETCHES the
// source there, while `into(dest)` requires an exact match and must reject it.
// That is what pins the `scan` call site below.
//
// This pins:
//   - the comparison type's width and signedness, and that it distinguishes the
//     exact pair that the `long` spelling conflated (compile time, so it holds on
//     every platform even though only LLP64 could observe the difference);
//   - a correctly-shaped `into(dest)` still works for all three producers, on
//     static and dynamic shapes alike;
//   - a MIS-SHAPED dest aborts for all three (the guard is wired in at each of the
//     three consolidated call sites, not just at two of them) -- including the
//     broadcastable source-extent-1 pair, which is the only one of these that
//     actually FAILS if `scan`'s call to the helper is removed;
//   - and that the large-extent mismatch aborts too;
//   - and (#434) that the axis-scoped `normalize<Axes...>(a, into(y))` holds the
//     SAME exact-shape rule as the whole-tensor form, on every spelling: correct
//     shapes unaffected, a dynamic mismatch aborts, a static one no longer
//     compiles (the repros are spelled out below `main`).
// The aborts are only expected when _TNY_CHECK is live (host, non-NDEBUG).
#include <teeny/teeny.h>
#include <cstdio>
#include <cstdint>
#ifndef NDEBUG
#include <unistd.h>
#include <sys/wait.h>
#include <csignal>
#endif

using namespace tny;
namespace cs = cuda::std;

struct sum_op { _TNY_API double operator()(double carry, double x) const { return carry + x; } };

// The pair the old `long` comparison conflated on LLP64: 2^32 + 5 and 5.
static constexpr long long BIG = (1LL << 32) + 5;   // 4294967301
static constexpr long long SMALL = 5;

// ---- the comparison type itself ------------------------------------------
// Width and signedness, spelled as a property rather than as a name, so swapping
// `ext_cmp_t` back to `long` (32-bit on LLP64) or to `cs::size_t` (32-bit on a
// 32-bit host) fails HERE rather than silently on one CI platform.
static_assert(sizeof(_md::ext_cmp_t) >= 8,
              "the into(dest) extent comparison must be at least 64 bits on every platform "
              "(teeny's default index type is int64_t)");
static_assert(cs::is_unsigned<_md::ext_cmp_t>::value,
              "extents are non-negative by construction; an unsigned comparison type sidesteps "
              "signed/unsigned mismatch between two differently-indexed participants");
// The truncation the old spelling suffered, demonstrated with a fixed-width type
// so every platform sees what LLP64 saw...
static_assert(static_cast<int32_t>(BIG) == static_cast<int32_t>(SMALL),
              "2^32+5 and 5 are indistinguishable in 32 bits -- this is the truncation");
// ...and the shared comparison telling them apart regardless.
static_assert(!_md::ext_eq(BIG, SMALL), "ext_eq must not truncate an extent above 2^31");
static_assert(!_md::ext_eq(SMALL, BIG), "...in either argument order");
static_assert(_md::ext_eq(BIG, BIG), "equal extents above 2^31 still compare equal");
static_assert(_md::ext_eq(0, 0) && !_md::ext_eq(0, 1), "and the ordinary small cases");

int main() {
    // ============ correctly-shaped into(dest): all three producers ==========
    // Static shapes (the static half of the guard is a no-op here: shapes agree).
    auto s4 = local<double, shape<4>>(); s4.iota_(1.0, 1.0);       // 1,2,3,4
    auto y4 = local<double, shape<4>>(); y4.zero_();
    s4.mul(2.0, into(y4));                                          // scalar rhs -> scalo_
    for (long i = 0; i < 4; ++i) if (y4(i) != 2.0*(i+1)) return 1;
    neg(s4, into(y4));                                              // unary      -> unaryo_
    for (long i = 0; i < 4; ++i) if (y4(i) != -1.0*(i+1)) return 2;
    y4.zero_();
    auto & ref = scan<0>(s4, 0.0, sum_op{}, into(y4));               // scan
    const double pre[4] = {1,3,6,10};
    for (long i = 0; i < 4; ++i) if (y4(i) != pre[i]) return 3;
    if (&ref != &y4) return 4;

    // Dynamic shapes (the runtime half of the guard is what runs here).
    auto ds = zeros<double>(shape<-1>{4}); ds.iota_(1.0, 1.0);
    auto dy = zeros<double>(shape<-1>{4});
    ds.mul(2.0, into(dy));
    for (long i = 0; i < 4; ++i) if (dy(i) != 2.0*(i+1)) return 5;
    exp(ds, into(dy));
    if (dy(0) <= 2.7 || dy(0) >= 2.8) return 6;                      // e^1
    dy.zero_();
    scan<0>(ds, 0.0, sum_op{}, into(dy));
    for (long i = 0; i < 4; ++i) if (dy(i) != pre[i]) return 7;

    // A dest whose INDEX TYPE differs from the source's: the comparison spans both
    // (this is why it is made in a type of its own rather than in either operand's).
    auto own = zeros<double>(shape<-1>{4});
    auto n32 = own.reindex<int32_t>();
    ds.mul(3.0, into(n32));
    for (long i = 0; i < 4; ++i) if (n32(i) != 3.0*(i+1)) return 8;
    n32.zero_();
    scan<0>(ds, 0.0, sum_op{}, into(n32));
    for (long i = 0; i < 4; ++i) if (n32(i) != pre[i]) return 9;

    // ---- the FOURTH caller: axis-scoped normalize (#434) --------------------
    // `normalize<Axes...>(a, into(y))` divides by a keepdim TENSOR, so it forwards
    // to the BROADCASTING engine -- whose dest gate (`bc_static_ok_dest`, #361)
    // asks "each operand axis equals the dest's extent OR IS 1", strictly weaker
    // than what this producer promises (its result IS `a`'s shape, since only the
    // divisor is reduced). So it states the exact rule itself, at the public
    // function, by calling the same shared helper. Correct shapes are unaffected:
    // rows of `nrm` are (3,4) and (6,8) -> each unit row is (0.6,0.8).
    auto nrm = local<double, shape<2,2>>();
    nrm(0,0)=3; nrm(0,1)=4; nrm(1,0)=6; nrm(1,1)=8;
    auto nout = local<double, shape<2,2>>();
    auto & nref = normalize<1>(nrm, into(nout));                    // static shapes
    if (nout(0,0) < 0.59 || nout(0,0) > 0.61) return 30;
    if (nout(1,1) < 0.79 || nout(1,1) > 0.81) return 31;
    if (&nref != &nout)                       return 32;
    auto dnrm = zeros<double>(shape<-1,-1>{2,2});                   // dynamic shapes
    dnrm(0,0)=3; dnrm(0,1)=4; dnrm(1,0)=6; dnrm(1,1)=8;
    auto dnout = zeros<double>(shape<-1,-1>{2,2});
    normalize(dnrm, axis<1>{}, into(dnout));                        // the value spelling too
    if (dnout(0,0) < 0.59 || dnout(0,0) > 0.61) return 33;
    if (dnout(1,1) < 0.79 || dnout(1,1) > 0.81) return 34;
    // ...and a dest of a DIFFERENT element type still works (the guard is shape-only)
    auto fnout = local<float, shape<2,2>>();
    nrm.normalize(axis<1>{}, into(fnout));
    if (fnout(0,0) < 0.59f || fnout(0,0) > 0.61f) return 35;

#ifndef NDEBUG
    // ================= a MIS-SHAPED dest aborts, at all three sites ==========
    // Dynamic shapes throughout: a fully static mismatch is a compile error (the
    // `static_assert` half), which no runtime suite can exercise -- same convention
    // as test_into.cpp's commented-out repros.
    // `code` indexes the producer so a failure names the site that stopped checking.
    // NB for `code == 2` this pair (source LARGER than dest in every axis) is also
    // refused by `copy_` a step later, so it does not by itself prove `scan`'s own
    // guard is wired in -- the source-extent-1 loop below is the one that does.
    for (int code = 0; code < 3; ++code) {
        pid_t p = fork();
        if (p == 0) {
            if (!freopen("/dev/null", "w", stderr)) _exit(2);   // hush the assert message
            auto a = zeros<double>(shape<-1,-1>{8,8});
            auto y = zeros<double>(shape<-1,-1>{2,2});          // 4 slots for a 64-element write
            if (code == 0) a.mul(2.0, into(y));                 // scalar rhs -> scalo_
            if (code == 1) exp(a, into(y));                     // unary      -> unaryo_
            if (code == 2) scan<0>(a, 0.0, sum_op{}, into(y));  // scan
            _exit(0);                                           // must NOT reach here
        }
        int st = 0;
        if (waitpid(p, &st, 0) < 0) return 10 + 2*code;
        if (!(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT)) return 11 + 2*code;
    }

    // ---- a BROADCASTABLE mismatch: source extent 1, dest extent 4 -----------
    // The pair that isolates the `into(dest)` guard from every other check in the
    // library, and the reason it is here: `bzip_` (so `copy_`, so everything the
    // tensor-rhs family does) accepts `ae[r] == ce[r] || ae[r] == 1` -- a source
    // axis of extent 1 is legally STRETCHED to the dest's extent. `into(dest)` is
    // not a broadcast: the caller handed us the buffer to write, its shape must be
    // the result's shape exactly, and a silently-stretched source is a caller
    // mistake we are supposed to name.
    // So this pair passes every downstream check and ONLY `check_into_same_shape`
    // rejects it -- which makes it the regression probe for the `scan` site
    // specifically: delete `scan`'s call to the helper and `code == 2` stops
    // aborting (its `copy_` broadcasts 1 -> 4 happily) and this test fails, whereas
    // the 8x8-into-2x2 pair above keeps passing on `copy_`'s check alone.
    for (int code = 0; code < 3; ++code) {
        pid_t p = fork();
        if (p == 0) {
            if (!freopen("/dev/null", "w", stderr)) _exit(2);
            auto a = zeros<double>(shape<-1>{1});               // ONE element...
            auto y = zeros<double>(shape<-1>{4});               // ...into four slots
            if (code == 0) a.mul(2.0, into(y));                 // scalar rhs -> scalo_
            if (code == 1) exp(a, into(y));                     // unary      -> unaryo_
            if (code == 2) scan<0>(a, 0.0, sum_op{}, into(y));  // scan
            _exit(0);                                           // must NOT reach here
        }
        int st = 0;
        if (waitpid(p, &st, 0) < 0) return 16 + 2*code;
        if (!(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT)) return 17 + 2*code;
    }

    // ---- the LARGE-EXTENT mismatch: 2^32+5 elements into a 5-element dest ----
    // The pair `scan`'s old `long` comparison could not tell apart on LLP64.
    // Nothing is allocated and nothing is read: the source is a VIEW claiming a
    // huge extent over a small buffer, and the guard fires before the first load.
    // Case 2 calls the shared helper DIRECTLY, so what aborts is unambiguously the
    // consolidated guard. Case 0 (`scan`) would be refused by its own `copy_` a
    // step later even without the guard -- see the loop above for the probe that
    // pins that site.
    for (int code = 0; code < 3; ++code) {
        pid_t p = fork();
        if (p == 0) {
            if (!freopen("/dev/null", "w", stderr)) _exit(2);
            double buf[8] = {0,0,0,0,0,0,0,0};
            auto huge = wrap(buf, shape<-1>{BIG});             // extent 2^32+5, never indexed
            auto tiny = wrap(buf, shape<-1>{SMALL});           // extent 5
            if (code == 0) scan<0>(huge, 0.0, sum_op{}, into(tiny));
            if (code == 1) huge.mul(2.0, into(tiny));
            if (code == 2) _md::check_into_same_shape(tiny, huge, cs::make_index_sequence<1>{});
            _exit(0);                                           // must NOT reach here
        }
        int st = 0;
        if (waitpid(p, &st, 0) < 0) return 22 + 2*code;
        if (!(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT)) return 23 + 2*code;
    }

    // ---- axis-scoped normalize: the same two mis-shaped dests abort (#434) ---
    // Both pairs used to be accepted SILENTLY -- not "caught only under a debug
    // build", but not caught at all: the broadcasting engine's runtime check is
    // broadcast-aware too, so a source extent of 1 legally stretched to the dest's
    // extent and the normalised row was replicated into a `y` of a shape the
    // caller's `a` never had. Dynamic shapes here for the same reason as above (a
    // fully static mismatch is now a compile error -- see the repros below main).
    //   code 0: source extent 1, dest extent 4 -- the broadcastable pair
    //   code 1: source 8x8, dest 2x2           -- the plain shorter-dest pair
    for (int code = 0; code < 2; ++code) {
        pid_t p = fork();
        if (p == 0) {
            if (!freopen("/dev/null", "w", stderr)) _exit(2);
            if (code == 0) {
                auto a = zeros<double>(shape<-1,-1>{1,3});      // ONE row...
                auto y = zeros<double>(shape<-1,-1>{4,3});      // ...into four
                normalize<1>(a, into(y));
            } else {
                auto a = zeros<double>(shape<-1,-1>{8,8});
                auto y = zeros<double>(shape<-1,-1>{2,2});
                normalize(a, axis<1>{}, into(y));               // the value spelling routes here too
            }
            _exit(0);                                           // must NOT reach here
        }
        int st = 0;
        if (waitpid(p, &st, 0) < 0) return 36 + 2*code;
        if (!(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT)) return 37 + 2*code;
    }
#endif
    return 0;
}

// ---- the STATIC half of #434, as compile-fail repros ------------------------
// A statically-known mis-shaped dest is now a COMPILE error for the axis form,
// matching the whole-tensor `normalize(a, into(y))` (which reaches the same helper
// through `scalo_`, since its divisor is a scalar). Enabling any line below fails
// to compile -- the same commented-out convention as test_into.cpp's #357/#361
// repros, since a compile error cannot live in a running test. Verified by hand on
// both clang++ and g++:
//
//   auto a43 = zeros<double>(shape<4,3>{});
//   auto y42 = zeros<double>(shape<4,2>{});
//   normalize<1>(a43, into(y42));      // extent mismatch -- was ALREADY a compile
//                                      //   error via bzip_'s #361 dest gate; the
//                                      //   message is now the into(dest) one.
//   auto a13 = zeros<double>(shape<1,3>{});
//   auto y53 = zeros<double>(shape<5,3>{});
//   normalize<1>(a13, into(y53));      // source extent 1 vs dest 5: was ACCEPTED
//                                      //   (statically and at run time) and silently
//                                      //   replicated the row. Now: compile error.
//   auto y213 = zeros<double>(shape<2,1,3>{});
//   normalize<1>(a13, into(y213));     // dest rank 3 vs source rank 2: was ACCEPTED
//                                      //   (bzip_ only needs operand rank <= dest
//                                      //   rank). Now: compile error on the rank
//                                      //   static_assert.
//   a13.normalize(axis<1>{}, into(y53));   // ...and every spelling forwards to the
//   a13.normalize<1>(into(y53));           //    one guarded overload, so all four
//   normalize(a13, axis<1>{}, into(y53));  //    fail identically.
