// The `into(dest)` EXACT-SHAPE guard, now one helper (#363).
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
// extent above 2^31: a source of extent 2^32+5 compared EQUAL to a dest of extent
// 5, the guard passed, and `scan` then copied four billion elements through a
// five-element buffer. The math engines' copy cast to `cs::size_t` and did not.
// All three now call `_md::check_into_same_shape`, whose runtime comparison type
// (`_md::ext_cmp_t`) is at least 64 bits on EVERY platform.
//
// This pins:
//   - the comparison type's width and signedness, and that it distinguishes the
//     exact pair that the `long` spelling conflated (compile time, so it holds on
//     every platform even though only LLP64 could observe the difference);
//   - a correctly-shaped `into(dest)` still works for all three producers, on
//     static and dynamic shapes alike;
//   - a MIS-SHAPED dest aborts for all three (the guard is wired in at each of the
//     three consolidated call sites, not just at two of them);
//   - and specifically that `scan` aborts on the large-extent mismatch above --
//     the case the narrow comparison used to wave through.
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
              "2^32+5 and 5 are indistinguishable in 32 bits -- this is the bug");
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

#ifndef NDEBUG
    // ================= a MIS-SHAPED dest aborts, at all three sites ==========
    // Dynamic shapes throughout: a fully static mismatch is a compile error (the
    // `static_assert` half), which no runtime suite can exercise -- same convention
    // as test_into.cpp's commented-out repros.
    // `code` indexes the producer so a failure names the site that stopped checking.
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

    // ---- the LARGE-EXTENT mismatch: 2^32+5 elements into a 5-element dest ----
    // The case `scan`'s old `long` comparison waved through on LLP64. Nothing is
    // allocated and nothing is read: the source is a VIEW claiming a huge extent
    // over a small buffer, and the guard fires before the first load.
    // Case 2 calls the shared helper DIRECTLY, so what aborts is unambiguously the
    // consolidated guard: `scan`'s own `copy_` would refuse this pair a step later
    // through `bzip_`'s broadcast check, whereas `scalo_` (case 1) and the helper
    // (case 2) have nothing else standing between them and a 2^32-element write.
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
        if (waitpid(p, &st, 0) < 0) return 16 + 2*code;
        if (!(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT)) return 17 + 2*code;
    }
#endif
    return 0;
}
