// wrap() trusts the strides you pass, so a stride-0 axis makes a SELF-OVERLAPPING
// view where several indices alias one element. Any WRITE into such a destination is
// wrong, in one of two ways: an in-place update applies to the aliased element
// repeatedly (`v.add_(b)` double-counts), and an out-of-place `into(dest)` store
// keeps only the last of the results that land in the slot (#364). So every engine
// that writes a destination host-debug-asserts that it has no extent>1/stride-0
// axis. This pins:
//   - a normal (dense) in-place op passes;
//   - an overlapping view read as an in-place SOURCE (broadcast-like) passes
//     (the guard is on the destination only, so a stretched RHS is never flagged);
//   - a correctly-shaped, non-overlapping `into(dest)` passes and is unaffected;
//   - an in-place WRITE into an overlapping destination ABORTS in a debug build;
//   - an OUT-OF-PLACE `into(dest)` write into one aborts too, for all three
//     producer families — tensor rhs, scalar rhs, and unary (#364: only the
//     tensor-rhs one used to).
// The abort is only expected when _TNY_CHECK is live (host, non-NDEBUG).
#include <teeny/teeny.h>
#include <cstdio>
#ifndef NDEBUG
#include <unistd.h>
#include <sys/wait.h>
#include <csignal>
#endif

using namespace tny;

int main() {
    double buf[6]; for (int i = 0; i < 6; ++i) buf[i] = i;

    // dense in-place op: fine.
    auto d = local<double, shape<2,3>>{}; d.iota_(0.0, 1.0); d.add_(1.0);
    if (d(0,0) != 1.0 || d(1,2) != 6.0) return 1;

    // an overlapping view read as an in-place SOURCE (broadcast-like) is fine — the
    // guard is on the DESTINATION only. `ov` has axis-0 stride 0, so both rows alias
    // buf[0..2]; adding it into a dense tensor reads (never writes) the aliased cells.
    auto ov = wrap(buf, shape<2,3>{}, {0,1});          // strides (0,1): rows alias
    auto e  = local<double, shape<2,3>>{}; e.zero_(); e.add_(ov);
    if (e(0,2) != buf[2] || e(1,2) != buf[2]) return 2;

    // a correctly-shaped, NON-overlapping into(dest) is untouched by the guard: all
    // three producer families still write every element (element identity, #364).
    auto ok  = local<double, shape<2,3>>{}; ok.iota_(1.0, 1.0);    // 1..6
    auto out = local<double, shape<2,3>>{}; out.zero_();
    ok.add(ok, into(out));                                         // tensor rhs
    for (int i = 0; i < 6; ++i) if (out.data()[i] != 2.0*(i+1)) return 9;
    ok.mul(2.0, into(out));                                        // scalar rhs
    for (int i = 0; i < 6; ++i) if (out.data()[i] != 2.0*(i+1)) return 10;
    neg(ok, into(out));                                            // unary
    for (int i = 0; i < 6; ++i) if (out.data()[i] != -1.0*(i+1)) return 11;
    // ...including into a strided (but non-overlapping) dest, which takes the general
    // decode rather than the linear fast path.
    double sbuf[12]; for (int i = 0; i < 12; ++i) sbuf[i] = -1.0;
    auto sd = wrap(sbuf, shape<2,3>{}, {6,2});                     // every other slot
    ok.mul(3.0, into(sd));
    for (int i = 0; i < 6; ++i) if (sd.data()[(i/3)*6 + (i%3)*2] != 3.0*(i+1)) return 12;
    neg(ok, into(sd));
    for (int i = 0; i < 6; ++i) if (sd.data()[(i/3)*6 + (i%3)*2] != -1.0*(i+1)) return 13;

#ifndef NDEBUG
    // in-place WRITE into the overlapping destination must trip the guard.
    pid_t pid = fork();
    if (pid == 0) {
        if (!freopen("/dev/null", "w", stderr)) _exit(2);  // hush the assert message
        double b[6];
        auto bad = wrap(b, shape<2,3>{}, {0,1});           // axis 0: extent 2, stride 0 -> overlap
        bad.add_(1.0);                                     // in-place write into aliased cells
        _exit(0);                                          // must NOT reach here
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return 3;
    if (!(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT)) return 4;  // must have aborted

    // a tensor RHS trips it too (bzip destination check): overlapping dest += dense.
    pid_t pid2 = fork();
    if (pid2 == 0) {
        if (!freopen("/dev/null", "w", stderr)) _exit(2);
        double b[6];
        auto bad = wrap(b, shape<2,3>{}, {0,1});
        auto src = local<double, shape<2,3>>{}; src.iota_(1.0, 1.0);
        bad.add_(src);                                     // in-place write into aliased dest
        _exit(0);
    }
    int status2 = 0;
    if (waitpid(pid2, &status2, 0) < 0) return 5;
    if (!(WIFSIGNALED(status2) && WTERMSIG(status2) == SIGABRT)) return 6;

    // an in-place UNARY into the overlapping dest trips it too (unary applies f twice
    // to an aliased element) — the guard is now symmetric with scal_/iota_.
    pid_t pid3 = fork();
    if (pid3 == 0) {
        if (!freopen("/dev/null", "w", stderr)) _exit(2);
        double b[6];
        auto bad = wrap(b, shape<2,3>{}, {0,1});
        bad.neg_();                                        // in-place unary write into aliased dest
        _exit(0);
    }
    int status3 = 0;
    if (waitpid(pid3, &status3, 0) < 0) return 7;
    if (!(WIFSIGNALED(status3) && WTERMSIG(status3) == SIGABRT)) return 8;

    // ...and the OUT-OF-PLACE into(dest) forms, whose destination is the caller's own
    // choice and so the only unguarded way to reach the single-source engines (#364).
    // Same overlapping `bad`, same three producer families as the in-place cases above:
    // a tensor rhs (already guarded before #364), a scalar rhs, and a unary. Each must
    // abort, not silently drop every result but the last landing in an aliased slot.
    // `code` indexes the case so a failure names which producer stopped tripping.
    for (int code = 0; code < 4; ++code) {
        pid_t p = fork();
        if (p == 0) {
            if (!freopen("/dev/null", "w", stderr)) _exit(2);
            double b[6];
            auto bad = wrap(b, shape<2,3>{}, {0,1});       // axis 0: extent 2, stride 0
            auto s   = local<double, shape<2,3>>{}; s.iota_(1.0, 1.0);
            if (code == 0) s.add(s, into(bad));            // tensor rhs   -> bzip
            if (code == 1) s.mul(2.0, into(bad));          // scalar rhs   -> scalo_
            if (code == 2) exp(s, into(bad));              // unary        -> unaryo_
            if (code == 3) clamp(s, 2.0, 4.0, into(bad));  // unary (2 args, same engine)
            _exit(0);                                      // must NOT reach here
        }
        int st = 0;
        if (waitpid(p, &st, 0) < 0) return 14 + 2*code;
        if (!(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT)) return 15 + 2*code;
    }
#endif
    return 0;
}
