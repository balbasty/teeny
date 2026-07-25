// wrap() trusts the strides you pass, so a stride-0 axis makes a SELF-OVERLAPPING
// view where several indices alias one element. An in-place WRITE into such a
// destination applies the update multiple times (`v.add_(b)` double-counts), so the
// in-place engines host-debug-assert that the DESTINATION has no extent>1/stride-0
// axis. This pins:
//   - a normal (dense) in-place op passes;
//   - an overlapping view read as an in-place SOURCE (broadcast-like) passes
//     (the guard is on the destination only, so a stretched RHS is never flagged);
//   - an in-place WRITE into an overlapping destination ABORTS in a debug build.
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
#endif
    return 0;
}
