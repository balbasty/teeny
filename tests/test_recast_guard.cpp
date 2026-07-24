// The recast layout-OVERRIDE ("I promise it's contiguous") now verifies, in a
// host-debug build, that the imposed strides actually match the source's — a false
// promise used to silently mis-address in every build. This pins that:
//   - a TRUE promise (a genuinely contiguous source) passes, and
//   - a FALSE promise (a transposed source claimed contiguous) ABORTS in debug.
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

    // TRUE promise: a real contiguous source reinterpreted as contiguous is fine.
    auto ok = wrap(buf, shape<-1,3>{2}).recast<shape<2,3>, ccontiguous>();
    if (ok(1,2) != buf[5]) return 1;
    // and keep_strides on a transposed source is always safe (no promise).
    auto tp = wrap(buf, shape<2,3>{}).permute<1,0>();      // 3x2, strides (1,3)
    auto tpr = tp.recast<shape<3,2>>();                     // keep_strides -> preserves (1,3)
    if (tpr(2,1) != tp(2,1)) return 2;

#ifndef NDEBUG
    // FALSE promise: the transposed view (strides (1,3)) is NOT row-major, so
    // recast<..., ccontiguous>() (which would impose (2,1)) must trip the guard.
    pid_t pid = fork();
    if (pid == 0) {
        if (!freopen("/dev/null", "w", stderr)) _exit(2);  // hush the assert message
        double b[6];
        auto bad = wrap(b, shape<2,3>{}).permute<1,0>();   // 3x2, strides (1,3)
        auto r = bad.recast<shape<3,2>, ccontiguous>();    // promises (2,1) — FALSE
        (void)r(0,0);                                       // force the mapping to be used
        _exit(0);                                           // must NOT reach here
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return 3;
    if (!(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT)) return 4;  // must have aborted
#endif
    return 0;
}
