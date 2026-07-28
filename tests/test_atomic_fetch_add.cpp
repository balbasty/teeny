// Host-side atomic fetch_add (#257): atomic_add_/atomic_sub_ now race-free on
// the host too (cuda::std::atomic_ref under the hood), not just documented as
// the caller's problem. Proven here with a genuine multi-threaded stress test
// (a plain non-atomic `+=` would lose updates under this workload), covering
// both the scalar-rhs (scal_) and tensor-rhs (bzip_) atomic engines — the
// single-threaded equivalence checks live in test_atomic_alias.cpp.
#include <teeny/teeny.h>
#include <thread>
#include <vector>

using namespace tny;

int main()
{
    // ---- many threads scatter-accumulate into ONE cell -----------------
    // A plain (non-atomic) `+=` here would lose the vast majority of updates
    // under contention; a real atomic must land every single one.
    {
        double buf[1] = {0.0};
        auto a = wrap(buf, shape<1>{});
        constexpr int nthreads = 8;
        constexpr int niter = 50000;
        std::vector<std::thread> ts;
        for (int t = 0; t < nthreads; ++t)
            ts.emplace_back([&]() { for (int i = 0; i < niter; ++i) a.at(0).atomic_add_(1.0); });
        for (auto & th : ts) th.join();
        if (buf[0] != static_cast<double>(nthreads) * niter) return 1;
    }

    // ---- same, via atomic_sub_ from a known total -----------------------
    {
        double buf[1] = {400000.0};
        auto a = wrap(buf, shape<1>{});
        constexpr int nthreads = 8;
        constexpr int niter = 50000;
        std::vector<std::thread> ts;
        for (int t = 0; t < nthreads; ++t)
            ts.emplace_back([&]() { for (int i = 0; i < niter; ++i) a.at(0).atomic_sub_(1.0); });
        for (auto & th : ts) th.join();
        if (buf[0] != 0.0) return 2;
    }

    // ---- many threads scatter into DIFFERENT overlapping cells of a small
    //      tensor (the actual push-kernel shape: several outputs, many
    //      writers per output) --------------------------------------------
    {
        double buf[4] = {0, 0, 0, 0};
        auto a = wrap(buf, shape<4>{});
        constexpr int nthreads = 8;
        constexpr int niter = 20000;
        std::vector<std::thread> ts;
        for (int t = 0; t < nthreads; ++t)
            ts.emplace_back([&, t]() {
                for (int i = 0; i < niter; ++i) a.at((t + i) % 4).atomic_add_(1.0);
            });
        for (auto & th : ts) th.join();
        double total = buf[0] + buf[1] + buf[2] + buf[3];
        if (total != static_cast<double>(nthreads) * niter) return 3;
    }

    // ---- TENSOR rhs atomic_add_ (bzip_'s w_add path, not scal_'s): every
    //      thread scatter-accumulates a whole broadcasting tensor rhs into
    //      the SAME shared destination, e.g. `out.atomic_add_(delta)` from a
    //      push kernel writing several channels per cell at once. This
    //      exercises the `a == c` aliasing case (atomic_add_(b) calls
    //      bzip<w_add>(*this, *this, b, rhs{})), distinct from the scalar
    //      overload the blocks above cover. -------------------------------
    {
        double buf[4] = {0, 0, 0, 0};
        auto a = wrap(buf, shape<4>{});
        double one_buf[4] = {1.0, 1.0, 1.0, 1.0};
        auto one = wrap(one_buf, shape<4>{});
        constexpr int nthreads = 8;
        constexpr int niter = 20000;
        std::vector<std::thread> ts;
        for (int t = 0; t < nthreads; ++t)
            ts.emplace_back([&]() { for (int i = 0; i < niter; ++i) a.atomic_add_(one); });
        for (auto & th : ts) th.join();
        if (buf[0] != static_cast<double>(nthreads) * niter ||
            buf[3] != static_cast<double>(nthreads) * niter) return 4;
    }

    return 0;
}
