// #175: in-place SINGLE-array ops (scalar / unary / iota / fill) take a contiguous
// linear fast path that auto-vectorizes with no __restrict__ (one array -> no aliasing).
// This checks the fast path and the strided fallback are element-identical, that the
// w_set guard keeps atomic scalar ops correct, and that half computes in float.
#include <teeny/teeny.h>
#include <cmath>
using namespace tny;

static bool feq(double a, double b) { return std::fabs(a - b) < 1e-9; }

int main() {
    // (1) contiguous in-place scalar: mul_/add_/compound — fast path.
    { auto a = arange<double>(12); a.mul_(2.0);
      for (long i = 0; i < 12; ++i) if (!feq(a(i), 2.0 * i)) return 1; }
    { auto a = arange<double>(12); a.add_(1.0);
      for (long i = 0; i < 12; ++i) if (!feq(a(i), i + 1.0)) return 2; }
    { auto a = arange<double>(12); a += 3.0; a *= 0.5;
      for (long i = 0; i < 12; ++i) if (!feq(a(i), (i + 3.0) * 0.5)) return 3; }

    // (2) contiguous in-place unary — fast path (single pointer).
    { auto a = arange<double>(8); a.neg_();
      for (long i = 0; i < 8; ++i) if (!feq(a(i), -double(i))) return 4; }
    { auto a = zeros<double>(shape<6>{}); a.add_(1.0); a.exp_();
      for (long i = 0; i < 6; ++i) if (!feq(a(i), std::exp(1.0))) return 5; }

    // (3) iota_ / fill_ / zero_ — fast path.
    { auto a = zeros<double>(shape<10>{}); a.iota_(5.0, 2.0);
      for (long i = 0; i < 10; ++i) if (!feq(a(i), 5.0 + 2.0 * i)) return 6; }
    { auto a = arange<double>(7); a.fill_(3.5);
      for (long i = 0; i < 7; ++i) if (!feq(a(i), 3.5)) return 7;
      a.zero_(); for (long i = 0; i < 7; ++i) if (!feq(a(i), 0.0)) return 8; }

    // (4) STRIDED destination -> the decode FALLBACK must give the same answer.
    //     A permuted (transposed) view is not C-contiguous, so the fast path is skipped.
    { double buf[12]; auto m = wrap(buf, shape<3,4>{});
      auto tr = m.permute(Int<1>(), Int<0>());   // (4,3), strides (1,4) — not contiguous
      tr.iota_(0.0, 1.0);                          // fills in tr's logical order via decode
      for (long i = 0; i < 4; ++i) for (long j = 0; j < 3; ++j)
        if (!feq(tr(i,j), double(i*3 + j))) return 9;
      tr.mul_(10.0);
      for (long i = 0; i < 4; ++i) for (long j = 0; j < 3; ++j)
        if (!feq(tr(i,j), 10.0 * (i*3 + j))) return 10; }

    // (5) atomic scalar (w_set-guarded off the fast path) still correct on a dense view.
    { auto a = arange<double>(5); a.atomic_add_(4.0);
      for (long i = 0; i < 5; ++i) if (!feq(a(i), i + 4.0)) return 11; }

    // (6) half in-place computes in float, stores half — fast path with a narrow type.
    { auto h = zeros<half>(shape<8>{}); h.add_(2.0); h.mul_(3.0);
      for (long i = 0; i < 8; ++i) if (!feq(double(float(h(i))), 6.0)) return 12; }

    // (7) integer bitwise scalar routes through the scalar engine too.
    { auto a = arange<int>(8); a |= 1;
      for (long i = 0; i < 8; ++i) if (a(i) != (int(i) | 1)) return 13; }

    return 0;
}
