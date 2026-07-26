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

    // (8) PERMUTED-dense destination: a scalar/unary in-place op is order-independent,
    //     so a transposed view (dense in a permuted order) takes the fast path and is
    //     still correct — every element is touched exactly once.
    { double buf[12]; auto m = wrap(buf, shape<3,4>{}); m.iota_(0.0, 1.0);   // row-major 0..11
      auto tr = m.permute(Int<1>(), Int<0>());   // (4,3), strides (1,4): dense, not C-order
      tr.mul_(2.0);                               // fast path via is_dense()
      for (long r = 0; r < 3; ++r) for (long c = 0; c < 4; ++c)
        if (!feq(m(r,c), 2.0 * (r*4 + c))) return 14;    // every element doubled
      tr.neg_();                                  // in-place unary, permuted-dense
      for (long r = 0; r < 3; ++r) for (long c = 0; c < 4; ++c)
        if (!feq(m(r,c), -2.0 * (r*4 + c))) return 15; }

    // (9) F-order (fcontiguous) dense destination — also dense, so the scalar in-place
    //     op takes the fast path; add is order-independent, so every logical element
    //     (iota fills logical row-major: f(i,j) = i*3 + j) gets +10.
    { double buf[6]; auto f = wrap<fcontiguous>(buf, shape<2,3>{});
      f.iota_(0.0, 1.0);       // logical row-major fill via decode (F-order isn't C-order)
      f.add_(10.0);            // scalar in-place: dense -> fast path
      for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j)
        if (!feq(f(i,j), 10.0 + (i*3 + j))) return 16; }

    // (10) NEGATIVE-stride (flipped) destination — the fast path's safety hinges on
    //      is_dense() EXCLUDING negative strides: a reversed view's physical block sits
    //      BEHIND data(), so a forward cp[i] walk would read out of bounds. is_dense()
    //      must return false here, sending scalar/unary in-place ops down the decode
    //      fallback, which handles the negative stride correctly. Guards that invariant.
    { double buf[6]; auto m = wrap(buf, shape<6>{}); m.iota_(0.0, 1.0);   // 0..5
      auto r = m.flip<0>();               // stride -1, data() at the last element
      r.mul_(2.0);                        // scalar in-place: NOT dense -> decode fallback
      for (long i = 0; i < 6; ++i) if (!feq(m(i), 2.0 * i)) return 17;    // every element doubled once
      r.neg_();                           // unary in-place on the same reversed view
      for (long i = 0; i < 6; ++i) if (!feq(m(i), -2.0 * i)) return 18; }

    // (11) a 2-D flip (negative stride on one axis, permuted) also falls back cleanly.
    { double buf[12]; auto m = wrap(buf, shape<3,4>{}); m.iota_(0.0, 1.0);
      auto r = m.flip<1>();               // reverse columns: stride (4,-1)
      r.add_(100.0);
      for (long i = 0; i < 3; ++i) for (long j = 0; j < 4; ++j)
        if (!feq(m(i,j), 100.0 + (i*4 + j))) return 19; }

    return 0;
}
