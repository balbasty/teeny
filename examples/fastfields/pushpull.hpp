// pushpull.hpp — rank-generic spline PULL (gather) and PUSH (scatter), the
// flagship fastfields kernel, written ONCE on teeny tensors.
//
// The separable tensor-product accumulation is a single recursion over the
// static spatial rank D (here D = MD::rank()); per axis the neighbours, weights
// (spline.hpp) and boundary index+sign (bounds.hpp) are formed once and shared.
// PUSH is the exact adjoint of PULL: same neighbours/weights, but it
// SCATTER-ACCUMULATES with tny::fetch_add (atomic on device) instead of reading.
//
// Single channel here for clarity; multi-channel is an inner `for c` loop that
// reuses the same neighbours (channel axis last, stride C) — see the handoff.
#ifndef FF_PUSHPULL_HPP
#define FF_PUSHPULL_HPP
#include <teeny/teeny.h>
#include <cmath>
#include "bounds.hpp"
#include "spline.hpp"

namespace ff {

// per-axis neighbourhood: K samples, their weights, bound-mapped indices, signs.
struct axis_nb { int K; double w[8]; long idx[8]; int sgn[8]; };

inline axis_nb make_axis(int order, bound b, double coord, long n) {
    axis_nb a; a.K = spline_support(order);
    long low = spline_low(order, coord);
    for (int k = 0; k < a.K; ++k) {
        long nb = low + k;
        a.w[k]   = spline_weight(order, std::fabs(coord - (double)nb));
        a.idx[k] = bound_index(b, nb, n);
        a.sgn[k] = bound_sign(b, nb, n);
    }
    return a;
}

// --- gather: acc = Σ_neighbourhood (Π weights) * (Π signs) * inp[·] ----------
template <int d, int D, class MD>
double pull_rec(const MD & inp, const axis_nb * nb, long off, double w, int sgn) {
    double acc = 0; const axis_nb & a = nb[d];
    for (int k = 0; k < a.K; ++k) {
        int s = sgn * a.sgn[k];
        if (s == 0) continue;                                 // zero boundary
        long   o  = off + a.idx[k] * (long)inp.stride(d);     // static stride folds
        double ww = w * a.w[k];
        if constexpr (d + 1 == D) acc += (s < 0 ? -inp.data()[o] : inp.data()[o]) * ww;
        else                      acc += pull_rec<d + 1, D>(inp, nb, o, ww, s);
    }
    return acc;
}

// --- scatter: out[·] += val * (Π weights) * (Π signs), atomic on device ------
template <int d, int D, class MD>
void push_rec(MD & out, const axis_nb * nb, long off, double wv, int sgn) {
    const axis_nb & a = nb[d];
    for (int k = 0; k < a.K; ++k) {
        int s = sgn * a.sgn[k];
        if (s == 0) continue;
        long   o  = off + a.idx[k] * (long)out.stride(d);
        double ww = wv * a.w[k];
        if constexpr (d + 1 == D) tny::fetch_add(out.data() + o, s < 0 ? -ww : ww);
        else                      push_rec<d + 1, D>(out, nb, o, ww, s);
    }
}

/** @brief Interpolate `inp` at coordinate `loc[D]` (order + per-axis bounds). */
template <class MD>
double pull(const MD & inp, const double * loc, int order, const bound * b) {
    constexpr int D = (int)MD::rank();
    axis_nb nb[D];
    for (int d = 0; d < D; ++d) nb[d] = make_axis(order, b[d], loc[d], (long)inp.extent(d));
    return pull_rec<0, D>(inp, nb, 0, 1.0, 1);
}

/** @brief Splat `val` into `out` at coordinate `loc[D]` (adjoint of pull). */
template <class MD>
void push(MD & out, const double * loc, double val, int order, const bound * b) {
    constexpr int D = (int)MD::rank();
    axis_nb nb[D];
    for (int d = 0; d < D; ++d) nb[d] = make_axis(order, b[d], loc[d], (long)out.extent(d));
    push_rec<0, D>(out, nb, 0, val, 1);
}

} // namespace ff
#endif // FF_PUSHPULL_HPP
