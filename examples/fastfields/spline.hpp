// spline.hpp — B-spline interpolation weights, transcribed from jitfields
// (csrc/lib/spline.h). Orders 0..3 here (the common cases); orders 4..7 follow
// the same shape and their weight polynomials are listed in the fastfields
// porting handoff. `x` is the ABSOLUTE distance |coord - neighbour|.
//
// The neighbourhood of a coordinate at interpolation `order` is `order+1`
// consecutive integer samples starting at `spline_low(order, coord)`; the weight
// of each is `spline_weight(order, |coord - sample|)`.
//
// Pure numeric substrate for pushpull — NOT part of teeny. The fastfields port
// lifts this file.
#ifndef FF_SPLINE_HPP
#define FF_SPLINE_HPP
#include <cmath>

namespace ff {

// number of neighbours contributing (the spline's support)
inline int spline_support(int order) { return order + 1; }

// index of the first (leftmost) contributing sample. Unifies all orders:
//   order 0 -> floor(x+0.5) = round(x);  1 -> floor(x);  2 -> floor(x-0.5);  3 -> floor(x-1)
inline long spline_low(int order, double x) {
    return (long)std::floor(x - (order - 1) * 0.5);
}

// weight as a function of absolute distance x = |coord - sample|
inline double spline_weight(int order, double x) {
    switch (order) {
    case 0:  // nearest
        return x < 0.5 ? 1.0 : 0.0;
    case 1:  // linear
        return x < 1.0 ? 1.0 - x : 0.0;
    case 2:  // quadratic
        if (x < 0.5) return 0.75 - x * x;
        if (x < 1.5) { double t = 1.5 - x; return 0.5 * t * t; }
        return 0.0;
    case 3:  // cubic
        if (x < 1.0) return (x * x * (x - 2.0) * 3.0 + 4.0) / 6.0;
        if (x < 2.0) { double t = 2.0 - x; return t * t * t / 6.0; }
        return 0.0;
    default:
        return 0.0;  // orders 4..7: see the fastfields handoff for the tables
    }
}

// derivative of the weight wrt the ORIENTED distance (sign applied by caller):
// used by the "grad" pushpull variant. Orders 0..3.
inline double spline_grad(int order, double x) {
    switch (order) {
    case 0: return 0.0;
    case 1: return x < 1.0 ? -1.0 : 0.0;
    case 2:
        if (x < 0.5) return -2.0 * x;
        if (x < 1.5) return x - 1.5;
        return 0.0;
    case 3:
        if (x < 1.0) return x * (1.5 * x - 2.0);
        if (x < 2.0) { double t = 2.0 - x; return -0.5 * t * t; }
        return 0.0;
    default: return 0.0;
    }
}

} // namespace ff
#endif // FF_SPLINE_HPP
