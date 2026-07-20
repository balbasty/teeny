// bounds.hpp — the 8 boundary conditions, transcribed faithfully from jitfields
// (csrc/lib/bounds.h). Each mode maps an out-of-range coordinate back to an
// in-range INDEX and returns a SIGN (+1 / 0 / -1) that encodes the symmetry
// (Zero -> 0 outside; DST1/DST2 flip sign on reflection). A gather reads
// `sign==0 ? 0 : sign<0 ? -data[i] : data[i]`; a scatter adds `sign*val`.
//
// This is pure numeric substrate for the pushpull kernels — it is NOT part of
// teeny (teeny is a tensor library). The fastfields port lifts this file.
#ifndef FF_BOUNDS_HPP
#define FF_BOUNDS_HPP
#include <cmath>

namespace ff {

enum class bound { zero, replicate, dct1, dct2, dst1, dst2, dft, nocheck };

// map a (possibly out-of-range) coordinate to an in-range index
inline long bound_index(bound b, long i, long n) {
    switch (b) {
    case bound::zero:
    case bound::nocheck:
        return i;                                     // caller relies on sign for zero
    case bound::replicate:
        return i <= 0 ? 0 : (i >= n ? n - 1 : i);
    case bound::dct1: {                               // reflect about border centres
        if (n == 1) return 0;
        long t = (n - 1) * 2, c = (i < 0 ? -i : i) % t;
        return c >= n ? t - c : c;
    }
    case bound::dct2:                                 // reflect about border edges
    case bound::dst2: {
        long t = n * 2;
        long c = i < 0 ? t - ((-i - 1) % t) - 1 : i % t;
        return c >= n ? t - c - 1 : c;
    }
    case bound::dst1: {
        long t = (n + 1) * 2;
        long c = (i == -1) ? 0 : (i < 0 ? -i - 2 : i);
        c %= t;
        return c == n ? n - 1 : (c > n ? t - c - 2 : c);
    }
    case bound::dft: {                                // circular wrap
        return i < 0 ? (n + i % n) % n : i % n;
    }
    }
    return i;
}

// symmetry sign for a coordinate under a boundary mode
inline int bound_sign(bound b, long i, long n) {
    switch (b) {
    case bound::zero:
        return (i < 0 || i >= n) ? 0 : 1;
    case bound::dst1: {
        long t = (n + 1) * 2;
        long c = (i < 0 ? n - i - 1 : i) % t;
        if (c % (n + 1) == n) return 0;
        return ((c / (n + 1)) % 2) ? -1 : 1;
    }
    case bound::dst2: {
        long c = i < 0 ? n - i - 1 : i;
        return ((c / n) % 2) ? -1 : 1;
    }
    default:                                          // replicate, dct1, dct2, dft, nocheck
        return 1;
    }
}

} // namespace ff
#endif // FF_BOUNDS_HPP
