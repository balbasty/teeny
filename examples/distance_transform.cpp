// distance_transform — a separable 1-D transform applied along the LAST axis,
// with an ARBITRARY number of leading batch axes peeled by the nd-peel utility.
//
// The point: the batch loop *is* the library. `peel<Axes...>(t)` peels the
// listed axes and hands you each remaining sub-view (here a 1-D line); you never
// write index2offset / sub2offset. Add or remove batch axes by changing the axis
// list, not the loop body.
#include <teeny/teeny.h>
#include <cstdio>

using namespace tny;
namespace cs = cuda::std;

static double vmin(double a, double b) { return a < b ? a : b; }

// The 1-D forward+backward L1 distance sweep, on a rank-1 view of ANY stride.
template <class Line>
static void l1_line(Line line, double w) {
    const long n = line.shape(0);
    if (n <= 1) return;
    double tmp = line(0);
    for (long i = 1;   i < n;  ++i) { tmp = vmin(tmp + w, line(i)); line(i) = tmp; }
    for (long i = n-2; i >= 0; --i) { tmp = vmin(tmp + w, line(i)); line(i) = tmp; }
}

// Whole kernel: peel every axis but the last, transform each line.
template <cs::size_t... Batch, class Tensor>
static void distance_l1(Tensor & t, double w) {
    for (auto line : peel<Batch...>(t)) l1_line(line, w);
}

int main() {
    // A (2,3,7) volume: batch axes 0,1; the transform runs along axis 2.
    double buf[2*3*7];
    for (int i = 0; i < 2*3*7; ++i) buf[i] = (i % 5 == 0) ? 0.0 : 1e9;
    auto t = wrap(buf, shape<2,3,7>{});

    distance_l1<0,1>(t, 1.0);                 // peel axes 0 and 1

    // reference: same sweep on each of the 6 contiguous rows of length 7
    double ref[2*3*7];
    for (int i = 0; i < 2*3*7; ++i) ref[i] = (i % 5 == 0) ? 0.0 : 1e9;
    for (int r = 0; r < 6; ++r) {
        double * f = ref + r*7; double tmp = f[0];
        for (int i=1;i<7;++i){ tmp = vmin(tmp+1.0, f[i]); f[i]=tmp; }
        for (int i=5;i>=0;--i){ tmp = vmin(tmp+1.0, f[i]); f[i]=tmp; }
    }
    for (int i = 0; i < 2*3*7; ++i) if (buf[i] != ref[i]) { std::printf("mismatch @%d\n", i); return 1; }

    // Different batch rank, SAME code: a (4,7) matrix, peel just axis 0.
    double buf2[4*7], ref2[4*7];
    for (int i = 0; i < 4*7; ++i) buf2[i] = ref2[i] = (i % 3 == 0) ? 0.0 : 1e9;
    auto m = wrap(buf2, shape<4,7>{});
    distance_l1<0>(m, 1.0);
    for (int r = 0; r < 4; ++r) {                        // reference: 4 rows of 7
        double * f = ref2 + r*7; double tmp = f[0];
        for (int i=1;i<7;++i){ tmp = vmin(tmp+1.0, f[i]); f[i]=tmp; }
        for (int i=5;i>=0;--i){ tmp = vmin(tmp+1.0, f[i]); f[i]=tmp; }
    }
    for (int i = 0; i < 4*7; ++i) if (buf2[i] != ref2[i]) { std::printf("rank-2 mismatch @%d\n", i); return 2; }

    std::printf("distance_transform: OK\n");
    return 0;
}
