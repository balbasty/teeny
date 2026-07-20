// pull_nd — N-D separable spline interpolation ("pull" / gather), rank-generic.
//
// This is the flagship fastfields idiom: the tensor-product accumulation over
// the interpolation neighbourhood is written ONCE as a recursion over the static
// rank, and instantiated for 1-D / 2-D / 3-D (linear here). teeny's tensor
// supplies stride(d) / data() so the same body works on any layout, and static
// strides fold to immediates.
//
// Boundary conditions and interpolation order are parameters, not forks: a real
// port would add more `bound` modes and higher `Order` weight tables, but the
// gather stays this one function.
#include <teeny/teeny.h>
#include <cuda/std/tuple>
#include <cmath>
#include <cstdio>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;

// ---- boundary conditions: map an out-of-range index back in range ----------
enum class bound { zero, replicate, reflect };

static long apply_bound(bound b, long i, long n) {
    if (i >= 0 && i < n) return i;
    switch (b) {
        case bound::zero:      return -1;                       // sentinel: contributes 0
        case bound::replicate: return i < 0 ? 0 : n - 1;
        case bound::reflect: {                                  // reflect at the edges
            if (n == 1) return 0;
            long p = 2 * (n - 1);
            long m = ((i % p) + p) % p;
            return m < n ? m : p - m;
        }
    }
    return 0;
}

// ---- per-dimension sample: Order+1 weights + (bounded) indices -------------
template <int Order> struct dim_sample { static constexpr int K = Order + 1; double w[K]; long idx[K]; };

template <int Order>
static dim_sample<Order> sample(double x, long n, bound b) {
    static_assert(Order == 1, "example ships linear only; add weight tables for more");
    dim_sample<Order> s;
    long   f    = (long)std::floor(x);
    double frac = x - (double)f;
    s.idx[0] = apply_bound(b, f,   n); s.w[0] = 1.0 - frac;
    s.idx[1] = apply_bound(b, f+1, n); s.w[1] = frac;
    return s;
}

// ---- the tensor product, written ONCE (recursion over the static rank) -----
template <int d, int D, class MD, class S>
static double convolve(const MD & inp, const S & s, long off, double w) {
    double acc = 0;
    const auto & sd = cs::get<d>(s);
    for (int k = 0; k < sd.K; ++k) {
        if (sd.idx[k] < 0) continue;                     // zero-boundary sentinel
        long   o  = off + sd.idx[k] * inp.stride(d);     // static stride folds here
        double ww = w   * sd.w[k];
        if constexpr (d + 1 == D) acc += inp.data()[o] * ww;
        else                      acc += convolve<d + 1, D>(inp, s, o, ww);
    }
    return acc;
}
template <class MD, class S>
static double pull(const MD & inp, const S & s) {
    return convolve<0, (int)MD::rank()>(inp, s, 0, 1.0);
}

// ---- references ------------------------------------------------------------
static bool close(double a, double b) { return std::fabs(a - b) < 1e-9; }

int main() {
    double buf[512];
    for (int i = 0; i < 512; ++i) buf[i] = std::sin(0.1 * i) * 10.0;

    // 2-D bilinear on a static-shape view; replicate boundary.
    auto img = view(buf, extents<long,5,6>{});           // strides (6,1) fold
    for (double x : {0.0, 1.4, 3.9, 4.9}) for (double y : {0.2, 2.5, 5.1}) {
        auto s = cs::make_tuple(sample<1>(x,5,bound::replicate),
                                sample<1>(y,6,bound::replicate));
        double got = pull(img, s);
        // hand reference
        long fx=(long)std::floor(x), fy=(long)std::floor(y); double tx=x-fx, ty=y-fy;
        auto cl=[](long i,long n){ return i<0?0:(i>=n?n-1:i); };
        auto at=[&](long i,long j){ return buf[cl(i,5)*6 + cl(j,6)]; };
        double ref=(at(fx,fy)*(1-ty)+at(fx,fy+1)*ty)*(1-tx)+(at(fx+1,fy)*(1-ty)+at(fx+1,fy+1)*ty)*tx;
        if (!close(got, ref)) { std::printf("2D mismatch\n"); return 1; }
    }

    // Same convolve, now 3-D trilinear on a fully-dynamic-extent view.
    using E3 = extents<long, cs::dynamic_extent, cs::dynamic_extent, cs::dynamic_extent>;
    auto vol = view(buf, E3{4,5,6});
    {
        double x=2.6,y=3.7,z=4.9;
        auto s = cs::make_tuple(sample<1>(x,4,bound::replicate),
                                sample<1>(y,5,bound::replicate),
                                sample<1>(z,6,bound::replicate));
        double got = pull(vol, s);
        long fx=2,fy=3,fz=4; double tx=x-fx,ty=y-fy,tz=z-fz;
        auto cl=[](long i,long n){ return i<0?0:(i>=n?n-1:i); };
        auto at=[&](long i,long j,long k){ return buf[(cl(i,4)*5+cl(j,5))*6+cl(k,6)]; };
        double ref=0;
        for(int a=0;a<2;++a)for(int bb=0;bb<2;++bb)for(int c=0;c<2;++c)
            ref+=at(fx+a,fy+bb,fz+c)*(a?tx:1-tx)*(bb?ty:1-ty)*(c?tz:1-tz);
        if (!close(got, ref)) { std::printf("3D mismatch\n"); return 2; }
    }

    // Zero boundary really zeroes out-of-range contributions (sample past the edge).
    {
        auto line = view(buf, extents<long,4>{});
        auto s = cs::make_tuple(sample<1>(3.5, 4, bound::zero));  // idx 3 (in) & 4 (out->0)
        double got = pull(line, s);
        if (!close(got, buf[3] * 0.5)) { std::printf("zero-bound mismatch\n"); return 3; }
    }

    std::printf("pull_nd: OK\n");
    return 0;
}
