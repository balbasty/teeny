// N-D spline "pull" (interpolation) on the md line: ONE compile-time-recursion
// convolve replaces jitfields' hand-unrolled 1d.h / 2d.h / 3d.h / nd.h. Here it
// is instantiated for D = 1, 2, 3 (linear / bilinear / trilinear) and validated
// against hand-written references. The md tensor supplies stride(d)/data(); the
// tensor-product accumulation is written once.
#include <teeny/md.h>
#include <cuda/std/tuple>
#include <cmath>

using namespace tny::md;
namespace cs = cuda::std;
using cs::extents;

// replicate-boundary clamp
static long clamp(long i, long n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

// per-dimension sample: Order+1 neighbours (here order 1 = linear).
template <int Order> struct dim_sample { static constexpr int K = Order + 1; double w[K]; long idx[K]; };
template <int Order> static dim_sample<Order> sample(double x, long n) {
    dim_sample<Order> s;
    long   f    = (long)std::floor(x);
    double frac = x - (double)f;
    s.idx[0] = clamp(f,   n); s.w[0] = 1.0 - frac;
    s.idx[1] = clamp(f+1, n); s.w[1] = frac;
    return s;
}

// -------- the tensor product, written ONCE (recursion over the static rank) --------
template <int d, int D, class MD, class S>
static double convolve(const MD & inp, const S & s, long off, double w) {
    double acc = 0;
    const auto & sd = cs::get<d>(s);
    for (int k = 0; k < sd.K; ++k) {
        long   o  = off + sd.idx[k] * inp.stride(d);   // static stride folds
        double ww = w   * sd.w[k];
        if constexpr (d + 1 == D) acc += inp.data()[o] * ww;   // gather leaf
        else                      acc += convolve<d + 1, D>(inp, s, o, ww);
    }
    return acc;
}
template <class MD, class S>
static double pull(const MD & inp, const S & s) {
    return convolve<0, (int)MD::rank()>(inp, s, 0, 1.0);
}

// -------- hand-written references --------
static double ref_lerp(const double * f, long nx, double x) {
    long f0 = (long)std::floor(x); double t = x - f0;
    return f[clamp(f0,nx)] * (1-t) + f[clamp(f0+1,nx)] * t;
}
static double ref_bilerp(const double * f, long nx, long ny, double x, double y) {
    long fx=(long)std::floor(x), fy=(long)std::floor(y); double tx=x-fx, ty=y-fy;
    auto at=[&](long i,long j){ return f[clamp(i,nx)*ny + clamp(j,ny)]; };
    return (at(fx,fy)*(1-ty)+at(fx,fy+1)*ty)*(1-tx) + (at(fx+1,fy)*(1-ty)+at(fx+1,fy+1)*ty)*tx;
}
static double ref_trilerp(const double * f, long nx,long ny,long nz, double x,double y,double z) {
    long fx=(long)std::floor(x),fy=(long)std::floor(y),fz=(long)std::floor(z);
    double tx=x-fx,ty=y-fy,tz=z-fz;
    auto at=[&](long i,long j,long k){ return f[(clamp(i,nx)*ny+clamp(j,ny))*nz+clamp(k,nz)]; };
    double c=0;
    for(int a=0;a<2;++a)for(int b=0;b<2;++b)for(int cc=0;cc<2;++cc)
        c += at(fx+a,fy+b,fz+cc) * (a?tx:1-tx) * (b?ty:1-ty) * (cc?tz:1-tz);
    return c;
}

static bool close(double a, double b) { return std::fabs(a-b) < 1e-9; }

int main()
{
    double buf[512];
    for (int i=0;i<512;++i) buf[i] = std::sin(0.1*i) * 10.0;

    // ---- D = 1 (linear): same convolve ----
    {
        auto inp = view(buf, extents<long,7>{});
        for (double x : {0.0, 0.3, 2.7, 5.9, 6.5}) {
            auto s = cs::make_tuple(sample<1>(x, 7));
            if (!close(pull(inp, s), ref_lerp(buf, 7, x))) return 1;
        }
    }
    // ---- D = 2 (bilinear): same convolve ----
    {
        auto inp = view(buf, extents<long,5,6>{});     // row-major, strides (6,1)
        for (double x : {0.0, 1.4, 3.9}) for (double y : {0.2, 2.5, 5.1}) {
            auto s = cs::make_tuple(sample<1>(x,5), sample<1>(y,6));
            if (!close(pull(inp, s), ref_bilerp(buf, 5, 6, x, y))) return 2;
        }
    }
    // ---- D = 3 (trilinear): same convolve ----
    {
        auto inp = view(buf, extents<long,4,5,6>{});   // strides (30,6,1)
        for (double x : {0.0, 2.6}) for (double y : {1.1, 3.7}) for (double z : {0.5, 4.9}) {
            auto s = cs::make_tuple(sample<1>(x,4), sample<1>(y,5), sample<1>(z,6));
            if (!close(pull(inp, s), ref_trilerp(buf, 4,5,6, x,y,z))) return 3;
        }
    }
    // ---- D = 3 also works on a fully-dynamic-extent view (strides loaded) ----
    {
        using E = extents<long, cs::dynamic_extent, cs::dynamic_extent, cs::dynamic_extent>;
        auto inp = view(buf, E{4,5,6});
        double x=2.6,y=3.7,z=4.9;
        auto s = cs::make_tuple(sample<1>(x,4), sample<1>(y,5), sample<1>(z,6));
        if (!close(pull(inp, s), ref_trilerp(buf, 4,5,6, x,y,z))) return 4;
    }

    return 0;
}
