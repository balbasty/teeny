// pushpull_adjoint — validate the fastfields pull/push reference two ways:
//   (1) numerically, linear pull == hand-written bilinear;
//   (2) the ADJOINT identity <P x, y> == <x, Pᵀ y> for random x, y, grid points,
//       across interpolation orders 0..3 and several boundary conditions.
// The adjoint test is the gold standard that pull and push share exactly the
// same interpolation weights (any mismatch in bounds/spline breaks it).
#include "fastfields/pushpull.hpp"
#include <teeny/teeny.h>
#include <cstdio>
#include <cmath>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;
using ff::bound;

static bool close(double a, double b) { return std::fabs(a - b) < 1e-9; }

// tiny deterministic PRNG (no <random> host-only surprises)
static unsigned rng_state = 12345;
static double frand() { rng_state = rng_state * 1103515245u + 12345u; return ((rng_state >> 8) & 0xffff) / 65535.0; }

int main() {
    constexpr long H = 5, W = 6;

    // ---- (1) linear pull == manual bilinear ---------------------------
    {
        auto img = local<double, extents<long,H,W>>();
        for (long i=0;i<H;++i) for (long j=0;j<W;++j) img(i,j) = std::sin(0.3*i)+std::cos(0.2*j);
        bound b[2] = { bound::replicate, bound::replicate };
        for (double x : {0.0, 1.4, 3.9, 4.9}) for (double y : {0.2, 2.5, 5.4}) {
            double loc[2] = {x,y};
            double got = ff::pull(img, loc, /*order=*/1, b);
            long fx=(long)std::floor(x), fy=(long)std::floor(y); double tx=x-fx, ty=y-fy;
            auto cl=[](long i,long n){ return i<0?0:(i>=n?n-1:i); };
            auto at=[&](long i,long j){ return img(cl(i,H),cl(j,W)); };
            double ref=(at(fx,fy)*(1-ty)+at(fx,fy+1)*ty)*(1-tx)+(at(fx+1,fy)*(1-ty)+at(fx+1,fy+1)*ty)*tx;
            if (!close(got, ref)) { std::printf("bilinear mismatch\n"); return 1; }
        }
    }

    // ---- (2) adjoint identity across orders and boundary conditions ----
    const bound bmodes[] = { bound::replicate, bound::dct2, bound::dft, bound::zero };
    const int   orders[] = { 0, 1, 2, 3 };
    const int   NP = 20;                                  // grid points

    for (bound bm : bmodes) for (int order : orders) {
        bound b[2] = { bm, bm };

        auto x = local<double, extents<long,H,W>>();      // "input volume"
        for (long i=0;i<H;++i) for (long j=0;j<W;++j) x(i,j) = frand()*2 - 1;

        double loc[NP][2], y[NP];
        for (int p=0;p<NP;++p) { loc[p][0]=frand()*(H+1)-1; loc[p][1]=frand()*(W+1)-1; y[p]=frand()*2-1; }

        // lhs = <P x, y> = Σ_p pull(x, loc_p) * y_p
        double lhs = 0;
        for (int p=0;p<NP;++p) lhs += ff::pull(x, loc[p], order, b) * y[p];

        // rhs = <x, Pᵀ y>  where Pᵀ y = Σ_p push(loc_p, y_p)
        auto pushed = local<double, extents<long,H,W>>(); pushed.zero_();
        for (int p=0;p<NP;++p) ff::push(pushed, loc[p], y[p], order, b);
        double rhs = dot(x, pushed);

        if (std::fabs(lhs - rhs) > 1e-9) {
            std::printf("adjoint FAIL: bound=%d order=%d  lhs=%.12g rhs=%.12g\n",
                        (int)bm, order, lhs, rhs);
            return 2;
        }
    }

    std::printf("pushpull_adjoint: OK (pull==bilinear; <Px,y>==<x,Pty> for orders 0-3, 4 bounds)\n");
    return 0;
}
