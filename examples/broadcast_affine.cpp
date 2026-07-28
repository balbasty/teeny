// broadcast_affine — per-channel affine transform (x * scale + bias) on a
// (C,H,W) image, with (C,1,1) parameters broadcast numpy-style over H and W.
//
// Shows: in-place broadcasting (add_/mul_ with a lower-effective-shape operand)
// and out-of-place broadcasting (a fully-static result folds to a stack tensor,
// host AND device).
#include <teeny/teeny.h>
#include <cstdio>

using namespace tny;

static bool close(double a, double b) { double d=a-b; return (d<0?-d:d) < 1e-9; }

int main() {
    constexpr long C=3, H=2, W=4;
    auto img   = local<double, shape<C,H,W>>();
    auto scale = local<double, shape<C,1,1>>();   // one per channel
    auto bias  = local<double, shape<C,1,1>>();

    for (long c=0;c<C;++c) { scale(c,0,0) = c+1; bias(c,0,0) = 10*(c+1); }
    for (long c=0;c<C;++c) for (long h=0;h<H;++h) for (long w=0;w<W;++w)
        img(c,h,w) = c*100 + h*10 + w;

    // in-place: img = img * scale + bias  (scale/bias broadcast over H,W)
    img.mul_(scale);
    img.add_(bias);

    for (long c=0;c<C;++c) for (long h=0;h<H;++h) for (long w=0;w<W;++w) {
        double want = (c*100 + h*10 + w) * (c+1) + 10*(c+1);
        if (!close(img(c,h,w), want)) { std::printf("in-place mismatch\n"); return 1; }
    }

    // out-of-place: a (C,1,1) column + a (1,H,W) plane -> (C,H,W) stack tensor
    auto col   = local<double, shape<C,1,1>>();
    auto plane = local<double, shape<1,H,W>>();
    for (long c=0;c<C;++c) col(c,0,0)=c;
    for (long h=0;h<H;++h) for (long w=0;w<W;++w) plane(0,h,w)=h*10+w;

    auto out = col + plane;
    static_assert(decltype(out)::rank()==3 && decltype(out)::is_static, "static broadcast -> stack");
    static_assert(decltype(out.shape(Int<0>()))::value==C, "result C");   // friendly static extent
    static_assert(decltype(out.shape(Int<1>()))::value==H, "result H");
    for (long c=0;c<C;++c) for (long h=0;h<H;++h) for (long w=0;w<W;++w)
        if (!close(out(c,h,w), col(c,0,0)+plane(0,h,w))) { std::printf("oop mismatch\n"); return 2; }

    std::printf("broadcast_affine: OK\n");
    return 0;
}
