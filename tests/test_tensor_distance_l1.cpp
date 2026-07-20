// Proof: port the core of the jitfields `distance_l1` CUDA kernel to teeny and
// check it produces bit-identical results to the original batching plumbing.
//
// The 1-D transform `algo` (from jitfields/csrc/lib/distance/l1.h) is unchanged
// -- it is not what teeny replaces. teeny replaces the per-kernel boilerplate
// around it: the `fillfrom` stack copies, `prod<nbatch>`, and
// `index2offset<nbatch>` (jitfields/csrc/lib/batch.h) that turn a launch index
// into a strided memory offset. With teeny that becomes: build a tensor view,
// drop the transformed dimension, ask the batch view for its numel and offsets.
//
// Builds on host with g++/clang++ at C++17 (no CUDA, no jitfields headers).

#include <teeny/tensor.h>
#include <cuda/std/type_traits>

using namespace tny;
using namespace tny::statix;

typedef long   offset_t;
typedef double scalar_t;

static inline scalar_t vmin(scalar_t a, scalar_t b) { return a < b ? a : b; }

// The 1-D L1 distance-transform pass, copied verbatim from l1.h (minus
// __device__). teeny does not touch this.
static void algo(scalar_t * f, offset_t size, offset_t stride, scalar_t w)
{
    if (size == 1) return;
    scalar_t tmp = *f;
    f += stride;
    for (offset_t i = 1; i < size; ++i, f += stride) { tmp = vmin(tmp + w, *f); *f = tmp; }
    f -= 2 * stride;
    for (offset_t i = size - 2; i >= 0; --i, f -= stride) { tmp = vmin(tmp + w, *f); *f = tmp; }
}

/* ============================ BEFORE (jitfields) ======================== *
 *  Faithful host copies of the batch.h helpers + the kernel() body.        */
namespace before {

template <int ndim> void fillfrom(offset_t * dst, const offset_t * src)
{ for (int i = 0; i < ndim; ++i) dst[i] = src[i]; }

template <int ndim> offset_t prod(const offset_t * s)
{ offset_t p = 1; for (int i = 0; i < ndim; ++i) p *= s[i]; return p; }

template <int ndim> offset_t index2offset(offset_t index, const offset_t * size, const offset_t * stride)
{
    offset_t ni = 0, ni1, cur = 1, nxt = 1;
    for (int i = 0; i < ndim; ++i) {
        nxt = cur * size[i];
        ni1 = (index % nxt) / cur;
        cur = nxt;
        ni += ni1 * stride[i];
    }
    return ni;
}

template <int ndim>
void run(scalar_t * f, scalar_t w, const offset_t * _size, const offset_t * _stride)
{
    constexpr int nbatch = ndim - 1;
    offset_t size  [ndim]; fillfrom<ndim>(size,   _size);
    offset_t stride[ndim]; fillfrom<ndim>(stride, _stride);
    offset_t n = size[nbatch];
    offset_t s = stride[nbatch];
    offset_t numel = prod<nbatch>(size);
    for (offset_t i = 0; i < numel; ++i) {
        offset_t offset = index2offset<nbatch>(i, size, stride);
        algo(f + offset, n, s, w);
    }
}

} // namespace before

/* ============================= AFTER (teeny) =========================== *
 *  Shape is a teeny `values` pack -- pass dynamic_values<ndim> for a fully  *
 *  dynamic kernel, or bake extents in as cvalue<> to specialize.           */
namespace after {

template <class Shape>
void run(scalar_t * f, scalar_t w, const offset_t * size, const offset_t * stride)
{
    auto t = make_tensor<Shape>(f, size, stride);
    constexpr size_t last = decltype(t)::ndim - 1;

    auto     batch = t.template sub<last>((offset_t)0);   // view over the batch dims
    offset_t n     = t.size(cptrdiff<-1>());              // transformed extent
    offset_t s     = t.stride_at(cptrdiff<-1>());         // its stride
    offset_t numel = batch.numel();                       // prod over batch dims

    for (offset_t i = 0; i < numel; ++i)
        algo(f + batch.foffset(i), n, s, w);              // index2offset, folded
}

} // namespace after

static void fill(scalar_t * f, offset_t n) {
    // deterministic pseudo-foreground/background pattern
    for (offset_t i = 0; i < n; ++i) f[i] = ((i * 37 + 5) % 11 < 3) ? 0.0 : 1e9;
}

template <class Shape>
static int check(const offset_t (&size)[3], const offset_t (&stride)[3], offset_t total)
{
    scalar_t a[512], b[512];
    fill(a, total); fill(b, total);
    before::run<3>(a, 1.5, size, stride);
    after::run<Shape>(b, 1.5, size, stride);
    for (offset_t i = 0; i < total; ++i)
        if (a[i] != b[i]) return 1;
    return 0;
}

int main()
{
    // F-contiguous [4,5,6]: stride = [1, 4, 20], 120 elements.
    const offset_t size[3]   = {4, 5, 6};
    const offset_t stride[3] = {1, 4, 20};
    const offset_t total     = 120;

    // (1) fully dynamic view -- one instantiation handles any shape.
    if (check<dynamic_values<3> >(size, stride, total)) return 1;

    // (2) fully static shape -- extents baked into the type (folds to
    //     constants); must still match the dynamic run bit-for-bit.
    using SS = tuple<cvalue<offset_t,4>, cvalue<offset_t,5>, cvalue<offset_t,6> >;
    if (check<SS>(size, stride, total)) return 2;

    // (3) mixed: batch extents static, transformed dim dynamic.
    using MS = tuple<cvalue<offset_t,4>, cvalue<offset_t,5>, cnone>;
    if (check<MS>(size, stride, total)) return 3;

    return 0;
}
