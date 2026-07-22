// distance_l1 ported onto tny::md, validated bit-identical against the jitfields
// batching plumbing. With the nd-peel the batch loop *is* the library: peel the
// leading axes and each slice is a 1-D line along the last axis -- no fillfrom,
// no prod<nbatch>, no index2offset.
#include <teeny/teeny.h>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;

typedef long   offset_t;
typedef double scalar_t;
static inline scalar_t vmin(scalar_t a, scalar_t b) { return a < b ? a : b; }

// The 1-D L1 transform, now on a rank-1 view of any stride (line(i) applies it).
template <class Line>
static void algo(Line line, scalar_t w) {
    const offset_t n = line.extent(0);
    if (n == 1) return;
    scalar_t tmp = line(0);
    for (offset_t i = 1;   i < n;  ++i) { tmp = vmin(tmp + w, line(i)); line(i) = tmp; }
    for (offset_t i = n-2; i >= 0; --i) { tmp = vmin(tmp + w, line(i)); line(i) = tmp; }
}

// The whole kernel body: peel the batch axes, transform each line.
template <class Tensor>
static void run_md(Tensor & t, scalar_t w) {
    for (auto line : peel<0,1>(t)) algo(line, w);          // rank-3: batch = axes 0,1
}

/* --- faithful jitfields reference (same as test_tensor_distance_l1) --- */
namespace before {
template <int nd> offset_t index2offset(offset_t idx, const offset_t * sz, const offset_t * st) {
    offset_t ni = 0, ni1, cur = 1, nxt = 1;
    for (int i = 0; i < nd; ++i) { nxt = cur*sz[i]; ni1 = (idx%nxt)/cur; cur = nxt; ni += ni1*st[i]; }
    return ni;
}
static void algo1(scalar_t * f, offset_t n, offset_t s, scalar_t w) {
    if (n == 1) return;
    scalar_t tmp = *f; f += s;
    for (offset_t i = 1;   i < n;  ++i, f += s) { tmp = vmin(tmp+w, *f); *f = tmp; }
    f -= 2*s;
    for (offset_t i = n-2; i >= 0; --i, f -= s) { tmp = vmin(tmp+w, *f); *f = tmp; }
}
template <int nd> void run(scalar_t * f, scalar_t w, const offset_t * sz, const offset_t * st) {
    constexpr int nb = nd - 1;
    offset_t n = sz[nb], s = st[nb], numel = 1;
    for (int i = 0; i < nb; ++i) numel *= sz[i];
    for (offset_t i = 0; i < numel; ++i) algo1(f + index2offset<nb>(i, sz, st), n, s, w);
}
}

static void fill(scalar_t * f, offset_t n) { for (offset_t i = 0; i < n; ++i) f[i] = ((i*37+5)%11 < 3) ? 0.0 : 1e9; }

int main()
{
    const offset_t size[3] = {4,5,6}, stride[3] = {1,4,20}, total = 120;  // F-contiguous
    scalar_t a[128], b[128];

    // (1) fully dynamic extents
    fill(a,total); fill(b,total);
    before::run<3>(a, 1.5, size, stride);
    {
        using E = extents<offset_t, cs::dynamic_extent, cs::dynamic_extent, cs::dynamic_extent>;
        auto t = wrap<cs::layout_left>(b, E{4,5,6});     // strides (1,4,20)
        run_md(t, 1.5);
    }
    for (offset_t i = 0; i < total; ++i) if (a[i] != b[i]) return 1;

    // (2) fully static extents (folds)
    fill(a,total); fill(b,total);
    before::run<3>(a, 1.5, size, stride);
    {
        auto t = wrap<cs::layout_left>(b, extents<offset_t,4,5,6>{});
        run_md(t, 1.5);
    }
    for (offset_t i = 0; i < total; ++i) if (a[i] != b[i]) return 2;

    return 0;
}
