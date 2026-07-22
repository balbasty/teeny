#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main()
{
    // ---- ellipsis expands to (rank - #other args) copies of `all` -----------
    double buf[2*3*4];
    for (int i = 0; i < 2*3*4; ++i) buf[i] = i;
    auto t = wrap(buf, shape<2,3,4>{});   // rank 3, contiguous

    // t(ellipsis) == the whole thing (rank 3 view, static extents preserved).
    auto a = t(ellipsis);
    static_assert(decltype(a)::rank() == 3, "ellipsis alone keeps rank");
    static_assert(decltype(a)::extents_type::static_extent(0) == 2, "extent preserved");
    static_assert(decltype(a)::extents_type::static_extent(2) == 4, "extent preserved");
    if (a(1,2,3) != buf[1*12+2*4+3]) return 1;

    // t(1, ellipsis) == t(1, all, all): drop axis 0, keep the rest (rank 2).
    auto b = t(1, ellipsis);
    static_assert(decltype(b)::rank() == 2, "leading index + ellipsis");
    if (b(2,3) != buf[12 + 2*4 + 3]) return 2;

    // t(ellipsis, 2) == t(all, all, 2): drop the LAST axis (rank 2).
    auto c = t(ellipsis, 2);
    static_assert(decltype(c)::rank() == 2, "ellipsis + trailing index");
    if (c(1,2) != buf[1*12 + 2*4 + 2]) return 3;

    // t(1, ellipsis, 2) == t(1, all, 2): ellipsis fills the ONE middle axis.
    auto d = t(1, ellipsis, 2);
    static_assert(decltype(d)::rank() == 1, "index + ellipsis + index");
    if (d(2) != buf[12 + 2*4 + 2]) return 4;

    // ellipsis covering ZERO axes, all-integer remainder -> an element T& (numpy
    // a[0, ..., 1] on a fully-indexed access).
    t(0, ellipsis, 1, 3) = 99.0;
    if (buf[0*12 + 1*4 + 3] != 99.0) return 5;
    buf[0*12 + 1*4 + 3] = 0*12 + 1*4 + 3;   // restore

    // a slice arg alongside the ellipsis still routes to a view.
    auto e = t(ellipsis, slice(1,3));
    static_assert(decltype(e)::rank() == 3, "ellipsis + range");
    if (e.extent(2) != 2) return 6;
    if (e(1,2,0) != buf[1*12 + 2*4 + 1]) return 7;

    // ---- assigning INTO a slice copies CONTENTS (numpy a[:] = b) -------------
    double dst[2*3*4] = {};
    auto D = wrap(dst, shape<2,3,4>{});
    double src[2*3*4];
    for (int i = 0; i < 2*3*4; ++i) src[i] = 100 + i;
    auto S = wrap(src, shape<2,3,4>{});

    D(ellipsis) = S;                         // whole-tensor copy
    for (int i = 0; i < 2*3*4; ++i) if (dst[i] != src[i]) return 8;

    // scalar fill through a slice
    D(0, all, all) = 7.0;                     // fill the first plane
    for (int j = 0; j < 12; ++j) if (dst[j] != 7.0) return 9;
    for (int j = 12; j < 24; ++j) if (dst[j] != src[12+j-12]) return 10;

    // partial region copy with broadcasting (a (4,) row into a (3,4) plane)
    double row[4] = {1,2,3,4};
    auto R = wrap(row, shape<1,4>{});         // broadcasts over the size-3 axis
    D(1, ellipsis) = R;
    for (int r = 0; r < 3; ++r)
        for (int col = 0; col < 4; ++col)
            if (dst[12 + r*4 + col] != row[col]) return 11;

    // `a = b` on a NAMED view still REBINDS (does not copy) -- the contrast.
    double p[3] = {1,2,3}, q[3] = {4,5,6};
    auto P = wrap(p, shape<3>{});
    auto Q = wrap(q, shape<3>{});
    P = Q;                                    // rebind: P now views q
    if (P.data() != q) return 12;
    P(0) = 42.0;                              // writes q, not p
    if (q[0] != 42.0 || p[0] != 1.0) return 13;

    return 0;
}
