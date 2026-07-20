#include <teeny/md.h>
#include <cuda/std/type_traits>

using namespace tny::md;
namespace cs = cuda::std;
using cs::extents;
using cs::dynamic_extent;

// A fully-static tensor view is exactly its data pointer (EBO on the mapping).
using static_view = tensor<double, extents<long,2,3,4>, cs::layout_right, own::view>;
static_assert(static_view::rank() == 3, "rank");
static_assert(static_view::is_static, "static");
static_assert(cs::is_trivially_copyable<static_view>::value, "view trivially copyable");
static_assert(sizeof(static_view) == sizeof(double*), "static view == just a pointer");

// A stack tensor stores exactly its elements.
using stack_33 = tensor<double, extents<long,3,3>, cs::layout_right, own::stack>;
static_assert(sizeof(stack_33) == 9 * sizeof(double), "stack tensor == its data");
static_assert(cs::is_trivially_copyable<stack_33>::value, "stack trivially copyable");

int main()
{
    double buf[24];
    for (long i = 0; i < 24; ++i) buf[i] = i;

    // ---- view over a contiguous buffer --------------------------------
    auto v = view(buf, extents<long,2,3,4>{});
    if (v(1,2,3) != 1*12 + 2*4 + 3) return 1;
    if (v.numel() != 24 || v.extent(1) != 3) return 2;
    v(0,0,0) = 99; if (buf[0] != 99) return 3;

    // ---- per-dim compile-time NON-contiguous strides (posdef case) ----
    // batch of 3x3 with a padded batch stride 16 (not the contiguous 9).
    double pad[64]; for (int i=0;i<64;++i) pad[i]=i;
    auto s = view_strided<16,3,1>(pad, extents<long,dynamic_extent,3,3>{4});
    if (s(2,1,0) != 2*16 + 1*3 + 0) return 4;
    static_assert(decltype(s)::rank() == 3, "strided rank");

    // ---- stack-owned, value semantics ---------------------------------
    auto m = local<double, extents<long,3,3>>();
    m(0,0) = 1; m(1,1) = 2; m(2,2) = 3;
    auto m2 = m;                       // deep copy
    m2(0,0) = 42; if (m(0,0) != 1) return 5;

    // ---- heap-owned (host), dynamic shape, move-only ------------------
    using DynE = extents<long,dynamic_extent,dynamic_extent>;
    auto h = owned<double, DynE>(DynE{2,3});
    h(1,2) = 7; if (h(1,2) != 7 || h.numel() != 6) return 6;
    auto h2 = static_cast<decltype(h)&&>(h);            // move
    if (h2(1,2) != 7 || h.data() != nullptr) return 7;

    // ---- .view() yields a plain cuda::std::mdspan ---------------------
    auto md = v.view();
    static_assert(cs::is_same<decltype(md), cs::mdspan<double, extents<long,2,3,4>, cs::layout_right>>(), "view() -> mdspan");
    if (md.extent(2) != 4) return 8;

    // ---- helpers: channel peel + batch offset -------------------------
    auto full = view(buf, extents<long,2,3,4>{});         // treat dim0 as "channel"
    auto sp = channel(full.view(), 1);                    // spatial (3,4) view of channel 1
    static_assert(decltype(sp)::rank() == 2, "channel peeled");
    if (sp(0,0) != full(1,0,0)) return 9;

    // batch_offset decodes an F-order index over the leading dims (0,1), last dim = 0.
    auto fv = full.view();                                // (2,3,4)
    for (long lin = 0; lin < 2*3; ++lin) {
        long i0 = lin % 2, i1 = lin / 2;                  // F-order over dims 0,1
        if (batch_offset(fv, lin) != fv.mapping()(i0, i1, 0)) return 10;
    }

    return 0;
}
