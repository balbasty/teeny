// Structural test of teeny/cuda.h using the malloc-backed fake cuda_runtime.h
// (see tests/fakecuda/). Validates allocation / move / free / factories for the
// gpu / pinned / mapped owning modes. Not a GPU test.
#include <teeny/cuda.h>
#include <teeny/teeny.h>
#include <teeny/dlpack.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;
using cs::dynamic_extent;
using DynE = extents<long, dynamic_extent, dynamic_extent>;

static_assert(own_is_owning(own::gpu) && own_is_owning(own::pinned) && own_is_owning(own::mapped), "owning");
static_assert(!own_is_host_accessible(own::gpu), "gpu memory not host-accessible");
static_assert( own_is_host_accessible(own::pinned) && own_is_host_accessible(own::mapped), "pinned/mapped accessible");

int main()
{
    // gpu tensor: owning, move-only. (Fake malloc makes the pointer usable.)
    auto d = gpu<double, DynE>(DynE{2,3});
    static_assert(decltype(d)::ownership == own::gpu, "gpu mode");
    static_assert(!cs::is_copy_constructible<decltype(d)>::value, "owning is move-only");
    if (d.numel() != 6 || d.data() == nullptr) return 1;
    auto d2 = static_cast<decltype(d)&&>(d);          // move transfers ownership
    if (d.data() != nullptr) return 2;

    // page-locked ("pinned") host memory: host-accessible, so we can read/write.
    auto h = pinned<double, DynE>(DynE{2,2});
    h(0,0) = 1; h(1,1) = 4;
    if (h(0,0) != 1 || h(1,1) != 4) return 3;

    // mapped (zero-copy) memory: same.
    auto p = mapped<float, extents<long,3>>(extents<long,3>{});
    p(0) = 7; if (p(0) != 7) return 4;

    // ---- #15: a VIEW of device memory carries its space (own::gpu_view) -------
    static_assert(own_is_device(own::gpu) && own_is_device(own::gpu_view), "device modes");
    static_assert(!own_is_host_accessible(own::gpu_view), "gpu_view not host-accessible");
    static_assert(!own_is_owning(own::gpu_view) && own_is_view(own::gpu_view), "gpu_view is a non-owning view");
    static_assert(own_view_of(own::gpu) == own::gpu_view, "view of gpu -> gpu_view");
    static_assert(own_view_of(own::gpu_view) == own::gpu_view, "view of gpu_view -> gpu_view");
    static_assert(own_view_of(own::heap) == own::view && own_view_of(own::stack) == own::view, "host source -> host view");
    static_assert(own_view_of(own::pinned) == own::view && own_view_of(own::mapped) == own::view, "pinned/mapped are host -> view");

    // slicing / structure / peel of a gpu tensor all yield gpu_view, not view.
    auto g = gpu<float, shape<4,5>>(shape<4,5>{});
    static_assert(decltype(g(1, all))::ownership       == own::gpu_view, "gpu slice -> gpu_view");
    static_assert(decltype(g(all, slice(1,4)))::ownership == own::gpu_view, "gpu range slice -> gpu_view");
    static_assert(decltype(g.at(0,0))::ownership       == own::gpu_view, "gpu .at -> gpu_view");
    static_assert(decltype(g.permute<1,0>())::ownership == own::gpu_view, "gpu permute -> gpu_view");
    static_assert(decltype(g.flip<0>())::ownership     == own::gpu_view, "gpu flip -> gpu_view");
    static_assert(decltype(g.unsqueeze<0>())::ownership == own::gpu_view, "gpu unsqueeze -> gpu_view");
    static_assert(decltype(peel_front_at<1>(g, 0))::ownership == own::gpu_view, "gpu peel_front_at -> gpu_view");
    static_assert(decltype(peel_at<0>(g, 0))::ownership == own::gpu_view, "gpu peel_at -> gpu_view");
    // a slice of a gpu_view stays a gpu_view (space is preserved through chains).
    auto gv = g(all, slice(0,3));
    static_assert(decltype(gv.permute<1,0>())::ownership == own::gpu_view, "gpu_view chain stays gpu_view");

    // ...and the rest of the view ops too (take_along / squeeze / reshape /
    // recast / flatten / peel range) — a contiguous gpu source, so all are valid.
    static_assert(decltype(g.take_along<0>(1))::ownership == own::gpu_view, "gpu take_along -> gpu_view");
    static_assert(decltype(g.unsqueeze<0>().squeeze<0>())::ownership == own::gpu_view, "gpu squeeze -> gpu_view");
    static_assert(decltype(g.reshape<2,10>())::ownership == own::gpu_view, "gpu reshape -> gpu_view");
    static_assert(decltype(g.recast<shape<4,5>>())::ownership == own::gpu_view, "gpu recast -> gpu_view");
    static_assert(decltype(g.flatten())::ownership     == own::gpu_view, "gpu flatten -> gpu_view");
    static_assert(decltype(peel<0>(g)[0])::ownership   == own::gpu_view, "gpu peel range -> gpu_view");

    // DLPack export of a device view now works and is labeled kDLCUDA (before #15
    // a gpu slice was own::view and exported as kDLCPU — a silent mislabel).
    auto * dl = to_dlpack(g(1, all));
    if (dl->dl_tensor.device.device_type != kDLCUDA) return 12;
    dl->deleter(dl);

    // contrast: a view of host-accessible owning memory (pinned) is a plain view.
    auto pm = pinned<float, shape<4,5>>(shape<4,5>{});
    static_assert(decltype(pm(1, all))::ownership == own::view, "pinned slice -> host view");
    auto * dlp = to_dlpack(pm(1, all));
    if (dlp->dl_tensor.device.device_type != kDLCPU) return 13;   // pinned view is host
    dlp->deleter(dlp);

    return 0;
}
