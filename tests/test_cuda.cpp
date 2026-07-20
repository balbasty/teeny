// Structural test of teeny/cuda.h using the malloc-backed fake cuda_runtime.h
// (see tests/fakecuda/). Validates allocation / move / free / factories for the
// device / host / pinned owning modes. Not a GPU test.
#include <teeny/cuda.h>
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;
using cs::dynamic_extent;
using DynE = extents<long, dynamic_extent, dynamic_extent>;

static_assert(own_is_owning(own::device) && own_is_owning(own::host) && own_is_owning(own::pinned), "owning");
static_assert(!own_is_host_accessible(own::device), "device memory not host-accessible");
static_assert( own_is_host_accessible(own::host) && own_is_host_accessible(own::pinned), "host/pinned accessible");

int main()
{
    // device tensor: owning, move-only. (Fake malloc makes the pointer usable.)
    auto d = device<double, DynE>(DynE{2,3});
    static_assert(decltype(d)::ownership == own::device, "device mode");
    static_assert(!cs::is_copy_constructible<decltype(d)>::value, "owning is move-only");
    if (d.numel() != 6 || d.data() == nullptr) return 1;
    auto d2 = static_cast<decltype(d)&&>(d);          // move transfers ownership
    if (d.data() != nullptr) return 2;

    // page-locked host memory: host-accessible, so we can read/write.
    auto h = host<double, DynE>(DynE{2,2});
    h(0,0) = 1; h(1,1) = 4;
    if (h(0,0) != 1 || h(1,1) != 4) return 3;

    // pinned memory: same.
    auto p = pinned<float, extents<long,3>>(extents<long,3>{});
    p(0) = 7; if (p(0) != 7) return 4;

    return 0;
}
