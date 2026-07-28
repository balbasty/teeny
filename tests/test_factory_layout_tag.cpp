// empty/zeros/ones/full: a layout value-tag (ccontiguous{}/fcontiguous{}),
// the twin of the explicit <..., Layout> template-argument spelling. (#271)
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    auto ef = empty(shape<3,3>{}, fcontiguous{});
    static_assert(cs::is_same<decltype(ef)::layout_type, fcontiguous>::value, "");
    static_assert(decltype(ef)::ownership == storage::stack, "");

    auto zf = zeros(shape<3,3>{}, fcontiguous{});
    static_assert(cs::is_same<decltype(zf)::layout_type, fcontiguous>::value, "");
    if (zf(0,0)!=0.0f || zf(2,2)!=0.0f)          return 1;

    auto of = ones<double>(shape<3,3>{}, fcontiguous{});
    static_assert(cs::is_same<decltype(of)::layout_type, fcontiguous>::value, "");
    if (of(0,0)!=1.0 || of(1,1)!=1.0)            return 2;

    auto ff = full(shape<4>{}, 7, fcontiguous{});
    static_assert(cs::is_same<decltype(ff)::layout_type, fcontiguous>::value, "");
    if (ff(0)!=7 || ff(3)!=7)                     return 3;

    // explicit backend override composes with the layout value tag (heap despite static shape)
    auto zfh = zeros<double, storage::heap>(shape<3,3>{}, fcontiguous{});
    static_assert(decltype(zfh)::ownership == storage::heap, "");
    static_assert(cs::is_same<decltype(zfh)::layout_type, fcontiguous>::value, "");
    if (zfh(1,1)!=0.0)                            return 4;

    // dynamic shape -> heap path (host-allocating _TNY_HOST branch)
    auto zd = zeros(shape<-1,3>{2}, fcontiguous{});
    if (zd.numel()!=6)                            return 5;

    return 0;
}
