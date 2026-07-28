// wrap(): a layout value-tag (ccontiguous{}/fcontiguous{}), the twin of the
// existing wrap<Layout> template-argument spelling. (#269)
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    double buf[6] = {1,2,3,4,5,6};   // row-major 2x3: [[1,2,3],[4,5,6]]

    auto vc = wrap(buf, shape<2,3>{}, ccontiguous{});
    static_assert(cs::is_same<decltype(vc)::layout_type, ccontiguous>::value, "");
    if (vc(0,0)!=1 || vc(1,2)!=6)             return 1;

    auto vf = wrap(buf, shape<2,3>{}, fcontiguous{});   // == wrap<fcontiguous>(buf, shape<2,3>{})
    static_assert(cs::is_same<decltype(vf)::layout_type, fcontiguous>::value, "");
    if (vf(0,0)!=1 || vf(1,0)!=2 || vf(0,1)!=3) return 2;   // column-major read of the same buffer

    auto vt = wrap<fcontiguous>(buf, shape<2,3>{});     // the explicit-template spelling: same values
    if (vf(0,0)!=vt(0,0) || vf(1,2)!=vt(1,2))  return 3;

    // value-tag layout composes with the trailing storage_c<Space> tag
    auto vfs = wrap(buf, shape<2,3>{}, fcontiguous{}, storage_c<storage::view>{});
    static_assert(cs::is_same<decltype(vfs)::layout_type, fcontiguous>::value, "");
    static_assert(decltype(vfs)::ownership == storage::view, "");

    // existing wrap overloads still resolve (no ambiguity introduced by the new form)
    auto vs = wrap(buf, shape<3,3>{}, strides<1,3>{});         // strides<> tag -> its own overload
    if (vs(0,1)!=4)                             return 4;
    double rt[6] = {0,0,0,0,0,0};
    auto vr = wrap(rt, shape<2,3>{}, {3,1});                   // runtime-strides array overload
    vr(0,0) = 9; if (rt[0]!=9)                  return 5;
    auto vmix = wrap<dynamic_stride,1>(buf, shape<3,3>{}, {3}); // mixed static/runtime strides overload
    if (vmix(1,0)!=4)                            return 6;

    return 0;
}
