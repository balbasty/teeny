// wrap()/factory ergonomics: a layout value-tag (ccontiguous{}/fcontiguous{}) on
// wrap and the creation factories, and wrap(mdspan) as a spelling of as_tensor.
// (#248)
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    double buf[6] = {1,2,3,4,5,6};   // row-major 2x3: [[1,2,3],[4,5,6]]

    // ---- wrap: layout value-tag ----------------------------------------
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

    // ---- wrap(mdspan): a spelling of as_tensor(mdspan) ------------------
    auto md = vc.mdspan();
    auto w1 = wrap(md);
    static_assert(cs::is_same<decltype(w1)::element_type, double>::value, "");
    if (w1(1,2)!=6)                              return 7;
    auto w2 = as_tensor(md);                     // as_tensor stays available, identical result
    if (w1(0,1)!=w2(0,1))                        return 8;
    auto ws = wrap(vc.permute(axis<1,0>{}).mdspan());   // works on a transposed submdspan too
    if (ws(2,1)!=6)                              return 9;

    // ---- factories: layout value-tag ------------------------------------
    auto ef = empty(shape<3,3>{}, fcontiguous{});
    static_assert(cs::is_same<decltype(ef)::layout_type, fcontiguous>::value, "");
    static_assert(decltype(ef)::ownership == storage::stack, "");

    auto zf = zeros(shape<3,3>{}, fcontiguous{});
    static_assert(cs::is_same<decltype(zf)::layout_type, fcontiguous>::value, "");
    if (zf(0,0)!=0.0f || zf(2,2)!=0.0f)          return 10;

    auto of = ones<double>(shape<3,3>{}, fcontiguous{});
    static_assert(cs::is_same<decltype(of)::layout_type, fcontiguous>::value, "");
    if (of(0,0)!=1.0 || of(1,1)!=1.0)            return 11;

    auto ff = full(shape<4>{}, 7, fcontiguous{});
    static_assert(cs::is_same<decltype(ff)::layout_type, fcontiguous>::value, "");
    if (ff(0)!=7 || ff(3)!=7)                     return 12;

    // explicit backend override composes with the layout value tag (heap despite static shape)
    auto zfh = zeros<double, storage::heap>(shape<3,3>{}, fcontiguous{});
    static_assert(decltype(zfh)::ownership == storage::heap, "");
    static_assert(cs::is_same<decltype(zfh)::layout_type, fcontiguous>::value, "");
    if (zfh(1,1)!=0.0)                            return 13;

    // dynamic shape -> heap path (host-allocating _TNY_HOST branch)
    auto zd = zeros(shape<-1,3>{2}, fcontiguous{});
    if (zd.numel()!=6)                            return 14;

    return 0;
}
