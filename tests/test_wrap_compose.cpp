// wrap()/make_local()/make_heap()/make_view(): the generic keyword-tag bag
// (#282) composed in ways not otherwise exercised by
// test_wrap_layout_tag.cpp/test_empty.cpp -- dtype x layout on
// make_local/make_heap (either order), and a layout tag + storage_c on wrap/
// make_view. A gap flagged by the #277 umbrella's cascade review: #282
// shipped without dedicated tests of its own.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    // make_local: dtype alone, dtype + layout (either order)
    auto l1 = make_local(shape<2,3>{}, dtype<double>{});
    static_assert(cs::is_same<typename decltype(l1)::element_type, double>::value, "");
    static_assert(decltype(l1)::ownership == storage::stack, "");
    l1.iota_(0.0, 1.0);
    if (l1(0,0)!=0.0 || l1(1,2)!=5.0) return 1;

    auto l2 = make_local(shape<2,3>{}, fcontiguous{}, dtype<float>{});
    static_assert(cs::is_same<typename decltype(l2)::element_type, float>::value, "");
    static_assert(cs::is_same<decltype(l2)::layout_type, fcontiguous>::value, "");
    l2.iota_(0.0f, 1.0f);   // iota_ fills in LOGICAL row-major order regardless of physical layout
    if (l2(0,0)!=0.0f || l2(1,0)!=3.0f) return 2;

    auto l3 = make_local(shape<2,3>{}, dtype<float>{}, fcontiguous{});   // reversed order
    static_assert(cs::is_same<decltype(l2), decltype(l3)>::value, "order-independent");

    // make_heap: same composition, dynamic shape
    auto h1 = make_heap(shape<-1,3>{4}, dtype<double>{});
    static_assert(cs::is_same<typename decltype(h1)::element_type, double>::value, "");
    static_assert(decltype(h1)::ownership == storage::heap, "");
    h1.iota_(0.0, 1.0);
    if (h1(0,0)!=0.0 || h1(3,2)!=11.0) return 3;

    auto h2 = make_heap(shape<-1,3>{4}, fcontiguous{}, dtype<int>{});
    static_assert(cs::is_same<typename decltype(h2)::element_type, int>::value, "");
    static_assert(cs::is_same<decltype(h2)::layout_type, fcontiguous>::value, "");

    // wrap: Layout is a distinct POSITIONAL slot (3rd arg, value tag or template
    // arg), not a composable keyword -- storage_c is the only trailing keyword
    // wrap accepts (#282; see CLAUDE.md). So the layout tag must come BEFORE the
    // trailing storage_c keyword; it cannot be reordered like a true keyword.
    double buf[6] = {1,2,3,4,5,6};
    auto v1 = wrap(buf, shape<2,3>{}, fcontiguous{}, storage_c<storage::view>{});
    static_assert(cs::is_same<decltype(v1)::layout_type, fcontiguous>::value, "");
    static_assert(decltype(v1)::ownership == storage::view, "");
    if (v1(0,0)!=1 || v1(1,0)!=2) return 4;

    // make_view: forwards its trailing keyword bag to wrap too (storage_c only --
    // like wrap's own bare-template-Layout overload, make_view's Layout is a
    // template argument, not a value-tag keyword; use the explicit form for F-order)
    auto v3 = make_view<fcontiguous>(buf, shape<2,3>{});
    static_assert(cs::is_same<decltype(v3)::layout_type, fcontiguous>::value, "");
    if (v3(0,0)!=1 || v3(1,0)!=2) return 5;
    auto v4 = make_view(buf, shape<2,3>{}, storage_c<storage::view>{});
    static_assert(cs::is_same<decltype(v4)::layout_type, ccontiguous>::value, "");
    if (v4(0,0)!=1 || v4(1,2)!=6) return 6;

    return 0;
}
