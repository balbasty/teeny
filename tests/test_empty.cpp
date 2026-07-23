// empty<T[, Space]>(shape) — the unified creation factory the make_* family
// fuses into. Deduces stack (static shape) / heap (dynamic), or takes an explicit
// backend as a template arg or a value-tag. (gpu/pinned/mapped live in test_cuda.)
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main()
{
    // ---- deduced ownership: static shape -> stack (host+device) --------------
    auto s = empty<float>(shape<2,3>{});
    static_assert(decltype(s)::ownership == own::stack, "static shape -> stack");
    static_assert(cs::is_same<decltype(s)::element_type, float>::value, "element type T");
    static_assert(decltype(s)::rank() == 2, "rank kept");
    s.fill_(7.f); if (s(1,2) != 7.f) return 1;                 // real, writable storage

    // T defaults to float
    auto sf = empty<>(shape<4>{});
    static_assert(cs::is_same<decltype(sf)::element_type, float>::value, "empty<> defaults to float");

    // ---- deduced ownership: any dynamic extent -> heap (host) ----------------
    auto h = empty<double>(shape<-1,3>{2});
    static_assert(decltype(h)::ownership == own::heap, "dynamic shape -> heap");
    static_assert(cs::is_same<decltype(h)::element_type, double>::value, "element type T");
    h.iota_(0.0, 1.0); if (h(1,2) != 5.0) return 2;

    // ---- explicit backend via template arg (overrides the deduction) ---------
    auto eh = empty<int, own::heap>(shape<2,2>{});             // heap even though static
    static_assert(decltype(eh)::ownership == own::heap, "empty<T,own::heap> forces heap");
    auto es = empty<int, own::stack>(shape<5>{});
    static_assert(decltype(es)::ownership == own::stack, "empty<T,own::stack> -> stack");

    // ---- explicit backend via value-tag: empty<T>(shape, own_c<...>{}) -------
    auto vh = empty<float>(shape<3,3>{}, own_c<own::heap>{});
    static_assert(decltype(vh)::ownership == own::heap, "value-tag own::heap -> heap");
    vh.fill_(1.f); if (vh(2,2) != 1.f) return 3;

    // ---- make_local / make_heap still work (thin spellings of empty) ---------
    auto ml = make_local<double>(shape<2,2>{});
    static_assert(decltype(ml)::ownership == own::stack, "make_local -> stack");
    auto mh = make_heap<int>(shape<-1>{5});
    static_assert(decltype(mh)::ownership == own::heap, "make_heap -> heap");
    mh.iota_(); if (mh(4) != 4) return 4;

    // ---- a non-default layout threads through ---------------------------------
    auto cf = empty<float, own::stack, forder>(shape<2,3>{});
    static_assert(cs::is_same<decltype(cf)::layout_type, forder>::value, "layout arg honoured");

    return 0;
}
