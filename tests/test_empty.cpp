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
    auto cf = empty<float, own::stack, fcontiguous>(shape<2,3>{});
    static_assert(cs::is_same<decltype(cf)::layout_type, fcontiguous>::value, "layout arg honoured");

    // ---- fill factories carry the same backend selector (host-accessible) ----
    auto zs = zeros<float>(shape<2,2>{});                      // deduced: static -> stack
    static_assert(decltype(zs)::ownership == own::stack, "zeros deduces stack for a static shape");
    if (zs(1,1) != 0.f) return 5;
    auto zh = zeros<float, own::heap>(shape<2,3>{});           // forced heap even though static
    static_assert(decltype(zh)::ownership == own::heap, "zeros<T,own::heap> -> heap");
    if (zh(1,2) != 0.f) return 6;
    auto oh = ones<double, own::heap>(shape<4>{});
    static_assert(decltype(oh)::ownership == own::heap, "ones<T,own::heap> -> heap");
    if (oh(3) != 1.0) return 7;
    auto fh = full<int, own::heap>(shape<2,2>{}, 9);
    static_assert(decltype(fh)::ownership == own::heap, "full<T,own::heap> -> heap");
    static_assert(cs::is_same<decltype(fh)::element_type, int>::value, "full<int> element type");
    if (fh(1,1) != 9) return 8;

    // value-tag backend forms on the fill factories
    auto zv = zeros<float>(shape<3>{}, own_c<own::heap>{});
    static_assert(decltype(zv)::ownership == own::heap, "zeros value-tag -> heap");
    auto fv = full<double>(shape<2>{}, 1.5, own_c<own::heap>{});
    static_assert(decltype(fv)::ownership == own::heap, "full value-tag -> heap");
    if (fv(1) != 1.5) return 9;

    // arange backend selector (dynamic 1-D); the static arange<T,N>() stays stack
    auto ah = arange<int, own::heap>(5);
    static_assert(decltype(ah)::ownership == own::heap, "arange<T,own::heap> -> heap");
    if (ah(4) != 4) return 10;
    auto av = arange<long>(4, own_c<own::heap>{});
    static_assert(decltype(av)::ownership == own::heap, "arange value-tag -> heap");
    if (av(3) != 3) return 11;
    auto asr = arange<double, 3>();                           // static form unaffected
    static_assert(decltype(asr)::ownership == own::stack, "arange<T,N>() still -> stack");
    if (asr(2) != 2.0) return 12;

    return 0;
}
