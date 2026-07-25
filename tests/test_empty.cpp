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
    static_assert(decltype(s)::ownership == storage::stack, "static shape -> stack");
    static_assert(cs::is_same<decltype(s)::element_type, float>::value, "element type T");
    static_assert(decltype(s)::rank() == 2, "rank kept");
    s.fill_(7.f); if (s(1,2) != 7.f) return 1;                 // real, writable storage

    // T defaults to float
    auto sf = empty<>(shape<4>{});
    static_assert(cs::is_same<decltype(sf)::element_type, float>::value, "empty<> defaults to float");

    // ---- deduced ownership: any dynamic extent -> heap (host) ----------------
    auto h = empty<double>(shape<-1,3>{2});
    static_assert(decltype(h)::ownership == storage::heap, "dynamic shape -> heap");
    static_assert(cs::is_same<decltype(h)::element_type, double>::value, "element type T");
    h.iota_(0.0, 1.0); if (h(1,2) != 5.0) return 2;

    // ---- explicit backend via template arg (overrides the deduction) ---------
    auto eh = empty<int, storage::heap>(shape<2,2>{});             // heap even though static
    static_assert(decltype(eh)::ownership == storage::heap, "empty<T,storage::heap> forces heap");
    auto es = empty<int, storage::stack>(shape<5>{});
    static_assert(decltype(es)::ownership == storage::stack, "empty<T,storage::stack> -> stack");

    // ---- explicit backend via value-tag: empty<T>(shape, storage_c<...>{}) -------
    auto vh = empty<float>(shape<3,3>{}, storage_c<storage::heap>{});
    static_assert(decltype(vh)::ownership == storage::heap, "value-tag storage::heap -> heap");
    vh.fill_(1.f); if (vh(2,2) != 1.f) return 3;

    // ---- make_local / make_heap still work (thin spellings of empty) ---------
    auto ml = make_local<double>(shape<2,2>{});
    static_assert(decltype(ml)::ownership == storage::stack, "make_local -> stack");
    auto mh = make_heap<int>(shape<-1>{5});
    static_assert(decltype(mh)::ownership == storage::heap, "make_heap -> heap");
    mh.iota_(); if (mh(4) != 4) return 4;

    // ---- a non-default layout threads through ---------------------------------
    auto cf = empty<float, storage::stack, fcontiguous>(shape<2,3>{});
    static_assert(cs::is_same<decltype(cf)::layout_type, fcontiguous>::value, "layout arg honoured");

    // ---- fill factories carry the same backend selector (host-accessible) ----
    auto zs = zeros<float>(shape<2,2>{});                      // deduced: static -> stack
    static_assert(decltype(zs)::ownership == storage::stack, "zeros deduces stack for a static shape");
    if (zs(1,1) != 0.f) return 5;
    auto zh = zeros<float, storage::heap>(shape<2,3>{});           // forced heap even though static
    static_assert(decltype(zh)::ownership == storage::heap, "zeros<T,storage::heap> -> heap");
    if (zh(1,2) != 0.f) return 6;
    auto oh = ones<double, storage::heap>(shape<4>{});
    static_assert(decltype(oh)::ownership == storage::heap, "ones<T,storage::heap> -> heap");
    if (oh(3) != 1.0) return 7;
    auto fh = full<int, storage::heap>(shape<2,2>{}, 9);
    static_assert(decltype(fh)::ownership == storage::heap, "full<T,storage::heap> -> heap");
    static_assert(cs::is_same<decltype(fh)::element_type, int>::value, "full<int> element type");
    if (fh(1,1) != 9) return 8;

    // value-tag backend forms on the fill factories
    auto zv = zeros<float>(shape<3>{}, storage_c<storage::heap>{});
    static_assert(decltype(zv)::ownership == storage::heap, "zeros value-tag -> heap");
    auto fv = full<double>(shape<2>{}, 1.5, storage_c<storage::heap>{});
    static_assert(decltype(fv)::ownership == storage::heap, "full value-tag -> heap");
    if (fv(1) != 1.5) return 9;

    // arange backend selector (dynamic 1-D); the static arange<T,N>() stays stack
    auto ah = arange<int, storage::heap>(5);
    static_assert(decltype(ah)::ownership == storage::heap, "arange<T,storage::heap> -> heap");
    if (ah(4) != 4) return 10;
    auto av = arange<long>(4, storage_c<storage::heap>{});
    static_assert(decltype(av)::ownership == storage::heap, "arange value-tag -> heap");
    if (av(3) != 3) return 11;
    auto asr = arange<double, 3>();                           // static form unaffected
    static_assert(decltype(asr)::ownership == storage::stack, "arange<T,N>() still -> stack");
    if (asr(2) != 2.0) return 12;

    // ---- empty is UNINITIALISED (numpy np.empty), but the zero-init paths stay zeroed ----
    // The storage_policy ctor split (value-init default + `_uninit` ctor) must NOT
    // cost stack triviality (only copy/move/dtor matter) — a stack empty is still
    // kernel-passable by value.
    static_assert(cs::is_trivially_copyable<decltype(s)>::value, "empty stack stays trivially copyable");

    // `local<...>{}` / `local(...)` / `zeros` VALUE-INITIALISE — every element zero.
    local<double, shape<3,3>> lz{};                            // brace-init -> zeroed
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) if (lz(i,j) != 0.0) return 13;
    auto lz2 = local<int, shape<4>>();                         // ()-init -> zeroed
    for (int i = 0; i < 4; ++i) if (lz2(i) != 0) return 14;
    auto zbig = zeros<double>(shape<-1>{16});                  // heap zeros -> zeroed
    for (int i = 0; i < 16; ++i) if (zbig(i) != 0.0) return 15;

    // `empty` yields real, writable storage of the right shape (contents indeterminate).
    auto eu = empty<double>(shape<-1,4>{3});
    static_assert(decltype(eu)::ownership == storage::heap, "dynamic empty -> heap");
    eu.fill_(2.5); if (eu(2,3) != 2.5) return 16;              // usable after a write

    return 0;
}
