// .to<T2>() — pytorch-like dtype conversion producing a dense owning copy.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main()
{
    // ---- static shape: .to<T2>() -> a stack copy of element type T2 ----------
    auto x = local<float, shape<2,3>>{}; x.iota_(0.f, 1.f);   // 0..5
    auto d = x.to<double>();
    static_assert(cs::is_same<decltype(d)::element_type, double>::value, "to<double> element type");
    static_assert(decltype(d)::ownership == storage::stack, "static shape -> stack");
    static_assert(decltype(d)::rank() == 2, "rank kept");
    if (d(1,2) != 5.0) return 1;
    d(0,0) = 99.0;                                             // it's an independent copy
    if (x(0,0) != 0.f) return 2;

    // .to<>() with a matching dtype (and no Force) is a NO-COPY borrow: a
    // read-only view over the same storage, keeping the source layout.
    auto c = x.to<>();
    static_assert(cs::is_same<decltype(c)::element_type, const float>::value, "to<> borrows (const view)");
    static_assert(decltype(c)::ownership == storage::view, "to<> matching dtype -> view, no copy");
    if (c(1,2) != 5.f || c.data() != x.data()) return 3;      // SAME storage (borrow)
    x(1,2) = 42.f; if (c(1,2) != 42.f) return 8; x(1,2) = 5.f;  // borrow reflects the source

    // Force a copy even when the dtype already matches.
    auto fc = x.to<float, true>();
    static_assert(decltype(fc)::ownership == storage::stack, "to<float,true> -> owning copy");
    if (fc(1,2) != 5.f || fc.data() == x.data()) return 9;   // distinct storage

    // A forced copy strips const from the destination dtype: `c` has a const
    // element type (it's a borrow), and the natural T2 to spell is that same
    // const type -> the owning copy must materialise a MUTABLE tensor, not fail.
    auto csym = c.to<const float, true>();
    static_assert(cs::is_same<decltype(csym)::element_type, float>::value, "forced copy dest is mutable");
    static_assert(decltype(csym)::ownership == storage::stack, "forced copy of a view lvalue -> stack");
    if (csym(1,2) != 5.f) return 12;

    // ---- dynamic shape: .to<T2>() -> a heap copy -----------------------------
    double buf[6]; auto v = wrap(buf, shape<-1,3>{2}); v.iota_(0.0, 1.0);
    auto h = v.to<int>();
    static_assert(decltype(h)::ownership == storage::heap, "dynamic shape -> heap");
    static_assert(cs::is_same<decltype(h)::element_type, int>::value, "to<int>");
    if (h(1,2) != 5) return 4;                                // 5.0 -> 5

    // ---- converts through a non-contiguous / permuted source ----------------
    auto p = x.permute<1,0>();                                // (3,2) view, non-C-order
    auto pc = p.to<double>();                                 // materialises dense
    static_assert(decltype(pc)::ownership == storage::stack, "permuted static -> stack");
    if (pc(2,1) != x(1,2)) return 5;
    if (!pc.is_contiguous<layout_right>()) return 6;          // result is dense C-order

    // ---- half round-trips through the compute type ---------------------------
    auto hf = x.to<half>();
    static_assert(cs::is_same<decltype(hf)::element_type, half>::value, "to<half>");
    if ((float)hf(1,2) != 5.f) return 7;

    // ---- rvalue .to<>() : an OWNING temporary can't be borrowed (would dangle) ->
    //      it forces a copy; a VIEW temporary borrows external storage safely ------
    auto rc = local<float, shape<2,2>>{}.to<>();              // owning stack rvalue -> copy
    static_assert(decltype(rc)::ownership == storage::stack, "owning rvalue .to<>() -> owning copy, not a borrow");
    static_assert(cs::is_same<decltype(rc)::element_type, float>::value, "rvalue .to<>() keeps the dtype");
    auto rh = full(shape<-1,3>{2}, 1.0).to<>();               // owning heap (dynamic) rvalue -> heap copy
    static_assert(decltype(rh)::ownership == storage::heap, "owning dynamic rvalue .to<>() -> heap copy");
    auto rv = wrap(buf, shape<-1,3>{2}).to<>();               // VIEW rvalue -> safe borrow (buf outlives it)
    static_assert(decltype(rv)::ownership == storage::view, "view rvalue .to<>() borrows (external storage, cannot dangle)");
    if ((void*)rv.data() != (void*)buf) return 10;            // same storage (borrow, not a copy)
    // borrow-of-a-borrow: chaining .to<>() on a (const-element) view temporary
    // must compile and stay a borrow, never attempt a const-writing copy.
    auto bob = x.to<>().to<>();
    static_assert(decltype(bob)::ownership == storage::view, "chained .to<>() on a view stays a borrow");
    if (bob.data() != x.data()) return 11;

    // ---- device-source guard (issue #126, finding #5): the dynamic-shape
    //      clone()/to() copying overloads run copy_ on the HOST, so they
    //      static_assert host-accessibility. A dynamic HOST tensor (view/heap)
    //      is host-accessible -> both still compile and run correctly here. ----
    double hb[6]; auto hv = wrap(hb, shape<-1,3>{2}); hv.iota_(0.0, 1.0);   // 0..5
    auto hclone = hv.clone();                                  // dynamic host source -> heap clone
    static_assert(decltype(hclone)::ownership == storage::heap, "dynamic host clone() -> heap");
    if (hclone(1,2) != 5.0 || (void*)hclone.data() == (void*)hb) return 13;  // independent copy
    hclone(0,0) = 77.0; if (hv(0,0) != 0.0) return 14;         // clone is independent of the source
    auto hto = hv.to<double>();                                // same-dtype dynamic host -> heap copy? no: Force=false, same dtype -> borrow
    static_assert(decltype(hto)::ownership == storage::view, "same-dtype dynamic host .to<>() borrows");
    auto htoi = hv.to<int>();                                  // differing dtype -> host heap copy (guarded, host OK)
    static_assert(decltype(htoi)::ownership == storage::heap, "dynamic host .to<int>() -> heap");
    if (htoi(1,2) != 5) return 15;

    // The guard rejects a DEVICE source at compile time. Left commented out
    // because a static_assert failure cannot be exercised by the runtime suite;
    // enabling either line is a compile error (use the free to<Space>(x) instead):
    //   gpu<float, shape<-1,3>> g(shape<-1,3>{2}); auto bad = g.clone();
    //   gpu<float, shape<-1,3>> g(shape<-1,3>{2}); auto bad = g.to<double>();

    return 0;
}
