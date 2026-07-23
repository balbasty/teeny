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
    static_assert(decltype(d)::ownership == own::stack, "static shape -> stack");
    static_assert(decltype(d)::rank() == 2, "rank kept");
    if (d(1,2) != 5.0) return 1;
    d(0,0) = 99.0;                                             // it's an independent copy
    if (x(0,0) != 0.f) return 2;

    // .to<>() with a matching dtype (and no Force) is a NO-COPY borrow: a
    // read-only view over the same storage, keeping the source layout.
    auto c = x.to<>();
    static_assert(cs::is_same<decltype(c)::element_type, const float>::value, "to<> borrows (const view)");
    static_assert(decltype(c)::ownership == own::view, "to<> matching dtype -> view, no copy");
    if (c(1,2) != 5.f || c.data() != x.data()) return 3;      // SAME storage (borrow)
    x(1,2) = 42.f; if (c(1,2) != 42.f) return 8; x(1,2) = 5.f;  // borrow reflects the source

    // Force a copy even when the dtype already matches.
    auto fc = x.to<float, true>();
    static_assert(decltype(fc)::ownership == own::stack, "to<float,true> -> owning copy");
    if (fc(1,2) != 5.f || fc.data() == x.data()) return 9;   // distinct storage

    // ---- dynamic shape: .to<T2>() -> a heap copy -----------------------------
    double buf[6]; auto v = wrap(buf, shape<-1,3>{2}); v.iota_(0.0, 1.0);
    auto h = v.to<int>();
    static_assert(decltype(h)::ownership == own::heap, "dynamic shape -> heap");
    static_assert(cs::is_same<decltype(h)::element_type, int>::value, "to<int>");
    if (h(1,2) != 5) return 4;                                // 5.0 -> 5

    // ---- converts through a non-contiguous / permuted source ----------------
    auto p = x.permute<1,0>();                                // (3,2) view, non-C-order
    auto pc = p.to<double>();                                 // materialises dense
    static_assert(decltype(pc)::ownership == own::stack, "permuted static -> stack");
    if (pc(2,1) != x(1,2)) return 5;
    if (!pc.is_contiguous<layout_right>()) return 6;          // result is dense C-order

    // ---- half round-trips through the compute type ---------------------------
    auto hf = x.to<half>();
    static_assert(cs::is_same<decltype(hf)::element_type, half>::value, "to<half>");
    if ((float)hf(1,2) != 5.f) return 7;

    // ---- rvalue .to<>() must NOT borrow (would dangle) -> forces a copy -------
    // A named lvalue with a matching dtype borrows (own::view, checked above);
    // the SAME call on a temporary instead materialises an owning copy, so the
    // result never points at freed storage.
    auto rc = local<float, shape<2,2>>{}.to<>();
    static_assert(decltype(rc)::ownership == own::stack, "rvalue .to<>() forces an owning copy, not a borrow");
    static_assert(cs::is_same<decltype(rc)::element_type, float>::value, "rvalue .to<>() keeps the dtype");
    auto rd = wrap(buf, shape<-1,3>{2}).to<>();               // dynamic rvalue -> heap copy
    static_assert(decltype(rd)::ownership == own::heap, "dynamic rvalue .to<>() -> heap copy");
    if (rd(1,2) != v(1,2)) return 10;

    return 0;
}
