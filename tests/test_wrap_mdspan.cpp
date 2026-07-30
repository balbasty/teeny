// wrap(mdspan): a spelling of as_tensor(mdspan) under the one factory name
// users already reach for. (#270)
//
// Since #370 it also honours the wrap family's memory-space contract: a plain
// backend (storage::gpu, ...) folds to its VIEW kind via storage_view_of, given
// either as the explicit template argument or as a trailing storage_c/storage_v
// keyword tag -- exactly like the four positional wrap forms. And the overload is
// constrained to things that actually look like an mdspan, so a 1-argument
// wrap(x) typo is a clean "no matching function" instead of an error inside
// as_tensor.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
using namespace tny;
namespace cs = cuda::std;

// Is `wrap(x)` a viable call at all? (expression SFINAE -- false when the
// mdspan-shaped constraint rejects every overload, instead of a hard error.)
template <class X, class = void>
struct wrap1_callable : cs::false_type {};
template <class X>
struct wrap1_callable<X, cs::void_t<decltype(wrap(cs::declval<const X &>()))>> : cs::true_type {};

int main() {
    double buf[6] = {1,2,3,4,5,6};   // row-major 2x3: [[1,2,3],[4,5,6]]
    auto vc = wrap(buf, shape<2,3>{});

    auto md = vc.mdspan();
    auto w1 = wrap(md);
    static_assert(cs::is_same<decltype(w1)::element_type, double>::value, "");
    if (w1(1,2)!=6)                              return 1;
    auto w2 = as_tensor(md);                     // as_tensor stays available, identical result
    if (w1(0,1)!=w2(0,1))                        return 2;
    auto ws = wrap(vc.permute(axis<1,0>{}).mdspan());   // works on a transposed submdspan too
    if (ws(2,1)!=6)                              return 3;

    // --- memory space (#370) -------------------------------------------------
    // default: a host view, byte-identical to as_tensor(md)
    static_assert(decltype(w1)::ownership == storage::view, "");
    static_assert(cs::is_same<decltype(w1), decltype(w2)>::value, "default == as_tensor");

    // the plain BACKEND folds to its view kind -- you never spell gpu_view
    auto g1 = wrap(md, storage_v<storage::gpu>);
    static_assert(decltype(g1)::ownership == storage::gpu_view, "");
    static_assert(decltype(g1)::is_device, "");
    auto g2 = wrap(md, storage_c<storage::gpu>{});           // braced spelling of the same tag
    static_assert(cs::is_same<decltype(g1), decltype(g2)>::value, "");
    auto g3 = wrap<storage::gpu>(md);                        // explicit template argument
    static_assert(cs::is_same<decltype(g1), decltype(g3)>::value, "");
    // ... and it matches the pointer form's result exactly
    auto gp = wrap(buf, shape<2,3>{}, storage_v<storage::gpu>);
    static_assert(cs::is_same<decltype(g1), decltype(gp)>::value, "mdspan form == pointer form");

    // pinned/mapped keep their space too (host-accessible, so still readable here)
    auto p1 = wrap(md, storage_v<storage::pinned>);
    static_assert(decltype(p1)::ownership == storage::pinned_view, "");
    static_assert(decltype(p1)::is_host_accessible, "");
    if (p1(1,0)!=4)                              return 4;
    auto m1 = wrap(md, storage_v<storage::mapped>);
    static_assert(decltype(m1)::ownership == storage::mapped_view, "");
    if (m1(0,2)!=3)                              return 5;

    // a view kind passed straight through is idempotent (back-compat spelling)
    auto g4 = wrap<storage::gpu_view>(md);
    static_assert(cs::is_same<decltype(g1), decltype(g4)>::value, "");

    // --- the 1-arg overload is constrained to mdspan-like types (#370) -------
    static_assert(wrap1_callable<decltype(md)>::value, "an mdspan is wrappable");
    static_assert(!wrap1_callable<decltype(vc)>::value, "a teeny tensor is NOT an mdspan");
    static_assert(!wrap1_callable<int>::value, "");
    static_assert(!wrap1_callable<double *>::value, "");

    return 0;
}
