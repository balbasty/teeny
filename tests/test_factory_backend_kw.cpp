// A LEADING explicit BACKEND template argument composes with the WHOLE trailing
// keyword bag -- any subset, any order -- not just a lone `dtype<T>{}` tag (#373).
//
//   empty<storage::heap>(e)                                  // no tag at all
//   empty<storage::heap>(e, dtype<double>{})                 // dtype only (worked before)
//   empty<storage::heap>(e, fcontiguous{})                   // layout only
//   empty<storage::heap>(e, dtype<double>{}, fcontiguous{})  // both, either order
//
// Every one of those used to be "no matching function for call to empty()" unless
// the bag was exactly one `dtype<T>{}`: the T-led entry point cannot bind a VALUE to
// its leading `class T`, so only a non-variadic `(Shape, dtype<T>)` forwarder could
// serve the spelling. The backend-led entry point now takes the same bag.
#include <teeny/teeny.h>

using namespace tny;
namespace cs = cuda::std;

template <class T, class Got> constexpr bool elem = cs::is_same<T, typename Got::element_type>::value;
template <class L, class Got> constexpr bool lay  = cs::is_same<L, typename Got::layout_type>::value;
template <storage O, class Got> constexpr bool own = (Got::ownership == O);

using E33 = shape<3,3>;
using D33 = shape<-1,-1>;

// ===========================================================================
//  empty -- static shape (stack-able backends stay _TNY_API) and dynamic shape
// ===========================================================================
// bag = {}                                                       (new: was an error)
static_assert(elem<float,  decltype(empty<storage::heap>(E33{}))>,        "empty<O>(e): T defaults to float");
static_assert(own<storage::heap, decltype(empty<storage::heap>(E33{}))>,  "empty<O>(e): O honoured");
static_assert(lay<ccontiguous, decltype(empty<storage::heap>(E33{}))>,    "empty<O>(e): layout defaults to C");
// bag = {dtype}                                                  (the one that worked)
static_assert(elem<double, decltype(empty<storage::heap>(E33{}, dtype<double>{}))>, "empty<O>(e, dtype)");
static_assert(own<storage::heap, decltype(empty<storage::heap>(E33{}, dtype<double>{}))>, "empty<O>(e, dtype): O");
// bag = {layout}                                                 (new: was an error)
static_assert(lay<fcontiguous, decltype(empty<storage::heap>(E33{}, fcontiguous{}))>, "empty<O>(e, fcontiguous)");
static_assert(elem<float, decltype(empty<storage::heap>(E33{}, fcontiguous{}))>,      "empty<O>(e, fcontiguous): T");
// bag = {dtype, layout} and {layout, dtype}                      (new: were errors)
static_assert(elem<double, decltype(empty<storage::heap>(E33{}, dtype<double>{}, fcontiguous{}))>, "empty<O>(e, dtype, layout): T");
static_assert(lay<fcontiguous, decltype(empty<storage::heap>(E33{}, dtype<double>{}, fcontiguous{}))>, "empty<O>(e, dtype, layout): L");
static_assert(elem<double, decltype(empty<storage::heap>(E33{}, fcontiguous{}, dtype<double>{}))>, "empty<O>(e, layout, dtype): T");
static_assert(lay<fcontiguous, decltype(empty<storage::heap>(E33{}, fcontiguous{}, dtype<double>{}))>, "empty<O>(e, layout, dtype): L");
// the _TNY_API (stack) arm of the same overload set
static_assert(own<storage::stack, decltype(empty<storage::stack>(E33{}, dtype<int>{}, fcontiguous{}))>, "empty<stack>(e, dtype, layout)");
static_assert(elem<int, decltype(empty<storage::stack>(E33{}, dtype<int>{}, fcontiguous{}))>,           "empty<stack>: T");
static_assert(lay<fcontiguous, decltype(empty<storage::stack>(E33{}, dtype<int>{}, fcontiguous{}))>,    "empty<stack>: L");
// an explicit LAYOUT template arg past the backend still works, with a dtype tag
static_assert(lay<fcontiguous, decltype(empty<storage::heap, fcontiguous>(E33{}, dtype<double>{}))>, "empty<O,L>(e, dtype)");

// a dynamic shape resolves to heap under a deduced backend; naming one still wins
static_assert(own<storage::heap, decltype(empty<storage::heap>(D33{2,3}, fcontiguous{}))>, "empty<heap>(dyn, layout)");

// ===========================================================================
//  zeros / ones / full
// ===========================================================================
static_assert(elem<double, decltype(zeros<storage::heap>(E33{}, dtype<double>{}, fcontiguous{}))>, "zeros<O>(e, dtype, layout): T");
static_assert(lay<fcontiguous, decltype(zeros<storage::heap>(E33{}, dtype<double>{}, fcontiguous{}))>, "zeros<O>(e, dtype, layout): L");
static_assert(lay<fcontiguous, decltype(zeros<storage::heap>(E33{}, fcontiguous{}))>, "zeros<O>(e, layout)");
static_assert(elem<float, decltype(zeros<storage::heap>(E33{}))>, "zeros<O>(e): T defaults to float");

static_assert(elem<double, decltype(ones<storage::heap>(E33{}, fcontiguous{}, dtype<double>{}))>, "ones<O>(e, layout, dtype): T");
static_assert(lay<fcontiguous, decltype(ones<storage::heap>(E33{}, fcontiguous{}, dtype<double>{}))>, "ones<O>(e, layout, dtype): L");

static_assert(elem<double, decltype(full<storage::heap>(E33{}, 2, dtype<double>{}, fcontiguous{}))>, "full<O>(e, v, dtype, layout): T");
static_assert(lay<fcontiguous, decltype(full<storage::heap>(E33{}, 2, dtype<double>{}, fcontiguous{}))>, "full<O>(e, v, dtype, layout): L");
static_assert(elem<int, decltype(full<storage::heap>(E33{}, 2, fcontiguous{}))>, "full<O>(e, v, layout): T from the VALUE");
static_assert(elem<int, decltype(full<storage::heap>(E33{}, 2))>, "full<O>(e, v): T from the VALUE");

// ===========================================================================
//  arange -- no layout keyword (1-D has no C/F distinction), but the bag composes
// ===========================================================================
static_assert(elem<double, decltype(arange<storage::heap>(4, dtype<double>{}))>, "arange<O>(n, dtype)");
static_assert(elem<cs::int64_t, decltype(arange<storage::heap>(4))>, "arange<O>(n): T defaults to int64");

// ===========================================================================
//  the T-led spelling is untouched: adding the backend-led overload must not
//  make the no-explicit-template-argument calls ambiguous.
// ===========================================================================
static_assert(elem<double, decltype(empty(E33{}, dtype<double>{}))>, "empty(e, dtype)");
static_assert(elem<double, decltype(empty<double>(E33{}, fcontiguous{}))>, "empty<T>(e, layout)");
static_assert(own<storage::heap, decltype(empty(E33{}, storage_c<storage::heap>{}, dtype<double>{}))>, "empty(e, storage_c, dtype)");
static_assert(elem<double, decltype(zeros(E33{}, dtype<double>{}, fcontiguous{}))>, "zeros(e, dtype, layout)");
static_assert(elem<double, decltype(full(E33{}, 2, dtype<double>{}, fcontiguous{}))>, "full(e, v, dtype, layout)");
static_assert(elem<double, decltype(arange(4, dtype<double>{}))>, "arange(n, dtype)");
static_assert(elem<double, decltype(make_local<double>(E33{}))>, "make_local<T>(e)");
static_assert(lay<fcontiguous, decltype(make_heap(E33{}, dtype<double>{}, fcontiguous{}))>, "make_heap(e, dtype, layout)");

int main()
{
    // ---- the values are right, not just the types --------------------------
    auto z = zeros<storage::heap>(E33{}, dtype<double>{}, fcontiguous{});
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) if (z(i,j) != 0.0) return 1;
    if (z.stride(0) != 1 || z.stride(1) != 3) return 2;   // F-order, as the tag asked

    auto o = ones<storage::heap>(E33{}, fcontiguous{}, dtype<double>{});
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) if (o(i,j) != 1.0) return 3;
    if (o.stride(0) != 1 || o.stride(1) != 3) return 4;

    auto f = full<storage::heap>(E33{}, 2, dtype<double>{}, fcontiguous{});
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) if (f(i,j) != 2.0) return 5;

    auto e = empty<storage::heap>(E33{}, dtype<double>{}, fcontiguous{});
    e.fill_(7.0);
    if (e(2,2) != 7.0) return 6;
    if (e.stride(0) != 1 || e.stride(1) != 3) return 7;

    auto r = arange<storage::heap>(4, dtype<double>{});
    if (r(0) != 0.0 || r(3) != 3.0) return 8;

    // a dynamic shape, the same bag
    auto zd = zeros<storage::heap>(D33{2,3}, dtype<double>{}, fcontiguous{});
    if (zd.shape(0) != 2 || zd.shape(1) != 3) return 9;
    if (zd(1,2) != 0.0) return 10;
    if (zd.stride(0) != 1 || zd.stride(1) != 2) return 11;

    // the stack arm is a real stack tensor and still fills
    auto s = full<storage::stack>(E33{}, 3, dtype<float>{});
    if (s(0,0) != 3.0f || s(2,2) != 3.0f) return 12;

    return 0;
}
