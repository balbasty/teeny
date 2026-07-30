// wrap()/make_view(): a duplicate (agreeing) layout tag given positionally
// twice must be a compile error, not silently accepted. (#394, a follow-up
// to #392/#374's positional layout overloads)
//
// Mechanism recap: wrap's positional-layout overload (`wrap(p, e, Layout,
// Tags... tags)`) consumes ONE layout tag into its `Layout` parameter; a
// second, agreeing tag used to slip through unnoticed -- either landing in
// `Tags...` (where only wrap's generic "unrecognised trailing argument"
// message caught it, not a message naming the actual mistake) or, when
// `make_view` forwards to `wrap<Layout, Space>(p, e, tags...)` with an
// explicit `Layout` template argument, being re-absorbed by wrap's own
// `Layout` parameter (partial ordering prefers that fixed parameter over its
// `Tags...` pack), so wrap's own Tags... never held the duplicate at all in
// that path. Fixed by asserting `!_kw::has<_is_layout_tag, Tags...>()` at
// BOTH wrap's own positional overload and make_view's -- make_view needed its
// OWN separate check: by the time it forwards to wrap via an explicit
// <Layout, Space>, the duplicate has already been swallowed by wrap's own
// Layout parameter instead of wrap's Tags pack, so wrap's assert alone is too
// late to see it.
//
// Like `test_into.cpp`'s mis-shaped-`into()` note and `test_math.cpp`'s `dot`
// note, no compile-fail harness exists in this repo to assert a call fails to
// compile -- the actual `wrap(...)`/`make_view(...)` duplicate calls below
// are commented out and were verified manually (both before this fix, where
// they wrongly compiled or -- for the direct `wrap` case -- compiled with the
// wrong, generic message; and after, where all four now fail with the named
// "a layout tag was already given positionally" message). What IS exercised
// here at compile time: the underlying `_kw::has<_is_layout_tag, ...>` guard
// against the exact `Tags...` packs each call site sees (mirroring
// `test_kwargs.cpp`/`test_kwargs_readers.cpp`'s convention of testing the
// `_kw` primitives directly), plus full runtime coverage of every legitimate
// call shape to confirm nothing else regressed.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

// ---- pin the predicate the new guard is built on (`_kw::has<_is_layout_tag,
//      Tags...>()`, added to both wrap's and make_view's positional-layout
//      overloads) against the shapes that matter here. The generic `_kw::has`
//      fold itself is already exercised with dummy tags in test_kwargs.cpp;
//      this just confirms the REAL layout tags classify as expected and that
//      an unrelated/absent trailing tag leaves the guard quiet ------------
static_assert(_kw::has<_is_layout_tag, fcontiguous>(), "fcontiguous is classified as a layout tag");
static_assert(_kw::has<_is_layout_tag, ccontiguous>(), "ccontiguous is classified as a layout tag");
static_assert(!_kw::has<_is_layout_tag>(), "no trailing tag at all -> guard stays quiet");
static_assert(!_kw::has<_is_layout_tag, storage_c<storage::view>>(),
              "a lone storage_c tag (no layout duplicate) -> guard stays quiet");

int main() {
    double buf[6] = {1,2,3,4,5,6};   // row-major 2x3: [[1,2,3],[4,5,6]]

    // ---- every legitimate call shape is UNAFFECTED -------------------------
    auto w0 = wrap(buf, shape<2,3>{});                       // no layout arg at all
    static_assert(cs::is_same<decltype(w0)::layout_type, ccontiguous>::value, "");
    if (w0(0,0)!=1 || w0(1,2)!=6)                return 1;

    auto wf = wrap(buf, shape<2,3>{}, fcontiguous{});        // single value-tag
    static_assert(cs::is_same<decltype(wf)::layout_type, fcontiguous>::value, "");
    if (wf(0,0)!=1 || wf(1,0)!=2 || wf(0,1)!=3)  return 2;   // column-major read

    auto wc = wrap(buf, shape<2,3>{}, ccontiguous{});
    static_assert(cs::is_same<decltype(wc)::layout_type, ccontiguous>::value, "");
    if (wc(0,0)!=1 || wc(1,2)!=6)                return 3;

    auto wt = wrap<fcontiguous>(buf, shape<2,3>{});          // explicit-template spelling
    static_assert(cs::is_same<decltype(wt), decltype(wf)>::value, "same result as the value tag");

    auto wfs = wrap(buf, shape<2,3>{}, fcontiguous{}, storage_c<storage::view>{});  // layout + storage compose
    static_assert(cs::is_same<decltype(wfs)::layout_type, fcontiguous>::value, "");
    static_assert(decltype(wfs)::ownership == storage::view, "");

    auto m0 = make_view(buf, shape<2,3>{});
    static_assert(cs::is_same<decltype(m0)::layout_type, ccontiguous>::value, "");

    auto mf = make_view(buf, shape<2,3>{}, fcontiguous{});
    static_assert(cs::is_same<decltype(mf), decltype(wf)>::value, "make_view == wrap, same tag");
    if (mf(1,0)!=2)                              return 4;

    auto mt = make_view<fcontiguous>(buf, shape<2,3>{});     // explicit-template spelling
    static_assert(cs::is_same<decltype(mt), decltype(mf)>::value, "");

    auto mfs = make_view(buf, shape<2,3>{}, fcontiguous{}, storage_c<storage::view>{});
    static_assert(cs::is_same<decltype(mfs), decltype(mf)>::value, "layout + storage compose");

    // ---- the bug itself: now correctly rejected, with a NAMED message ------
    // (commented out -- see the file header: no compile-fail harness exists.
    //  Verified manually, both before and after this fix.)
    //   auto bad1 = wrap(buf, shape<2,3>{}, fcontiguous{}, fcontiguous{});
    //     // BEFORE: compiled silently (bug). AFTER: "wrap(): a layout tag was
    //     // already given positionally — remove the duplicate".
    //   auto bad2 = wrap(buf, shape<2,3>{}, ccontiguous{}, ccontiguous{});
    //     // same, C-order.
    //   auto bad3 = make_view(buf, shape<2,3>{}, fcontiguous{}, fcontiguous{});
    //     // BEFORE: compiled silently (the make_view-specific manifestation --
    //     // see the file header). AFTER: "make_view(): a layout tag was
    //     // already given positionally — remove the duplicate".
    //   auto bad4 = wrap(buf, shape<2,3>{}, ccontiguous{}, fcontiguous{});
    //     // the pre-existing DISAGREEING pair -- already errored before this
    //     // fix (with the generic "unrecognised trailing argument" message);
    //     // still errors, now with the same clearer named message.

    return 0;
}
