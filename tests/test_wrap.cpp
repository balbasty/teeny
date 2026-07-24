// wrap() carries an optional memory-space tag (#120), and the tag is the plain
// BACKEND the memory lives in (storage::gpu/pinned/mapped) — since wrap always yields a
// VIEW, the space folds to its view kind via storage_view_of, so users never spell the
// _view kinds (#124). Default (no tag) stays storage::view. storage_v is the no-braces tag.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

template <class Tn, storage O> constexpr bool is_own = (Tn::ownership == O);

int main() {
    double buf[24]; for (int i = 0; i < 24; ++i) buf[i] = i;

    // ---- default: still a host storage::view (backward compatible) --------------
    auto hv = wrap(buf, shape<2,3,4>{});
    static_assert(cs::is_same<decltype(hv), tensor<double, shape<2,3,4>, ccontiguous, storage::view>>::value,
                  "default wrap -> storage::view");
    static_assert(decltype(hv)::is_view && !decltype(hv)::is_device && decltype(hv)::is_host_accessible,
                  "host view flags");
    if (hv(1,2,3) != buf[12+8+3]) return 1;

    // ---- device pointer: pass the BACKEND storage::gpu; it folds to gpu_view -----
    auto dv = wrap(buf, shape<2,3,4>{}, storage_c<storage::gpu>{});
    static_assert(cs::is_same<decltype(dv), tensor<double, shape<2,3,4>, ccontiguous, storage::gpu_view>>::value,
                  "wrap + storage::gpu tag folds to gpu_view");
    static_assert(decltype(dv)::is_view && decltype(dv)::is_device && !decltype(dv)::is_host_accessible,
                  "device view flags");
    // storage_v is the no-braces spelling of the same tag; the _view kind is idempotent
    auto dv2 = wrap(buf, shape<2,3,4>{}, storage_v<storage::gpu>);
    static_assert(cs::is_same<decltype(dv2), decltype(dv)>::value, "storage_v<storage::gpu> == storage_c<storage::gpu>{}");
    auto dv3 = wrap(buf, shape<2,3,4>{}, storage_c<storage::gpu_view>{});           // still accepted (idempotent)
    static_assert(cs::is_same<decltype(dv3), decltype(dv)>::value, "storage::gpu_view folds to itself");
    // (do NOT host-dereference a gpu_view — it models device memory)

    // ---- pinned / mapped host views (host-accessible, keep their space) ------
    auto pv = wrap(buf, shape<2,3,4>{}, storage_v<storage::pinned>);
    static_assert(cs::is_same<decltype(pv), tensor<double, shape<2,3,4>, ccontiguous, storage::pinned_view>>::value,
                  "storage::pinned folds to pinned_view");
    static_assert(decltype(pv)::is_view && !decltype(pv)::is_device && decltype(pv)::is_host_accessible,
                  "pinned view is host-accessible");
    if (pv(0,0,1) != buf[1]) return 2;                         // pinned is host-dereferenceable
    auto mv = wrap(buf, shape<2,3,4>{}, storage_v<storage::mapped>);
    static_assert(cs::is_same<decltype(mv), tensor<double, shape<2,3,4>, ccontiguous, storage::mapped_view>>::value,
                  "storage::mapped folds to mapped_view");
    if (mv(1,0,0) != buf[12]) return 3;

    // ---- the tag composes with an explicit Layout (two-arg template form) ----
    auto fv = wrap<fcontiguous, storage::gpu>(buf, shape<2,3,4>{});
    static_assert(cs::is_same<decltype(fv), tensor<double, shape<2,3,4>, fcontiguous, storage::gpu_view>>::value,
                  "wrap<Layout, storage::gpu> folds to gpu_view");
    auto fh = wrap<fcontiguous>(buf, shape<2,3,4>{});          // still defaults to view
    static_assert(is_own<decltype(fh), storage::view>, "wrap<Layout> default view");

    // ---- runtime-stride overload (layout_stride) ----------------------------
    auto sh = wrap(buf, shape<-1,-1,-1>{2,3,4}, {12,4,1});     // default view
    static_assert(is_own<decltype(sh), storage::view>, "runtime-stride default view");
    if (sh(1,2,3) != buf[12+8+3]) return 4;
    auto sd = wrap(buf, shape<-1,-1,-1>{2,3,4}, {12,4,1}, storage_v<storage::gpu>);
    static_assert(cs::is_same<decltype(sd), tensor<double, shape<-1,-1,-1>, cs::layout_stride, storage::gpu_view>>::value,
                  "runtime-stride + storage::gpu -> gpu_view");

    // ---- compile-time strides<...> overload ---------------------------------
    auto cd = wrap(buf, shape<2,3,4>{}, strides<12,4,1>{}, storage_v<storage::gpu>);
    static_assert(cs::is_same<decltype(cd), tensor<double, shape<2,3,4>, strides<12,4,1>, storage::gpu_view>>::value,
                  "strides<> + storage::gpu -> gpu_view");
    static_assert(decltype(cd)::is_device, "strides<> device view");
    auto ch = wrap(buf, shape<2,3,4>{}, strides<12,4,1>{});    // default view, still folds
    static_assert(is_own<decltype(ch), storage::view>, "strides<> default view");
    if (ch(1,2,3) != buf[12+8+3]) return 5;

    // ---- mixed static/runtime strides overload (explicit <S...>) ------------
    auto md = wrap<dynamic_stride, 4, 1>(buf, shape<2,3,4>{}, {12}, storage_v<storage::gpu>);
    static_assert(cs::is_same<decltype(md), tensor<double, shape<2,3,4>, strides<dynamic_stride,4,1>, storage::gpu_view>>::value,
                  "mixed strides + storage::gpu -> gpu_view");
    auto mh = wrap<dynamic_stride, 4, 1>(buf, shape<2,3,4>{}, {12});   // default view
    static_assert(is_own<decltype(mh), storage::view>, "mixed strides default view");
    if (mh(1,2,3) != buf[12+8+3]) return 6;
    static_assert(decltype(mh.stride(Int<2>()))::value == 1, "mixed static slot still folds");

    // ---- make_view mirrors wrap's tag + fold (deduces E) --------------------
    auto mkh = make_view(buf, shape<2,3,4>{});                            // default view
    static_assert(is_own<decltype(mkh), storage::view>, "make_view default view");
    auto mkd = make_view<fcontiguous>(buf, shape<2,3,4>{}, storage_v<storage::pinned>);
    static_assert(cs::is_same<decltype(mkd), tensor<double, shape<2,3,4>, fcontiguous, storage::pinned_view>>::value,
                  "make_view<Layout> + storage::pinned -> pinned_view");
    if (mkh(1,2,3) != buf[12+8+3]) return 7;

    return 0;
}
