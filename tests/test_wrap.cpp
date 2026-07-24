// wrap() carries an optional memory-space tag (#120): a trailing own_c<Space>{}
// tags the view kind, so a device/pinned/mapped pointer is not mis-typed as a
// host view. Default (no tag) stays own::view. Works on every wrap overload.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

template <class Tn, own O> constexpr bool is_own = (Tn::ownership == O);

int main() {
    double buf[24]; for (int i = 0; i < 24; ++i) buf[i] = i;

    // ---- default: still a host own::view (backward compatible) --------------
    auto hv = wrap(buf, shape<2,3,4>{});
    static_assert(cs::is_same<decltype(hv), tensor<double, shape<2,3,4>, ccontiguous, own::view>>::value,
                  "default wrap -> own::view");
    static_assert(decltype(hv)::is_view && !decltype(hv)::is_device && decltype(hv)::is_host_accessible,
                  "host view flags");
    if (hv(1,2,3) != buf[12+8+3]) return 1;

    // ---- device pointer -> gpu_view (contiguous overload) -------------------
    auto dv = wrap(buf, shape<2,3,4>{}, own_c<own::gpu_view>{});
    static_assert(cs::is_same<decltype(dv), tensor<double, shape<2,3,4>, ccontiguous, own::gpu_view>>::value,
                  "wrap + gpu_view tag");
    static_assert(decltype(dv)::is_view && decltype(dv)::is_device && !decltype(dv)::is_host_accessible,
                  "device view flags");
    // (do NOT host-dereference a gpu_view — it models device memory)

    // ---- pinned / mapped host views (host-accessible, keep their space) ------
    auto pv = wrap(buf, shape<2,3,4>{}, own_c<own::pinned_view>{});
    static_assert(cs::is_same<decltype(pv), tensor<double, shape<2,3,4>, ccontiguous, own::pinned_view>>::value,
                  "wrap + pinned_view tag");
    static_assert(decltype(pv)::is_view && !decltype(pv)::is_device && decltype(pv)::is_host_accessible,
                  "pinned view is host-accessible");
    if (pv(0,0,1) != buf[1]) return 2;                         // pinned is host-dereferenceable
    auto mv = wrap(buf, shape<2,3,4>{}, own_c<own::mapped_view>{});
    static_assert(cs::is_same<decltype(mv), tensor<double, shape<2,3,4>, ccontiguous, own::mapped_view>>::value,
                  "wrap + mapped_view tag");
    if (mv(1,0,0) != buf[12]) return 3;

    // ---- the tag composes with an explicit Layout ---------------------------
    auto fv = wrap<fcontiguous>(buf, shape<2,3,4>{}, own_c<own::gpu_view>{});
    static_assert(cs::is_same<decltype(fv), tensor<double, shape<2,3,4>, fcontiguous, own::gpu_view>>::value,
                  "wrap<Layout> + space tag");
    auto fh = wrap<fcontiguous>(buf, shape<2,3,4>{});          // still defaults to view
    static_assert(is_own<decltype(fh), own::view>, "wrap<Layout> default view");

    // ---- runtime-stride overload (layout_stride) ----------------------------
    auto sh = wrap(buf, shape<-1,-1,-1>{2,3,4}, {12,4,1});     // default view
    static_assert(is_own<decltype(sh), own::view>, "runtime-stride default view");
    if (sh(1,2,3) != buf[12+8+3]) return 4;
    auto sd = wrap(buf, shape<-1,-1,-1>{2,3,4}, {12,4,1}, own_c<own::gpu_view>{});
    static_assert(cs::is_same<decltype(sd), tensor<double, shape<-1,-1,-1>, cs::layout_stride, own::gpu_view>>::value,
                  "runtime-stride + gpu_view tag");

    // ---- compile-time strides<...> overload ---------------------------------
    auto cd = wrap(buf, shape<2,3,4>{}, strides<12,4,1>{}, own_c<own::gpu_view>{});
    static_assert(cs::is_same<decltype(cd), tensor<double, shape<2,3,4>, strides<12,4,1>, own::gpu_view>>::value,
                  "strides<> + gpu_view tag");
    static_assert(decltype(cd)::is_device, "strides<> device view");
    auto ch = wrap(buf, shape<2,3,4>{}, strides<12,4,1>{});    // default view, still folds
    static_assert(is_own<decltype(ch), own::view>, "strides<> default view");
    if (ch(1,2,3) != buf[12+8+3]) return 5;

    // ---- mixed static/runtime strides overload (explicit <S...>) ------------
    auto md = wrap<dynamic_stride, 4, 1>(buf, shape<2,3,4>{}, {12}, own_c<own::gpu_view>{});
    static_assert(cs::is_same<decltype(md), tensor<double, shape<2,3,4>, strides<dynamic_stride,4,1>, own::gpu_view>>::value,
                  "mixed strides + gpu_view tag");
    auto mh = wrap<dynamic_stride, 4, 1>(buf, shape<2,3,4>{}, {12});   // default view
    static_assert(is_own<decltype(mh), own::view>, "mixed strides default view");
    if (mh(1,2,3) != buf[12+8+3]) return 6;
    static_assert(decltype(mh.stride(Int<2>()))::value == 1, "mixed static slot still folds");

    // ---- make_view mirrors wrap's tag (deduces E) ---------------------------
    auto mkh = make_view(buf, shape<2,3,4>{});                            // default view
    static_assert(is_own<decltype(mkh), own::view>, "make_view default view");
    auto mkd = make_view<fcontiguous>(buf, shape<2,3,4>{}, own_c<own::gpu_view>{});
    static_assert(cs::is_same<decltype(mkd), tensor<double, shape<2,3,4>, fcontiguous, own::gpu_view>>::value,
                  "make_view<Layout> + space tag");
    if (mkh(1,2,3) != buf[12+8+3]) return 7;

    return 0;
}
