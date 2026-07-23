// Structural test of teeny/cuda.h using the malloc-backed fake cuda_runtime.h
// (see tests/fakecuda/). Validates allocation / move / free / factories for the
// gpu / pinned / mapped owning modes. Not a GPU test.
#include <teeny/cuda.h>
#include <teeny/teeny.h>
#include <teeny/dlpack.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;
using cs::dynamic_extent;
using DynE = extents<long, dynamic_extent, dynamic_extent>;

static_assert(own_is_owning(own::gpu) && own_is_owning(own::pinned) && own_is_owning(own::mapped), "owning");
static_assert(!own_is_host_accessible(own::gpu), "gpu memory not host-accessible");
static_assert( own_is_host_accessible(own::pinned) && own_is_host_accessible(own::mapped), "pinned/mapped accessible");

int main()
{
    // gpu tensor: owning, move-only. (Fake malloc makes the pointer usable.)
    auto d = gpu<double, DynE>(DynE{2,3});
    static_assert(decltype(d)::ownership == own::gpu, "gpu mode");
    static_assert(!cs::is_copy_constructible<decltype(d)>::value, "owning is move-only");
    if (d.numel() != 6 || d.data() == nullptr) return 1;
    auto d2 = static_cast<decltype(d)&&>(d);          // move transfers ownership
    if (d.data() != nullptr) return 2;

    // page-locked ("pinned") host memory: host-accessible, so we can read/write.
    auto h = pinned<double, DynE>(DynE{2,2});
    h(0,0) = 1; h(1,1) = 4;
    if (h(0,0) != 1 || h(1,1) != 4) return 3;

    // mapped (zero-copy) memory: same.
    auto p = mapped<float, extents<long,3>>(extents<long,3>{});
    p(0) = 7; if (p(0) != 7) return 4;

    // ---- #15: a VIEW of device memory carries its space (own::gpu_view) -------
    static_assert(own_is_device(own::gpu) && own_is_device(own::gpu_view), "device modes");
    static_assert(!own_is_host_accessible(own::gpu_view), "gpu_view not host-accessible");
    static_assert(!own_is_owning(own::gpu_view) && own_is_view(own::gpu_view), "gpu_view is a non-owning view");
    static_assert(own_view_of(own::gpu) == own::gpu_view, "view of gpu -> gpu_view");
    static_assert(own_view_of(own::gpu_view) == own::gpu_view, "view of gpu_view -> gpu_view");
    static_assert(own_view_of(own::heap) == own::view && own_view_of(own::stack) == own::view, "host source -> host view");
    // #41: a VIEW of pinned/mapped keeps its space (pinned_view/mapped_view), still
    // host-accessible & non-owning, so to_dlpack can label it kDLCUDAHost.
    static_assert(own_view_of(own::pinned) == own::pinned_view && own_view_of(own::mapped) == own::mapped_view, "pinned/mapped view keeps space");
    static_assert(own_view_of(own::pinned_view) == own::pinned_view && own_view_of(own::mapped_view) == own::mapped_view, "idempotent");
    static_assert(own_is_view(own::pinned_view) && own_is_view(own::mapped_view), "pinned_view/mapped_view are views");
    static_assert(own_is_host_accessible(own::pinned_view) && own_is_host_accessible(own::mapped_view), "pinned_view/mapped_view host-accessible");
    static_assert(!own_is_device(own::pinned_view) && !own_is_owning(own::mapped_view), "not device, not owning");
    // all four view kinds share one pointer storage: trivially copyable (kernel-
    // passable) and exactly one pointer wide (EBO with the mapping base intact).
    static_assert(cs::is_trivially_copyable<tensor<float, shape<2,3>, ccontiguous, own::view>>::value, "view trivially copyable");
    static_assert(cs::is_trivially_copyable<tensor<float, shape<2,3>, ccontiguous, own::gpu_view>>::value, "gpu_view trivially copyable");
    static_assert(cs::is_trivially_copyable<tensor<float, shape<2,3>, ccontiguous, own::pinned_view>>::value, "pinned_view trivially copyable");
    static_assert(cs::is_trivially_copyable<tensor<float, shape<2,3>, ccontiguous, own::mapped_view>>::value, "mapped_view trivially copyable");
    static_assert(sizeof(tensor<float, shape<2,3>, ccontiguous, own::gpu_view>) == sizeof(float*), "view tensor is one pointer wide");

    // compile-time memory-space flags (members + free trait forms).
    static_assert(gpu<float, shape<2>>::is_device && !gpu<float, shape<2>>::is_host_accessible, "gpu is device");
    static_assert(gpu<float, shape<2>>::is_owning && !gpu<float, shape<2>>::is_view, "gpu owns");
    static_assert(local<float, shape<2>>::is_host_accessible && !local<float, shape<2>>::is_device, "stack host");
    static_assert(is_device_v<gpu<float, shape<2>>> && !is_view_v<gpu<float, shape<2>>>, "free traits");

    // slicing / structure / peel of a gpu tensor all yield gpu_view, not view.
    auto g = gpu<float, shape<4,5>>(shape<4,5>{});
    static_assert(decltype(g(1, all))::is_device && decltype(g(1, all))::is_view, "gpu slice is a device view");
    static_assert(!decltype(g(1, all))::is_host_accessible, "gpu view not host-accessible");
    static_assert(decltype(g(1, all))::ownership       == own::gpu_view, "gpu slice -> gpu_view");
    static_assert(decltype(g(all, slice(1,4)))::ownership == own::gpu_view, "gpu range slice -> gpu_view");
    static_assert(decltype(g.at(0,0))::ownership       == own::gpu_view, "gpu .at -> gpu_view");
    static_assert(decltype(g.permute<1,0>())::ownership == own::gpu_view, "gpu permute -> gpu_view");
    static_assert(decltype(g.flip<0>())::ownership     == own::gpu_view, "gpu flip -> gpu_view");
    static_assert(decltype(g.unsqueeze<0>())::ownership == own::gpu_view, "gpu unsqueeze -> gpu_view");
    static_assert(decltype(peel_front_at<1>(g, 0))::ownership == own::gpu_view, "gpu peel_front_at -> gpu_view");
    static_assert(decltype(peel_at<0>(g, 0))::ownership == own::gpu_view, "gpu peel_at -> gpu_view");
    // a slice of a gpu_view stays a gpu_view (space is preserved through chains).
    auto gv = g(all, slice(0,3));
    static_assert(decltype(gv.permute<1,0>())::ownership == own::gpu_view, "gpu_view chain stays gpu_view");

    // ...and the rest of the view ops too (take_along / squeeze / reshape /
    // recast / flatten / peel range) — a contiguous gpu source, so all are valid.
    static_assert(decltype(g.take_along<0>(1))::ownership == own::gpu_view, "gpu take_along -> gpu_view");
    static_assert(decltype(g.unsqueeze<0>().squeeze<0>())::ownership == own::gpu_view, "gpu squeeze -> gpu_view");
    static_assert(decltype(g.reshape<2,10>())::ownership == own::gpu_view, "gpu reshape -> gpu_view");
    static_assert(decltype(g.recast<shape<4,5>>())::ownership == own::gpu_view, "gpu recast -> gpu_view");
    static_assert(decltype(g.flatten())::ownership     == own::gpu_view, "gpu flatten -> gpu_view");
    static_assert(decltype(peel<0>(g)[0])::ownership   == own::gpu_view, "gpu peel range -> gpu_view");

    // DLPack export of a device view now works and is labeled kDLCUDA (before #15
    // a gpu slice was own::view and exported as kDLCPU — a silent mislabel).
    auto * dl = to_dlpack(g(1, all));
    if (dl->dl_tensor.device.device_type != kDLCUDA) return 12;
    dl->deleter(dl);

    // #41: a view of host-accessible owning memory (pinned) keeps its space
    // (pinned_view) and exports as kDLCUDAHost — not a silent kDLCPU downgrade.
    auto pm = pinned<float, shape<4,5>>(shape<4,5>{});
    static_assert(decltype(pm(1, all))::ownership == own::pinned_view, "pinned slice -> pinned_view");
    auto * dlp = to_dlpack(pm(1, all));
    if (dlp->dl_tensor.device.device_type != kDLCUDAHost) return 13;   // space preserved
    dlp->deleter(dlp);

    // an OWNING pinned/mapped export keeps its space label: both are page-locked
    // HOST memory -> kDLCUDAHost (mapped is zero-copy host, NOT managed/UVM).
    auto * dpn = to_dlpack(pinned<float, shape<3>>(shape<3>{}));
    if (dpn->dl_tensor.device.device_type != kDLCUDAHost) return 14;
    dpn->deleter(dpn);
    auto * dmp = to_dlpack(mapped<float, shape<3>>(shape<3>{}));
    if (dmp->dl_tensor.device.device_type != kDLCUDAHost) return 15;   // NOT kDLCUDAManaged
    dmp->deleter(dmp);

    // ---- memory-backend to<Space, ET, Force>(x) ------------------------------
    // (fake gpu memory is malloc-backed, so a download round-trips the values.)
    auto host = local<float, shape<2,3>>{}; host.iota_(0.f, 1.f);   // 0..5

    // host -> gpu (upload), then gpu -> heap (download): values survive.
    auto gu = to<own::gpu>(host);
    static_assert(decltype(gu)::ownership == own::gpu, "to<gpu> -> gpu");
    static_assert(cs::is_same<decltype(gu)::element_type, float>::value, "dtype kept");
    auto back = to<own::heap>(gu);
    static_assert(decltype(back)::ownership == own::heap, "to<heap> -> heap");
    if (back(1,2) != 5.f) return 5;

    // convert dtype AND upload in one call.
    auto gd = to<own::gpu, double>(host);
    static_assert(cs::is_same<decltype(gd)::element_type, double>::value, "to<gpu,double> converts");
    auto backd = to<own::heap>(gd);
    if (backd(0,1) != 1.0) return 6;

    // no-copy: source already gpu<float>, no Force -> a DEVICE view (gpu_view) borrow.
    auto vv = to<own::gpu>(gu);
    static_assert(decltype(vv)::ownership == own::gpu_view, "already there -> gpu_view, no copy");
    if (vv.data() != gu.data()) return 7;

    // Force a fresh gpu copy even though it already is gpu<float>.
    auto fg = to<own::gpu, void, true>(gu);
    static_assert(decltype(fg)::ownership == own::gpu, "forced -> owning gpu");
    if (fg.data() == gu.data()) return 8;                // distinct storage

    // download into a stack tensor (static shape).
    auto sstk = to<own::stack>(gu);
    static_assert(decltype(sstk)::ownership == own::stack, "to<stack> -> stack");
    if (sstk(1,1) != 4.f) return 9;

    // #15+#29 integration: downloading a device VIEW (a strided gpu_view slice)
    // now takes the download path (own_is_device) instead of host-dereferencing.
    auto gslice = gu(all, slice(0,2));                   // gpu_view, 2x2 window of gu (2x3)
    static_assert(decltype(gslice)::ownership == own::gpu_view, "gpu slice is a device view");
    auto dslice = to<own::heap>(gslice);
    static_assert(decltype(dslice)::ownership == own::heap, "download -> heap");
    if (dslice(0,0) != 0.f || dslice(1,1) != 4.f) return 10;   // strided device view downloaded correctly

    // a FLIPPED (negative-stride) device view: x.data() points at the axis's LAST
    // element, so the download must walk back to the region start (not read past
    // the end). gu is 2x3 = [[0,1,2],[3,4,5]]; flip<0> -> rows reversed.
    auto gflip = gu.flip<0>();                           // gpu_view, row 0 <-> row 1
    static_assert(decltype(gflip)::ownership == own::gpu_view, "flipped gpu view");
    auto dflip = to<own::heap>(gflip);
    if (dflip(0,0) != 3.f || dflip(0,2) != 5.f || dflip(1,0) != 0.f || dflip(1,2) != 2.f) return 16;

    // const-element source composes: x.to<>() borrows as tensor<const T>, and
    // to<Space>(that) must strip the const (else it fails to compile / write const).
    auto cb = host.to<>();                         // tensor<const float, ...> borrow
    static_assert(cs::is_same<decltype(cb)::element_type, const float>::value, "borrow is const");
    auto gcb = to<own::gpu>(cb);
    static_assert(cs::is_same<decltype(gcb)::element_type, float>::value, "to<gpu> strips const");
    if (to<own::heap>(gcb)(1,2) != 5.f) return 17;

    // an F-order (column-major) gpu source must NOT be silently transposed on
    // download: stage into a layout-matching host buffer, then densify.
    auto gf = gpu<float, shape<2,3>, fcontiguous>(shape<2,3>{});
    wrap<fcontiguous>(gf.data(), shape<2,3>{}).iota_(0.f, 1.f);   // logical (i,j) = i*3+j, stored F-order
    auto bf = to<own::heap>(gf);
    static_assert(decltype(bf)::ownership == own::heap, "download -> heap");
    if (bf(0,2) != 2.f || bf(1,0) != 3.f || bf(1,2) != 5.f) return 11;   // values, not transposed

    // rvalue source (a temporary) must never be BORROWED (no dangling /
    // freed-device-memory view) — it is moved or copied into an owning result.
    auto tv = to<own::gpu>(make_gpu<float>(shape<2,3>{}));
    static_assert(decltype(tv)::ownership == own::gpu, "rvalue source -> owning, not a view");

    // ---- #58: device data bound for the device stays on the device -----------
    auto memc   = [](cudaMemcpyKind k){ return tny_fakecuda::memcpy_count[static_cast<unsigned>(k)]; };
    auto resetc = []{ for (int i = 0; i < 5; ++i) tny_fakecuda::memcpy_count[i] = 0; };

    auto hsrc = zeros<float, own::heap>(shape<3,4>{}); hsrc.iota_(0.f, 1.f);   // 0..11 (host)
    auto gsrc = to<own::gpu>(hsrc);                          // upload -> owning gpu
    static_assert(decltype(gsrc)::ownership == own::gpu, "upload -> gpu");

    // (a) a gpu_view slice sent to the device BORROWS — no copy, stays a gpu_view
    auto gview = gsrc(1, all);
    static_assert(decltype(gview)::ownership == own::gpu_view, "slice of gpu -> gpu_view");
    resetc();
    auto gb = to<own::gpu>(gview);                           // already on device -> borrow
    static_assert(decltype(gb)::ownership == own::gpu_view, "to<gpu>(gpu_view) borrows");
    if (memc(cudaMemcpyDeviceToDevice) || memc(cudaMemcpyDeviceToHost) || memc(cudaMemcpyHostToDevice)) return 27;
    if (gb.data() != gview.data()) return 28;               // aliases the source, no round-trip

    // (b) a FORCED device->device copy of a dense source is ONE device-to-device
    //     memcpy — never a host round-trip
    resetc();
    auto gcopy = to<own::gpu, void, true>(gsrc);
    static_assert(decltype(gcopy)::ownership == own::gpu, "forced -> owning gpu");
    if (memc(cudaMemcpyDeviceToDevice) != 1) return 29;
    if (memc(cudaMemcpyDeviceToHost) != 0)  return 30;      // no download
    if (gcopy.data() == gsrc.data())        return 31;      // distinct storage
    if (to<own::heap>(gcopy)(2,3) != 11.f)  return 32;      // values preserved

    // (c) an rvalue owning gpu of the same space is MOVED — buffer stolen, zero memcpy
    auto gtmp = to<own::gpu, void, true>(gsrc);
    resetc();
    auto gmv = to<own::gpu>(cs::move(gtmp));
    static_assert(decltype(gmv)::ownership == own::gpu, "moved rvalue -> owning gpu");
    if (memc(cudaMemcpyDeviceToDevice) || memc(cudaMemcpyDeviceToHost) || memc(cudaMemcpyHostToDevice)) return 33;
    if (to<own::heap>(gmv)(0,0) != 0.f || to<own::heap>(gmv)(2,3) != 11.f) return 34;   // stole the right buffer

    // (d) a STRIDED device source forced to the device can't flat-densify, so it
    //     falls back to the host densify (D2H + H2D), NOT a wrong flat D2D
    auto gcol = gsrc(all, 1);                                // column -> stride-4 gpu_view (non-contiguous)
    static_assert(decltype(gcol)::ownership == own::gpu_view, "column -> gpu_view");
    resetc();
    auto gcolc = to<own::gpu, void, true>(gcol);
    if (memc(cudaMemcpyDeviceToDevice) != 0) return 35;     // NOT a flat D2D (would mis-densify)
    if (memc(cudaMemcpyDeviceToHost) < 1 || memc(cudaMemcpyHostToDevice) < 1) return 36;   // host round-trip
    if (to<own::heap>(gcolc)(2) != 9.f) return 37;          // column 1 = [1,5,9]

    // rank-0 device view download: read one element back from the GPU. Must
    // COMPILE (dense_host's span loop guards rank-0 now, #55) and round-trip.
    auto gr0 = gu.at(1,1);                               // rank-0 gpu_view (gu(1,1) == 4)
    static_assert(decltype(gr0)::ownership == own::gpu_view, "at() of gpu -> rank-0 gpu_view");
    auto dr0 = to<own::heap>(gr0);
    static_assert(decltype(dr0)::rank() == 0, "rank-0 download stays rank-0");
    if (dr0.item() != 4.f) return 18;

    // ---- unified empty<T, Space>(...) factory reaches the CUDA backends -------
    auto eg = empty<float, own::gpu>(shape<2,3>{});
    static_assert(decltype(eg)::ownership == own::gpu, "empty<T,own::gpu> -> gpu");
    auto ep = empty<float, own::pinned>(shape<-1,3>{2});          // dynamic pinned
    static_assert(decltype(ep)::ownership == own::pinned, "empty<T,own::pinned> -> pinned");
    ep.fill_(2.f);                                                // pinned is host-accessible
    if (ep(1,2) != 2.f) return 19;
    auto em = empty<double>(shape<4>{}, own_c<own::mapped>{});    // value-tag backend form
    static_assert(decltype(em)::ownership == own::mapped, "empty value-tag -> mapped");
    em.zero_(); if (em(3) != 0.0) return 20;

    // ---- fill factories reach host-accessible CUDA backends (pinned/mapped) --
    auto zp = zeros<float, own::pinned>(shape<2,3>{});            // pinned zeros (host fill ok)
    static_assert(decltype(zp)::ownership == own::pinned, "zeros<T,own::pinned> -> pinned");
    if (zp(1,2) != 0.f) return 21;
    auto ap = arange<int, own::pinned>(4);                        // pinned [0..3]
    static_assert(decltype(ap)::ownership == own::pinned, "arange<T,own::pinned> -> pinned");
    if (ap(3) != 3) return 22;
    auto fm = full<double>(shape<2>{}, 2.0, own_c<own::mapped>{}); // mapped, value-tag
    static_assert(decltype(fm)::ownership == own::mapped, "full value-tag -> mapped");
    if (fm(1) != 2.0) return 23;

    // ---- #41: a slice/view of pinned/mapped keeps its space, and to_dlpack
    //           labels it kDLCUDAHost (not kDLCPU) ------------------------------
    auto pt = pinned<float, shape<4,5>>(shape<4,5>{}); pt.zero_();
    auto pv = pt(1, all);                                         // a view of pinned memory
    static_assert(decltype(pv)::ownership == own::pinned_view, "slice of pinned -> pinned_view");
    static_assert(decltype(pv)::is_view && decltype(pv)::is_host_accessible, "pinned_view: host-accessible view");
    pv(2) = 9.f; if (pt(1,2) != 9.f) return 24;                  // host-writable, aliases the source
    DLManagedTensor * mp = to_dlpack(pv);
    if (mp->dl_tensor.device.device_type != kDLCUDAHost) return 25;   // space preserved through the view
    mp->deleter(mp);
    auto mt = mapped<double, shape<3>>(shape<3>{}); mt.zero_();
    auto mv = mt.flip<0>();                                      // a structural view of mapped memory
    static_assert(decltype(mv)::ownership == own::mapped_view, "view of mapped -> mapped_view");
    DLManagedTensor * mm = to_dlpack(mv);
    if (mm->dl_tensor.device.device_type != kDLCUDAHost) return 26;
    mm->deleter(mm);

    // ---- #50: run-wise gather of a strided device view -----------------------
    // A padded sub-block (rows contiguous, separated by the parent row pitch) is
    // downloaded run-by-run into a dense buffer instead of dragging the whole span.
    double sb50[16]; for (int i = 0; i < 16; ++i) sb50[i] = i;
    auto g50  = to<own::gpu>(wrap(sb50, shape<4,4>{}));          // device 4x4, row-major
    auto sub50 = g50(slice(0,2), slice(0,3));                    // 2x3, strides (4,1); span 7 > numel 6
    resetc();
    auto hsub50 = to<own::heap>(sub50);
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j)
        if (hsub50(i,j) != (double)(4*i + j)) return 40;        // gathered values correct
    if (memc(cudaMemcpyDeviceToHost) != 2) return 41;           // 2 runs (rows), not one span copy

    resetc();                                                    // fully contiguous -> single span memcpy
    auto hfull50 = to<own::heap>(g50);
    if (hfull50(3,3) != 15.0) return 42;
    if (memc(cudaMemcpyDeviceToHost) != 1) return 43;

    resetc();                                                    // strided column (no unit inner) -> span fallback
    auto hcol50 = to<own::heap>(g50(all, 1));
    if (hcol50(2) != 9.0) return 44;
    if (memc(cudaMemcpyDeviceToHost) != 1) return 45;

    return 0;
}
