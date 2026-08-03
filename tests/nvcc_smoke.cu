// Compile-only smoke test for nvcc's DEVICE pass. It is NOT run (that needs a
// GPU) — the value is that nvcc compiles teeny's `__host__ __device__` code FOR
// THE DEVICE, catching `__CUDA_ARCH__` pitfalls (e.g. #59) and device-unsafe
// engine code the host-only g++/clang++ suite cannot see. Built by
// .github/workflows/nvcc-compile.yaml with `nvcc --compile` (no link, no run).
#include <teeny/teeny.h>   // pulls <teeny/cuda.h> (self-guarded, active under nvcc)

using namespace tny;

// Views are trivially copyable, so they pass into a kernel by value. Exercise
// checked and unchecked element access plus in-place math in the DEVICE pass.
template <class V>
__global__ void axpy_kernel(V y, V x, float a) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < y.extent(0)) {
        y(i)      += a * x(i);            // checked access + compound-assign
        y.uget(i)  = y.uget(i) * 2.f;     // unchecked access
    }
}

// #389: a `strides<...>`-layout view -- what EVERY slice / slice_along / peel /
// index_select / scan yields -- used from DEVICE code, reading a stride at a
// RUNTIME rank. `mapping::stride(rank_type)` takes its rank at run time, so while
// the static stride pack lived in a host `static constexpr` array data member
// there was no device-side object to index and this kernel could not compile
// ("identifier `S_` is undefined in device code") -- i.e. NO strided view was
// device-usable at all, its `_TNY_API` annotation notwithstanding.
template <class V>
__global__ void strided_kernel(V v) {
    // #401: `v.rank()` called directly from a `__global__` function. Before #401,
    // `tensor::rank()` was a plain constexpr HOST function -- calling it from a
    // `__host__ __device__` function was merely nvcc warning #20013-D, but calling
    // it from a `__global__` function -- exactly what this line does -- was a hard
    // error ("calling a constexpr __host__ function from a __global__ function is
    // not allowed"), and the call had to be routed through `V::extents_type::rank()`
    // (CCCL's own, already __host__ __device__) instead. Now that `rank()` carries
    // `_TNY_API`, this is device-legal directly -- keep it spelled this way so a
    // regression (a missing `_TNY_API` re-creeping in) fails the device compile.
    long r = (long) threadIdx.x % (long) v.rank();
    if (v.extent(0) > 0 && v.extent(1) > 0) {
        v(0, 0) += (float) v.stride((int) r);   // runtime-rank stride() + strided access
        auto row = v.slice_along(axis<0>{}, 0);  // gather a NEW strides<...> view on device
        row(0) += 1.f;
        // #401 sibling: `shape()`'s array-like view (`_geom_view`) has its own
        // `rank()` (used by its range-for `end()`), same bug, same fix.
        long n = 0;
        for (auto e : v.shape()) n += (e > 0 ? 1 : 0);
        if (n > 0) v(0, 0) += 0.f;
    }
}

// #389: peel an anyrank whose cell layout is a MIXED `strides<...>` pack (a static
// inner stride right-aligned behind a dynamic one), so the cell builder takes its
// dynamic-slot branch -- the third place that read the pack at a runtime index.
template <class AR>
__global__ void strided_cell_kernel(AR at) {
    auto cell = at.template peel_front_at<-2>(0);
    if (cell.extent(0) > 0 && cell.extent(1) > 0) cell(0, 0) += 1.f;
}

// #435: axis-scoped `normalize` / `normalize_` FROM DEVICE CODE. Both divide by
// `norm<Axes...>(v)`, which is a static(stack, _TNY_API) / dynamic(heap, _TNY_HOST)
// overload pair — so each axis-scoped normalize is split the same way, and only its
// `_TNY_API` half may appear here. A fully static source keeps every allocation on
// the stack: the reduced norm AND the source-shaped result.
template <class V>
__global__ void normalize_kernel(V v) {
    v.template normalize_<0>();                 // in place (divides by a stack norm)
    auto u = tny::normalize<0>(v);              // allocating -> a STACK result
    auto w = v.template normalize<1>();         // ...the method spelling
    v(0, 0) += u(0, 0) + w(0, 0);
    tny::normalize<1>(v, tny::into(u));         // into(dest): no allocation but the norm
}

// ...and the case that tells the two split KEYS apart: a source with a DYNAMIC axis,
// reduced over exactly that axis. The reduced norm is a static `shape<3>` stack
// tensor, so the in-place / into(dest) forms stay device-callable even though the
// ALLOCATING form on this same source would be host-only (its result is
// source-shaped, hence heap). Keying those two families identically would make this
// kernel fail to compile.
template <class V>
__global__ void normalize_dyn_kernel(V v) {
    v.template normalize_<0>();
    if (v.extent(0) > 0) v(0, 0) += 1.f;
}

// A device-safe functor for `scan`/`scan_` (lambda-free, per teeny's convention).
struct scan_sum { _TNY_API float operator()(float carry, float x) const { return carry + x; } };

// Peel a device-passable (copy-carrier) anyrank ON THE DEVICE.
template <class AR>
__global__ void peel_kernel(AR at) {
    auto cell = at.template peel_front_at<-1>(0);   // keep the last dim
    if (cell.extent(0) > 0) cell(0) += 1.f;
}

void smoke() {
    float * dy = nullptr, * dx = nullptr;           // compile-only: never dereferenced
    auto y = wrap(dy, shape<-1>{8});
    auto x = wrap(dx, shape<-1>{8});
    axpy_kernel<<<1, 32>>>(y, x, 2.f);

    // A view-carrier anyrank is HOST-ONLY; use it from host code in a TU that also
    // compiles device code (the exact #59 scenario). If the `__CUDA_ARCH__` guard
    // over-fires in the device pass, this TU fails to compile under nvcc.
    long sh[3] = {2, 3, 4}, st[3] = {12, 4, 1};
    float * hp = nullptr;
    auto host_carrier = as_anyrank(hp, sh, st, 3);
    auto v3 = host_carrier.fixed<3>();  (void) v3;
    for (auto cell : host_carrier.peel_front<-2>()) (void) cell;

    // A copy-carrier anyrank IS device-passable: hand it to a kernel, peel on device.
    auto dev_carrier = as_anyrank(hp, sh, st, 3, copy_meta);
    peel_kernel<<<1, 1>>>(dev_carrier);

    // #467: narrow the WHOLE carrier host-side, before the launch, then peel on the
    // device. The narrowed carrier must stay a POD kernel parameter (with half the
    // inline meta store) and its cells must be int32-indexed in the device pass.
    peel_kernel<<<1, 1>>>(dev_carrier.reindex<cuda::std::int32_t>());

    // #389: slice on BOTH sides of the static/dynamic line and run each resulting
    // `strides<...>` view through a kernel. The dynamic source folds only its unit
    // inner stride (a MIXED pack -> `stride()` exercises both arms, including the
    // dynamic-slot lookup); the static source folds every stride (an all-static
    // pack -> an EBO mapping).
    //
    // NB `tny::all` MUST stay qualified here. CUDA's own headers declare the legacy
    // warp-vote intrinsic `all()` in the GLOBAL namespace, so under `using namespace
    // tny;` a bare `all` is ambiguous ("error: `all` is ambiguous") -- but ONLY in a
    // .cu translation unit, since a host compiler never sees those headers. Do not
    // "simplify" these back to a bare `all`: the host suite cannot catch the relapse.
    auto dyn_sliced = wrap(dx, shape<-1,-1>{4, 4})(tny::all, slice(1, 3));
    strided_kernel<<<1, 32>>>(dyn_sliced);
    auto sta_sliced = wrap(dx, shape<4,4>{})(tny::all, slice<1,3>());
    strided_kernel<<<1, 32>>>(sta_sliced);
    // (the folded pack each of those two slices produces is pinned by static_assert
    //  in tests/test_strides.cpp -- this file stays a pure device-compile smoke test)

    // ...and an anyrank whose peel cells carry a folded (mixed) `strides<...>` pack:
    // the static-tail tag + layout tag fold the inner stride, and right-aligning that
    // 1-dim tail into a rank-2 cell leaves the outer slot dynamic.
    auto tail_carrier = as_anyrank(hp, sh, st, 3, copy_meta, anyshape<etc,4>{}, ccontiguous{});
    strided_cell_kernel<<<1, 1>>>(tail_carrier);

    // #375/#388/#399: the axis<> VALUE forms of index_select and scan each forward to
    // a static/dynamic overload PAIR whose static arm is _TNY_API (stack result) and
    // whose dynamic arm is _TNY_HOST (heap result, `new[]`). A single unsplit
    // _TNY_API forwarder would be a __host__ __device__ function that resolves to a
    // __host__ allocator on the dynamic path -- exactly what golden rule 4 forbids,
    // and something ONLY the device pass can see (both macros expand to nothing
    // without __CUDACC__, so the host-only suite is blind to it). These DYNAMIC-shaped
    // calls are the ones that pick the _TNY_HOST arm; a static shape takes the
    // _TNY_API arm and was always fine. Host code, but in a TU nvcc compiles for the
    // device too -- so an unsplit forwarder fails here (#389 blocked this coverage
    // until strided-view device support landed; see strided_kernel above).
    long ibuf[2] = {0, 1};
    auto isel_idx = wrap(ibuf, shape<-1>{2});
    auto isel_src = wrap(hp, shape<-1,3>{5, 3});
    auto isel = isel_src.index_select(isel_idx, axis<0>{});    // -> _TNY_HOST arm
    (void) isel;
    auto scanned = scan(isel_src, 0.f, scan_sum{}, axis<1>{}); // -> _TNY_HOST arm
    (void) scanned;
    // #450: the SAME dynamic-shaped source through an `into(dest)` call stays on
    // scan's _TNY_API arm instead -- the refined host/device split this PR adds
    // keys on ALLOCATION (dynamic shape AND no `into`), not on shape alone, so a
    // dynamic source with a preallocated dest never clone()s. A wrong split that
    // still routed a dynamic-source `into(dest)` call through the _TNY_HOST
    // clone()-based forwarder would make this specific call fail nvcc's device
    // pass (a __host__ __device__ function calling a __host__-only allocator) --
    // exactly the failure mode #389/#399 above exist to catch for the plain
    // axis<> value form.
    auto isel_dest = wrap(hp, shape<-1,3>{5, 3});
    auto & scanned_into = scan(isel_src, 0.f, scan_sum{}, axis<1>{}, into(isel_dest)); // -> _TNY_API arm
    (void) scanned_into;
    // #451: index_select's own axis/into pair now rides the same generic `_kw` bag,
    // so its tagged forwarder gained the identical allocation-keyed split -- and the
    // identical exposure. A dynamic-shaped source through `into(dest)` must stay on
    // the _TNY_API arm (nothing is allocated), and the keywords compose in EITHER
    // order, so pin the SWAPPED spelling here: only nvcc's device pass can see a
    // split that wrongly sent it through the heap-allocating _TNY_HOST forwarder.
    auto isel_into_dest = wrap(hp, shape<-1,3>{2, 3});
    auto & isel_into = isel_src.index_select(isel_idx, into(isel_into_dest), axis<0>{}); // -> _TNY_API arm
    (void) isel_into;
    // ...and the static-shaped spellings, which must stay on the _TNY_API arm.
    auto isel_sidx = local<long, shape<2>>{};
    auto isel_ssrc = local<float, shape<5,3>>{};
    auto isel_s = isel_ssrc.index_select(isel_sidx, axis<0>{});    // -> _TNY_API arm
    (void) isel_s;
    auto scanned_s = scan(isel_ssrc, 0.f, scan_sum{}, axis<1>{});  // -> _TNY_API arm
    (void) scanned_s;

    // #435: the two axis-scoped normalize kernels above, plus the HOST-side calls
    // that pick the `_TNY_HOST` halves. As with index_select/scan, a host compiler
    // cannot see any of this (both annotations expand to nothing without __CUDACC__),
    // so an unsplit `_TNY_API` normalize only fails HERE.
    normalize_kernel<<<1, 1>>>(wrap(dx, shape<3,4>{}));
    normalize_dyn_kernel<<<1, 1>>>(wrap(dx, shape<-1,3>{2, 3}));
    auto nrm_dyn = wrap(hp, shape<-1,-1>{2, 3});
    auto nrm_u  = tny::normalize<1>(nrm_dyn);              // -> _TNY_HOST arm (heap result)
    auto nrm_u2 = nrm_dyn.normalize(axis<1>{});            // -> _TNY_HOST arm (method, value form)
    tny::normalize<1>(nrm_dyn, tny::into(nrm_u));          // -> _TNY_HOST into arm
    nrm_dyn.normalize_<1>();                               // -> _TNY_HOST in-place arm
    (void) nrm_u2;

    // int32 offset dispatch (#115): the narrowed (shape32) view must stay a POD,
    // device-passable view. dispatch_index instantiates the kernel for both widths;
    // launching from each arm exercises the int32-view device pass under nvcc.
    dispatch_index(y, [&](auto v){ axpy_kernel<<<1, 32>>>(v, v, 1.f); });
}

int main() { smoke(); return 0; }
