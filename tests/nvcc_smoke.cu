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

// #389: a `strides<...>`-layout view -- what EVERY slice / take_along / peel /
// index_select / scan yields -- used from DEVICE code, reading a stride at a
// RUNTIME rank. `mapping::stride(rank_type)` takes its rank at run time, so while
// the static stride pack lived in a host `static constexpr` array data member
// there was no device-side object to index and this kernel could not compile
// ("identifier `S_` is undefined in device code") -- i.e. NO strided view was
// device-usable at all, its `_TNY_API` annotation notwithstanding.
template <class V>
__global__ void strided_kernel(V v) {
    long r = (long) threadIdx.x % (long) v.rank();
    if (v.extent(0) > 0 && v.extent(1) > 0) {
        v(0, 0) += (float) v.stride((int) r);   // runtime-rank stride() + strided access
        auto row = v.take_along(axis<0>{}, 0);  // gather a NEW strides<...> view on device
        row(0) += 1.f;
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

    // #389: slice on BOTH sides of the static/dynamic line and run each resulting
    // `strides<...>` view through a kernel. The dynamic source folds only its unit
    // inner stride (a MIXED pack -> `stride()` exercises both arms, including the
    // dynamic-slot lookup); the static source folds every stride (an all-static
    // pack -> an EBO mapping).
    auto dyn_sliced = wrap(dx, shape<-1,-1>{4, 4})(all, slice(1, 3));
    strided_kernel<<<1, 32>>>(dyn_sliced);
    auto sta_sliced = wrap(dx, shape<4,4>{})(all, slice<1,3>());
    strided_kernel<<<1, 32>>>(sta_sliced);

    // ...and an anyrank whose peel cells carry a folded (mixed) `strides<...>` pack:
    // the static-tail tag + layout tag fold the inner stride, and right-aligning that
    // 1-dim tail into a rank-2 cell leaves the outer slot dynamic.
    auto tail_carrier = as_anyrank(hp, sh, st, 3, copy_meta, anyshape<etc,4>{}, ccontiguous{});
    strided_cell_kernel<<<1, 1>>>(tail_carrier);

    // int32 offset dispatch (#115): the narrowed (shape32) view must stay a POD,
    // device-passable view. dispatch_index instantiates the kernel for both widths;
    // launching from each arm exercises the int32-view device pass under nvcc.
    dispatch_index(y, [&](auto v){ axpy_kernel<<<1, 32>>>(v, v, 1.f); });
}

int main() { smoke(); return 0; }
