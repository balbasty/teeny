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

    // int32 offset dispatch (#115): the narrowed (shape32) view must stay a POD,
    // device-passable view. dispatch_index instantiates the kernel for both widths;
    // launching from each arm exercises the int32-view device pass under nvcc.
    dispatch_index(y, [&](auto v){ axpy_kernel<<<1, 32>>>(v, v, 1.f); });
}

int main() { smoke(); return 0; }
