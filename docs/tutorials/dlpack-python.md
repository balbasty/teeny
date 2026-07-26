# Tutorial: CPU + CUDA kernels for DLPack → Python

**Goal.** Write one numeric kernel — say **batched matrix inversion** — that runs
on CPU *and* CUDA, ingests an `ndarray`-like input (NumPy / PyTorch / CuPy /
JAX via **DLPack**), and returns an `ndarray`-like output. The inner matrix is
small (`C×C`, C ∈ {2,3,4}); the input is `(*batch, C, C)` with an **arbitrary
number of batch axes**, all only known at run time — `(C,C)`, `(B,C,C)`,
`(B0,B1,C,C)`, … all go through the *same* code.

The `C×C` math compiles to unrolled code (C is static) while the batch stays
dynamic. The trick that makes the batch rank irrelevant is **`peel_front<-2>`**:
*keep the trailing two axes static, iterate everything in front as one runtime
loop* — **without flattening**, so it works on any stride and never copies the
input. Four steps.

---

## 1. The kernel core (shape-generic, folds when static)

Write the numerics **once**, on teeny views. Because `C` is a compile-time
constant in the view's type, the loops unroll and the strides fold to immediates.

```cpp
#include <teeny/teeny.h>
using namespace tny;

// Invert one static C×C matrix in place (Gauss–Jordan, no pivoting — assumes the
// matrix is non-singular). `A` and `out` are (C,C) views of ANY stride; `C` is a
// compile-time constant in their type, so every loop below unrolls.
template <class MatA, class MatOut>
_TNY_API void invert(const MatA & A, MatOut & out) {
    constexpr int C = (int) decltype(A.extent(Int<0>()))::value;   // static extent -> folds

    double m[C][C], inv[C][C];
    for (int i = 0; i < C; ++i)
        for (int j = 0; j < C; ++j) {
            m[i][j]   = A(i, j);
            inv[i][j] = (i == j) ? 1.0 : 0.0;              // inv starts as the identity
        }

    for (int col = 0; col < C; ++col) {
        const double pivot = 1.0 / m[col][col];
        for (int j = 0; j < C; ++j) { m[col][j] *= pivot; inv[col][j] *= pivot; }
        for (int row = 0; row < C; ++row) {
            if (row == col) continue;
            const double f = m[row][col];
            for (int j = 0; j < C; ++j) {
                m[row][j]   -= f * m[col][j];
                inv[row][j] -= f * inv[col][j];
            }
        }
    }

    for (int i = 0; i < C; ++i)
        for (int j = 0; j < C; ++j) out(i, j) = inv[i][j];
}
```

`_TNY_API` = `__host__ __device__`, so the *same* `invert` is callable from a CPU
thread and from a CUDA kernel.

---

## 2. Peel one batch element

Given a rank-erased `(*batch, C, C)` carrier (`in`, built in §3), the `i`-th
matrix is one `peel_front_at<-2>` away. `peel_front<-2>` means *"keep the last two
axes, peel everything in front"* — so this is rank-agnostic: `(C,C)`, `(B,C,C)`,
`(B0,B1,C,C)` all work unchanged, and the batch keeps its real (possibly strided)
layout — nothing is flattened or copied.

```cpp
// Invert the i-th batch element. peel_front_at<-2> folds the batch index `i` into
// the pointer and leaves a rank-2 view; recast(shape<c,c>{}) restores the two static
// extents so `invert` sees a compile-time C×C. (`c` is the static C from §3.)
// The value-form recast(shape<c,c>{}) needs no `.template`; peel_front_at is a
// count selector, so it keeps it on this dependent receiver.
template <long c, class In, class Out>
_TNY_API void invert_cell(const In & in, Out & out, long i) {
    auto A  = in .template peel_front_at<-2>(i).recast(shape<c, c>{});
    auto Oi = out.template peel_front_at<-2>(i).recast(shape<c, c>{});
    invert(A, Oi);
}
```

!!! note "Random access vs. the incremental range-for"
    `peel_front_at<-2>(i)` re-derives the cell's pointer by decoding `i` over the
    batch dims — `O(#batch dims)` integer ops **per call**. That's the right tool for
    a **grid-stride** loop (`i += nthreads`), where consecutive `i` for one worker are
    a stride apart. When a worker instead sweeps a **contiguous** block, prefer the
    range-for `for (auto cell : in.peel_front<-2>())`, which is *incremental* — it
    advances the pointer and reuses the cell mapping (O(1) per cell, no re-decode) — and
    `.subrange(lo, hi)` for one worker's chunk. Next to a `C×C` inversion the decode is
    noise, but for a light per-cell body it's a 2–3× swing. See
    [Efficient kernels §3](../efficient-kernels.md#3-the-batch-batch-idiom).

---

## 3. The two drivers (CPU threads and CUDA), one line apart

`size_front<-2>()` is the **batch count** — the product of every axis but the last
two — read straight from the shape, without building a peel range. Both drivers
index a single flat `[0, n)` loop no matter how many batch axes the caller passed.

```cpp
#include <thread>
#include <vector>

// CPU: split the batch across threads (grid-stride so any n / nthreads works).
template <long c, class In, class Out>
void invert_cpu(const In & in, Out & out) {
    const long n = in.template size_front<-2>();                       // product of all batch axes
    unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());   // never 0
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nthreads; ++t)
        pool.emplace_back([&, t] {
            for (long i = t; i < n; i += nthreads) invert_cell<c>(in, out, i);
        });
    for (auto & th : pool) th.join();
}

// CUDA: one thread per matrix, grid-stride. The carriers pass BY VALUE into the
// kernel, so they must be copy_meta carriers (shape/stride inline — see §4).
#ifdef __CUDACC__
template <long c, class In, class Out>
__global__ void invert_kernel(In in, Out out) {
    const long n = in.template size_front<-2>();
    for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += gridDim.x * blockDim.x)
        invert_cell<c>(in, out, i);
}
template <long c, class In, class Out>
void invert_cuda(const In & in, Out & out) {
    const long n = in.template size_front<-2>();
    if (n == 0) return;                                   // a 0-block launch is a CUDA error
    const int block = 256, grid = (int)((n + block - 1) / block);
    invert_kernel<c><<<grid, block>>>(in, out);
    // NB async on the default stream — synchronize before Python reads (see §4).
}
#endif
```

!!! tip "Random access vs. contiguous chunks"
    The grid-stride split above (`i += nthreads`) is right when each worker's cells are
    a stride apart — it needs the random-access `peel_front_at(i)`, used here for *both*
    `in` and `out` at the same index. If a worker instead owns a **contiguous** block of
    the batch, a single-tensor sweep can use the incremental range-for and
    `for (auto cell : t.peel_front<-2>().subrange(lo, hi))` — seed once, then O(1) per
    cell (a 2–3× swing for a light body; see
    [Efficient kernels §3](../efficient-kernels.md#3-the-batch-batch-idiom)). For a two-input
    kernel like this one, either keep the shared index (as above) or advance one range
    incrementally alongside a running index for the other.

This tutorial deliberately applies the [Efficient-kernels](../efficient-kernels.md)
playbook: a **static inner block** via `recast(shape<c,c>{})` (§1), the **`peel_front`
batch idiom** (§3), **snapshotting inputs through a `local`** before writing the result
(§7), and **`dispatch_value`/`dispatch_index`** at the boundary (§4). The `C×C` body
nests its unit-stride axis innermost and its `m`/`inv` workspaces are fully overwritten
(so an [`empty`](../efficient-kernels.md#7-small-static-c-kernels-three-codegen-rules)
scratch would skip the zero-fill) — the §8 strided-loop rules.

---

## 4. The boundary: array metadata → teeny carrier → dispatch

At the boundary you get plain array metadata — a `data` pointer, a shape array, a
strides array (in **elements**), a rank, and a device. Wrap it in an **`anyrank`**
(a rank-erased carrier) and let teeny peel the batch; **do not flatten**. Two
runtime facts become static exactly where they need to:

- the **rank** disappears into `peel_front<-2>` (any number of batch axes);
- the trailing **`C`** is turned into a compile-time `c` with `dispatch_value`.

```cpp
// `in`/`out` are anyrank carriers over the SAME (*batch, C, C) layout — the batch
// keeps its real shape (no flatten, no contiguity assumption). The carrier's memory
// SPACE (host vs device) lives in its TYPE, so `if constexpr (In::is_device)` picks
// the driver — no runtime device flag threads through.
template <class In, class Out>
void invert_nd(const In & in, Out & out) {
    const int C = (int) in.size(in.ndim - 1);             // trailing C×C; runtime C

    dispatch_value<2, 3, 4>(C, [&](auto CC) {
        // `CC` is a cuda::std::integral_constant — `CC.value` is a constexpr, so it
        // can be a template argument / array bound. (You need the *value*, not `CC`
        // itself, precisely there: a compile-time shape<c,c> below.)
        constexpr long c = CC.value;
#ifdef __CUDACC__
        if constexpr (In::is_device) { invert_cuda<c>(in, out); return; }
#endif
        invert_cpu<c>(in, out);
    });
}
```

`anyrank` carries a compile-time memory **space** (see
[Dynamic dispatch](../dispatch.md#memory-space-of-the-data)): a host pointer peels
into host views, a device pointer (`as_anyrank<storage::gpu_view>` / a `kDLCUDA`
capsule via `from_dlpack<T, storage::gpu_view>`) peels into `gpu_view` views. That is
why `invert_nd` can branch on `In::is_device` — the space is in the type, not a
runtime flag.

---

## 5. Bind to Python

Any framework that speaks DLPack (`np.from_dlpack`, `torch.from_dlpack`,
`cupy.from_dlpack`) can hand teeny a zero-copy array. Use
[nanobind](https://nanobind.readthedocs.io), whose `nb::ndarray<>` **speaks DLPack
natively** — an `ndarray` parameter accepts any numpy/torch/cupy/jax array (CPU or
CUDA), and an `ndarray` return value is auto-exported via `__dlpack__`.

We take the input at **any stride** (no `c_contig` — the peel handles arbitrary
strides, so nanobind never has to copy a non-contiguous batch) and write a **dense**
output on the same device.

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>          // cudaMalloc / cudaFree / cudaDeviceSynchronize
namespace nb = nanobind;

// (`invert_nd` is the boundary function from §4.)

nb::ndarray<> invert(nb::ndarray<double> x) {     // any (*batch, C, C) float64, any stride
    const int  ndim    = (int) x.ndim();
    const bool on_cuda = x.device_type() == nb::device::cuda::value;

    // Copy nanobind's metadata into int64 arrays (DLPack strides are in ELEMENTS).
    // The FULL (*batch, C, C) shape is kept — no flatten.
    std::vector<int64_t> shape(ndim), stride(ndim);
    for (int d = 0; d < ndim; ++d) {
        shape[d]  = (int64_t) x.shape(d);
        stride[d] = (int64_t) x.stride(d);
    }

    // A DENSE row-major output of the same shape.
    int64_t total = 1;
    for (int d = 0; d < ndim; ++d) total *= shape[d];
    std::vector<int64_t> ostride(ndim);
    { int64_t s = 1; for (int d = ndim - 1; d >= 0; --d) { ostride[d] = s; s *= shape[d]; } }

    double* out = nullptr;
    if (on_cuda) cudaMalloc((void**) &out, sizeof(double) * (size_t) total);
    else         out = new double[(size_t) total];
    nb::capsule owner = on_cuda
        ? nb::capsule(out, [](void* p) noexcept { cudaFree(p); })
        : nb::capsule(out, [](void* p) noexcept { delete[] (double*) p; });

    // Build the carriers and dispatch. A device carrier COPIES its shape/stride
    // inline (copy_meta) so it survives the trip into the kernel, and is tagged
    // storage::gpu_view so its peeled cells are device views; the host carrier WRAPS the
    // metadata (the vectors outlive the synchronous CPU call).
    if (on_cuda) {
#ifdef __CUDACC__
        auto in  = as_anyrank<TNY_MAX_RANK, storage::gpu_view>(x.data(), shape.data(),  stride.data(),  ndim, copy_meta);
        auto od  = as_anyrank<TNY_MAX_RANK, storage::gpu_view>(out,      shape.data(),  ostride.data(), ndim, copy_meta);
        invert_nd(in, od);
        cudaDeviceSynchronize();                  // finish before Python reads the result
#endif
    } else {
        auto in = as_anyrank(x.data(), shape.data(), stride.data(),  ndim);   // host wrap carrier
        auto od = as_anyrank(out,      shape.data(), ostride.data(), ndim);
        invert_nd(in, od);
    }

    std::vector<size_t> oshape(shape.begin(), shape.end());
    return nb::ndarray<>(out, (size_t) ndim, oshape.data(), owner, /*dense strides*/ nullptr,
                         nb::dtype<double>(),
                         on_cuda ? nb::device::cuda::value : nb::device::cpu::value);
}

NB_MODULE(fastinvert, m) { m.def("invert", &invert); }
```

nanobind parses the DLPack capsule for you, so you read `x.data()` / `x.shape(i)` /
`x.stride(i)` / `x.device_type()` and hand them to `as_anyrank`. The return
`ndarray` is auto-exported via `__dlpack__`, so `np.from_dlpack` /
`torch.from_dlpack` pick it up zero-copy. **With nanobind you never touch a
`DLManagedTensor` — nor teeny's `from_dlpack`/`to_dlpack`** (those in
`<teeny/dlpack.h>` are for a raw-C boundary that isn't nanobind).

```python
import numpy as np, torch, fastinvert
a = np.random.rand(1000, 3, 3) + 3*np.eye(3)      # 1 batch axis
inv = np.from_dlpack(fastinvert.invert(a))         # -> CPU kernel, shape (1000,3,3)
b = np.random.rand(4, 8, 3, 3) + 3*np.eye(3)      # 2 batch axes — SAME binding
binv = np.from_dlpack(fastinvert.invert(b))        # -> CPU kernel, shape (4,8,3,3)
bt = torch.as_strided(torch.rand(8, 3, 3), (4, 3, 3), (0, 3, 1))  # non-contiguous batch
btinv = np.from_dlpack(fastinvert.invert(bt))      # -> no copy: peel reads the real strides
g  = torch.rand(16, 100, 2, 2, device="cuda") + 3*torch.eye(2, device="cuda")
ginv = torch.from_dlpack(fastinvert.invert(g))     # -> CUDA kernel, zero-copy
```

That's the whole shape of a teeny binding: **framework-agnostic memory in via
DLPack → `anyrank` carrier (no flatten) → `peel_front<-2>` the batch +
`dispatch_value` to a static inner shape → one `_TNY_API` kernel core → DLPack
out.** The numerics are written once, work for any number of batch axes at any
stride, and run on CPU threads or CUDA with no shape/stride overhead.

See `examples/batched_inverse.cpp` in the repo for the runnable core (the CPU
path plus the CUDA kernel), including the assembly proof that a static `C×C`
access folds to constant-offset loads.
