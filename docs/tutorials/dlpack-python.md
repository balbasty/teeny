# Tutorial: CPU + CUDA kernels for DLPack → Python

**Goal.** Write one numeric kernel — say **batched matrix inversion** — that runs
on CPU *and* CUDA, ingests an `ndarray`-like input (NumPy / PyTorch / CuPy /
JAX via **DLPack**), and returns an `ndarray`-like output. The inner matrix is
small (`C×C`, C ∈ {2,3,4}); the input is `(*batch, C, C)` with an **arbitrary
number of batch axes**, all only known at run time — `(C,C)`, `(B,C,C)`,
`(B0,B1,C,C)`, … all go through the *same* code.

The `C×C` math compiles to unrolled code (C is static) while the batch stays
dynamic. The trick that makes the batch rank irrelevant is `peel_front<-2>`:
*keep the trailing two axes static, flatten everything in front into one runtime
loop*. Four steps.

---

## 1. The kernel core (shape-generic, folds when static)

Write the numerics **once**, on teeny views. Because `C` is a compile-time
constant in the view's type, the loops unroll and the strides fold to immediates.

```cpp
#include <teeny/teeny.h>
using namespace tny;

// invert one static C×C matrix (Gauss–Jordan). A, out: (C,C) views of any stride.
template <class MatA, class MatO>
_TNY_API void invert(const MatA & A, MatO & out) {
    constexpr int C = (int)decltype(A.extent(Int<0>()))::value;  // static extent, folds
    double m[C][C], inv[C][C];
    for (int i=0;i<C;++i) for (int j=0;j<C;++j){ m[i][j]=A(i,j); inv[i][j]=(i==j); }
    for (int c=0;c<C;++c){
        double p = 1.0/m[c][c];
        for (int j=0;j<C;++j){ m[c][j]*=p; inv[c][j]*=p; }
        for (int r=0;r<C;++r) if(r!=c){ double f=m[r][c];
            for (int j=0;j<C;++j){ m[r][j]-=f*m[c][j]; inv[r][j]-=f*inv[c][j]; } }
    }
    for (int i=0;i<C;++i) for (int j=0;j<C;++j) out(i,j)=inv[i][j];
}

// the i-th batch element: peel EVERY leading batch axis -> a (C,C) view.
// peel_front<-2> = "keep the last two axes, peel all the rest", so this is
// rank-agnostic: it works for (C,C), (B,C,C), (B0,B1,C,C), ... unchanged.
template <class In, class Out>
_TNY_API void invert_at(const In & in, Out & out, long i) {
    auto A  = peel_front_at<-2>(in,  i);
    auto Oi = peel_front_at<-2>(out, i);
    invert(A, Oi);
}
```

`_TNY_API` = `__host__ __device__`, so the *same* `invert_at` is callable from a
thread and from a CUDA kernel.

---

## 2. The two drivers (CPU threads and CUDA), one line apart

`peel_front<-2>(x).size()` is the **flattened batch count** — the product of every
axis but the last two — so both drivers index a single flat `[0, n)` loop no matter
how many batch axes the caller passed.

```cpp
// CPU: split the flattened batch across threads.
template <class In, class Out>
void invert_cpu(const In & in, Out & out) {
    const long n = peel_front<-2>(in).size();          // product of all batch axes
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());  // never 0
    std::vector<std::thread> pool;
    for (unsigned t=0;t<nt;++t)
        pool.emplace_back([&,t]{ for (long i=t;i<n;i+=nt) invert_at(in,out,i); });
    for (auto & th : pool) th.join();
}

// CUDA: one thread per matrix, grid-stride. Views are trivially copyable, so
// they pass by value into the kernel.
#ifdef __CUDACC__
template <class In, class Out>
__global__ void invert_kernel(In in, Out out) {
    const long n = peel_front<-2>(in).size();
    for (long i = blockIdx.x*blockDim.x+threadIdx.x; i < n;
             i += gridDim.x*blockDim.x)
        invert_at(in, out, i);
}
template <class In, class Out>
void invert_cuda(const In & in, Out & out) {
    const long n = peel_front<-2>(in).size();
    if (n == 0) return;                       // a 0-block launch is a CUDA error
    int b=256, g=(int)((n+b-1)/b);
    invert_kernel<<<g,b>>>(in, out);
    // NB: the kernel is async on the default stream. nanobind hands the result
    // back to Python without synchronizing — call cudaDeviceSynchronize() (or
    // sync the stream) before the caller reads it if you're not on a blocking
    // stream. (torch.from_dlpack consumers that touch the data will sync anyway.)
}
#endif
```

---

## 3. The boundary: array metadata → teeny view

At the boundary you get plain array metadata — a `void* data`, an `int64_t*` shape
array, an `int64_t*` strides array (in **elements**, or `null` = C-contiguous), a
rank, and a device. (That's exactly what nanobind hands you in §4, and what DLPack
carries on the wire.) The rank is arbitrary, so **flatten every batch axis into one**
(a C-contiguous `(*batch, C, C)` block is the same memory as `(∏batch, C, C)`), then
**dispatch the runtime `C` to a static type** so the inner `C×C` folds:

```cpp
// `dims` = {*batch, C, C}, length `ndim`; `strides` in ELEMENTS (or null = dense).
// (Name it `dims`, not `shape` — a local `shape` would shadow teeny's shape<...>.)
static void invert_nd(double* in, double* out, const int64_t* dims,
                      const int64_t* strides, int ndim, bool on_device) {
    long n = 1;
    for (int d = 0; d < ndim - 2; ++d) n *= dims[d];  // flatten ALL batch axes
    const int C = (int)dims[ndim - 1];                // trailing C×C; runtime C

    dispatch_value<2,3,4>(C, [&](auto CC) {  // runtime C -> compile-time c
        constexpr long c = CC.value;
        // static inner dims, one dynamic (flattened) batch axis.
        auto vin  = wrap(in,  shape<dynamic_extent, c, c>{n});
        auto vout = wrap(out, shape<dynamic_extent, c, c>{n});
#ifdef __CUDACC__
        if (on_device) { invert_cuda(vin, vout); return; }
#endif
        (void)on_device;
        invert_cpu(vin, vout);
    });
}
```

The flatten keeps this branch-free in the batch rank: whether Python sent a
`(3,3)`, a `(1000,3,3)`, or a `(4,8,3,3)` array, `invert_nd` sees one dynamic batch
axis and one static `c×c` tile.

!!! tip "Two boundary facts to get right"
    - **DLPack strides are in elements**, NumPy's `__array_interface__` strides
      are in **bytes** — divide by the itemsize before handing them to teeny.
    - **The flatten above assumes the batch is contiguous.** For a *non-contiguous*
      or fully runtime-strided input you can't collapse the batch axes, so keep them
      dynamic: `as_anyrank(data, shape, stride, ndim)` wraps the metadata with no
      copy, and `at.peel_front<2>()` yields one rank-2 cell per matrix over *any*
      batch rank — the exact anyrank mirror of `peel_front<-2>` above. Fold the inner
      dims with `cell.recast<shape<-1,c,c>>()` (or the single matrix with
      `.recast<shape<c,c>>()`) so the `c`s become immediates. A single fixed inner
      stride pair can instead be baked into the type with
      `wrap(ptr, shape, strides<S...>{})`.

That last point is the crux: `recast` turns a runtime `(…,C,C)` view into a static
`shape<…,c,c>` one so the `c`s become immediates in the kernel — the abstraction
stays free even when the data came from Python at an unknown rank.

---

## 4. Bind to Python

Any framework that speaks DLPack (`np.from_dlpack`, `torch.from_dlpack`,
`cupy.from_dlpack`) can hand teeny a zero-copy array. Use
[nanobind](https://nanobind.readthedocs.io), whose `nb::ndarray<>` **speaks DLPack
natively** — an `ndarray` parameter accepts any numpy/torch/cupy/jax array (CPU or
CUDA), and an `ndarray` return value is auto-exported via `__dlpack__`, so there's
no `PyCapsule`/`__dlpack__()` dance on our side:

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>          // cudaMalloc/cudaFree for the CUDA output
namespace nb = nanobind;

// (`invert_nd` is the boundary function defined in §3, above in this file.)

// Accepts any (*batch, C, C) float64 DLPack array of arbitrary rank >= 2 — numpy /
// torch / cupy / jax, CPU or CUDA — and returns one of the same shape. Dropping
// `nb::ndim<K>` lets the parameter take any rank; `nb::c_contig` keeps the batch
// flatten in §3 valid. nanobind fills it from the caller's DLPack capsule for us.
nb::ndarray<> invert(nb::ndarray<double, nb::c_contig> x) {
    const int  ndim    = (int) x.ndim();                           // (*batch, C, C)
    const bool on_cuda = x.device_type() == nb::device::cuda::value;

    long nmat = 1;                                                 // flattened batch
    for (int d = 0; d < ndim - 2; ++d) nmat *= (long) x.shape(d);
    const long C = (long) x.shape(ndim - 1);                       // trailing C×C

    // DLPack strides are in ELEMENTS; copy nanobind's metadata into int64 arrays.
    std::vector<int64_t> shape(ndim), stride(ndim);
    for (int d = 0; d < ndim; ++d) { shape[d] = (int64_t) x.shape(d); stride[d] = (int64_t) x.stride(d); }

    // allocate the output on the SAME device; a (non-capturing) capsule deleter
    // frees it when Python drops the returned array.
    const size_t total = (size_t)(nmat * C * C);
    double* out = nullptr;
    if (on_cuda) cudaMalloc((void**) &out, sizeof(double) * total);
    else         out = new double[total];
    nb::capsule owner = on_cuda
        ? nb::capsule(out, [](void* p) noexcept { cudaFree(p); })
        : nb::capsule(out, [](void* p) noexcept { delete[] (double*) p; });

    invert_nd(x.data(), out, shape.data(), stride.data(), ndim, on_cuda);  // §3 dispatch + kernel

    // return with the SAME (arbitrary-rank) shape as the input, densely strided.
    std::vector<size_t> oshape(shape.begin(), shape.end());
    return nb::ndarray<>(out, (size_t) ndim, oshape.data(), owner, /*strides=dense*/ nullptr,
                         nb::dtype<double>(),
                         on_cuda ? nb::device::cuda::value : nb::device::cpu::value);
}

NB_MODULE(fastinvert, m) { m.def("invert", &invert); }
```

nanobind parses the DLPack capsule for you, so you read `x.data()` / `x.shape(i)` /
`x.stride(i)` / `x.device_type()` and hand them straight to §3's `invert_nd` (which
does the batch flatten + `dispatch_value` inside). The return `ndarray` is auto-exported
via `__dlpack__`, so `np.from_dlpack` / `torch.from_dlpack` pick it up zero-copy.
**With nanobind you never touch a `DLManagedTensor` — nor teeny's `from_dlpack`/
`to_dlpack`** (those in `<teeny/dlpack.h>` are for a raw-C boundary that isn't
nanobind).

```python
import numpy as np, torch, fastinvert
a = np.random.rand(1000, 3, 3) + 3*np.eye(3)      # 1 batch axis
inv = np.from_dlpack(fastinvert.invert(a))         # -> CPU kernel, shape (1000,3,3)
b = np.random.rand(4, 8, 3, 3) + 3*np.eye(3)      # 2 batch axes — same binding
binv = np.from_dlpack(fastinvert.invert(b))        # -> CPU kernel, shape (4,8,3,3)
g  = torch.rand(16, 100, 2, 2, device="cuda") + 3*torch.eye(2, device="cuda")
ginv = torch.from_dlpack(fastinvert.invert(g))     # -> CUDA kernel, zero-copy
```

That's the whole shape of a teeny binding: **framework-agnostic memory in via
DLPack → flatten the batch → `dispatch_value` to a static inner shape → one
`_TNY_API` kernel core → DLPack out.** The numerics are written once, work for any
number of batch axes, and run on CPU threads or CUDA with no shape/stride overhead.

See `examples/batched_inverse.cpp` in the repo for the runnable core (the CPU
path plus the CUDA kernel), including the assembly proof that a static `C×C`
access folds to constant-offset loads.
