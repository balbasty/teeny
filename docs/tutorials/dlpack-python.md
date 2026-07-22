# Tutorial: CPU + CUDA kernels for DLPack → Python

**Goal.** Write one numeric kernel — say **batched matrix inversion** — that runs
on CPU *and* CUDA, ingests an `ndarray`-like input (NumPy / PyTorch / CuPy /
JAX via **DLPack**), and returns an `ndarray`-like output. The inner matrix is
small (`C×C`, C ∈ {2,3,4}); the batch is arbitrary and only known at run time.

The `C×C` math compiles to unrolled code (C is static) while the batch stays
dynamic. Four steps.

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

// the i-th batch element: peel the leading batch axis -> a (C,C) view.
template <class In, class Out>
_TNY_API void invert_at(const In & in, Out & out, long i) {
    auto A  = peel_front_at<1>(in,  i);
    auto Oi = peel_front_at<1>(out, i);
    invert(A, Oi);
}
```

`_TNY_API` = `__host__ __device__`, so the *same* `invert_at` is callable from a
thread and from a CUDA kernel.

---

## 2. The two drivers (CPU threads and CUDA), one line apart

```cpp
// CPU: split the batch across threads.
template <class In, class Out>
void invert_cpu(const In & in, Out & out) {
    const long n = in.extent(0);
    unsigned nt = std::thread::hardware_concurrency();
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
    for (long i = blockIdx.x*blockDim.x+threadIdx.x; i < in.extent(0);
             i += gridDim.x*blockDim.x)
        invert_at(in, out, i);
}
template <class In, class Out>
void invert_cuda(const In & in, Out & out) {
    int b=256, g=(int)((in.extent(0)+b-1)/b);
    invert_kernel<<<g,b>>>(in, out);
}
#endif
```

---

## 3. The boundary: DLPack in, teeny view out

A DLPack array arrives as a `DLTensor`: a `void* data`, an `int64_t* shape`, an
`int64_t* strides` (in **elements**, or `null` = C-contiguous), an `int ndim`,
and a device. Turn it into a teeny view, then **dispatch the runtime `C` to a
static type** so the inner `C×C` folds:

```cpp
// `shape` = {n, C, C}, `strides` in ELEMENTS (or contiguous if null).
static void invert_dlpack(double* in, double* out, const int64_t* shape,
                          const int64_t* strides, bool on_device) {
    const long n = shape[0];
    const int  C = (int)shape[1];  // known only at run time

    dispatch_value<2,3,4>(C, [&](auto CC) {  // runtime C -> compile-time c
        constexpr long c = CC.value;
        // static inner dims, dynamic batch; strides<...> if non-contiguous.
        auto vin  = view(in,  shape<dynamic_extent, c, c>{n});
        auto vout = view(out, shape<dynamic_extent, c, c>{n});
#ifdef __CUDACC__
        if (on_device) { invert_cuda(vin, vout); return; }
#endif
        (void)on_device;
        invert_cpu(vin, vout);
    });
}
```

!!! tip "Two boundary facts to get right"
    - **DLPack strides are in elements**, NumPy's `__array_interface__` strides
      are in **bytes** — divide by the itemsize before handing them to teeny.
    - If the input is non-contiguous, build the view with a strided layout:
      `view_strided<...>` for compile-time strides, or pass the runtime strides
      to a `strides<dynamic_stride,...>` mapping. For a *fully* runtime-strided,
      runtime-rank input use `as_anyrank(data, shape, stride, ndim)` +
      [`dispatch_rank`](../dispatch.md), then `recast<shape<-1,c,c>>()` to
      recover the static inner dims.

That last point is the crux: `recast` turns a runtime `(n,C,C)` view into a
`shape<-1,c,c>` view so the `c`s become immediates in the kernel — the abstraction
stays free even when the data came from Python.

---

## 4. Bind to Python

Any framework that speaks DLPack (`np.from_dlpack`, `torch.from_dlpack`,
`cupy.from_dlpack`) can hand teeny a zero-copy array. A minimal
[pybind11](https://pybind11.readthedocs.io) wrapper:

```cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

py::object invert(py::object x) {
    py::capsule cap = x.attr("__dlpack__")();  // zero-copy DLPack handle
    DLManagedTensor* t = cap.get_pointer<DLManagedTensor>();
    const DLTensor& a = t->dl_tensor;

    // allocate an output of the same shape/device (elided: use the DLPack
    // allocator or your framework's), wrap it as a DLManagedTensor `outmanaged`.
    invert_dlpack((double*)a.data, (double*)out_data, a.shape, a.strides,
                  a.device.device_type == kDLCUDA);

    return py::reinterpret_steal<py::object>(  // hand it back as DLPack
        PyCapsule_New(outmanaged, "dltensor", nullptr));
}

PYBIND11_MODULE(fastinvert, m) { m.def("invert", &invert); }
```

```python
import numpy as np, torch, fastinvert
a = np.random.rand(1000, 3, 3) + 3*np.eye(3)      # numpy
inv = np.from_dlpack(fastinvert.invert(a))         # -> CPU kernel
g  = torch.rand(1000, 3, 3, device="cuda") + 3*torch.eye(3, device="cuda")
ginv = torch.from_dlpack(fastinvert.invert(g))     # -> CUDA kernel, zero-copy
```

That's the whole shape of a teeny binding: **framework-agnostic memory in via
DLPack → `dispatch_value` to a static inner shape → one `_TNY_API` kernel core →
DLPack out.** The numerics are written once and run on CPU threads or CUDA with
no shape/stride overhead.

See `examples/batched_inverse.cpp` in the repo for the runnable core (the CPU
path plus the CUDA kernel), including the assembly proof that a static `C×C`
access folds to constant-offset loads.
