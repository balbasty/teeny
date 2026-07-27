# teeny

!!! warning "Vibe-coded"
    teeny was written by an AI coding assistant and reviewed by a human. Treat it
    as experimental — expect rough edges, and pin a commit if you depend on it.

Header-only C++17 tensor library for host and CUDA device, built on NVIDIA
CCCL's `cuda::std::mdspan`.

One tensor type whose shape and strides may each be **static, dynamic, or a mix**,
independently per dimension. A kernel written once specializes to fully-dynamic,
fully-static (shapes and strides become compile-time constants baked into the
code), or anything in between, from a single source. Intended for compact numeric
C++/CUDA kernels: spline interpolation, distance transforms, small linear algebra,
batched solves.

```cpp
#include <teeny/teeny.h>
using namespace tny;

auto img = wrap(ptr, shape<480, 640>{});   // non-owning view; static 480×640
auto vol = wrap(ptr, shape<-1, 3, 3>{n});  // dynamic batch, static 3×3

img(0, -1);                            // element access -> T& ; negative index wraps
img(all, slice(0, 32, 2));             // strided sub-view (no copy)
auto m = local<double, shape<3,3>>{};  // stack-owned, exactly 9 doubles
m.fill_(0.0); m.add_(other);           // in-place math, broadcasting
auto c = a + b;                        // out-of-place (promotes types)
```

## Properties

- **Static folds.** Access to an item in a tensor with static strides compiles to
  a simple pointer dereference, without requiring any stride math. Tensors with a
  static shape and contiguous layout have static strides. The same kernel body
  works whether a stride is a compile-time or a runtime value; dynamism is paid for
  only where used.
- **mdspan does the layout.** Shapes are `extent`s, strides are `layout`s, and both
  come from `cuda::std`. Teeny adds per-dimension static strides, owned storage (as
  opposed to views), a math layer with broadcasting, and other sweeteners (iteration
  across inner views using `peel`, automated static dispatch from runtime values,
  etc.).
- **Host and device.** No virtuals, exceptions, or RTTI; engines are lambda-free,
  so they build under `nvcc` without `--extended-lambda`. A `view` is trivially
  copyable — pass it into a `__global__` kernel by value.

## Pages

- **[Getting started](getting-started.md)** — build flags and a first program.

Guide:

- **[Tensors & ownership](tensors.md)** — the `tensor` class, data views and data owners.
- **[Shapes & strides](shapes-strides.md)** — `shape<...>`, `strides<...>`, static vs dynamic.
- **[Indexing & slicing](indexing.md)** — elements, `at`, slices, `none`/step, negatives.
- **[Math & broadcasting](math.md)** — in-place / out-of-place, promotion, reductions.
- **[Views & structure](structure.md)** — `permute`/`flip`/`reshape`/`peel`.
- **[Dispatch & the anyrank boundary](dispatch.md)** — runtime rank/value → static.
- **[mdspan vs teeny](mdspan-vs-teeny.md)** — the vocabulary map, for readers who know mdspan.
- **[Performance](performance.md)** — where the dynamic cost lands and how to keep it off the hot path.
- **[Writing efficient kernels](efficient-kernels.md)** — the fast-kernel playbook.
- **[Half precision](half.md)** — `half` / `bfloat16` element types.
- **[CUDA & CCCL compatibility](cuda-compat.md)** — vendored CCCL, `nvcc` ranges, wheel strategy.

Tutorials:

- **[DLPack → Python](tutorials/dlpack-python.md)** — CPU+CUDA kernels over dlpack inputs (numpy, cupy, pytorch, …).

API:

- **[Cheat sheet](cheatsheet.md)** — the whole surface on one screen.
- **[API reference](reference.md)** — the full surface, curated.
- **[Autodoc](api/index.md)** — every symbol's extracted signature.
