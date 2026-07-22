# teeny

!!! warning "Vibe-coded"
    teeny was written by an AI coding assistant and reviewed by a human. Treat it
    as experimental — expect rough edges, and pin a commit if you depend on it.

Header-only C++17 tensor library for host and CUDA device, built on NVIDIA
CCCL's `cuda::std::mdspan`.

One tensor type whose shape and strides may be **static, dynamic, or a mix — per
dimension**. A kernel written once specializes to fully-dynamic, fully-static
(everything folds to immediates), or anything in between, from a single source.
Intended for compact numeric C++/CUDA kernels: spline interpolation, distance
transforms, small linear algebra, batched solves.

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

- **Static folds.** A `shape<3,3>` access compiles to a load at a constant
  offset — no stride math. The same kernel body works whether an extent is a
  compile-time `3` or a runtime value; dynamism is paid for only where used.
- **mdspan does the layout.** Extents, layouts, and offset mapping come from
  `cuda::std`. teeny adds per-dimension static strides, owning storage, a
  valarray-like math layer with broadcasting, and kernel ergonomics (nd-peel,
  runtime→static dispatch).
- **Host and device.** No virtuals, exceptions, or RTTI; engines are lambda-free,
  so they build under `nvcc` without `--extended-lambda`. A `view` is trivially
  copyable — pass it into a `__global__` kernel by value.

## Pages

- **[Getting started](getting-started.md)** — build flags and a first program.
- **[Tensors & ownership](tensors.md)** — `view` first, then the owning variants.
- **[Shapes & strides](shapes-strides.md)** — `shape<...>`, `strides<...>`, static vs dynamic.
- **[Indexing & slicing](indexing.md)** — elements, `at`, slices, `none`/step, negatives.
- **[Math & broadcasting](math.md)** — in-place / out-of-place, promotion, reductions.
- **[Views & structure](structure.md)** — `permute`/`flip`/`reshape`/`peel`.
- **[Dispatch & the ndarray boundary](dispatch.md)** — runtime rank/value → static.
- **[Tutorial: DLPack → Python](tutorials/dlpack-python.md)** — CPU+CUDA kernels over `ndarray` inputs.
- **[API reference](reference.md)** — the full surface, curated.
