# teeny

A **teeny, header-only C++17 tensor library** for host *and* CUDA device, built on
NVIDIA CCCL's `cuda::std::mdspan`.

teeny is **one tensor type** whose shape and strides may be **static, dynamic, or a
mix — per dimension**. A kernel written once specializes to fully-dynamic,
fully-static (everything folds to immediates), or anything in between, from a
single source. It exists to make numeric C++/CUDA kernels — spline interpolation,
distance transforms, small linear algebra, batched solves — compact, readable,
and *fast*.

```cpp
#include <teeny/teeny.h>
using namespace tny;

// a view over caller memory (non-owning, trivially copyable, kernel-passable)
auto img = view(ptr, shape<480, 640>{});      // static 480×640
auto vol = view(ptr, shape<-1, 3, 3>{n});     // dynamic batch, static 3×3

img(0, -1);                 // element access, python-style negative index
img(all, slice(0, 32, 2));  // a strided sub-view (no copy)
auto m = local<double, shape<3,3>>();          // stack-owned, exactly 9 doubles
m.fill_(0.0); m.add_(other);                   // in-place math, broadcasting
auto c = a + b;                                // out-of-place (promotes types)
```

## Why teeny

- **Zero-overhead when static.** A `shape<3,3>` access compiles to a load at a
  constant offset — no stride math. On a static shape, `A(0,0)+A(1,1)+A(2,2)`
  is literally `movsd (%rdi); addsd 32(%rdi); addsd 64(%rdi); ret`.
- **One source, static *or* dynamic.** The same kernel body works whether the
  extent is a compile-time `3` or a runtime value; you pay for dynamism only
  where you use it.
- **mdspan does the heavy lifting.** Extents, layouts, offset mapping and
  `submdspan` come from `cuda::std`. teeny adds only what mdspan lacks:
  per-dimension static strides, owning storage, a valarray-like math layer,
  broadcasting, and kernel ergonomics (nd-peel, runtime→static dispatch).
- **Host and device.** No virtuals, exceptions, or RTTI; engines are lambda-free
  so they build under `nvcc` without `--extended-lambda`. A `view` is trivially
  copyable — pass it straight into a `__global__` kernel.

## Where to next

- **[Getting started](getting-started.md)** — build flags and your first program.
- **[Tensors & ownership](tensors.md)** — `view` first, then what owning variants add.
- **[Shapes & strides](shapes-strides.md)** — `shape<...>`, `strides<...>`, static vs dynamic.
- **[Indexing & slicing](indexing.md)** — elements, slices, `none`/step, negatives.
- **[Math & broadcasting](math.md)** — in-place / out-of-place, promotion, reductions.
- **[Views & structure](structure.md)** — `permute`/`flip`/`reshape`/`peel`/…
- **[Dispatch & the ndarray boundary](dispatch.md)** — runtime rank/value → static.
- **[Tutorial: DLPack → Python](tutorials/dlpack-python.md)** — write efficient
  CPU+CUDA kernels over `ndarray`-like inputs and bind them to Python.
- **[API reference](reference.md)** — the full surface, curated.
