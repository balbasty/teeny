# teeny

A teeny-tiny, header-only tensor library that works on host _and_ device (CUDA),
built on NVIDIA CCCL's `cuda::std::mdspan`.

teeny is one tensor type whose **shape and strides may be static, dynamic, or a
mix — per dimension** — so a kernel specializes to fully-dynamic, fully-static
(everything folds to immediates), or anything in between, from a single source.
It is meant to be `#include`d into C++/CUDA kernels (e.g.
[jitfields](https://github.com/balbasty/jitfields)) to make them compact.

mdspan (from CCCL) does the heavy lifting — per-dimension `extents`, layouts,
offset mapping, and `submdspan`. teeny adds only what mdspan lacks: per-dimension
compile-time strides, owning storage (stack / heap / CUDA memory), a
valarray-like math layer, and kernel ergonomics (nd-peel, dynamic-rank dispatch).

## Requirements

- **C++17** (the floor set by CCCL, which refuses to build lower).
- A host compiler (tested: g++ 13, clang++ 18) and/or `nvcc`.
- CCCL, vendored as a submodule:
  ```sh
  git submodule update --init --depth 1 external/cccl
  ```
  Add `-I include -I external/cccl/libcudacxx/include`.

## At a glance

```cpp
#include <teeny/teeny.h>
using namespace tny;
namespace cs = cuda::std;

// one tensor type, ownership as a parameter (view / stack / heap / device / ...):
auto v = view(ptr, cs::extents<long,2,3,4>{});     // non-owning, kernel-passable
auto m = local<double, cs::extents<long,3,3>>();   // stack-owned (static shape)
auto h = owned<double, DynE>(DynE{2,3});           // heap-owned (host, move-only)

// mdspan's static/dynamic extents + a custom per-dim static-stride layout:
auto s = view_strided<16,3,1>(ptr, cs::extents<long,cs::dynamic_extent,3,3>{n});

// valarray-like math (broadcasting, numpy-style):
m.add_(other); m.mul_(2.0);                        // in-place, any tensor/view
auto c = a + b;           // out-of-place: static -> stack (host+device),
auto d = a.add(b);        //               dynamic -> heap (host only)
auto e = exp(a); a.sqrt_();                         // unary math (out-/in-place)
sum(m); prod(m); max(m); min(m); dot(a,b);

// indexing / slicing (python-like: negatives wrap, `all`/`rng` slice):
t(0, -1, slice(1,4));       // element or sub-view
t.take_along<0,2>(i, all);// bind named axes, keep the rest
t.permute<2,0,1>();       // reorder axes
t.unsqueeze<2>();         // insert a size-1 axis (numpy newaxis)
for (auto line : peel<0,1>(t)) work(line);       // nd-peel: iterate a subset of axes
auto v3 = at.fixed<3>();  // dynamic-rank dispatch at the host boundary
```

## Headers (`include/teeny/`)

| Header | Contents |
|---|---|
| `teeny.h` | umbrella (everything except `cuda.h`) |
| `storage.h` | `own` modes + storage policies (`owning_storage<T,Alloc>`) |
| `layout.h` | `layout_static_stride<S...>` — per-dim compile-time strides |
| `tensor.h` | `tensor<T, Extents, Layout, own>` + `view`/`local`/`owned` + slicing / `take_along` / `permute` / `unsqueeze` |
| `math.h` | in-place / out-of-place ops (broadcasting) + unary math, `sum`/`prod`/`max`/`min`/`dot` |
| `iterate.h` | nd-peel: `peel<Axes…>` / `peel_at<Axes…>` |
| `helpers.h` | `batch_offset` (index2offset), `channel` |
| `dynamic.h` | `any_tensor` + `dispatch_rank` (runtime rank) |
| `cuda.h` | opt-in device / host / pinned memory (needs the CUDA runtime) |

## Building & testing

```sh
make run-test              # build + run all tests (default toolchain)
make CXX=clang++ run-test  # or pick a compiler
```

Tests under `tests/` mix `static_assert` batteries with host runtime checks;
`test_distance_l1`, `test_pull`, and `test_posdef` port real jitfields kernels
and validate them numerically. `test_cuda` exercises the CUDA storage against a
malloc-backed fake runtime (`tests/fakecuda/`).

## Status

Implemented and tested on host under clang++ and g++ at C++17. Device (`nvcc`)
compilation is intended — the headers avoid virtuals, exceptions, and (on the
device path) allocation — but a real GPU CI leg is still to be added, and
`teeny/cuda.h` is so far validated only structurally.
