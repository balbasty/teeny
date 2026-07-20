# teeny

A teeny-tiny, header-only tensor library that works on host _and_ device (CUDA).

teeny's distinguishing feature is **hybrid static/dynamic shapes and strides**:
every extent and stride of a tensor may be known at *compile time*, at *run
time*, or a mix — decided **per dimension**. Dimensions known statically fold
their index arithmetic into immediates (zero registers, zero loads); dynamic
ones are carried as ordinary values. The same source specializes to a fully
dynamic kernel, a fully static one, or anything in between — no separate
`1d`/`2d`/`3d`/`nd` copies.

It is meant to be `#include`d into C++/CUDA kernels (e.g. the ones in
[jitfields](https://github.com/balbasty/jitfields)) to make them compact.

## Requirements

- **C++17** (the floor is set by the CCCL dependency, which refuses to build
  below 17).
- A host compiler (tested: g++ 13, clang++ 18) and/or `nvcc` for device code.
- [NVIDIA CCCL](https://github.com/NVIDIA/cccl) for `cuda::std::*` on host and
  device, vendored as a submodule:

  ```sh
  git submodule update --init --depth 1 external/cccl
  ```

Everything is header-only under `include/teeny`; add `-I include` and
`-I external/cccl/libcudacxx/include`.

## Layers

| Header | What it gives you |
|---|---|
| `teeny/core.h` | host/device macros, `cuda::std` integer aliases |
| `teeny/statix.h` | `tny::statix` — compile-time metaprogramming: `pack`, `carray`, `tuple`, a uniform pack API (`get`/`at`/`head`/`tail`/`cat`/`erase`/`reversed`/…), `_math` (`prod`/`sum`/`cumprod`/`shifted_cumprod`/…), Python-style indices & slices |
| `teeny/xarray.h` | `tny::xarray<T, values>` — the hybrid 1-D array, plus its algebra and structural ops |
| `teeny/tensor.h` | `tny::tensor<T, Offset, Shape, Stride>` — the N-D strided view |

### `xarray<T, values>`

A 1-D array where each slot is static (`cvalue<T,X>`) or dynamic (`cnone`).
**Only the dynamic elements are stored** (`sizeof == max(1, num_dynamic *
sizeof(T))`); static values live in the type. It is trivially copyable.

```cpp
using namespace tny; using namespace tny::statix;

// [ ?, 3, 4 ] : one runtime extent, two compile-time ones.
xarray<long, tuple<cnone, cvalue<long,3>, cvalue<long,4>>> a;
a[csize<0>()] = 5;          // write the dynamic slot (returns long&)
a[csize<1>()];              // -> 3, a compile-time prvalue
a.front(); a.back();        // negative/static indices supported
```

Algebra (`tny::` free functions; fully-static inputs fold to a compile-time
`cvalue`, otherwise a runtime value):

```cpp
tny::prod(a);               // numel;  tny::sum, tny::max, tny::dot(idx, stride)
tny::for_each(a, f);        // unrolled loop: f(csize<I> tag, element)
tny::from_pointer<V>(p);    // build from a raw size[]/stride[] vector
tny::select<2,0,1>(a);      // gather / permute
tny::erase<1>(a);           // drop a slot (squeeze); tny::reversed(a)
```

### `tensor<T, Offset, Shape, Stride>`

A non-owning N-D strided view = `{ T* data; xarray shape; xarray stride; }`.
Trivially copyable, so it passes into a `__global__` kernel by value.

```cpp
auto t = tny::make_tensor<Shape>(data, sizes, strides);   // Stride defaults dynamic
t.numel();                          // folds to a constant for a static shape
t(i, j, k);                         // mixed runtime / static (csize<>) indices
t.offset_at(lin);  t.foffset(lin);  // row-major / column-major linear decode
t.sub<D>(i);                        // drop a dim -> (ndim-1) view
t.permute<2,0,1>();                 // reorder dims
```

For a fully-static shape+stride, `t(i,j,k)` compiles to the same immediate/shift
address arithmetic as hand-written code — no shape or stride is loaded from
memory.

## Building & testing

```sh
make run-test              # builds and runs everything (default toolchain)
make CXX=clang++ run-test  # or pick a compiler
make run-test-xarray       # a subset: statix / xarray / tensor
```

Tests are header-only translation units under `tests/`, mixing `static_assert`
batteries (type-level results, `sizeof`/EBO layout locks, trivially-copyable)
with host runtime checks. `tests/test_tensor_distance_l1.cpp` ports the core of
a real jitfields kernel and asserts bit-identical results against the original
batching plumbing.

## Status

`statix`, `xarray` (+ algebra + structural ops), and `tensor` are implemented
and tested on host under clang++ and g++ at C++17. Device (`nvcc`) compilation
is intended and the headers avoid virtuals, exceptions, and device-side
allocation, but a real GPU CI leg is still to be added. Slicing, strides-from-
shape derivation, and further kernel ports are the natural next steps.
