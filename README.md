# teeny

> ⚠️ **Vibe-coded.** teeny was written by an AI coding assistant and reviewed by a
> human. Treat it as experimental — expect rough edges, and pin a commit if you
> depend on it.

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

## Using it in your build

teeny is header-only, so you only need the include paths above. Two convenient
ways to wire it in:

**CMake** — teeny ships an `INTERFACE` target `teeny::teeny` that carries the
include dirs, the C++17 requirement, and the CCCL dependency.

```cmake
# In-tree (a submodule / vendored copy), or via FetchContent:
add_subdirectory(teeny)
target_link_libraries(my_app PRIVATE teeny::teeny)

# ...or after `cmake --install`:
find_package(teeny CONFIG REQUIRED)   # re-resolves CCCL via find_package(CCCL)
target_link_libraries(my_app PRIVATE teeny::teeny)
```

If CCCL is installed it is found automatically; otherwise the vendored submodule
is used for in-tree builds. Run the test suite with `-DTEENY_BUILD_TESTS=ON` then
`ctest`; `-DTEENY_BUILD_EXAMPLES=ON` builds each [`examples/`](examples) program as
`ex_<name>`, matching the Makefile's binary names. (The [`Makefile`](Makefile)
remains the reference build for the tests.)

**Plain compiler flags** — `-std=c++17 -I include -I external/cccl/libcudacxx/include`.

## At a glance

```cpp
#include <teeny/teeny.h>
using namespace tny;

// one tensor type, ownership as a parameter (view / stack / heap / device / ...).
// `shape<...>` is the python-friendly extents type (int64 index, matches DLPack):
auto v = wrap(ptr, shape<2,3,4>{});                           // non-owning, kernel-passable
auto m = local<double, shape<3,3>>();                         // stack-owned (static shape)
auto h = owned<double, shape<-1,3>>(shape<-1,3>{n});   // heap-owned (host); -1 == dynamic

// static / dynamic sizes mix per dimension, plus a per-dim static-stride layout:
auto s = wrap(ptr, shape<dynamic_extent,3,3>{n}, strides<16,3,1>{});   // compile-time strides

// geometry: t.shape() / t.shape(d) (or t.extents() / t.extent(d)); t.rank(); t.numel();

// valarray-like math (broadcasting, numpy-style; float16 > float32 > float64):
m.add_(other); m.mul_(2.0);  // in-place, any tensor/view
auto c = a + b;              // out-of-place: static -> stack (host+device),
auto d = a.add(b);           // dynamic -> heap (host only)
auto e = exp(a); a.sqrt_();  // unary math (out-/in-place)
sum(m); prod(m); max(m); min(m); dot(a,b);

// indexing / slicing (python-like: negatives wrap; none = open end; 3rd arg = step):
t(0, -1, slice(1,4));                                // element, or a sub-view
t(all, slice(none,4), slice(1,none,2));              // keep axis / open ends / strided
t.slice_along(axis<0,2>{}, i, all);                   // bind named axes, keep the rest
t.permute(Int<2>(), Int<0>(), Int<1>());             // reorder axes
t.unsqueeze(Int<2>());                               // insert a size-1 axis (numpy newaxis)
for (auto line : peel(t, axis<0,1>{})) work(line);   // nd-peel: iterate a subset of axes
for (auto v : peel_front<Nbatch>(t)) work(v);        // peel arbitrary leading batch dims
dispatch_value<1,2,3>(D, [&](auto d){ kernel<d.value>(v); });  // runtime -> static

// half precision: `half` / `bfloat16` element types (native CUDA types under nvcc)
auto hh = local<half, shape<64,64>>();
```

## Headers (`include/teeny/`)

| Header | Contents |
|---|---|
| `teeny.h` | umbrella (includes everything; `cuda.h` is pulled in, self-guarded — a no-op without the CUDA runtime). `dlpack.h` is the one opt-in extra |
| `alias.h` | `shape<...>`, `Int<V>`/… static ints, `all`, mdspan vocabulary in `tny::` |
| `half.h` | `half` (binary16) + `bfloat16` element types + `compute_type` |
| `storage.h` | `storage` modes + storage policies (`owning_storage<T,Alloc>`) |
| `layout.h` | `strides<S...>` (`= layout_static_stride`) — per-dim static/dynamic strides |
| `indexing.h` | slice vocabulary: `slice()`/`none`, axis/index wrapping |
| `axis.h` | view builders: `permute`/`flip`/`squeeze`/`unsqueeze` |
| `tensor.h` | `tensor<T, Extents, Layout, storage>` + `view`/`local`/`owned` + slicing / `slice_along` / `at` |
| `math.h` | in-place / out-of-place ops (broadcasting) + unary math + reductions |
| `iterate.h` | nd-peel: `peel` / `peel_at` / `peel_front` / `peel_front_at` |
| `dynamic.h` | `anyrank` (rank-erased carrier) + `peel_front<-Sr>` + `dispatch_rank` |
| `cuda.h` | opt-in gpu / pinned / mapped memory (needs the CUDA runtime) |

## Building & testing

```sh
make run-test              # build + run all tests (default toolchain)
make CXX=clang++ run-test  # or pick a compiler
```

Tests under `tests/` mix `static_assert` batteries with host runtime checks;
`test_distance_l1`, `test_pull`, and `test_posdef` port real jitfields kernels
and validate them numerically. `test_cuda` exercises the CUDA storage against a
malloc-backed fake runtime (`tests/fakecuda/`).

## Contributing

Bug reports and PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
workflow (issue-based, conventional commits, both-compiler test gate) and coding
style; [`CLAUDE.md`](CLAUDE.md) has the deep design rules.

## Status

Implemented and tested on host under clang++ and g++ at C++17. Device (`nvcc`)
compilation is intended — the headers avoid virtuals, exceptions, and (on the
device path) allocation — but a real GPU CI leg is still to be added, and
`teeny/cuda.h` is so far validated only structurally.
