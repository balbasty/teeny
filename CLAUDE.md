# CLAUDE.md — working on teeny

teeny is a **header-only C++17 tensor library for host + CUDA device**, built on
NVIDIA CCCL's `cuda::std::mdspan`. One tensor type, with **per-dimension static
and/or dynamic shape and strides**, so a single kernel source folds to
immediates when shapes are static and stays generic when they are dynamic. It
exists to make numeric C++/CUDA kernels (spline interpolation, distance
transforms, small linear algebra) compact and readable.

## Golden rules

1. **mdspan does the heavy lifting.** Extents, layouts, offset mapping, and
   `submdspan` come from `cuda::std`. teeny adds *only* what mdspan lacks. Before
   writing new machinery, check whether `cuda::std::mdspan`/`submdspan`/`extents`
   already does it. Do not reinvent layouts or offset math.
2. **Everything must compile on both `clang++` and `g++` at `-std=c++17`.** Run
   `make CXX=clang++ run-test && make CXX=g++ run-test` before every commit. The
   floor is C++17 because CCCL refuses lower.
3. **Device-safe by construction.** No virtuals, no exceptions, no RTTI, no
   `Date.now`-style host-only calls in `_TNY_API` code. Engines are **lambda-free**
   (index-sequence folds + tiny functors) so they instantiate under `nvcc`
   *without* `--extended-lambda`. Keep them that way.
4. **`_TNY_API` = host+device, `_TNY_HOST` = host-only.** Anything that
   allocates (heap/CUDA storage, out-of-place ops on dynamic shapes) is
   `_TNY_HOST`. Element access, in-place math, views, static-shape out-of-place
   ops are `_TNY_API`. A `_TNY_API` function must not call a `_TNY_HOST` one on
   the device path.
5. **Keep it teeny.** Every header is small and single-purpose. Prefer deleting
   code to adding it. The whole point is compactness; resist feature creep that
   does not serve real kernels.

## Layout of the repo

```
include/teeny/
  _core/defines.h  macros: _TNY_API / _TNY_HOST, namespace open/close
  alias.h          pulls cs:: vocabulary into tny:: + Int<V>/Long<V>/... static ints + `all`
  half.h           `half` (IEEE binary16) + `bfloat16` element types + compute_type
  storage.h        `own` enum + storage policies (owning_storage<T,Alloc>, cpp_alloc)
  layout.h         layout_static_stride<S...> — per-dim compile-time strides
  tensor.h         the tensor class + view/local/owned/view_t aliases + as_tensor
                   + indexing/slicing, take_along, permute, unsqueeze/squeeze
  math.h           in-place & out-of-place elementwise (broadcasting) + unary math
                   + reductions (sum/prod/max/min/dot). Members declared in
                   tensor.h, DEFINED here.
  iterate.h        nd-peel: peel<Axes...> / peel_at<Axes...>
  helpers.h        batch_offset (index2offset), channel
  dynamic.h        any_tensor + dispatch_rank (runtime-rank host boundary)
  cuda.h           OPT-IN device/host/pinned memory (needs <cuda_runtime.h>);
                   NOT included by teeny.h
  teeny.h          umbrella (everything except cuda.h)
tests/             one file per feature; `make run-test` builds + runs all
examples/          standalone example algorithms (see examples/README.md)
external/cccl/     vendored CCCL 3.3.0 (libcudacxx). -I external/cccl/libcudacxx/include
```

`namespace tny` is the public namespace; `tny::_md` and `tny::_detail` are
internal. `namespace cs = cuda::std;` throughout.

## The tensor type

```cpp
template <class T, class Extents, class Layout = layout_right, own O = own::view>
struct tensor;
```

- **`T`** element type. Any arithmetic type, plus teeny's `half` (IEEE binary16)
  and `bfloat16` (`half.h`) — half-precision math computes/accumulates in `float`
  (`compute_type<T>`) so reductions don't lose precision.
- **`Extents`** = `cuda::std::extents<Idx, E0, E1, ...>`; each `Ei` is a compile-time
  size or `dynamic_extent`. Mix freely per dimension.
- **`Layout`** = `layout_right` (default, C-order), `layout_left` (F-order),
  `layout_stride` (runtime strides), or teeny's `layout_static_stride<S...>`
  (compile-time strides).
- **`O`** ownership: `view` (non-owning, trivially copyable, kernel-passable),
  `stack` (inline array, requires fully static shape), `heap` (host `new[]`,
  move-only), or CUDA `device`/`host`/`pinned` (from `cuda.h`).

The mapping lives in an **empty base** (private inheritance → EBO), so a
fully-static stack tensor is *exactly* `sizeof` its data.

### Aliases (prefer these over spelling out `tensor<...>`)

```cpp
view_t<T,E,L>   // non-owning view type
local<T,E,L>    // stack-owned (static shape)     e.g. local<double, extents<long,3,3>>{}
owned<T,E,L>    // heap-owned (host, move-only)    e.g. owned<double, DynE>(DynE{2,3})
device/host/pinned<T,E,L>   // from cuda.h
```

Factories: `view(ptr, extents)` / `view<Layout>(ptr, extents)`,
`view_strided<S...>(ptr, extents)` (compile-time strides),
`as_tensor(any_mdspan)` (wrap a submdspan/mdspan result as a view).

## API cheat-sheet

```cpp
// --- geometry (static index -> integral_constant, runtime index -> value) ---
t.rank();  t.numel();
t.extent(Int<0>());   // static lookup -> integral_constant when the extent is static
t.extent(0);          // runtime lookup -> index_type
t.stride(Int<1>());   // static when derivable (static-stride layout, or contiguous+static)
t.data();  t.view();  t.extents();  t.mapping();

// --- indexing / slicing (python-like) ---
t(1, 2, 3);           // element access; negative indices wrap (count from the back)
t(0, all, slice(1,4));  // any slice arg -> a lower-/same-rank VIEW. all = keep axis,
                      //   slice(a,b) = half-open [a,b). Integer args drop that axis.
t(0, slice(none,4), slice(1,none,2));  // python-like: none = open end, 3rd arg = step.
                      //   negative bounds wrap. Two open-end sentinels: static `none`
                      //   (folds: all == slice(none,none), keeps static extent) and
                      //   runtime `rnone` (whole axis resolved at run time -> dynamic).
                      //   NB a range routes through layout_stride (ranged axis -> dynamic;
                      //   `all`-kept axes stay static). See the CCCL note in tensor.h.
t.take_along<0,2>(i, slice(1,4));  // bind named axes only; keep every other axis
t.permute<2,0,1>();   // reorder axes (a permutation of 0..N-1) -> view
t.unsqueeze<2>();     // insert size-1 axis at pos 2 (numpy newaxis) -> rank+1 view
t.squeeze<3>();       // drop a size-1 axis -> rank-1 view

// --- math (in-place: any tensor/view; mutates *this) ---
a.add_(b); a.sub_(b); a.mul_(b); a.div_(b);   // tensor rhs BROADCASTS numpy-style
a.add_(2.0); a.mul_(0.5);                     // scalar rhs
a.neg_(); a.abs_(); a.exp_(); a.log_();       // unary in-place
a.sin_(); a.cos_(); a.sqrt_(); a.tanh_(); a.pow_(3.0);

// --- assignment / scatter (kernel prologue/epilogue) ---
a.fill_(0.0); a.zero_(); a.copy_(b);          // b broadcasts into a
a.add_at(v, i, j);                            // scatter-accumulate: a(i,j) += v,
                                              //   ATOMIC on device (push/splat write)

// --- math (out-of-place -> NEW tensor; static shape -> stack, else heap/host) ---
auto c = a + b;  auto c = a.add(b);   // tensor+tensor (broadcasts) or tensor+scalar
auto c = a * 2.0;  auto c = 2.0 * a;  // scalar ops (+ and * are commutative)
auto c = a.pow(b);                    // element-wise power
auto e = exp(a); auto e = sqrt(a);    // unary free functions (neg/abs/exp/log/sin/cos/sqrt/tanh)

// --- reductions -> scalar ---
sum(a); prod(a); max(a); min(a); dot(a,b);

// --- nd-peel: iterate a SUBSET of axes, each yielding a lower-rank view ---
for (auto line : peel<0,1>(t)) f(line);   // peel axes 0,1; each `line` is a view
auto s = peel_at<0,1>(t, i);               // the i-th peeled sub-view (grid-stride style)

// --- nd-peel: peel the FIRST N axes (arbitrary batch rank) ---
for (auto v : peel_front<N>(t)) f(v);      // v is (*spatial, C); N = #batch dims
auto v = peel_front_at<N>(t, i);            // the i-th (grid-stride style)

// --- dynamic-rank / dynamic-value host boundary ---
auto at = any(data, shape, stride, ndim);    // rank-erased, bounded MaxRank (default 8)
dispatch_rank(at, [&](auto v){ kernel(v); });  // instantiates kernel once per rank
auto v3 = at.fixed<3>();                      // or force a known rank
dispatch_value<1,2,3>(D, [&](auto d){ kern<d.value>(v); });  // runtime value -> static
```

### Static vs runtime values (important idiom)

The API accepts **runtime integers, static integers (`integral_constant`), and
slices of either**, and dispatches to the right output type:

- `alias.h` provides short names for `cuda::std::integral_constant`:
  `Int<V>`, `Long<V>`, `Size<V>`, `Uint<V>`, `Int32<V>`, `Int64<V>`, `Diff<V>`,
  `Bool<V>`, and `ic<V>`. Each converts implicitly to a runtime integer *and*
  carries `::value`.
- `extent(Int<0>())` returns an `integral_constant` when that extent is static
  (folds into later arithmetic); `extent(0)` returns a runtime `index_type`.
- Same for `stride(...)`. A static-stride layout, or a contiguous layout over
  static extents, yields a static stride.

Rule of thumb: **pass a static index (`Int<k>()`) when you want the compiler to
fold**, a plain `int`/`long` when the value is only known at run time.

## How the hard parts work (so you don't re-derive them)

- **Broadcasting** (`math.h`, `_md::bzip`): operands are aligned by rank; a
  dimension of extent 1 gets stride 0 on that axis (it is stretched). The result
  extent per axis is computed at compile time by `bc1`/`bcast_extents`
  (`dynamic_extent` if either operand is dynamic). Out-of-place: a fully static
  result → `own::stack` (host+device); any dynamic → `own::heap` (host only). The
  SFINAE keys on `bcast_extents<...>::rank_dynamic()`, **not** on instantiating a
  stack tensor (that would fire the "stack needs static shape" `static_assert`).
- **nd-peel** (`iterate.h`): a linear index over the peeled axes is decoded to a
  multi-index, then `submdspan` binds those axes and keeps the rest with
  `full_extent`. This replaces jitfields' hand-written `index2offset`.
- **layout_static_stride** (`layout.h`): the one thing mdspan lacks — strides
  baked into the type. NOTE: `submdspan` (and therefore `peel`/`take_along`/
  `permute`) is only defined by CCCL for the *standard* layouts, so it does NOT
  work on `layout_static_stride`. Use it for whole-tensor access with folded
  strides; use `layout_right`/`left`/`stride` when you need to slice.
- **EBO**: `tensor : private Layout::mapping<Extents>`. `mapping()` returns
  `*this`. Do not add non-static data members besides `store_`.

## Adding a feature — checklist

1. Is it really missing from mdspan? If not, alias it in `alias.h` instead.
2. Member function? **Declare** it in `tensor.h`, **define** it in `math.h`
   (math) or inline in `tensor.h` (structural). Watch declaration-before-use:
   `math.h` has a single `_md` engine block first, then all member/operator
   definitions — keep new engines in that block.
3. Keep it lambda-free and `_TNY_API` unless it allocates (then `_TNY_HOST`).
4. Add a `tests/test_<feature>.cpp` mixing `static_assert` (compile-time shape
   checks) and runtime asserts; wire it into `Makefile`'s `TESTS` list.
5. `make CXX=clang++ run-test && make CXX=g++ run-test` — all green.
6. Commit with a focused message.

## Testing

`make run-test` builds and runs every `tests/test_*.cpp`, printing `PASS`/`FAIL`.
Tests return non-zero on failure (the return code is the failing check number).
`test_cuda` compiles against a malloc-backed fake CUDA runtime in
`tests/fakecuda/`. `test_pull`, `test_distance_l1`, `test_posdef` port real
kernels and validate them numerically against hand-written references — when you
change math or layout code, these are the ones that catch regressions.

## Downstream: fastfields

teeny is the substrate for the `fastfields` kernels (spline pull/push, distance
transforms, small SPD solves). Those kernels care about: arbitrary **batch**
rank (shape `(*batch, *spatial, C)`, not a fixed number of batch dims), a small
set of **spatial** ranks (1/2/3D) worth dispatching to specialized code, and
configurable **boundary conditions** and **interpolation order**. Design teeny
changes so those stay expressible and fast.

- **Idioms:** `examples/` — `pushpull_adjoint.cpp` is the flagship (pull+push).
- **Reference numerics:** `examples/fastfields/{bounds,spline,pushpull}.hpp`
  (the 8 boundary conditions + spline weight tables + separable gather/scatter,
  transcribed from jitfields — domain code that lives outside teeny core).
- **Full porting plan:** `docs/fastfields-port.md` — the dispatch architecture
  (`(*batch,*spatial,C)` via `peel_front`/`dispatch_value`), the complete
  boundary/order spec, kernel-by-kernel mechanics, and repo mapping.
