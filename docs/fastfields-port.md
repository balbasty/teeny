# Porting fastfields onto teeny — handoff

This document is a self-contained plan for reimplementing the **fastfields**
kernels on top of **teeny**. It exists because the session that built teeny
could not reach the `fastfields/*` repositories (see *Execution* below), so the
work is specified here for a session that can.

Everything needed to be *faithful* to the original algorithms is captured in
this repo:

- **Algorithm spec** — §5 (boundary conditions) and §6 (interpolation orders)
  are transcribed from the jitfields source; §4 gives each kernel's mechanics.
- **Working reference** — `examples/fastfields/{bounds,spline,pushpull}.hpp` +
  `examples/pushpull_adjoint.cpp` implement and validate pull/push on teeny.
- **teeny idioms** — §2, the repo's `CLAUDE.md`, `docs/efficient-kernels.md` (the
  perf idioms consolidated), `docs/dispatch.md`, and `examples/`.

The guiding instruction from the project owner: *understand the point of the
algorithms and their "truthness"; simplify and abstract rather than copying
conservatively, but stay **at least as flexible** as jitfields (same boundary
conditions, interpolation orders, arbitrary batch rank).*

---

## 1. Execution (read first)

**Do the port in its own session.** teeny changes fast; this plan is written from
the teeny side. Start a fresh session focused on the refactor — don't graft it onto a
teeny-maintenance thread.

**Getting the repos into scope.** Run `list_repos` to see what's reachable, then
`add_repo` each `fastfields/*` repo you'll touch (and `balbasty/teeny` if it isn't
already). If a repo is a different tier/org and `add_repo` refuses it, vendor teeny in
instead — copy `include/teeny/` + `examples/fastfields/{bounds,spline,pushpull}.hpp`,
or add teeny as a submodule / CMake dependency (teeny now ships a CMake `INTERFACE`
target `teeny::teeny`, see its `README`).

**Read teeny's docs first — they carry this week's semantics that this plan assumes:**
`CLAUDE.md` (design + the API cheat-sheet), `docs/efficient-kernels.md` (the perf
idioms in one place), `docs/dispatch.md` (the ndarray boundary + the batch idiom),
`docs/shapes-strides.md` (int32 offsets), and `examples/` (`pushpull_adjoint.cpp` is
the flagship).

**Branch.** Develop on `claude/teeny` (or a task branch) in each fastfields repo.

**First action in that session.** `list_repos` + read each fastfields repo's
`README`/`CLAUDE.md` to confirm the role mapping in §7 — the org **may have changed**
since this plan was written; treat §7 as a starting hypothesis, not fact — and to see
how the existing csrc is structured and built.

---

## 2. What teeny gives you (idiom playbook)

teeny is one tensor type, `tensor<T, Shape, Layout, storage>` (the shape is exposed
as `shape_type`/`extents_type`), with per-dimension static/dynamic shape and strides on
`cuda::std::mdspan`. The kernel-relevant surface (value-form sugar shown; every
`method<...>()` also has a deduced value twin — `permute(Int<2>(),Int<0>(),Int<1>())`,
`recast(shape<...>{})` — which avoids `.template` on a dependent receiver):

| Need | teeny |
|---|---|
| kernel-passable strided view | `wrap(ptr, extents)`, `wrap<fcontiguous>(...)`, `wrap(ptr, ext, strides<S...>{})`, `as_tensor(submdspan_result)` |
| element / folded stride | `t(i,j,k)`, `t.data()[off]`, `t.stride(Int<d>())` (static) / `t.stride(d)` (runtime) |
| **scatter (push)** | `t.at(i...).atomic_add_(v)` — **atomic on host and device** |
| assign / init | `t.copy_(src)` (broadcasts), `t.fill_(v)`, `t.zero_()`, `t.iota_(a,b)` |
| in-place / reduce math | `t.add_(x)/mul_(x)/…` (broadcasts), `sum/dot/min/max`. Contiguous out-of-place and in-place scalar/unary ops **auto-vectorize** (see `efficient-kernels.md`) |
| **peel arbitrary batch** | `peel_front(Int<-Sr>())` on an `anyrank` (keep the trailing `Sr` dims static, peel the runtime batch) → `dextents<_,Sr>` cells; `peel_front_at(i, Int<-Sr>())` for a grid-stride index; `size_front(Int<-Sr>())` = cell count. (Each also has the explicit-template spelling `peel_front<-Sr>()` etc.; the `Int<>` value form is deduced, so an impl-layer helper taking the carrier as a template parameter needs no `.template`.) On a static-rank tensor with a *known* batch count, positive `peel_front<Nbatch>(t)` |
| recover static inner dims | `cell.recast(shape<-1, C, C>{})` — fold the known trailing dims of a peeled cell (no copy, preserves strides) |
| peel named axes | `peel(t, axis<0,1>{})`, `slice_along(axis<0,2>{}, …)`, `permute(Int<...>()…)`, `flip(Int<d>())` |
| add/drop size-1 axis | `unsqueeze(Int<Ax>())`, `squeeze(Int<Ax>())` |
| **runtime→static dispatch** | `dispatch_value<1,2,3>(D, f)` (one knob); `dispatch_values(f, candidates<1,2,3>(D), candidates<0,…,7>(order), candidates<0,…,7>(bnd))` (several at once — same nesting, one call, enums need no cast); `dispatch_rank(as_anyrank(...), f)` (total rank — prefer `peel_front<-Sr>` per §3) |
| **narrow device offsets (int32)** | `dispatch_index(v, f)` at the launch site → int32 arm when `v.index_fits<int32_t>()`, else int64. Halves a dynamic view's register footprint + 32-bit address math (a GPU occupancy win). `reindex<int32_t>()` is the raw retype. For a CUDA launch narrow the **carrier** instead — `at.index_fits<int32_t>()` / `at.reindex<int32_t>()` (or `dispatch_index(at, f)`): once, host-side, before the launch, and every cell `peel_front<-Sr>` then hands out is already int32-indexed (the batch idiom keeps `ndim` runtime, so per-cell narrowing can't be the device mechanism) |
| host ndarray boundary | `as_anyrank(data, shape, stride, ndim)` → `anyrank`; `.fixed<R>()`. **Device data:** `as_anyrank<storage::gpu_view>(dptr, …)` so cells are device-tagged; DLPack: `from_dlpack<T[,Space]>` / `dispatch_dlpack<Space>` set+check the space from the capsule |
| **DLPack dtype dispatch (batch idiom)** | `dispatch_dlpack_dtype<Space>(m, f)` — dtype-dispatch that hands `f` the **typed `anyrank`** (rank preserved), so `f` drives `peel_front<-Sr>` (kernel per `Sr`, not per total rank). `dispatch_dlpack<Space>(m, f)` is the rank-collapsing sibling. `f` must be generic over its element type |
| owning buffers | `local<T,E>` (stack, static), `owned<T,E>(e)` (heap host), `gpu/pinned/mapped<T,E>(e)` (from `teeny/cuda.h`); `empty<T[,storage::Space]>(e)` deduces stack/heap from the shape |

What teeny deliberately does **not** do (kept out to stay tiny) and therefore
lives in the fastfields layer: boundary-condition index maps, spline weight
tables, host↔device copies, the parallel-for driver, and atomics selection
beyond `fetch_add`. The reference `examples/fastfields/*.hpp` is where these go.

---

## 3. Dispatch architecture (the core design)

fastfields tensors are **`(*batch, *spatial, C)`**: an arbitrary number of batch
dims, then 1/2/3 spatial dims, then a trailing channel `C`. jitfields does this
with hand-written `index2offset` batch plumbing and giant per-rank switch
statements. On teeny it becomes: dispatch the *spatial rank* static, peel the
runtime batch into the pointer, and keep the trailing `Sr = D+1` (spatial + channel)
dims static — **the kernel instantiates once per `Sr`, not once per total rank.**

```text
host ndarray (numpy/cupy/torch/dlpack)  ──as_anyrank(data,shape,stride,ndim)──►  anyrank
   │  strides are in ELEMENTS (dlpack); numpy's __array_interface__ gives BYTES — /itemsize first
   │  device data: as_anyrank<storage::gpu_view>(dptr,…)  (or from_dlpack<T,storage::gpu_view>)
   ▼
dispatch_values([&](auto D, auto O, auto B){             // spatial rank / order / bound -> static
    constexpr long Sr = D.value + 1;                     // spatial + channel dims to keep static
    for (auto cell : at.peel_front<-Sr>()) {             // NEGATIVE front: peel the runtime batch
        auto v = cell.recast(shape<-1, …static inner…>{});   // fold known inner dims (e.g. C)
        kernel<D.value, O.value, B.value>(v, grid_cell, …);  // parallelise this loop
    }
  },
  candidates<1,2,3>(spatial_ndim),          // spatial rank
  candidates<0,1,2,3,4,5,6,7>(order),       // interpolation order  (optional)
  candidates<0,1,2,3,4,5,6,7>(bnd));        // boundary condition — an enum, no cast
```

One `dispatch_values` call replaces the `dispatch_value` nesting pyramid (it *is* that
nesting: one comparison per parameter, `f` instantiated once per combination), and it
puts the whole instantiation budget — the product of the candidate lists — in one
readable place. Reach for the single-parameter `dispatch_value` when there is only one
knob.

- Note the **negative** `peel_front<-Sr>()` on `anyrank`: it keeps the last `Sr` dims
  static and folds *however many* batch dims there are into each cell's data pointer.
  (A *positive* front-count would leave a runtime rank — a `static_assert`.) If you
  truly need the whole rank static, `dispatch_rank(at, f)` → `fixed<R>()` then a
  positive `peel_front<R - Sr>(t)`, but that instantiates per total rank — avoid it.
- On the **device**, wrap each `kernel(...)` launch in `dispatch_index(v, f)` so a view
  whose element span fits int32 runs the kernel in 32-bit offsets (fewer registers per
  thread → occupancy). Opt in per launch site.

Key facts carried from jitfields (§4.1):

- The **channel axis is never parallelised** — parallelise the flat
  `(*batch, *spatial)` index; loop channels *inside* the kernel. Neighbours and
  weights are computed once per spatial location and reused across channels.
- **Only spatial strides** go to the interpolator; the batch offset is folded into the
  base pointer — exactly what `peel_front<-Sr>` produces (each cell already has the
  batch offset baked into its data handle).
- Choose **static specialisation for D ∈ {1,2,3}** (the common case) and fall back to a
  generic-D path otherwise. `dispatch_value` gives the static D with no hand switch.

For the CPU driver, replace jitfields' `parallel_for(0, numel, grain, …)` with your
platform's parallel-for over the cell range: `size_front<-Sr>()` is the cell count and
`peel_front_at<-Sr>(i)` the i-th cell (grid-stride / worker split). For CUDA, one thread
per cell; `fetch_add` handles push races.

---

## 4. Kernel-by-kernel port

### 4.1 pushpull (pull / push / count / grad / hess + backwards) — PRIMARY

**Reference:** `examples/fastfields/pushpull.hpp` (rank-generic pull+push) and
`examples/pushpull_adjoint.cpp` (validation). Port these, then add the variants.

**Data layout.** `inp`/coeff volume `(*batch, *spatial_in, C)`; `grid`
`(*batch, *spatial_grid, D)` (last axis = the D coordinate components);
`out` `(*batch, *spatial_grid, C)`. grad output appends a `D` axis; hess appends
`D*(D+1)/2` (packed symmetric).

**pull** = separable tensor product: per spatial axis `d` form `order+1`
neighbours at `spline_low(order, loc[d])` with weights
`spline_weight(order, |loc[d]-nb|)`, boundary-map each neighbour to an index +
sign (§5). Gather: `out[c] = Σ_neighbourhood (Πweights)·(Πsigns)·inp[offset + c·stride_C]`.
`sign==0` ⇒ that neighbour contributes 0 (zero boundary); `sign<0` ⇒ negate
(DST bounds).

**push** = the adjoint: same neighbours/weights, but
`out.at(idx...).atomic_add_(val·Πweights·Πsigns)` (atomic). This is `push_rec` in the
reference. **Correctness gate:** the adjoint identity `<pull x, y> == <x, push y>`
must hold (the reference test checks it across orders 0–3 and 4 bounds — keep
that test).

**count** = push of constant `1` (splat the weights themselves; no channel
loop). **grad** = pull using `spline_grad` on one axis at a time: component `d`
is `Σ cget(inp,·)·(Π_{e≠d} w_e)·g_d`; write to the appended `D` axis. **hess**
uses second-derivative weights (jitfields implements them for orders 0–4 only;
5–7 fall back to 0). **pull_backward / push_backward / grad_backward** combine a
push of the incoming grad with a grad-weighted gather (see the jitfields
extraction; all are the same separable recursion with different leaf ops).

**extrapolate** (FOV test) ∈ {1 = always in, 0 = limits at voxel centres
`[-tiny, n-1+tiny]`, -1 = at voxel edges `[-0.5-tiny, n-0.5+tiny]`, tiny=5e-2}:
out-of-FOV pull writes 0.

**Spatial-rank dispatch:** specialise D=1,2,3 via `dispatch_value<1,2,3>`; the
reference recursion already folds for any static D, so you mostly get this free —
only add hand-unrolled D-specific paths if profiling demands it.

**Multi-channel:** wrap the leaf in `for (long c=0;c<C;++c)` using the channel
stride; neighbours/weights are hoisted above the channel loop.

### 4.2 distance (l1, euclidean)

Separable: a 1-D sweep along one axis, batched over the rest. On teeny:
`for (auto line : peel_front<Nbatch>(vol)) sweep(line, w)` — or peel all-but-one
axis and sweep the last. A full transform runs the sweep once per axis (permute
so the target axis is innermost, or peel a different axis each pass).

- **L1** (`examples/distance_transform.cpp` already implements this): forward
  `tmp=min(tmp+w, f[i])` then backward. Two passes, in place — write them as the
  [forward + backward sweep](structure.md#the-forward-backward-sweep-two-pass-line-recurrences),
  `scan_(t, inf, minplus{w}, ax); scan_(t.flip(ax), inf, minplus{w}, ax);`, which
  batches over every other axis itself: no peel loop, no batch-axis list.
- **Euclidean (squared):** lower-envelope-of-parabolas (Felzenszwalb–Huttenlocher).
  Needs 3 scratch buffers of length `n` per worker (`local<double, shape<N>>`
  if N static, else `owned`). Vertices `v`, breakpoints `z`, copy `d`; intersection
  `s = (f[q]-f[v]+w²(q²-v²)) / (2w²(q-v))`; then fill `f[q]=d[v]+w²(q-v)²`.

### 4.3 posdef (cholesky + solve)

Small dense/packed SPD systems per voxel. Storage types auto-detected from the
element count: `Eye`(1), `Diag`(C), `ESTATICS`(2C-1), `Sym`(C(C+1)/2, packed
diag-then-rows), `Full`(C²). Port:

- `cholesky_solve.cpp` already implements dense Cholesky factor + solve on a
  the `wrap(..., strides<S...>{})` matrix with `local` work tensors — that is the `Full`/`Sym`
  path once you expand the packed matrix `tofull` into a `local<T, shape<C,C>>`.
- Add `Diag`/`Eye`/`ESTATICS` fast paths (trivial). Keep the `1e-40` pivot floor
  and the `1.000001` diagonal ridge for conditioning.
- Static-C specialisation via `dispatch_value<1,2,3,4,...>(C, …)`; the 2×2/3×3
  closed-form inverses are worth keeping as specialisations.
- For the per-voxel `C×C` inner loops to fold to hand quality, follow the two static-C
  codegen rules in [efficient-kernels §7](efficient-kernels.md): mark the small static
  loops `TNY_UNROLL` (gcc ignores bare `#pragma unroll`), and **snapshot inputs through
  a `local`** before writing outputs so the compiler doesn't reload them per store.

### 4.4 splinc (spline prefilter)

1-D IIR pole recursion (Unser/Thévenaz) along each axis, batched like distance.
Poles per order (order≤1 none; 2: √8-3; 3: √3-2; 4/5: two; 6/7: three), gain
`Π(1-p)(1-1/p)`. Per pole: scale by gain, causal init + `f[i]+=p·f[i-1]`,
anticausal init + `f[i]=p·(f[i+1]-f[i])`. Only the causal/anticausal boundary
*initialisation* depends on the boundary mode (DCT1/DCT2/DFT implemented in
jitfields). The causal + anticausal pair is the same
[forward + backward sweep](structure.md#the-forward-backward-sweep-two-pass-line-recurrences)
the L1 transform uses — one `scan_` per direction, one functor per pole, with the
boundary initialiser supplying each pass's `init`.

### 4.5 others (resize/restrict, regularisers)

`resize`/`restrict` are pull/push with a scale factor (prolongation/restriction);
build on 4.1. `regularisers` (field/flow, 1d/2d/3d) are stencil operators —
express the stencil with `t(i+di, j+dj, …)` element access (boundary-mapped) and,
for the adjoint/`push`-like accumulation, `at(i...).atomic_add_(v)`. These are lower priority.

---

## 5. Boundary conditions (complete, from jitfields `bounds.h`)

Faithfully implemented in `examples/fastfields/bounds.hpp`. Enumeration:
`{ zero, replicate, dct1, dct2, dst1, dst2, dft, nocheck }`. Each maps an
out-of-range coordinate to an in-range **index** and a **sign** (+1/0/-1); a
gather reads `sign==0 ? 0 : sign<0 ? -data[i] : data[i]`, a scatter adds `sign·v`.

| mode | index map (integral) | sign | note |
|---|---|---|---|
| zero | identity | `(i<0‖i≥n)?0:1` | out-of-range ⇒ 0 |
| replicate | `i≤0?0 : i≥n?n-1 : i` | 1 | nearest edge (clip) |
| dct1 | reflect about border **centres**, period `(n-1)·2`: `t=(n-1)·2; c=abs(i)%t; c≥n?t-c:c` | 1 | `-1→1, n→n-2` |
| dct2 | reflect about border **edges**, period `n·2`: `t=n·2; c=i<0?t-((-i-1)%t)-1:i%t; c≥n?t-c-1:c` | 1 | Neumann; `-1→0, n→n-1` |
| dst1 | `t=(n+1)·2; c=(i==-1)?0:(i<0?-i-2:i); c%=t; c==n?n-1:(c>n?t-c-2:c)` | `t=(n+1)·2; c=(i<0?n-i-1:i)%t; c%(n+1)==n?0:((c/(n+1))%2?-1:1)` | antisymmetric about first OOB centre |
| dst2 | same index as dct2 | `c=i<0?n-i-1:i; (c/n)%2?-1:1` | Dirichlet |
| dft | circular: `i<0?(n+i%n)%n:i%n` | 1 | wrap; `-1→n-1, n→0` |
| nocheck | identity | 1 | assumes in-bounds |

The prefilter (splinc) only needs DCT1/DCT2/DFT boundary initialisers; the DST
family there is not implemented in jitfields.

---

## 6. Interpolation orders (complete, from jitfields `spline.h`)

`{ Nearest=0, Linear=1, Quadratic=2, Cubic=3, FourthOrder=4, …, SeventhOrder=7 }`.
Support = `order+1` neighbours starting at `spline_low(order,x)=floor(x-(order-1)/2)`.
Weight is a function of the **absolute** distance `x=|coord-sample|`. Orders 0–3
are implemented in `examples/fastfields/spline.hpp`; the full set:

- **0 Nearest** (support 1): `w = x<0.5 ? 1 : 0`. grad 0.
- **1 Linear** (2): `w = 1-x` (x<1). grad `-1`.
- **2 Quadratic** (3): `x<0.5: 0.75-x²`; `x<1.5: 0.5(1.5-x)²`. grad `x<0.5:-2x`; `x<1.5:x-1.5`.
- **3 Cubic** (4): `x<1: (3x²(x-2)+4)/6`; `x<2: (2-x)³/6`. grad `x<1: x(1.5x-2)`; `x<2: -0.5(2-x)²`.
- **4** (5): `x<0.5: x⁴/4 - 0.625x² + 115/192`; `x<1.5: x(x(x(5-x)/6 - 1.25)+5/24)+55/96`; `x<2.5: (x-2.5)⁴/24`. (hess implemented)
- **5** (6, f=x²): `x<1: f(f(0.25 - x/12) - 0.5)+0.55`; `x<2: x(x(x(x(x/24-0.375)+1.25)-1.75)+0.625)+0.425`; `x<3: (3-x)⁵/120`.
- **6** (7): `x<0.5: (x²)²(7/48 - x²/36) - (77/192)x² + 5887/11520`; `x<1.5: deg-6 poly + 7861/15360`; `x<2.5: deg-6 poly + 1379/7680`; `x<3.5: (x-3.5)⁶/720`.
- **7** (8, f=x²): `x<1: f³(x/144 - 1/36) + f²/9 - f/3 + 151/315`; `x<2: deg-7 poly + 103/210`; `x<3: deg-7 poly - 139/630`; `x<4: (4-x)⁷/5040`.

For orders 4–7 the exact interior-polynomial coefficients (the `deg-6`/`deg-7`
terms above) should be copied verbatim from jitfields `csrc/lib/spline.h`
(`_spline::weight4..7`, `grad4..7`, `hess0..4`) when implementing — treat the
formulas above as the shape/anchor, not a substitute for the source constants.
`hess` (second derivative) exists only for orders 0–4; 5–7 use `hess=0`.

---

## 7. Repo mapping (inferred — CONFIRM in the port session)

From the org's repo list; roles are guessed from names and must be verified by
reading each repo:

| repo | likely role | port action |
|---|---|---|
| `fastfields-kernels` | core header-only C++/CUDA numerics | **primary**: reimplement on teeny (bounds/spline/pushpull/distance/posdef/splinc) |
| `fastfields-cpu-impl`, `fastfields-cuda-impl` | backend implementations / instantiations | wire the teeny kernels to CPU parallel-for and CUDA launch; the two share one kernel source |
| `fastfields-cpu-lib`, `fastfields-cuda-lib`, `fastfields-lib` | packaged libraries | build/packaging over the impls |
| `fastfields-bind-py`, `fastfields-csrc-*` | binding glue | the `any(...)`/`dispatch_rank` host boundary lives here (element vs byte strides!) |
| `fastfields-numpy/-cupy/-torch` | python bindings (the goal) | consume the built lib; pass `ndarray_like` → `anyrank` |
| `fastfields`, `.github` | umbrella / org profile | docs only |

Vendor teeny (`include/teeny/` + `examples/fastfields/{bounds,spline}.hpp`) into
`fastfields-kernels`, or depend on `balbasty/teeny` if the build allows.

---

## 8. Testing / correctness ("truthness")

1. **Adjoint identity** for pull/push (and grad/push_backward): `<Px,y>=<x,Pᵀy>`.
   Already in `examples/pushpull_adjoint.cpp` — port it per rank/order/bound.
2. **Numeric vs jitfields**: if the port session also has jitfields, diff a grid
   of pull/push/distance/cholesky outputs against the original (bit-close). The
   teeny tests `test_pull`/`test_distance_l1`/`test_posdef` already do this style
   of check against hand references.
3. **Static-shape folding**: assert `sizeof(local<...>)` and that specialised
   D/order paths compile to folded strides (teeny's `stride(Int<d>())`).
4. **Dispatch coverage**: every legal (rank, D, order, bound) combination must
   instantiate; test the `dispatch_value` chains at their bounds.

---

## 9. Simplifications teeny enables (do these rather than copy conservatively)

- Delete `index2offset`/`sub2offset`/`index2sub` batch plumbing → `peel_front`.
- Delete the per-rank hand-unrolled `1d/2d/3d/nd.h` gather trees → one separable
  recursion over static D (the reference `pull_rec`/`push_rec`).
- Delete `Pointer<T,S>` static-stride pointer → `wrap(ptr, ext, strides<S...>{})` /
  `stride(Int<d>())`.
- Replace the `has_atomic_add` fork with `fetch_add` (atomic on device via
  `atomicAdd`, and atomic on the host too via `cuda::std::atomic_ref` for
  arithmetic element types) — keep the batch-parallel/spatial-sequential
  path only for the non-arithmetic (`half`/`bfloat16`) element types that have
  no atomic representation to route through.
- Keep the giant order/bound/rank switches only at the **dispatch boundary**
  (`dispatch_value`), not threaded through every function.
