# API reference

A thorough, type-annotated reference for the public API. For a one-screen scan
of the whole surface, see the [Cheat sheet](cheatsheet.md); for the raw
Doxygen-extracted signatures, see [Autodoc](api/index.md).

Everything is in `namespace tny` (`namespace cs = cuda::std`). Include
`<teeny/teeny.h>` for all of it; `<teeny/cuda.h>` adds the CUDA memory spaces.

Legend: **`Idx`** = the extents' `index_type` (`int64_t` for `shape<...>`).
**`T`** = element type. A *static index* is an `integral_constant` (`Int<k>()`);
a *runtime index* is a plain integer. "→ view" means a non-owning
`tensor<…, storage::view>` aliasing the same memory (no copy).

---

## The tensor type

```cpp
template <class T, class Shape, class Layout = ccontiguous, storage O = storage::view>
struct tensor;
```

| Parameter | Meaning | Typical value |
|---|---|---|
| `T` | element type | any arithmetic type, `half`, `bfloat16` |
| `Shape` | the **shape** (exposed as `shape_type` / `extents_type`) | `shape<2,3>` (a `cs::extents<int64_t,…>`; `-1` = dynamic) |
| `Layout` | memory order | `ccontiguous` (C, default), `fcontiguous` (F), `dynamic_strides` (runtime), `strides<S...>` (static/mixed) |
| `O` | ownership | `storage::view` (default), `storage::stack`, `storage::heap`, `storage::gpu`/`pinned`/`mapped`, `storage::gpu_view` |

Slicing / permuting / peeling / `.at()` of a `gpu` tensor yields a `storage::gpu_view`
(a non-owning view of *device* memory), so a device pointer is never mistaken for
a host one in the type. Helpers: `storage_is_device` (gpu/gpu_view),
`storage_is_host_accessible`, `storage_is_view` (view/gpu_view/pinned_view/mapped_view),
`storage_view_of(O)` (the view kind that preserves a source's space). `pinned`/`mapped`
are host-accessible, but a view of one keeps its space — `pinned_view` / `mapped_view`
(so DLPack can label it `kDLCUDAHost`) — while behaving like a plain `view` otherwise.

### Ownership aliases

| Alias | Ownership | Notes |
|---|---|---|
| `view<T,E,L=ccontiguous>` | none (host view) | trivially copyable, kernel-passable; the bare `tensor` is this |
| `local<T,E,L>` | stack | requires a fully static shape; `sizeof` == its data |
| `owned<T,E,L>` | heap (host) | move-only |
| `gpu<T,E,L>` / `pinned<T,E,L>` / `mapped<T,E,L>` | CUDA | from `<teeny/cuda.h>`; a view of a `gpu` is `storage::gpu_view` |

---

## Factories

Wrap existing memory (→ view):

| Call | Returns | Notes |
|---|---|---|
| `wrap(ptr, shape)` | `view<T,E>` | C-order (`ccontiguous`) |
| `wrap<Layout>(ptr, shape)` | `view<T,E,Layout>` | chosen layout |
| `wrap(ptr, shape, {s0,s1,…})` | `view<T,E,dynamic_strides>` | **runtime** strides (elements; may be negative). A **stride-0** axis self-overlaps — fine to read (broadcast), but an in-place write is a host-debug error (`clone()` first) |
| `wrap<S...>(ptr, shape, {dyn…})` | `view<T,E,strides<S...>>` | **mixed** static/runtime strides (`dynamic_stride` slots) |
| `wrap(ptr, shape, strides<S...>{})` | `view<T,E,strides<S...>>` | **compile-time** strides (fold into the type) |
| `wrap(…, storage_v<storage::gpu>)` | view in that space | trailing **memory-space** tag on any overload above (default `storage::view`); pass the plain backend — `storage::gpu`/`pinned`/`mapped` — and it folds to the view kind (`gpu_view`/…), since `wrap` always views. `storage_c<S>{}` / `storage_v<S>` are the braced / no-braces spellings |
| `as_tensor(md)` | `view<…>` | wrap any `cs::mdspan`/`submdspan` result |
| `make_view(ptr, shape)` | `view<T,E>` | an alias of `wrap` that deduces `E` (`make_view<Layout>` for the layout); takes the same trailing `storage_c<Space>{}` tag |

**Input → output type — `wrap` view facets.** The element type `T` is deduced
from the pointer (a `const T*` gives a read-only view); the extents come from the
passed `shape` (each axis static/dynamic exactly as in its `E`); the ownership is
**always a view**. Only the *layout* and the *space* vary with the call form
(`storage_view_of` folds a named backend to its view kind, so `wrap` never yields
an owner):

| Argument form | Output layout | Output space |
|---|---|---|
| `wrap(ptr, e)` | `ccontiguous` | `storage::view` |
| `wrap<Layout>(ptr, e)` | `Layout` (e.g. `fcontiguous`) | `storage::view` |
| `wrap(ptr, e, {s0,s1,…})` | `dynamic_strides` (all runtime) | `storage::view` |
| `wrap<S...>(ptr, e, {dyn…})` | `strides<S...>` (mixed; `dynamic_stride` slots runtime) | `storage::view` |
| `wrap(ptr, e, strides<S...>{})` | `strides<S...>` (all compile-time, folded) | `storage::view` |
| `wrap(…, storage_v<storage::gpu>)` | any of the above | `storage::gpu_view` |
| `wrap(…, storage_v<storage::pinned>)` / `…<storage::mapped>` | any of the above | `pinned_view` / `mapped_view` |

Allocate new storage — element type **`T` defaults to `float`**; static shape →
stack (host+device), dynamic shape → heap (host only):

| Call | Returns | Element type |
|---|---|---|
| `empty<T>(shape)` | `local`/`owned` (deduced) | `T` (=`float`); UNINITIALISED |
| `empty<T, storage::S>(shape)` | owner in space `S` | name a backend: `stack`/`heap`/`gpu`/`pinned`/`mapped` |
| `empty<T>(shape, storage_c<storage::S>{})` | owner in space `S` | value-tag backend form (same result) |
| `make_local<T>(shape)` | `local<T,E>` | `T` (=`float`); = `empty<T,storage::stack>` |
| `make_heap<T>(shape)` | `owned<T,E>` | `T` (=`float`); = `empty<T,storage::heap>` |
| `make_gpu<T>(shape)` / `make_pinned<T>` / `make_mapped<T>` | CUDA owner | `T` (=`float`); = `empty<T,storage::gpu/…>` |
| `zeros<T>(shape)` / `ones<T>(shape)` | stack or heap | `T` (=`float`) |
| `full(shape, v)` | stack or heap | **the value's type** (`full<T>(…)` to force) |
| `arange<T>(n)` | `owned<T, shape<-1>>` | `T` (=`int64_t`); 1-D `[0,n)` |
| `arange<T,N>()` / `arange<T>(Int<N>())` | `local<T, shape<N>>` | static 1-D `[0,N)` |
| `zeros<T, storage::S>(shape)` (also `ones`/`full`/`arange`) | owner in space `S` | host-accessible backend (`stack`/`heap`/`pinned`/`mapped`); `storage::gpu` `static_assert`s → `to<storage::gpu>(zeros<T>(shape))`. Value-tag: `zeros<T>(shape, storage_c<storage::S>{})` |

---

## Geometry

| Call | Returns | Notes |
|---|---|---|
| `t.rank()` | `size_t` (constexpr) | number of axes |
| `t.numel()` | `integral_constant` if fully static, else `Idx` | number of elements (product of the shape); folds when static |
| `t.shape()` | array-like accessor | the tensor's **shape** (numpy/pytorch spelling): `shape()[Int<k>()]` folds (integral_constant), `shape()[i]` is runtime; has `rank()`, is iterable, converts to the raw `Extents` |
| `t.shape(d)` | `integral_constant`/`Idx` | size of axis `d` — static `Int<k>()` folds, runtime `d` is a value |
| `t.strides()` | array-like accessor | the tensor's **strides**: twin of `shape()` — `strides()[Int<k>()]` folds where derivable, `strides()[i]` is runtime |
| `t.stride(d)` | `integral_constant` if derivable, else `Idx` | stride of axis `d`; static-stride / contiguous layouts fold |
| `t.is_contiguous()` | `bool` | **C-order** (numpy/pytorch default) — what `reshape`/`flatten` need; `is_contiguous(fcontiguous{})` for F |
| `t.is_dense()` | `bool` | dense block in **some** axis order (C, F, or permuted); `is_dense(L{})` is the exact-layout check and `is_contiguous()` == `is_dense(ccontiguous{})` |
| `t.data()` | `T*` | base pointer |
| `t.view()` | `view<T,E,L>` (`gpu_view` if device) | non-owning teeny view aliasing `t`'s memory (no copy) |
| `t.extent(d)` / `t.extent(Int<k>())` | `Idx` / `integral_constant` if static | mdspan-side per-axis size (== `t.shape(d)`); folds when static |
| `t.extents()` / `t.mapping()` | `const Extents&` / `const mapping&` | the **raw mdspan objects** (interop escape hatch) |
| `t.mdspan()` | `cs::mdspan<T,E,L>` | the raw `cuda::std::mdspan` |

!!! note "mdspan equivalent"
    `t.shape()` and `t.strides()` are teeny's **array-like accessors** — indexable
    with a static `Int<k>()` (folds to an `integral_constant`) or a runtime
    integer, iterable, and rank-aware — and are the primary spelling. `t.extents()`
    and `t.mapping()` hand back the **raw `cuda::std` objects** (`cs::extents`, the
    layout mapping) for mdspan interop. Per-axis, `t.extent(d) == t.shape(d)`, and
    `t.stride(d)` is the strides twin. Reach for `extent`/`extents`/`mapping`/
    `mdspan()` only when you need the underlying mdspan types.

---

## Indexing & slicing

Axis args accept a runtime integer, a static `Int<k>()`, or a slice specifier.
Negative integer indices wrap (count from the back).

| Call | Returns | Notes |
|---|---|---|
| `t(i, j, k)` (all integers) | `T&` | element access |
| `t.at(i, j, k)` (all integers) | rank-0 view | scalar-like: converts to/from `T`, `.item()`, has `atomic_add_` |
| `t(0, all, slice(1,4))` (any slice arg) | → view | lower-/same-rank |
| `t(1, ellipsis, 2)` | → view or `T&` | `ellipsis` = `rank − #other args` copies of `all` (≤1 per call) |
| `t.take_along<Axes...>(args...)` | → view | bind named axes only, keep the rest |
| `t.take_along(axis<Axes...>{}, args...)` | → view | value form — `axis<...>` selector (numpy-like), no `.template` on a dependent receiver |

Slice specifiers:

| Spelling | Meaning |
|---|---|
| `all` | keep the whole axis (== `slice(none,none)`; folds, keeps static extent) |
| `slice(a, b)` / `slice(a, b, step)` | half-open `[a,b)`, optional (negative) step |
| `slice<a,b>()` / `slice<a,b,step>()` | compile-time bounds (fold like `all`) |
| `none` | open slice end (python `None`) |
| `ellipsis` | fill the middle with `all` |

Assignment **into** a slice copies (broadcasts); on a **named** view it rebinds:

| Statement | Effect |
|---|---|
| `a = b` (named view) | rebind `a` to `b`'s memory (shallow — nothing copied) |
| `a(ellipsis) = b` / `a(0,all) = b` | copy `b`'s elements into the region (broadcasts) |
| `a(...) = 5.0` | fill the region |

**Input → output type — `operator()` per-axis arg.** Element type is unchanged
(`const` if the source is), the output layout is a folded `strides<...>`, and the
space is `storage_view_of(O)`. Each axis argument independently decides whether
that axis survives and, if so, its output extent/stride staticity. The output
**rank** = source rank − (number of integer args); `ellipsis` expands to
`rank − #other args` copies of `all` before this table applies:

| Axis arg | Axis in output? | Output extent | Output stride |
|---|---|---|---|
| `i` (runtime int) | dropped (folded into base offset) | — | — |
| `Int<k>()` (static int) | dropped | — | — |
| `all` (== `slice(none,none)`) | kept | the source extent (static stays static) | source stride, folded where derivable |
| `slice<a,b>()` / `slice<a,b,step>()` (static bounds) | kept | **static** when the source extent is static (`b−a`, ceil-div by `step`) | folded (`source_static_stride × step`) where derivable |
| `slice(a,b)` / `slice(a,b,step)` (runtime bounds) | kept | **dynamic** (`-1`) | folded where derivable, else runtime |

So `t(i, all, slice<1,4>())` on `<A,B,C>` → `<B,3>`; `t(i, all, slice(a,b))` on
`<A,B,C>` → `<B,-1>`. A negative `step` (`slice(none,none,-1)`) reverses the axis
(a negative folded stride); a `strides<...>` source stays fully sliceable.

---

## Structure (views)

All return a view and work on any source layout (incl. `strides<...>`); axis
template args are signed (negatives count from the back). Each `<Ax>` op also has
a **value form** — pass `Int<Ax>()` in place of the template argument
(`t.squeeze(Int<1>()) == t.squeeze<1>()`, likewise `permute`/`flip`/`unsqueeze`/
`reshape`/`recast`); the value spelling is the preferred one. See
[Views & structure](structure.md).

**Type inference — what the output *type* keeps.** A view op transforms the input
type along four independent facets; staticity is preserved wherever it is
derivable, so a kernel written against a static shape stays static through these
ops (no accidental fallback to runtime extents/strides):

| Facet | Rule |
|---|---|
| **extents** | each *kept* axis keeps its source extent — static stays static. Peeling the batch dims off `shape<-1,-1,M,N>` yields a `<M,N>` cell; a *static-bounds* slice/range keeps a static extent, a runtime range is dynamic; `unsqueeze` inserts a static `1` |
| **strides** | folded to a compile-time `strides<...>` slot wherever `source_stride × step` is known at compile time — a static source keeps folded strides, and a partially-dynamic *contiguous* stride folds from the static extents it spans (`shape<-1,3,3>` → `stride0 = 9`); otherwise runtime |
| **layout** | the view ops (`operator()`, `take_along`, `permute`, `flip`, `un/squeeze`, `peel`) output a folded `strides<...>`. `recast<keep_strides>` (the default) instead **preserves the source layout type** (`ccontiguous` stays `ccontiguous`, `strides<>` stays `strides<>`) |
| **space** | `storage_view_of(source)`: a `gpu`/`gpu_view` source → `gpu_view`, `pinned`/`mapped` → the matching `_view`, else `view` — a view never loses its memory space |

Worked input→output shapes (`E` = source extents):

| Call | `E` | → output |
|---|---|---|
| `t(i, all, slice<1,4>())` | `<A,B,C>` | `<B,3>` — integer drops the axis, `all` keeps `B`, a static range keeps a static extent |
| `t(i, all, slice(a,b))` | `<A,B,C>` | `<B,-1>` — a runtime range's extent is dynamic |
| `peel_front<N>(t)` cell | `<*batch(N),M,N>` | `<M,N>` — trailing extents **and** strides stay static even when the batch is dynamic |
| `t.recast<NewE>()` | `<-1,M,N>` (any layout) | `NewE` with the static dims recovered; **strides + layout preserved** |
| `sum<Ax>(t)` | `<A,B,C>` | `<A,C>` — named axis removed (static result → stack, any dynamic → heap) |

| Call | Returns | Notes |
|---|---|---|
| `t.permute<Perm...>()` | → view | reorder axes (a permutation of `0..N-1`) |
| `t.flip<Ax>()` | → view | reverse an axis (negative-stride) |
| `t.unsqueeze<Ax>()` | → view, rank+1 | insert a size-1 axis |
| `t.squeeze<Ax>()` / `t.squeeze()` | → view, rank−1 / − all size-1 | drop size-1 axis / axes |
| `t.reshape<NewExt...>()` | → view | contiguous reshape (one `-1` inferred) |
| `t.flatten()` | → 1-D view | ravel; needs C-contiguous |
| `t.recast<NewShape[, NewLayout]>()` | → view | reinterpret with a more-static same-rank extents; **`NewLayout` defaults to `keep_strides`** (preserve the source strides AND layout type, any layout, no copy). `ccontiguous`/`fcontiguous` = reinterpret AS that order (derive+fold the strides — the "I promise it's contiguous" form; a **debug build verifies** the imposed strides match the source's and aborts a false promise, symmetric with the extent check — UB only under `-DNDEBUG`); `strides<S...>` = impose them. Functional form `t.recast(shape{…}, layout{…})` |
| `t.reindex<Idx2>()` (free: `reindex<Idx2>(t)`) | → view | no-copy, **layout-preserving** retype of the offset **index width** to `Idx2`: same pointer + layout kind, extents & dynamic strides narrowed to `Idx2` (a `strides<...>` literal pack unchanged). Narrows a boundary view to `shape32` — halves the by-value footprint, 32-bit device offset math. Orthogonal to `recast` (which staticizes the extent *values*); they compose. Debug-checks `index_fits<Idx2>()`; UB if the caller lies |
| `t.index_fits<Idx2>()` (free: `index_fits<Idx2>(t)`) | `bool` | do all element offsets fit `Idx2`? Signed reach `Σ_{s>0}(e−1)s` / `Σ_{s<0}(e−1)s` ⊆ `Idx2` (handles negative-stride/flipped views; broadcast stride 0 adds 0) |
| `t.clone()` | owning (stack/heap) | materialise a dense row-major copy. Copies on the HOST → the **dynamic-shape** overload `static_assert`s the source is host-accessible; for a `gpu`/`gpu_view` tensor use the free `to<Space>(t)` below |
| `t.to<T2>()` | view (no-copy) or owning | **dtype** convert. Matching dtype (no `Force`) → a read-only borrow (`gpu_view` if `t` is on the device, else `view`); differing dtype or `t.to<T2,true>()` → a dense owning copy (static→stack, dyn→heap). The **dynamic-shape** copy runs on the HOST and `static_assert`s host-accessibility — convert a `gpu`/`gpu_view` tensor via the free `to<Space>(t)` |
| `to<Space>(t)` (`cuda.h`) | view (no-copy) or owning | **memory-space** move: `to<Space, ET, Force>(t)` — `Space` ∈ `storage::gpu`/`pinned`/`mapped`/`heap`/`stack`. Same no-copy/`Force` rule; a device source (gpu/`gpu_view`) downloads via `cudaMemcpy`. rvalue source → always copies |

`reshape`/`flatten` need exact C-contiguity (`is_contiguous()`);
`clone()` first if the source isn't. `recast` does **not** — it only re-types the
extents and keeps the source strides, so it works on any layout.

**Input → output type — `recast<NewE[, NewL]>()`.** No copy in any form; the
element type `T` and the space are preserved (the result is `storage_view_of(O)`),
the rank is unchanged, and every static dim of `NewE` recovers (folds into the
type) a possibly-dynamic source dim — e.g. `<-1,-1>` → `.recast<shape<-1,3,3>>()`
gives `<-1,3,3>`. The `NewL` argument chooses the *strides + layout type*:

| `NewL` | Output extents | Output strides | Output layout type |
|---|---|---|---|
| `keep_strides` (default) | `NewE` | the source strides, unchanged | **the source `L`** (`ccontiguous`→`ccontiguous`, `strides<S...>`→`strides<S...>`, …) |
| `ccontiguous` / `fcontiguous` | `NewE` | derived + folded from `NewE` (C / F packing) | `ccontiguous` / `fcontiguous` |
| `strides<S...>` | `NewE` | the imposed `S...` (dynamic slots filled from the source) | `strides<S...>` |

An explicit `NewL` is a *reinterpretation* ("I promise it's laid out this way");
a debug build verifies the imposed/derived strides against the source's actual
strides (symmetric with the static-extent check), UB only under `-DNDEBUG`.

**Input → output type — `to` / `clone`.** Whether the result is a borrow or an
owning copy, and its ownership, per call:

| Call | Copy? | Output type | Notes |
|---|---|---|---|
| `t.to<>()` (same dtype, no `Force`) | no — borrow | `view` of `const T` (`gpu_view` if device), **source `L` kept** | read-only alias, no allocation |
| `t.to<T2>()` (differing dtype) | yes | static → `local<T2,E>`; dynamic → `owned<T2,E>` (host) | dense `ccontiguous` copy |
| `t.to<T2,true>()` (`Force`) | yes | as above (even when `T2==T`) | force-materialise |
| `t.clone()` | yes | static → `local<T,E>`; dynamic → `owned<T,E>` (host) | dense row-major (`ccontiguous`) copy |
| `to<Space>(t)` (`cuda.h`) | no-copy if already in `Space`, else yes | borrow (`view`-kind of `Space`) or owner in `Space` | memory-space move; `Space` ∈ `gpu`/`pinned`/`mapped`/`heap`/`stack`; rvalue source always copies |

The `to<T2>()`/`clone()` **copying** paths run `copy_` on the HOST, so they cannot
dereference DEVICE memory; the dynamic-shape (`_TNY_HOST`) overloads `static_assert`
the source is host-accessible. To copy or convert a `gpu`/`gpu_view` tensor use the
free **`to<Space>(t)`** (`cuda.h`) above, which is device-aware.

### nd-peel (iteration)

| Call | Returns | Notes |
|---|---|---|
| `peel<Axes...>(t)` | a range of views | iterate the named axes; each item is a lower-rank view. The **range-for is incremental** — it advances the base pointer and reuses the loop-invariant cell mapping (O(1) per step), not an O(#peeled-axes) decode per cell |
| `peel<Axes...>(t).subrange(lo,hi)` | a `[lo,hi)` sub-range | seed the incremental cursor once at `lo`, then O(1)/step — split `[0,size())` across **CPU threads / device blocks**, each sweeping its own contiguous chunk |
| `peel_at<Axes...>(t, i)` | → view | the `i`-th sub-view — **random access**, decoded from scratch; this is what a device **grid-stride** loop (`i += nthreads`) needs |
| `peel_front<N>(t)` | a range of views | `N≥0`: peel the first `N` axes; `N<0`: keep the last `|N|`. Same incremental range-for + `subrange(lo,hi)` |
| `peel_front_at<N>(t, i)` | → view | the `i`-th (random access / grid-stride) |
| `size_front<N>(t)` | → index | # cells `peel_front<N>` yields (product of the peeled extents), no range built |

**Input → output type — peel cell.** Each yielded cell is a **view**
(`storage_view_of(O)` — `gpu`/`gpu_view` source → `gpu_view`), element type `T`
unchanged, layout a folded `strides<...>`. The *kept* axes carry their source
extents (static stays static) and folded strides; a `peel_front<N>` cell over
`<*batch(N), M, N>` is `<M, N>` with the trailing extents **and** strides static
even when the batch is dynamic. `peel<Axes...>` drops the named axes; the cell
rank is source rank − #peeled axes.

---

## Math

Element type of results follows `promote(A,B)` (C++ rules, but among floats the
**lower** width wins — pytorch-style; `-DTNY_STD_PROMOTION` opts out). The result's
offset **index type** is the *wider* of the two operands' (a mixed `int32`/`int64`
broadcast → `int64`), independent of the element promotion.

### In-place (mutates `*this`, returns `tensor&`)

A tensor rhs broadcasts (numpy-style; needs equal rank — `unsqueeze` first); a
scalar rhs applies to every element.

| Call | Notes |
|---|---|
| `a.add_(x)` `a.sub_(x)` `a.mul_(x)` `a.div_(x)` | `x` = tensor (broadcasts) or scalar |
| `a.atomic_add_(x)` `a.atomic_sub_(x)` | atomic accumulate on device (`x` = tensor or scalar); underlying form is `add_<Atomic>`/`sub_<Atomic>` |
| `a += x` `a -= x` `a *= x` `a /= x` | compound-assign sugar |
| `++a` `--a` | prefix, in place |
| `a++` | postfix → pre-value **stack copy** (static shape only) |
| `a.neg_/abs_/exp_/log_/sin_/cos_/sqrt_/tanh_()` | unary, in place |
| `a.floor_/ceil_/round_/trunc_/sign_()` `a.pow_(e)` `a.clamp_(lo,hi)` | unary, in place |
| `a & b` `a \| b` `a ^ b` `~a` `a &= b` … | bitwise (**integer** element types only) |
| `a.fill_(v)` `a.zero_()` `a.copy_(b)` `a.iota_(start,step)` | assignment / init (`b` broadcasts) |
| `a.map_(f)` `a.zip_with_(g, b)` | user functor (a device-safe struct, not a lambda) |
| `a.at(i...).atomic_add_(v)` | scatter-accumulate `a(i...) += v` (atomic on device) |

### Out-of-place (→ new tensor; static shape → stack, else heap)

| Call | Returns |
|---|---|
| `a + b` / `a.add(b)`, `a - b`, `a * b`, `a / b` | new tensor (tensor+tensor broadcasts, or +scalar) |
| `2.0 * a`, `2.0 - a`, `1.0 / a`, `-a` | scalar–tensor / unary minus |
| `a.pow(b)` | element-wise power |
| `neg/abs/exp/log/sin/cos/sqrt/tanh/floor/ceil/round/trunc/sign(a)` | unary free functions |
| `minimum(a,b)` `maximum(a,s)` `clamp(a,lo,hi)` | elementwise min/max/clamp |
| `a.map(f)` | new tensor from a user functor |

### Comparisons → a `bool` tensor (broadcast), reduced with `.all()`/`.any()`

| Call | Returns |
|---|---|
| `a < b`, `a == 2.0`, `3.0 < a`, … (`== != < <= > >=`) | `bool` tensor |
| `(a > 0).all()` / `(a > 3).any()` | `bool` (members) |

### Reductions

**Accumulate** in the *reduce type* — `double` for small floats
(`float`/`double`/`half`; a wider float keeps its own type), the item type for
integers — then **cast the result back to the tensor's element type** `T`
(`sum(float_tensor)` → `float`, computed in `double`). A leading **type** argument
makes that type both the accumulator **and** the result.

| Call | Returns | Notes |
|---|---|---|
| `sum(a)` `prod(a)` `max(a)` `min(a)` `mean(a)` | `T` (accumulated wide) | over all axes |
| `dot(a, b)` | `promote(Ta,Tb)` (accumulated wide) | inner product; extents must match **exactly** (no broadcast) |
| `sum<Acc>(a)`, `mean<Acc>(a)`, `dot<Acc>(a,b)` | `Acc` | force the accumulator/return type |
| `allclose(a, b, rtol=1e-5, atol=1e-8)` | `bool` | `\|a−b\| ≤ atol+rtol·\|b\|` everywhere (broadcasts) |
| `sum<Axes...>(a)` `mean<Axes...>` `max`/`min`/`prod<Axes...>` | lower-rank tensor | remove the named axes (negatives wrap) |
| `sum<Acc, Axes...>(a)` | lower-rank tensor | leading **type** = accumulator, leading **int** = axis |
| `sum(a, axis<Axes...>{})` (also `mean`/`max`/`min`/`prod`) | lower-rank tensor | value form — numpy `axis=` selector; `sum<Acc>(a, axis<...>{})` for the accumulator; no `.template` on a dependent receiver |

Axis reductions: a fully static result → stack (host+device); any dynamic result
→ heap (host only).

**Input → output type — result element type.** The *accumulator* and the *result
dtype* are separate. Default accumulator = `reduce_type_t<T>` (`double` for a
small float `float`/`double`/`half`/`bfloat16`; a wider float keeps itself;
`int64_t`/`uint64_t` for a narrow integer, so `sum`/`prod`/`dot` can't overflow
mid-accumulation; an already-≥8-byte integer keeps itself). A leading **type**
arg makes that type *both* the accumulator and the result.

| Call | Default result dtype | With leading `Acc` |
|---|---|---|
| `sum(a)` `prod(a)` `max(a)` `min(a)` | `T` (item type — accumulate wide, cast back; `sum(int8)` → `int8`) | `Acc` (e.g. `sum<int64_t>(a)` keeps the untruncated wide value) |
| `mean(a)` | `T` for a **floating** `T`; **`double`** for an **integer** `T` (numpy: integer mean is float64, divided in `double`) | `Acc` |
| `dot(a, b)` | `promote(Ta,Tb)` | `Acc` |
| `sum<Axes...>(a)` `prod`/`max`/`min<Axes...>` | `T` (lower-rank tensor) | `Acc` |
| `mean<Axes...>(a)` | `T` for floating `T`; **`double`** for integer `T` (lower-rank tensor) | `Acc` |

The axis forms follow the same element-type rule; only the shape changes (named
axes removed) and the ownership splits static → stack / dynamic → heap.

---

## Half precision

| Name | Meaning |
|---|---|
| `half` / `bfloat16` | IEEE binary16 / bfloat16 (native `__half`/`__nv_bfloat16` under nvcc) |
| `compute_type<T>` / `compute_type_t<T>` | `float` for half types, else `T` (math computes here) |

---

## Dispatch (the ndarray boundary)

| Call | Returns | Notes |
|---|---|---|
| `as_anyrank(data, shape, stride, ndim)` | `anyrank` (view store) | **wraps** the arrays, no copy (default; host only) |
| `as_anyrank(data, shape, stride, ndim, copy_meta)` | `anyrank` (inline store) | **copies** into a `TNY_MAX_RANK` store (device-passable); `as_anyrank<N>(…,copy_meta)` sets capacity |
| `as_anyrank<Space>(…)` | space-tagged `anyrank` | `Space` = the data's memory space (default `storage::view` = host); pass `storage::gpu_view` for a device pointer so `fixed`/`peel_front` yield `gpu_view` views |
| `as_anyrank(…, anyshape<etc,-1,-1,3>{})` | `anyrank` with a static **tail** | bake the static TRAILING extents into the type (`anyshape<etc,...>`: `etc` = the erased batch, the dims after it = the static tail) — `fixed`/`peel_front` cells fold those inner dims (no per-call `recast`). Checked vs the runtime shape once at the boundary, then trusted. Also on the `copy_meta` form (`…, copy_meta, anyshape<etc,…>{}`). Bare `anyshape<etc>` ⇒ today's fully-dynamic carrier |
| `from_dlpack<T[, Space]>(m)` / `from_dlpack<T, R[, Space]>(m)` | `anyrank` / rank-`R` view | import; `m` may be a `DLManagedTensor*`, a **bare `DLTensor*`** (unmanaged — nothing to free, caller owns all lifetime), or a **`DLManagedTensorVersioned*`** (DLPack 1.0+). `Space` (default host) is **checked against `m→device`** — a `kDLCUDA` tensor needs `storage::gpu_view` |
| `from_dlpack<T, anyshape<etc,-1,-1,3>[, Space]>(m)` | `anyrank` with a static **tail** | import with the static TRAILING extents baked in — the payload's trailing dims are checked against the tag at the import boundary, then every peeled cell folds them. Accepts all three carriers |
| `to_dltensor(t, shape_out, strides_out[, dev])` | `DLTensor` | export to a **bare `DLTensor`** (no capsule/deleter/alloc): borrows `t`'s data AND the two caller-supplied `int64_t[≥rank]` buffers it fills. Caller keeps the tensor + buffers alive. (`to_dlpack(t)` is the managed-capsule export.) |
| `at.fixed<R>()` | rank-`R` `dynamic_strides` view | requires `ndim == R` |
| `dispatch_rank(at, f)` | `bool` | call `f` with a fixed-rank view chosen by runtime `ndim` (one instantiation per total rank) |
| `dispatch_rank<narrow_index>(at, f)` | `bool` | ...and narrow each fixed cell's offset width to `int32` when `index_fits` (rank outer, width inner; the leaf doubles). Default `Narrow=false` is the plain dispatch |
| `dispatch_index<Idx2=int32_t>(v, f)` | `void` | narrow a fixed-rank view `v` to `Idx2` offsets when its span fits, else keep it — then call `f`; instantiates `f` for both widths |
| `dispatch_layout(v, f)` | `void` | the LAYOUT sibling: runtime-classify a `dynamic_strides` view as `ccontiguous`/`fcontiguous`/(else)`dynamic_strides` (`is_dense<…>()`, a stride compare) and call `f` with the **retyped** view — so `recast<shape<-1,c,c>>()` folds the inner strides **safely**. The C-order arm is the payoff (inner strides fold); `f` runs on up to 3 view types. **Opt-in** (triples instantiations) |
| `at.peel_front<N>()` / `at.peel_front_at<N>(i)` | range / view | batch idiom: keep the last `\|N\|` dims static, peel the rest (one kernel per `\|N\|`) |
| `at.size_front<N>()` | → offset | flattened batch count `peel_front<N>` yields (product of the peeled leading extents), no range built; `N < 0` |
| `dispatch_value<Vs...>(v, f)` | `bool` | call `f(Int<k>{})` for the matching candidate `k == v` |

**Input → output type — fixed-rank views.** The rank-erased carrier hands out a
`dyn_tensor<T, offset_t, R, Space>` = `tensor<T, cs::dextents<offset_t, R>,
cs::layout_stride, Space>` — a **fully dynamic**, arbitrarily-strided view whose
`Space` is the carrier's view space (`storage::view` by default; `storage::gpu_view`
for a device carrier):

| Call | Output type | Notes |
|---|---|---|
| `at.fixed<R>()` | `dyn_tensor<T, offset_t, R, Space>` | requires `ndim == R`; extents + strides all runtime |
| `at.peel_front_at<-Sr>(i)` | `dyn_tensor<T, offset_t, Sr, Space>` | one batch cell (rank `Sr`); `.recast<shape<-1,c,c>>()` recovers static inner dims |
| `at.peel_front_at<NewE[, NewL]>(i)` / `at.peel_front_at(i, shape<-1,c,c>{}[, ccontiguous{}])` | cell with shape `NewE` | **fused** peel+recast: the cell has the target trailing shape (rank = `NewE::rank()`) DIRECTLY — static inner extents fold, `-1` stays dynamic, no separate `recast`. Removes the hand-kept `Sr ≡ shape-rank` invariant. **Strides**: `keep_strides` (default) keeps the carrier's runtime strides; a layout arg folds them (debug-checked promise), or `dispatch_layout` proves it |
| `from_dlpack<T, R[, Space]>(m)` | `dyn_tensor<T, offset_t, R, Space>` | rank-`R` view; `Space` checked against `m→device` |
| `from_dlpack<T[, Space]>(m)` | `anyrank` (space-tagged) | rank-erased; call `fixed`/`peel_front`/`dispatch_rank` on it |

Recover static extents with `.recast<NewE>()` at the kernel boundary (see the
`recast` type mapping) to fold the inner shape back into the type.

DLPack strides are in **elements**; numpy `__array_interface__` in **bytes**
(divide by the itemsize first).

---

## Compile flags

| Flag | Effect |
|---|---|
| `-DTNY_STD_PROMOTION` | standard C++ float promotion (wider wins) instead of lower-wins |
| `-DTNY_NO_NEGATIVE_INDEX` | drop python-style negative-index wrap from `operator()` (tightest codegen) |
| `-DTNY_PORTABLE_HALF` | force the portable software `half`/`bfloat16` even under nvcc |
| `-DTNY_MAX_RANK=N` | inline `anyrank` capacity / dispatch bound (default 32) |
| `-DNDEBUG` | strip debug shape/precondition checks (host-only; already off on device) |
