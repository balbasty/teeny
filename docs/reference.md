# API reference

A thorough, type-annotated reference for the public API. For a one-screen scan
of the whole surface, see the [Cheat sheet](cheatsheet.md); for the raw
Doxygen-extracted signatures, see [Autodoc](api/index.md).

Everything is in `namespace tny` (`namespace cs = cuda::std`). Include
`<teeny/teeny.h>` for all of it; `<teeny/cuda.h>` adds the CUDA memory spaces.

Legend: **`Idx`** = the shape's `index_type` (`int64_t` for `shape<...>`).
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
| `wrap<Layout>(ptr, shape)` / `wrap(ptr, shape, Layout{})` | `view<T,E,Layout>` | chosen layout — template arg, or the value-tag spelling (`ccontiguous{}`/`fcontiguous{}`, deduced, no `.template`) |
| `wrap(ptr, shape, {s0,s1,…})` | `view<T,E,dynamic_strides>` | **runtime** strides (elements; may be negative). A **stride-0** axis self-overlaps — fine to read (broadcast), but an in-place write is a host-debug error (`clone()` first) |
| `wrap<S...>(ptr, shape, {dyn…})` | `view<T,E,strides<S...>>` | **mixed** static/runtime strides (`dynamic_stride` slots) |
| `wrap(ptr, shape, strides<S...>{})` | `view<T,E,strides<S...>>` | **compile-time** strides (fold into the type) |
| `wrap(…, storage_v<storage::gpu>)` | view in that space | trailing **memory-space** tag on **every** overload here, the `mdspan` one included (default `storage::view`); pass the plain backend — `storage::gpu`/`pinned`/`mapped` — and it folds to the view kind (`gpu_view`/…), since `wrap` always views. `storage_c<S>{}` / `storage_v<S>` are the braced / no-braces spellings |
| `as_tensor(md)` / `wrap(md)` | `view<…>` | wrap any `cs::mdspan`/`submdspan` result — both spellings, same result, both public on purpose: `wrap` is the one factory name for viewing existing memory whatever you hold (so the mdspan case doesn't force a different name), `as_tensor` the mdspan-adapting primitive under it (what teeny's own view-producing ops call internally, and the [mdspan interop](mdspan-vs-teeny.md) spelling). Element type, extents and layout all come from the mdspan; only the space is yours to name — `wrap(md, storage_v<storage::gpu>)` (or `wrap<storage::gpu>(md)`, whose explicit template argument is the **space**, since the mdspan already carries the layout) |
| `make_view(ptr, shape)` | `view<T,E>` | an alias of `wrap` that deduces `E`; takes the layout the same two ways (`make_view(ptr, shape, fcontiguous{})` or `make_view<Layout>(ptr, shape)`) and the same trailing `storage_c<Space>{}` tag |

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
| `wrap(md)` | the mdspan's own layout | `storage::view` |
| `wrap(…, storage_v<storage::gpu>)` | any of the above | `storage::gpu_view` |
| `wrap(…, storage_v<storage::pinned>)` / `…<storage::mapped>` | any of the above | `pinned_view` / `mapped_view` |
| `wrap(…, storage_v<storage::heap>)` / `…<storage::stack>` | any of the above | `storage::view` — `heap`/`stack` name no memory *space*, so they fold to a plain view like any other host-accessible tag, not to an owning tensor; use `empty<T, storage::heap>(e)`/`make_heap<T>(e)` for that |

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
| `make_local`/`make_heap`/`make_gpu`/`make_pinned`/`make_mapped(shape, dtype<T>{}/layout tag)` | same, T/layout as given | `make_*` also forwards a trailing `dtype`/layout tag bag to `empty` (#282) — `make_heap(shape, dtype<double>{}, fcontiguous{})` |
| `zeros<T>(shape)` / `ones<T>(shape)` | stack or heap | `T` (=`float`) |
| `full(shape, v)` | stack or heap | **the value's type** (`full<T>(…)` to force) |
| `arange<T>(n)` | `owned<T, shape<-1>>` | `T` (=`int64_t`); 1-D `[0,n)` |
| `arange<T,N>()` / `arange<T>(Int<N>())` | `local<T, shape<N>>` | static 1-D `[0,N)` |
| `zeros<T, storage::S>(shape)` (also `ones`/`full`/`arange`) | owner in space `S` | host-accessible backend (`stack`/`heap`/`pinned`/`mapped`); `storage::gpu` `static_assert`s → `to<storage::gpu>(zeros<T>(shape))`. Value-tag: `zeros<T>(shape, storage_c<storage::S>{})` |
| `empty(shape, dtype<T>{})` (also `zeros`/`ones`/`full`/`arange`) | same as `<T>` | value-tag element-type form — deduces `T` from the tag instead of an explicit `<T>` template argument, so a type-dependent receiver needs no `.template`. A backend override stays a *leading* explicit template arg since `T` is now deduced: `empty<storage::gpu>(shape, dtype<T>{})` |
| `empty(shape, dtype<T>{}, storage_c<S>{})` / `empty(shape, storage_c<S>{}, dtype<T>{})` (also `zeros`/`ones`/`full`/`arange`) | same as `<T,S>` | **both** value tags at once, either order — no explicit template argument needed at all |
| `empty(shape, ccontiguous{}/fcontiguous{})`, composed with `dtype<T>{}`/`storage_c<S>{}` in ANY order/subset (also `zeros`/`ones`/`full`/`arange`, except `arange` has no layout) | same, layout as given | the generic keyword mechanism (#277-#281): a bare layout tag, alone or alongside `dtype`/`storage_c` in any order — `empty(shape, fcontiguous{}, dtype<double>{})`, `zeros(shape, storage_c<storage::heap>{}, fcontiguous{}, dtype<double>{})`, `arange(n, storage_c<storage::pinned>{}, dtype<double>{})`. `wrap` (#282) is on the same mechanism for its `storage_c` tag, but has no `dtype` keyword (`T` is deduced from the pointer) and keeps `Layout` as a distinct positional slot (a 3rd-argument value tag or template arg), not a composable keyword |
| `empty<storage::S>(shape, …keywords…)` (also `zeros`/`ones`/`full`/`arange`) | owner in space `S` | the **leading backend** spelling takes that *same* keyword bag — any subset, any order (#373): `zeros<storage::pinned>(shape)`, `zeros<storage::pinned>(shape, fcontiguous{})`, `empty<storage::gpu>(shape, dtype<double>{}, fcontiguous{})`. The one keyword it rejects is `storage_c<…>{}` — that *is* the backend, so naming it twice is a "pick one" `static_assert` |

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
| `t(m)` / `t.at(m)` / `t.uget(m)` / `t.uat(m)` | same as the unpacked call | **tuple-unpack**: ONE tuple-like argument (`cuda::std::array` / `cuda::std::tuple`) carrying the whole index list — numpy's `x[(a,b,c)] == x[a,b,c]`. Elements may be anything the variadic call accepts (int, `Int<k>()`, `all`, `slice(...)`, `none`, one `ellipsis`); result type is identical. Single-argument only (never mixed with other positional args); C++23 `t[m]` forwards too. Pairs with `peel(...).enumerate()`, whose multi-index *is* a `cuda::std::array`: `for (auto [m, cell] : peel(a, axis<0,1,2>{}).enumerate()) b(m) = f(cell);` |
| `t.slice_along<Axes...>(args...)` | → view | bind named axes only, keep the rest — pytorch `select`/`narrow` over several axes at once. NOT numpy's `take_along_axis`/pytorch's `take_along_dim` (an index-tensor gather; that's `index_select`) |
| `t.slice_along(axis<Axes...>{}, args...)` | → view | value form — `axis<...>` selector (numpy-like), no `.template` on a dependent receiver |
| `t.subsample<Axes...>(k, starts...)` | → view | coloured/strided sub-lattice — sugar for `slice_along` + `slice(start,none,k)` per named axis, one shared step `k`; `k`/`starts` accept runtime or `Int<>` (folds static) |
| `t.subsample(axis<Axes...>{}, k, starts...)` | → view | value form, same placement as `slice_along`'s |
| `t.unfold<Axis>(size, step=1)` | → view | pytorch `Tensor.unfold` — appends a new trailing axis of width `size`, stepped by `step` along `Axis`; `Axis`'s extent shrinks to `(shape(Axis)-size)/step+1`; `size`/`step` accept runtime or `Int<>` (folds static); ND windows compose by chaining one `unfold` per axis; `size`∈`[1,shape(Axis)]`/`step>=1` checked (`static_assert` or debug-time) |
| `t.unfold(Int<Axis>(), size, step=1)` | → view | value form — single-axis `Int<k>()` selector (like `flip`/`squeeze`), no `.template` on a dependent receiver |
| `t.index_select<Axis>(idx)` | → new tensor (static idx shape → stack, else heap) | gather along `Axis` by a rank-1 integer index TENSOR (runtime data, unlike `slice_along`); `idx` values wrap negative; static result → stack (host+device, works on a `gpu`/`gpu_view` source, like `clone()`), dynamic result → heap (host only: source must be host-accessible) |
| `t.index_select<Axis>(idx, into(dest))` | `dest&` | no-allocation, device-safe form; `dest`'s axis-`Axis` extent must equal `idx`'s (checked) and must not alias `t` |
| `t.index_select(idx, axis<Axis>{})` `t.index_select(idx, axis<Axis>{}, into(dest))` | same as above | value form — no `.template` on a dependent receiver |

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
`reshape`/`recast`); the value spelling is the preferred one. The axis-**list**
ops — `permute`, `flip`, `squeeze`, `unsqueeze` — additionally take an `axis<...>{}` tag
(`t.squeeze(axis<0,2>{}) == t.squeeze<0,2>()`, `t.flip(axis<0,2>{}) == t.flip<0,2>()`),
the same one `peel`/`slice_along`/
the reductions use. An **empty** list, `axis<>{}`, names no axis and is therefore a
**no-op** for `squeeze`/`unsqueeze`/`flip` (numpy's `axis=()` rule) — distinct from the
no-argument `squeeze()`/`unsqueeze()`/`flip()`, which keep their own meanings; `permute`
accepts it only at rank 0. The **reductions** read it the same way — `sum(a,
axis<>{})` reduces over no axis and keeps `a`'s shape, where the plain `sum(a)`
(no axis argument at all) reduces over every axis, see
[Math & reductions](math.md#an-empty-axis-list-reduces-over-no-axis).
See [Views & structure](structure.md).

**Type inference — what the output *type* keeps.** A view op transforms the input
type along four independent facets; staticity is preserved wherever it is
derivable, so a kernel written against a static shape stays static through these
ops (no accidental fallback to runtime extents/strides):

| Facet | Rule |
|---|---|
| **extents** | each *kept* axis keeps its source extent — static stays static. Peeling the batch dims off `shape<-1,-1,M,N>` yields a `<M,N>` cell; a *static-bounds* slice/range keeps a static extent, a runtime range is dynamic; `unsqueeze` inserts a static `1` |
| **strides** | folded to a compile-time `strides<...>` slot wherever `source_stride × step` is known at compile time — a static source keeps folded strides, and a partially-dynamic *contiguous* stride folds from the static extents it spans (`shape<-1,3,3>` → `stride0 = 9`); otherwise runtime |
| **layout** | the view ops (`operator()`, `slice_along`, `permute`, `flip`, `un/squeeze`, `peel`) output a folded `strides<...>`. `recast<keep_strides>` (the default) instead **preserves the source layout type** (`ccontiguous` stays `ccontiguous`, `strides<>` stays `strides<>`) |
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
| `t.flip<Ax0,Ax1,...>()` | → view | reverse **several** axes at once (numpy `flip(a, axis=(0,2))`); distinct, any order — flips commute, so `flip<0,2>` == `flip<2,0>` == `flip<0>().flip<2>()`, built in one pass (#349) |
| `t.unsqueeze<Ax>()` | → view, rank+1 | insert a size-1 axis |
| `t.unsqueeze<Ax0,Ax1,...>()` | → view, rank+k | insert **several** size-1 axes at once; positions are relative to the *final* rank, distinct, any order (#275) |
| `t.squeeze<Ax>()` / `t.squeeze()` | → view, rank−1 / − all size-1 | drop size-1 axis / axes |
| `t.squeeze<Ax0,Ax1,...>()` | → view, rank−k | drop **several** size-1 axes at once; positions are relative to the *source* rank, distinct, any order (#275) |
| `t.reshape<NewExt...>()` | → view | contiguous reshape (one `-1` inferred) |
| `t.flatten()` | → 1-D view | ravel; needs C-contiguous |
| `t.recast<NewShape[, NewLayout]>()` | → view | reinterpret with a more-static same-rank extents; **`NewLayout` defaults to `keep_strides`** (preserve the source strides AND layout type, any layout, no copy). `ccontiguous`/`fcontiguous` = reinterpret AS that order (derive+fold the strides — the "I promise it's contiguous" form; a **debug build verifies** the imposed strides match the source's and aborts a false promise, symmetric with the extent check — UB only under `-DNDEBUG`); `strides<S...>` = impose them. Functional form `t.recast(shape{…}, layout{…})` |
| `t.reindex<Idx2>()` (free: `reindex<Idx2>(t)`) | → view | no-copy, **layout-preserving** retype of the offset **index width** to `Idx2`: same pointer + layout kind, extents & dynamic strides narrowed to `Idx2` (a `strides<...>` literal pack unchanged). Narrows a boundary view to `shape32` — halves the by-value footprint, 32-bit device offset math. Orthogonal to `recast` (which staticizes the extent *values*); they compose. Debug-checks `index_fits<Idx2>()`; UB if the caller lies |
| `t.index_fits<Idx2>()` (free: `index_fits<Idx2>(t)`) | `bool` | do all element offsets fit `Idx2`? Signed reach `Σ_{s>0}(e−1)s` / `Σ_{s<0}(e−1)s` ⊆ `Idx2` (handles negative-stride/flipped views; broadcast stride 0 adds 0) |
| `t.clone()` | owning (stack/heap) | materialise a dense row-major copy. Copies on the HOST → the **dynamic-shape** overload `static_assert`s the source is host-accessible; for a `gpu`/`gpu_view` tensor use the free `to<Space>(t)` below |
| `t.to<T2>()` | view (no-copy) or owning | **dtype** convert. Matching dtype (no `Force`) → a read-only borrow (`gpu_view` if `t` is on the device, else `view`); differing dtype or `t.to<T2,true>()` → a dense owning copy (static→stack, dyn→heap). The **dynamic-shape** copy runs on the HOST and `static_assert`s host-accessibility — convert a `gpu`/`gpu_view` tensor via the free `to<Space>(t)` |
| `t.to(dtype<T2>{})` | same as `t.to<T2>()` | value-tag form — deduces `T2` from the tag instead of an explicit `<T2>` template argument |
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
| `peel<Axes...>(t).enumerate()` | a range of `{index, cell}` | range-for that ALSO yields the peeled **multi-index**: `for (auto [m, cell] : …)` with `m[d]` = the coordinate of peeled axis `d` (row-major, last listed fastest) — for a per-axis table `axtab[d][m[d]]` or a write-by-coord. **Opt-in** (the bare `peel` cell stays lean — no coordinate words); composes with `subrange` (`.enumerate().subrange(lo,hi)`). The raw iterator also exposes `it.index(d)` / `it.index()` |
| `peel_at<Axes...>(t, i)` | → view | the `i`-th sub-view — **random access**, decoded from scratch; this is what a device **grid-stride** loop (`i += nthreads`) needs |
| `peel_front<N>(t)` | a range of views | `N≥0`: peel the first `N` axes; `N<0`: keep the last `abs(N)`. Same incremental range-for + `subrange(lo,hi)` |
| `peel_front_at<N>(t, i)` | → view | the `i`-th (random access / grid-stride) |
| `size_front<N>(t)` | → index | # cells `peel_front<N>` yields (product of the peeled extents), no range built |
| `peel_zip<Axes...>(a, b[, c])` | a range of `cs::tuple<ViewA,ViewB[,ViewC]>` | zip-peel 2 or 3 tensors' named axes in **lock-step** — a distinct name from `peel` (1 tensor → a view per step, 2+ → a tuple is a silent arity-driven return-type bifurcation, so it gets its own name, mirroring python's `zip()`). Operands may differ in shape if **broadcast-compatible** (numpy right-align, the rule `a+b` uses); `Axes...` are in the **broadcast rank's** numbering. Operands may also differ in **index type** — the cells carry one wide (and, on mixed signedness, signed) enough to address every operand exactly, so a reversed operand zipped against an unsigned-indexed one still steps backwards. Decodes fresh each step (no incremental cursor yet) |
| `peel_zip(a, b[, c], axis<Axes...>{})` | same | value form — axis tag **trailing** (after every positional tensor), unlike `slice_along`/`peel_at`'s leading tag |
| `peel_zip<Axes...>(a, b[, c]).enumerate()` | a range of `{index, tuple}` | same shape as `peel(...).enumerate()` |
| `peel_zip<Axes...>(a, b[, c]).subrange(lo,hi)` | a `[lo,hi)` sub-range | same shape as `peel(...).subrange()` |
| `scan_<Axis>(t, init, f)` | `void` (in-place) | sequential fold along `Axis` — `carry=init`, then `carry=f(carry,x); x=carry` for each element (increasing order), batched (peeled) over every other axis; `f` is a device-safe functor (like `map_`'s own convention). Reverse sweep: `scan_<Axis>(t.flip<Axis>(), init, f)` (a temporary view binds fine — `scan_` has lvalue and rvalue overloads) |
| `scan_(t, init, f, axis<Axis>{})` | same | value form — the axis tag is **trailing**, like `index_select`'s and the reduction family's (#348). The leading spelling `scan_(t, axis<Axis>{}, init, f)` was **removed**, with no deprecated alias |
| `scan<Axis>(t, init, f)` | → new tensor | out-of-place: fresh dense copy, scanned (static→stack, dynamic→heap host-only, built on `clone()`); `t` itself untouched |
| `scan<Axis>(t, init, f, into(dest))` | `dest&` | no fresh allocation beyond the copy into `dest`; `dest`'s shape must match `t`'s EXACTLY (checked — `static_assert` when both are static, `_TNY_CHECK` otherwise), unlike `copy_`'s own broadcast rule, since `scan_` then walks `dest`'s own axis numbering |
| `scan(t, init, f, axis<Axis>{}[, into(dest)])` | → new tensor / `dest&` | value form — `scan`'s two trailing keywords ride the generic keyword bag, so `axis<...>{}` and `into(dest)` compose in **any subset, any order** (`scan(t, init, f, into(dest), axis<0>{})` is the same call) |

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
offset **index type** is one that covers both operands' — the *wider* of the two for
the usual (all-signed) pair, so a mixed `int32`/`int64` broadcast → `int64`; a pair
that disagrees in signedness steps up to a *signed* type wide enough for both ranges.
Either way it is independent of the element promotion.

### In-place (mutates `*this`, returns `tensor&`)

A tensor rhs broadcasts (numpy-style; needs equal rank — `unsqueeze` first); a
scalar rhs applies to every element. An in-place op keeps `*this`'s own offset
index type (there is no new result to widen); the offset math runs internally in a
type that covers every participant — both in width and in signedness — so a
narrow-indexed destination is safe against a wide-indexed rhs, and an
unsigned-indexed tensor is safe against a flipped (negative-stride) signed-indexed
one — see [Mixing widths](shapes-strides.md#mixing-widths-in-a-broadcast).

| Call | Notes |
|---|---|
| `a.add_(x)` `a.sub_(x)` `a.mul_(x)` `a.div_(x)` | `x` = tensor (broadcasts) or scalar |
| `a.minimum_(x)` `a.maximum_(x)` | running min/max update (#325): `*this = min/max(*this, x)`, `x` = tensor (broadcasts) or scalar — the running-nearest-distance idiom `best.minimum_(candidate)` |
| `y.add_(x, alpha)` `y.sub_(x, alpha)` | fused scaled accumulate (BLAS axpy): `y ± alpha*x`, `x` broadcasts; scaled copy `y = alpha*x` is `y.zero_().add_(x, alpha)` |
| `a.atomic_add_(x)` `a.atomic_sub_(x)` | atomic accumulate, host and device (`x` = tensor or scalar); underlying form is `add_<Atomic>`/`sub_<Atomic>` |
| `a += x` `a -= x` `a *= x` `a /= x` | compound-assign sugar |
| `++a` `--a` | prefix, in place |
| `a++` | postfix → pre-value **stack copy** (static shape only) |
| `a.neg_/abs_/exp_/log_/sin_/cos_/sqrt_/tanh_()` | unary, in place |
| `a.floor_/ceil_/round_/trunc_/sign_()` `a.pow_(e)` `a.clamp_(lo,hi)` | unary, in place |
| `a & b` `a \| b` `a ^ b` `~a` `a &= b` … | bitwise (**integer** element types only) |
| `a.fill_(v)` `a.zero_()` `a.copy_(b)` `a.iota_(start,step)` | assignment / init (`b` broadcasts) |
| `a.map_(f)` `a.zip_with_(g, b)` | user functor (a device-safe struct, not a lambda) |
| `a.at(i...).atomic_add_(v)` | scatter-accumulate `a(i...) += v` (atomic, host and device) |

### Out-of-place (→ new tensor; static shape → stack, else heap)

| Call | Returns |
|---|---|
| `a + b` / `a.add(b)`, `a - b`, `a * b`, `a / b` | new tensor (tensor+tensor broadcasts, or +scalar) |
| `2.0 * a`, `2.0 - a`, `1.0 / a`, `-a` | scalar–tensor / unary minus |
| `a.pow(b)` | element-wise power |
| `neg/abs/exp/log/sin/cos/sqrt/tanh/floor/ceil/round/trunc/sign(a)` | unary free functions — **also methods** (`a.exp()`, …) |
| `minimum(a,b)` `maximum(a,s)` `clamp(a,lo,hi)` | elementwise min/max/clamp — **also methods** (`a.minimum(b)`, …). `clamp` follows pytorch's name over numpy's `clip` — see `CLAUDE.md`'s "Naming policy" |
| `normalize(a)` `cross(a,b)` | unit vector / 3D cross — **also methods** (`a.normalize()`, `a.cross(b)`) |
| `normalize<Axes...>(a)` / `normalize(a, axis<...>{})` | unit vectors over the named axes — **also methods** (`a.normalize<1>()`, `a.normalize(axis<1>{})`) |
| `a.add(b, alpha)` `a.sub(b, alpha)` | fused out-of-place axpy: `a ± alpha*b` (b broadcasts); the in-place twin is `add_(b, alpha)` |
| `a.map(f)` | new tensor from a user functor |

**`into(dest)` — write into a preallocated buffer.** Pass `into(y)` as the last
argument to any producer above to write the result into `y` (one fused pass, no
allocation) and return `y&`, instead of a fresh tensor — the kernel-friendly `out=`.
`into(y)` is a distinct type, so it never collides with a scalar argument (which is
what lets `add`/`sub` also take the fused `alpha`). `y` may alias an operand and may
have a different dtype — the arithmetic still runs in the **operands'** precision
(scalar right-hand side and axpy coefficient included) and only the **result** is
cast to `y`'s type, so `a.op(b, into(y))` produces exactly the numbers
`y.copy_(a.op(b))` would — including when the source is `half`/`bfloat16`: `into(y)`
rounds through the twin's own `promote_t` (a 16-bit float there) before casting to
`y`, the same extra rounding step the twin's temporary result goes through.

**`y` may be a slice**, written straight out of a view-producing op —
`cross(a, b, into(N(i, all)))`, `sum(a, into(cells.at(i, j)))`,
`x.add(y, into(z.permute<1,0>()))`. Those ops return their view by value, and
`into()` accepts such a temporary, so a slot of a bigger output needs no named
intermediate; the write lands in the storage the slice refers to. Use the call for
its effect and don't keep the `dest&` it returns (the temporary view is gone at the
end of the statement). A temporary *owning* tensor is rejected at compile time
(`into(zeros<double>(shape<3>{}))`) — its storage would die with the statement, so
the result would be discarded.

**`y`'s shape is checked.** It must match the result the producer would have
allocated — the source's own shape for a unary or scalar-rhs op, the broadcast
shape for a tensor-rhs one. There is no stretching on the way *out*: only the
operands broadcast, never the destination, so a `y` smaller in any axis is
rejected, not written repeatedly. Every producer catches it the same way: a
**compile error** whenever the extents in play are static, and a debug-time check
(`assert`, off under `-DNDEBUG`) when any of them is dynamic. A **unary or
scalar-rhs** producer wants **exact** equality (it has nothing to stretch); a
**tensor-rhs** one applies its broadcast rule to the *operands* only — each operand
axis must equal `y`'s or be 1.

```cpp
auto a = zeros(shape<8,8>{});
auto y = zeros(shape<2,2>{});
a.mul(2.0, into(y));   // compile error: dest's shape must match the source's
exp(a, into(y));       // same
a.add(a, into(y));     // same, for the broadcasting tensor-rhs form

auto row = zeros(shape<1,3>{});
auto out = zeros(shape<2,3>{});
out.add(row, into(out));   // fine: the (1,3) OPERAND stretches over (2,3)
out.add(out, into(row));   // compile error: a (1,3) DEST does not stretch
```

**`y` must not self-overlap** either — no `extent > 1` axis with stride 0 (only a
hand-written `wrap(ptr, e, strides-with-a-0)` makes such a view). It would take many
results into one element and keep just the last, so a debug-time check rejects it,
for every `into(dest)` producer alike — the same guard that rejects an in-place write
into one (see [Math & broadcasting](math.md#in-place-ops-mutate-this)).

`y` may
also carry a different offset **index type** from the operands' (a narrower one
included) — the widening needed to walk both safely happens inside the engine, in
width and in signedness alike, and `y` keeps its own type; see
[Mixing widths](shapes-strides.md#mixing-widths-in-a-broadcast).

| Call | Returns |
|---|---|
| `a.add(b, into(y))` `a.mul(b, into(y))` `a.add(s, into(y))` … | `y&` (elementwise, tensor or scalar rhs) |
| `a.add(b, alpha, into(y))` `a.sub(b, alpha, into(y))` | `y&` (fused axpy into `y`) |
| `exp(a, into(y))` … (every unary) | `y&` |
| `minimum(a,b,into(y))` `maximum(a,s,into(y))` `clamp(a,lo,hi,into(y))` | `y&` |
| `normalize(a, into(y))` `normalize<1>(a, into(y))` `normalize(a, axis<1>{}, into(y))` | `y&` (`y` must match `a`'s full shape **exactly** — only the divisor is reduced; no broadcasting leeway for the axis form either) |
| `a.map(f, into(y))` | `y&` (user functor, one fused pass) |
| `cross(a,b,into(y))` `cross(a,b,into(N(i,all)))` | `y&` / the slice's `dest&` |
| `sum(a, into(cell))` … `dot(a,b,into(cell))` `sqdist(a,b,into(cell))` | `cell&` (full reduction → **rank-0** dest) |
| `sum<0>(m, into(buf))` `mean(m, axis<1>{}, into(buf))` | `buf&` (axis reduction → lower-rank dest) |

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

Every reduction here (plus `sqnorm`/`norm`/`sqdist`/`dist` below, and `allclose`)
is **also a method** — `a.sum()`, `a.mean()`, `a.dot(b)`, `a.sqdist(b)`,
`a.allclose(b)`, `a.sum<0>()`, `a.mean(axis<1>{})`, `a.norm()`,
`a.sum(into(cell))` — with the same overload shapes as the free form. The free
`sum(a)` spelling stays (generic code and the numpy reader both use it).

| Call | Returns | Notes |
|---|---|---|
| `sum(a)` `prod(a)` `max(a)` `min(a)` `mean(a)` | `T` (accumulated wide) | over all axes |
| `dot(a, b)` | `promote(Ta,Tb)` (accumulated wide) | inner product; extents must match **exactly** (no broadcast). The two operands may differ in offset index type — see [Mixing widths](shapes-strides.md#mixing-widths-in-a-broadcast); `sqdist`/`dist` likewise |
| `sum<Acc>(a)`, `mean<Acc>(a)`, `dot<Acc>(a,b)` | `Acc` | force the accumulator/return type |
| `sum(a, dtype<Acc>{})`, `dot(a, b, dtype<Acc>{})` | `Acc` | value-tag form of the above — deduces `Acc` from the tag (numpy's `dtype=`); composes freely with an axis list, `keepdims`, and `into(dest)`, in any subset/order (see below) |
| `allclose(a, b, rtol=1e-5, atol=1e-8)` | `bool` | `\|a−b\| ≤ atol+rtol·\|b\|` everywhere (broadcasts); also a method, `a.allclose(b, rtol, atol)` |
| `allclose<Acc>(a, b)`, `allclose(a, b, dtype<Acc>{})` | `bool` | carry the comparison out in `Acc` instead of the operands' compute type |
| `allclose(a, b, into(cell))`, `allclose(a, b, rtol, atol, dtype<Acc>{}, into(cell))` | `dest&` | the `dot`/`sqdist` keyword bag: `into(dest)` writes the answer into a **rank-0** cell (a `bool` cell keeps it, another dtype takes the 0/1 cast). The tolerances stay positional, ahead of the bag — pass any prefix of `(rtol, atol)` and omit the rest |
| `sum<Axes...>(a)` `mean<Axes...>` `max`/`min`/`prod<Axes...>` | lower-rank tensor | remove the named axes (negatives wrap) |
| `sum<Acc, Axes...>(a)` | lower-rank tensor | leading **type** = accumulator, leading **int** = axis |
| `sum(a, axis<Axes...>{})` (also `mean`/`max`/`min`/`prod`) | lower-rank tensor | value form — numpy `axis=` selector; `sum<Acc>(a, axis<...>{})` for the accumulator; no `.template` on a dependent receiver |
| `sum<Axes...>(a, keepdims)` (also as the value form, or with a leading `Acc`) | same rank | numpy `keepdims=True` — the named axes stay size-1 instead of being removed, so the result broadcasts back against the input |

Axis reductions: a fully static result → stack (host+device); any dynamic result
→ heap (host only). `keepdims` applies to `sum`/`prod`/`max`/`min`/`mean`/`sqnorm`/`norm`.
Axis lists must be **distinct** but may be given in **any order** — with or without
`keepdims` (`sum<2,0>(a, keepdims)` == `sum<0,2>(a, keepdims)`), like every other
axis-list op. A repeated axis is a **compile error** in every spelling —
`sum<0,0>(a)`, `sum(a, axis<0,0>{})`, `a.sum<0,0>()`, `mean<1,-2>(a)` on a rank-3
tensor (negatives are normalised before the check) — not a silently dropped
duplicate.
The explicit `<Acc, Axes...>` template split stays (C++17 has no universal template
parameter to unify "leading type = accumulator" vs "leading int = axis"), but past
that split every TRAILING keyword — `dtype<Acc>{}`, `axis<...>{}`, `keepdims`,
`into(dest)` — composes generically, in any subset and any order (`tny::_kw` in
`kwargs.h`): `sum(a, dtype<double>{}, axis<0>{}, keepdims, into(dest))` ==
`sum<double, 0>(a, keepdims, into(dest))`. `dot` composes the same way (`dtype`
and `into`; it has no axis concept).

Reductions also take **`into(dest)`** (the `out=` spelling). A **full** reduction
writes its scalar into a **rank-0** destination — `sum(a, into(cell))`,
`dot(a, b, into(cell))`, where `cell` is a `local<T, shape<>>{}` or a rank-0 view
over an address `wrap(&x, shape<>{})` (a non-rank-0 dest is a `static_assert`, so
forgetting the `<axes>` fails to compile instead of splatting the total). An
**axis** reduction copies its lower-rank result into `dest` — `sum<0>(m, into(buf))`,
and the `axis<...>` value form takes `into` too (`mean(m, axis<1>{}, into(buf))`);
its `dest` is broadcast-compatible with the reduced shape (it goes through `copy_`).
Dtype casts — for every pair of element types, `half` into `bfloat16` and back
included — and the destination is returned by reference.

### Vector algebra & geometry

Contained exact linear-algebra / geometry on views (no solves, inversion, or
optimisation — those stay out of teeny).

| Call | Returns | Notes |
|---|---|---|
| `sqnorm(a)` | `T` (accumulated wide) | Σ aᵢ² over all axes; == `dot(a, a)`. `sqnorm<Acc>` forces accumulator+result. `sqnorm`/`sqdist` are deliberately invented names — no numpy/pytorch precedent to diverge from (`CLAUDE.md`'s "Naming policy") |
| `sqnorm<Axes...>(a)` / `sqnorm(a, axis<...>{})` | lower-rank tensor | Σ aᵢ² over the named axes (reduction API, like `sum`); `sqnorm<Acc,Axes...>` |
| `norm(a)` | floating (`T` for float `T`, **`double`** for integer `T`) | √Σ aᵢ² (L2 / Frobenius over all axes); `norm<Acc>` forces accumulator+result |
| `norm<Axes...>(a)` / `norm(a, axis<...>{})` | lower-rank tensor (floating) | √Σ aᵢ² over the named axes; `norm<Acc,Axes...>` |
| `sqdist(a, b)` | `promote(Ta,Tb)` (accumulated wide) | Σ(aᵢ-bᵢ)²; mathematically `sqnorm(a-b)`, one fused pass (bit-exact only for `double` operands). Binary only (no axis form, like `dot`); `sqdist<Acc>` forces accumulator+result |
| `dist(a, b)` | floating (`double` for integer operands) | √Σ(aᵢ-bᵢ)²; mathematically `norm(a-b)`, one fused pass (bit-exact only for `double` operands); `dist<Acc>` forces accumulator+result |
| `a.normalize_()` | `tensor&` | in place `a /= norm(a)`; floating element types only; zero vector → NaN |
| `a.normalize_<Axes...>()` | `tensor&` | in place, over the named axes (keepdim broadcast); axes distinct, any order |
| `normalize(a)` | new tensor (floating; integer → `double`) | out-of-place unit vector; static → stack, dynamic → heap |
| `normalize<Axes...>(a)` / `normalize(a, axis<...>{})` | new tensor (floating) | out-of-place, over the named axes (keepdim). Also methods (`a.normalize<1>()`, `a.normalize(axis<1>{})`), and every spelling takes `into(y)` |
| `cross(a, b)` | new stack 3-vector `promote(Ta,Tb)` | 3D cross product; operands rank-1, length 3 |
| `a.cross_(b)` | `tensor&` | in place: `a` becomes `a × b` (rank-1, length 3; aliasing-safe). Into a separate slot: `slot.copy_(cross(a,b))` |

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
| `dot(a, b)` `sqdist(a, b)` | `promote(Ta,Tb)` | `Acc` |
| `dist(a, b)` | `double` for integer operands (mean rule) | `Acc` |
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

## Dispatch (the anyrank boundary)

| Call | Returns | Notes |
|---|---|---|
| `as_anyrank(data, shape, stride, ndim)` | `anyrank` (view store) | **wraps** the arrays, no copy (default; host only) |
| `as_anyrank(data, shape, stride, ndim, copy_meta)` | `anyrank` (inline store) | **copies** into a `TNY_MAX_RANK` store (device-passable); `as_anyrank<N>(…,copy_meta)` sets capacity |
| `as_anyrank<Space>(…)` | space-tagged `anyrank` | `Space` = the data's memory space (default `storage::view` = host); pass `storage::gpu_view` for a device pointer so `fixed`/`peel_front` yield `gpu_view` views |
| `as_anyrank(…, anyshape<etc,-1,-1,3>{}[, layout])` | `anyrank` with a static **tail** | bake the static TRAILING extents into the type (`anyshape<etc,...>`: `etc` = the erased batch, the dims after it = the static tail) — `fixed`/`peel_front` cells fold those inner dims (no per-call `recast`). An optional **layout tag** (`ccontiguous`/`fcontiguous`/`strides<…>`, like `recast`'s 2nd arg; default `keep_strides`) also folds the trailing STRIDES — a contiguous inner block's cell mapping is then empty (EBO). Extents and strides are checked vs the runtime shape/strides once at the boundary, then trusted. Also on the `copy_meta` form. Bare `anyshape<etc>` ⇒ today's fully-dynamic carrier |
| `as_anyrank(…, anyshape<3,etc,5>{})` | `anyrank` with a static **Head + Tail** | dims *before* `etc` are a static leading **Head** (anchored at 0), after it the **Tail** — `anyshape<3,etc,5>` = `(C_in=3, *spatial, C_out=5)`. The Head folds in `fixed`/`dispatch_rank` (full-rank window); `peel_front<-Sr>` stays trailing (the Head is peeled into the batch). Head **extents** fold; head **strides** stay runtime (a leading stride spans the dynamic middle). Empty Head (`etc` first) ⇒ the trailing-only carrier |
| `from_dlpack<T[, Space]>(m)` / `from_dlpack<T, R[, Space]>(m)` | `anyrank` / rank-`R` view | import; `m` may be a `DLManagedTensor*`, a **bare `DLTensor*`** (unmanaged — nothing to free, caller owns all lifetime), or a **`DLManagedTensorVersioned*`** (DLPack 1.0+). `Space` (default host) is **checked against `m→device`** — a `kDLCUDA` tensor needs `storage::gpu_view` |
| `from_dlpack<T, anyshape<etc,-1,-1,3>[, Space]>(m[, layout])` | `anyrank` with a static **tail** | import with the static TRAILING extents baked in — the payload's trailing dims are checked against the tag at the import boundary, then every peeled cell folds them. Pass a **layout tag by value** (`m, ccontiguous{}`) to also fold the inner strides (checked vs the payload's strides — the "input is contiguous" precondition). Accepts all three carriers |
| `to_dltensor(t, shape_out, strides_out[, dev])` | `DLTensor` | export to a **bare `DLTensor`** (no capsule/deleter/alloc): borrows `t`'s data AND the two caller-supplied `int64_t[≥rank]` buffers it fills. Caller keeps the tensor + buffers alive. (`to_dlpack(t)` is the managed-capsule export.) |
| `at.fixed<R>()` | rank-`R` `dynamic_strides` view | requires `ndim == R` |
| `dispatch_rank(at, f)` | `bool` | call `f` with a fixed-rank view chosen by runtime `ndim` (one instantiation per total rank) |
| `dispatch_rank<narrow_index>(at, f)` | `bool` | ...and narrow each fixed cell's offset width to `int32` when `index_fits` (rank outer, width inner; the leaf doubles). Default `Narrow=false` is the plain dispatch |
| `dispatch_index<Idx2=int32_t>(v, f)` | `void` | narrow a fixed-rank view `v` to `Idx2` offsets when its span fits, else keep it — then call `f`; instantiates `f` for both widths |
| `dispatch_layout(v, f)` | `void` | the LAYOUT sibling: runtime-classify a `dynamic_strides` view as `ccontiguous`/`fcontiguous`/(else)`dynamic_strides` (`is_dense<…>()`, a stride compare) and call `f` with the **retyped** view — so `recast<shape<-1,c,c>>()` folds the inner strides **safely**. The C-order arm is the payoff (inner strides fold); `f` runs on up to 3 view types. **Opt-in** (triples instantiations) |
| `at.peel_front<N>()` / `at.peel_front_at<N>(i)` | range / view | batch idiom: keep the last `\|N\|` dims static, peel the rest (one kernel per `\|N\|`) |
| `at.peel_front<-Sr>().enumerate()` | a range of `{index, cell}` | range-for that ALSO yields the **batch multi-index**: `for (auto [m, cell] : …)` with `m[d]` = coord of batch axis `d`, `m.rank()` (runtime) = #batch axes, `m.linear()` = the flat batch index — for a per-batch-axis table `param[d][m[d]]`. **Opt-in** (the bare cell stays lean); `m` is a view of the live odometer (valid within the loop body). Raw iterator: `it.index(d)` / `it.nbatch()` / `it.linear()`. Composes with `subrange` |
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
