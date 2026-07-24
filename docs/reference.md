# API reference

A thorough, type-annotated reference for the public API. For a one-screen scan
of the whole surface, see the [Cheat sheet](cheatsheet.md); for the raw
Doxygen-extracted signatures, see [Autodoc](api/index.md).

Everything is in `namespace tny` (`namespace cs = cuda::std`). Include
`<teeny/teeny.h>` for all of it; `<teeny/cuda.h>` adds the CUDA memory spaces.

Legend: **`Idx`** = the extents' `index_type` (`int64_t` for `shape<...>`).
**`T`** = element type. A *static index* is an `integral_constant` (`Int<k>()`);
a *runtime index* is a plain integer. "→ view" means a non-owning
`tensor<…, own::view>` aliasing the same memory (no copy).

---

## The tensor type

```cpp
template <class T, class Extents, class Layout = ccontiguous, own O = own::view>
struct tensor;
```

| Parameter | Meaning | Typical value |
|---|---|---|
| `T` | element type | any arithmetic type, `half`, `bfloat16` |
| `Extents` | the **shape** | `shape<2,3>` (a `cs::extents<int64_t,…>`; `-1` = dynamic) |
| `Layout` | memory order | `ccontiguous` (C, default), `fcontiguous` (F), `dynamic_strides` (runtime), `strides<S...>` (static/mixed) |
| `O` | ownership | `own::view` (default), `own::stack`, `own::heap`, `own::gpu`/`pinned`/`mapped`, `own::gpu_view` |

Slicing / permuting / peeling / `.at()` of a `gpu` tensor yields an `own::gpu_view`
(a non-owning view of *device* memory), so a device pointer is never mistaken for
a host one in the type. Helpers: `own_is_device` (gpu/gpu_view),
`own_is_host_accessible`, `own_is_view` (view/gpu_view), `own_view_of(O)` (the
view kind that preserves a source's space). `pinned`/`mapped` are host-accessible,
so their views are plain `view`.

### Ownership aliases

| Alias | Ownership | Notes |
|---|---|---|
| `view<T,E,L=ccontiguous>` | none (host view) | trivially copyable, kernel-passable; the bare `tensor` is this |
| `local<T,E,L>` | stack | requires a fully static shape; `sizeof` == its data |
| `owned<T,E,L>` | heap (host) | move-only |
| `gpu<T,E,L>` / `pinned<T,E,L>` / `mapped<T,E,L>` | CUDA | from `<teeny/cuda.h>`; a view of a `gpu` is `own::gpu_view` |

---

## Factories

Wrap existing memory (→ view):

| Call | Returns | Notes |
|---|---|---|
| `wrap(ptr, shape)` | `view<T,E>` | C-order (`ccontiguous`) |
| `wrap<Layout>(ptr, shape)` | `view<T,E,Layout>` | chosen layout |
| `wrap(ptr, shape, {s0,s1,…})` | `view<T,E,dynamic_strides>` | **runtime** strides (elements; may be negative) |
| `wrap<S...>(ptr, shape, {dyn…})` | `view<T,E,strides<S...>>` | **mixed** static/runtime strides (`dynamic_stride` slots) |
| `wrap(ptr, shape, strides<S...>{})` | `view<T,E,strides<S...>>` | **compile-time** strides (fold into the type) |
| `wrap(…, own_v<own::gpu>)` | view in that space | trailing **memory-space** tag on any overload above (default `own::view`); pass the plain backend — `own::gpu`/`pinned`/`mapped` — and it folds to the view kind (`gpu_view`/…), since `wrap` always views. `own_c<S>{}` / `own_v<S>` are the braced / no-braces spellings |
| `as_tensor(md)` | `view<…>` | wrap any `cs::mdspan`/`submdspan` result |
| `make_view(ptr, shape)` | `view<T,E>` | an alias of `wrap` that deduces `E` (`make_view<Layout>` for the layout); takes the same trailing `own_c<Space>{}` tag |

Allocate new storage — element type **`T` defaults to `float`**; static shape →
stack (host+device), dynamic shape → heap (host only):

| Call | Returns | Element type |
|---|---|---|
| `empty<T>(shape)` | `local`/`owned` (deduced) | `T` (=`float`); UNINITIALISED |
| `empty<T, own::S>(shape)` | owner in space `S` | name a backend: `stack`/`heap`/`gpu`/`pinned`/`mapped` |
| `empty<T>(shape, own_c<own::S>{})` | owner in space `S` | value-tag backend form (same result) |
| `make_local<T>(shape)` | `local<T,E>` | `T` (=`float`); = `empty<T,own::stack>` |
| `make_heap<T>(shape)` | `owned<T,E>` | `T` (=`float`); = `empty<T,own::heap>` |
| `make_gpu<T>(shape)` / `make_pinned<T>` / `make_mapped<T>` | CUDA owner | `T` (=`float`); = `empty<T,own::gpu/…>` |
| `zeros<T>(shape)` / `ones<T>(shape)` | stack or heap | `T` (=`float`) |
| `full(shape, v)` | stack or heap | **the value's type** (`full<T>(…)` to force) |
| `arange<T>(n)` | `owned<T, shape<-1>>` | `T` (=`int64_t`); 1-D `[0,n)` |
| `arange<T,N>()` / `arange<T>(Int<N>())` | `local<T, shape<N>>` | static 1-D `[0,N)` |
| `zeros<T, own::S>(shape)` (also `ones`/`full`/`arange`) | owner in space `S` | host-accessible backend (`stack`/`heap`/`pinned`/`mapped`); `own::gpu` `static_assert`s → `to<own::gpu>(zeros<T>(shape))`. Value-tag: `zeros<T>(shape, own_c<own::S>{})` |

---

## Geometry

| Call | Returns | Notes |
|---|---|---|
| `t.rank()` | `size_t` (constexpr) | number of axes |
| `t.numel()` | `integral_constant` if fully static, else `Idx` | product of extents; folds when static |
| `t.extent(d)` | `Idx` | runtime axis size |
| `t.extent(Int<k>())` | `integral_constant` if static, else `Idx` | folds when static |
| `t.shape()` | array-like accessor | `shape()[Int<k>()]` folds (integral_constant), `shape()[i]` is runtime; `rank()`, iterable, converts to `Extents` |
| `t.shape(d)` | `integral_constant`/`Idx` | per-axis shorthand (== `extent(d)`) |
| `t.strides()` | array-like accessor | twin of `shape()` for strides: `strides()[Int<k>()]` folds where derivable, `strides()[i]` runtime |
| `t.stride(d)` | `Idx` | runtime axis stride |
| `t.stride(Int<k>())` | `integral_constant` if derivable, else `Idx` | folds for static-stride / contiguous layouts |
| `t.is_dense()` | `bool` | dense block in **some** axis order (C, F, or permuted) |
| `t.is_dense<L>()` | `bool` | exact: strides equal `L`'s packing (`ccontiguous`=C, `fcontiguous`=F) |
| `t.is_contiguous()` | `bool` | **C-order** (numpy/pytorch default); `is_contiguous<fcontiguous>()` for F — a thin alias of `is_dense<L>()`. This is what `reshape`/`flatten` need |
| `t.data()` | `T*` | base pointer |
| `t.view()` | `view<T,E,L>` (`gpu_view` if device) | non-owning teeny view aliasing `t`'s memory (no copy) |
| `t.mdspan()` | `cs::mdspan<T,E,L>` | the raw mdspan |
| `t.extents()` / `t.mapping()` | `const Extents&` / `const mapping&` | |

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

---

## Structure (views)

All return a view and work on any source layout (incl. `strides<...>`); axis
template args are signed (negatives count from the back).

**Type inference — what the output *type* keeps.** A view op transforms the input
type along four independent facets; staticity is preserved wherever it is
derivable, so a kernel written against a static shape stays static through these
ops (no accidental fallback to runtime extents/strides):

| Facet | Rule |
|---|---|
| **extents** | each *kept* axis keeps its source extent — static stays static. Peeling the batch dims off `shape<-1,-1,M,N>` yields a `<M,N>` cell; a *static-bounds* slice/range keeps a static extent, a runtime range is dynamic; `unsqueeze` inserts a static `1` |
| **strides** | folded to a compile-time `strides<...>` slot wherever `source_stride × step` is known at compile time — a static source keeps folded strides, and a partially-dynamic *contiguous* stride folds from the static extents it spans (`shape<-1,3,3>` → `stride0 = 9`); otherwise runtime |
| **layout** | the view ops (`operator()`, `take_along`, `permute`, `flip`, `un/squeeze`, `peel`) output a folded `strides<...>`. `recast<keep_strides>` (the default) instead **preserves the source layout type** (`ccontiguous` stays `ccontiguous`, `strides<>` stays `strides<>`) |
| **space** | `own_view_of(source)`: a `gpu`/`gpu_view` source → `gpu_view`, `pinned`/`mapped` → the matching `_view`, else `view` — a view never loses its memory space |

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
| `t.clone()` | owning (stack/heap) | materialise a dense row-major copy |
| `t.to<T2>()` | view (no-copy) or owning | **dtype** convert. Matching dtype (no `Force`) → a read-only borrow (`gpu_view` if `t` is on the device, else `view`); differing dtype or `t.to<T2,true>()` → a dense owning copy (static→stack, dyn→heap) |
| `to<Space>(t)` (`cuda.h`) | view (no-copy) or owning | **memory-space** move: `to<Space, ET, Force>(t)` — `Space` ∈ `own::gpu`/`pinned`/`mapped`/`heap`/`stack`. Same no-copy/`Force` rule; a device source (gpu/`gpu_view`) downloads via `cudaMemcpy`. rvalue source → always copies |

`reshape`/`flatten` need exact C-contiguity (`is_contiguous()`);
`clone()` first if the source isn't. `recast` does **not** — it only re-types the
extents and keeps the source strides, so it works on any layout.

### nd-peel (iteration)

| Call | Returns | Notes |
|---|---|---|
| `peel<Axes...>(t)` | a range of views | iterate the named axes; each item is a lower-rank view |
| `peel_at<Axes...>(t, i)` | → view | the `i`-th sub-view (grid-stride style) |
| `peel_front<N>(t)` | a range of views | `N≥0`: peel the first `N` axes; `N<0`: keep the last `|N|` |
| `peel_front_at<N>(t, i)` | → view | the `i`-th (grid-stride style) |
| `size_front<N>(t)` | → index | # cells `peel_front<N>` yields (product of the peeled extents), no range built |

---

## Math

Element type of results follows `promote(A,B)` (C++ rules, but among floats the
**lower** width wins — pytorch-style; `-DTNY_STD_PROMOTION` opts out).

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

Axis reductions: a fully static result → stack (host+device); any dynamic result
→ heap (host only).

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
| `as_anyrank<Space>(…)` | space-tagged `anyrank` | `Space` = the data's memory space (default `own::view` = host); pass `own::gpu_view` for a device pointer so `fixed`/`peel_front` yield `gpu_view` views |
| `from_dlpack<T[, Space]>(m)` / `from_dlpack<T, R[, Space]>(m)` | `anyrank` / rank-`R` view | import a capsule; `Space` (default host) is **checked against `m→device`** — a `kDLCUDA` capsule needs `own::gpu_view` |
| `at.fixed<R>()` | rank-`R` `dynamic_strides` view | requires `ndim == R` |
| `dispatch_rank(at, f)` | `bool` | call `f` with a fixed-rank view chosen by runtime `ndim` (one instantiation per total rank) |
| `at.peel_front<N>()` / `at.peel_front_at<N>(i)` | range / view | batch idiom: keep the last `\|N\|` dims static, peel the rest (one kernel per `\|N\|`) |
| `at.size_front<N>()` | → offset | flattened batch count `peel_front<N>` yields (product of the peeled leading extents), no range built; `N < 0` |
| `dispatch_value<Vs...>(v, f)` | `bool` | call `f(Int<k>{})` for the matching candidate `k == v` |

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
