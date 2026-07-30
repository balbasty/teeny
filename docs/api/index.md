# Autodoc

Generated from the header Doxygen comments (`doxygen` + `moxygen`). For a curated view see the [Reference](../reference.md) and [Cheat sheet](../cheatsheet.md).

## Classes

| Name | Description |
|------|-------------|
| [`anyrank`](#anyrank) | A rank-erased tensor for the host/ndarray dispatch boundary. |
| [`anyrank_front`](#anyrank_front) | A range of fixed-rank-`Sr` sub-views over an `anyrank`'s batch axes. |
| [`compute_type`](#compute_type) | The type math should ACCUMULATE / compute in for element type `T`. |
| [`compute_type< bfloat16 >`](#compute_typebfloat16) |  |
| [`compute_type< half >`](#compute_typehalf) |  |
| [`copy_meta_t`](#copy_meta_t) | Tag for `as_anyrank(..., copy_meta)`: COPY shape/stride into an inline, device-passable store instead of wrapping the caller's arrays. |
| [`cpp_alloc`](#cpp_alloc) | Host allocator using C++ `new[]` / `delete[]`. |
| [`cuda_gpu_alloc`](#cuda_gpu_alloc) | Device (GPU) memory (`cudaMalloc`). |
| [`cuda_mapped_alloc`](#cuda_mapped_alloc) | Page-locked + device-mapped (zero-copy) host memory (`cudaHostAlloc`). |
| [`cuda_pinned_alloc`](#cuda_pinned_alloc) | Page-locked ("pinned") host memory (`cudaMallocHost`). |
| [`ellipsis_t`](#ellipsis_t) | Ellipsis sentinel — teeny's `...` (python `a[..., 0]` / numpy `Ellipsis`). |
| [`none_t`](#none_t) | Open-ended slice sentinel — teeny's `None` (python `a[:n]` / `a[m:]`). |
| [`owning_storage`](#owning_storage) | Generic owning storage (move-only, no ref-counting), parameterised by an allocator policy. |
| [`peel_range`](#peel_range) | A range of sub-views obtained by peeling `Axes...`. |
| [`storage`](#storage) |  |
| [`gpu, N >`](#gpun) |  |
| [`gpu_view, N >`](#gpu_viewn) |  |
| [`heap, N >`](#heapn) |  |
| [`mapped, N >`](#mappedn) |  |
| [`pinned, N >`](#pinnedn) |  |
| [`stack, N >`](#stackn) |  |
| [`view, N >`](#viewn) |  |
| [`storage_size`](#storage_size) | Storage element count for a stack tensor (0 for view/owning). |
| [`storage_size< Mapping, true >`](#storage_sizemappingtrue) |  |
| [`strides`](#strides) | An mdspan layout policy with **per-dimension static or dynamic strides** — the stride analogue of `extents`/`shape`. |
| [`tensor`](#tensor) | One N-dimensional tensor, parameterised by ownership. |

## Enumerations

| Name | Description |
|------|-------------|
| [`own`](#own)  | Ownership / memory-space of a tensor's storage. |

---

### own

```cpp
enum own
```

Ownership / memory-space of a tensor's storage.

`view` / `stack` need no allocator. The owning modes differ only in where the memory lives and how it is (de)allocated:

* `heap` : ordinary C++ `new[]` / `delete[]` (host memory).

* `gpu` : `cudaMalloc` (device memory; not host-dereferenceable).

* `pinned` : `cudaMallocHost` (page-locked host memory — pytorch's "pinned").

* `mapped` : `cudaHostAlloc` (page-locked + device-mapped / zero-copy). The `gpu`/`pinned`/`mapped` storage is defined in the opt-in `[teeny/cuda.h](#cudah)` (which needs the CUDA runtime); using them without it is a compile error.

| Value | Description |
|-------|-------------|
| `view` |  |
| `stack` |  |
| `heap` |  |
| `gpu` |  |
| `pinned` |  |
| `mapped` |  |
| `gpu_view` |  |

## Typedefs

| Return | Name | Description |
|--------|------|-------------|
| `layout_right` | [`corder`](#corder)  | numpy-style spellings of the contiguous layouts: `corder` is C-order (row-major, `layout_right`), `forder` is Fortran-order (column-major, `layout_left`). |
| `layout_left` | [`forder`](#forder)  |  |
| `layout_stride` | [`dynamic_strides`](#dynamic_strides)  | `dynamic_strides` — the all-runtime strided layout (`layout_stride`, a full runtime stride array). |
| `integral_constant< int, V >` | [`Int`](#int)  |  |
| `integral_constant< long, V >` | [`Long`](#long)  |  |
| `integral_constant< size_t, V >` | [`Size`](#size)  |  |
| `integral_constant< unsigned, V >` | [`UInt`](#uint)  |  |
| `integral_constant< ptrdiff_t, V >` | [`Diff`](#diff)  |  |
| `integral_constant< bool, V >` | [`Bool`](#bool)  |  |
| `integral_constant< int8_t, V >` | [`Int8`](#int8)  |  |
| `integral_constant< int16_t, V >` | [`Int16`](#int16)  |  |
| `integral_constant< int32_t, V >` | [`Int32`](#int32)  |  |
| `integral_constant< int64_t, V >` | [`Int64`](#int64)  |  |
| `integral_constant< uint8_t, V >` | [`UInt8`](#uint8)  |  |
| `integral_constant< uint16_t, V >` | [`UInt16`](#uint16)  |  |
| `integral_constant< uint32_t, V >` | [`UInt32`](#uint32)  |  |
| `integral_constant< uint64_t, V >` | [`UInt64`](#uint64)  |  |
| `Int8< V >` | [`I1`](#i1)  |  |
| `Int16< V >` | [`I2`](#i2)  |  |
| `Int32< V >` | [`I4`](#i4)  |  |
| `Int64< V >` | [`I8`](#i8)  |  |
| `UInt8< V >` | [`U1`](#u1)  |  |
| `UInt16< V >` | [`U2`](#u2)  |  |
| `UInt32< V >` | [`U4`](#u4)  |  |
| `UInt64< V >` | [`U8`](#u8)  |  |
| `int8_t` | [`i1`](#i1-1)  |  |
| `int16_t` | [`i2`](#i2-1)  |  |
| `int32_t` | [`i4`](#i4-1)  |  |
| `int64_t` | [`i8`](#i8-1)  |  |
| `uint8_t` | [`u1`](#u1-1)  |  |
| `uint16_t` | [`u2`](#u2-1)  |  |
| `uint32_t` | [`u4`](#u4-1)  |  |
| `uint64_t` | [`u8`](#u8-1)  |  |
| `float` | [`f4`](#f4)  |  |
| `double` | [`f8`](#f8)  |  |
| `extents< int64_t, _dyn_extent(E)... >` | [`shape`](#shape)  | User-friendly shape type: `shape<2,3,4>` == `extents<int64_t, 2,3,4>`. |
| `dextents< int64_t, N >` | [`rank`](#rank)  | Fully-dynamic shape of a given rank: `rank<3>` == `shape<-1,-1,-1>` == `extents<int64_t, dynamic_extent, dynamic_extent, dynamic_extent>`. |
| `tensor< T, Shape, Layout, own::gpu >` | [`gpu`](#gpu)  | Owning tensor in device (GPU) memory (move-only). |
| `tensor< T, Shape, Layout, own::pinned >` | [`pinned`](#pinned)  | Owning tensor in page-locked ("pinned") host memory (move-only). |
| `tensor< T, Shape, Layout, own::mapped >` | [`mapped`](#mapped)  | Owning tensor in mapped (zero-copy) host memory (move-only). |
| `tensor< T, dextents< offset_t, R >, layout_stride, own::view >` | [`dyn_tensor`](#dyn_tensor)  | A fixed-rank, fully-dynamic, arbitrarily-strided tensor view. |
| `__half` | [`half`](#half)  | IEEE binary16 — the native CUDA `__half` under nvcc. |
| `__nv_bfloat16` | [`bfloat16`](#bfloat16)  | bfloat16 — the native CUDA `__nv_bfloat16` under nvcc. |
| `typename compute_type< T >::type` | [`compute_type_t`](#compute_type_t)  |  |
| `half` | [`f2`](#f2)  |  |
| `bfloat16` | [`bf16`](#bf16)  |  |
| `strides< S... >` | [`layout_static_stride`](#layout_static_stride)  | Back-compat alias: the original all-static-stride layout name. |
| `typename _promote< A, B, true >::type` | [`promote_t`](#promote_t)  |  |
| `conditional_t<(is_floating_point< T >::value||!is_same< compute_type_t< T >, T >::value), conditional_t<(sizeof(T) > 8), T, double >, T >` | [`reduce_type_t`](#reduce_type_t)  | Default accumulator type for a reduction over element type `T`. |
| `tensor< T, Shape, Layout, own::view >` | [`view`](#view)  | A non-owning view type. |
| `tensor< T, Shape, Layout, own::stack >` | [`local`](#local)  | Stack-owned tensor (fully static shape). |
| `tensor< T, Shape, Layout, own::heap >` | [`owned`](#owned)  | Heap-owned tensor (host only, move-only). |

---

### corder

```cpp
using corder = layout_right
```

numpy-style spellings of the contiguous layouts: `corder` is C-order (row-major, `layout_right`), `forder` is Fortran-order (column-major, `layout_left`).

Use wherever a `Layout` is expected.

---

### forder

```cpp
using forder = layout_left
```

---

### dynamic_strides

```cpp
using dynamic_strides = layout_stride
```

`dynamic_strides` — the all-runtime strided layout (`layout_stride`, a full runtime stride array).

The all-/mixed-static sibling is teeny's `strides<S...>` (folds known strides to immediates). Same name shape as `strides<...>` so the two read as a pair.

---

### Int

```cpp
using Int = integral_constant< int, V >
```

---

### Long

```cpp
using Long = integral_constant< long, V >
```

---

### Size

```cpp
using Size = integral_constant< size_t, V >
```

---

### UInt

```cpp
using UInt = integral_constant< unsigned, V >
```

---

### Diff

```cpp
using Diff = integral_constant< ptrdiff_t, V >
```

---

### Bool

```cpp
using Bool = integral_constant< bool, V >
```

---

### Int8

```cpp
using Int8 = integral_constant< int8_t, V >
```

---

### Int16

```cpp
using Int16 = integral_constant< int16_t, V >
```

---

### Int32

```cpp
using Int32 = integral_constant< int32_t, V >
```

---

### Int64

```cpp
using Int64 = integral_constant< int64_t, V >
```

---

### UInt8

```cpp
using UInt8 = integral_constant< uint8_t, V >
```

---

### UInt16

```cpp
using UInt16 = integral_constant< uint16_t, V >
```

---

### UInt32

```cpp
using UInt32 = integral_constant< uint32_t, V >
```

---

### UInt64

```cpp
using UInt64 = integral_constant< uint64_t, V >
```

---

### I1

```cpp
using I1 = Int8< V >
```

---

### I2

```cpp
using I2 = Int16< V >
```

---

### I4

```cpp
using I4 = Int32< V >
```

---

### I8

```cpp
using I8 = Int64< V >
```

---

### U1

```cpp
using U1 = UInt8< V >
```

---

### U2

```cpp
using U2 = UInt16< V >
```

---

### U4

```cpp
using U4 = UInt32< V >
```

---

### U8

```cpp
using U8 = UInt64< V >
```

---

### i1

```cpp
using i1 = int8_t
```

---

### i2

```cpp
using i2 = int16_t
```

---

### i4

```cpp
using i4 = int32_t
```

---

### i8

```cpp
using i8 = int64_t
```

---

### u1

```cpp
using u1 = uint8_t
```

---

### u2

```cpp
using u2 = uint16_t
```

---

### u4

```cpp
using u4 = uint32_t
```

---

### u8

```cpp
using u8 = uint64_t
```

---

### f4

```cpp
using f4 = float
```

---

### f8

```cpp
using f8 = double
```

---

### shape

```cpp
using shape = extents< int64_t, _dyn_extent(E)... >
```

User-friendly shape type: `shape<2,3,4>` == `extents<int64_t, 2,3,4>`.

The fixed-size `int64_t` index type matches DLPack's `shape` exactly, so it drops straight onto ndarray bindings. A dynamic dimension can be spelled either `dynamic_extent` or, numpy-style, **`-1`** — so `shape<-1,2,3>` == `shape<dynamic_extent,2,3>` == `extents<int64_t, dynamic_extent, 2, 3>`. Use it in place of `extents<...>`: `local<double, shape<3,3>>`, `owned<float, shape<-1,4>>`.

---

### rank

```cpp
using rank = dextents< int64_t, N >
```

Fully-dynamic shape of a given rank: `rank<3>` == `shape<-1,-1,-1>` == `extents<int64_t, dynamic_extent, dynamic_extent, dynamic_extent>`.

Handy for a rank-N view whose sizes are all runtime: `view<float, rank<3>>`. `rank<0>` is the rank-0 (scalar) shape.

---

### gpu

```cpp
using gpu = tensor< T, Shape, Layout, own::gpu >
```

Owning tensor in device (GPU) memory (move-only).

`gpu<T,E>(extents)`.

---

### pinned

```cpp
using pinned = tensor< T, Shape, Layout, own::pinned >
```

Owning tensor in page-locked ("pinned") host memory (move-only).

`pinned<T,E>(extents)` — pytorch's `pin_memory`.

---

### mapped

```cpp
using mapped = tensor< T, Shape, Layout, own::mapped >
```

Owning tensor in mapped (zero-copy) host memory (move-only).

`mapped<T,E>(extents)`.

---

### dyn_tensor

```cpp
using dyn_tensor = tensor< T, dextents< offset_t, R >, layout_stride, own::view >
```

A fixed-rank, fully-dynamic, arbitrarily-strided tensor view.

---

### half

```cpp
using half = __half
```

IEEE binary16 — the native CUDA `__half` under nvcc.

---

### bfloat16

```cpp
using bfloat16 = __nv_bfloat16
```

bfloat16 — the native CUDA `__nv_bfloat16` under nvcc.

---

### compute_type_t

```cpp
using compute_type_t = typename compute_type< T >::type
```

---

### f2

```cpp
using f2 = half
```

---

### bf16

```cpp
using bf16 = bfloat16
```

---

### layout_static_stride

```cpp
using layout_static_stride = strides< S... >
```

Back-compat alias: the original all-static-stride layout name.

---

### promote_t

```cpp
using promote_t = typename _promote< A, B, true >::type
```

---

### reduce_type_t

```cpp
using reduce_type_t = conditional_t<(is_floating_point< T >::value||!is_same< compute_type_t< T >, T >::value), conditional_t<(sizeof(T) > 8), T, double >, T >
```

Default accumulator type for a reduction over element type `T`.

`double` for floating-point types of at most 8 bytes (`float`, `double`, `half`, `bfloat16`) — enough headroom that summing many low-precision values doesn't lose catastrophically; a *wider* floating type (`long double`) keeps itself; every other type (integers, ...) accumulates in its own item type. Half types are spotted via `[compute_type](#compute_type)` (the only `T` whose compute type differs from itself). Override per call, e.g. `sum<float>(a)`.

---

### view

```cpp
using view = tensor< T, Shape, Layout, own::view >
```

A non-owning view type.

Construct as `view<T,E>(ptr, extents)`.

---

### local

```cpp
using local = tensor< T, Shape, Layout, own::stack >
```

Stack-owned tensor (fully static shape).

Use `local<T,E>{}`.

---

### owned

```cpp
using owned = tensor< T, Shape, Layout, own::heap >
```

Heap-owned tensor (host only, move-only).

Use `owned<T,E>(extents)`.

## Functions

| Return | Name | Description |
|--------|------|-------------|
| `auto` | [`make_gpu`](#make_gpu)  |  |
| `auto` | [`make_pinned`](#make_pinned)  |  |
| `auto` | [`make_mapped`](#make_mapped)  |  |
| `auto` | [`to`](#to)  | Move `x` to memory space `Space` (`[own::gpu](#namespacetny_1a6a432f80fb491dbcb5d4b0692616b918a0aa0be2a866411d9ff03515227454947)`/`pinned`/`mapped`/`heap`), optionally converting the element type to `ET` — the memory-backend half of pytorch's `.to`. |
| `auto` | [`to`](#to-1)  | Rvalue overload of `to<Space>`: a temporary source cannot be borrowed (the no-copy branch would dangle — and for a `gpu` temporary would point at freed device memory), so this always **forces a fresh owning copy**. |
| `DLManagedTensor *` | [`to_dlpack`](#to_dlpack)  | Export a **view** (host `view` or device `gpu_view`) to a `DLManagedTensor` (borrows the data — the caller must keep the underlying memory alive; only the metadata is owned by the capsule). |
| `DLManagedTensor *` | [`to_dlpack`](#to_dlpack-1)  | Export an **owning** tensor, TRANSFERRING ownership of the buffer into the capsule (the tensor is moved-from; the capsule's `deleter` frees the buffer). |
| `anyrank< T, int64_t >` | [`from_dlpack`](#from_dlpack)  | Import a `DLManagedTensor` of known element type `T` as an `anyrank` (runtime rank). |
| `dyn_tensor< T, int64_t, R >` | [`from_dlpack`](#from_dlpack-1)  | Import as a **fixed-rank** view (requires `m->dl_tensor.ndim == R`). |
| `bool` | [`dispatch_dlpack`](#dispatch_dlpack)  | Import + dispatch: read the dtype/rank from the `DLManagedTensor` and call `f` with a fixed-rank typed view (one instantiation per (dtype, rank)). |
| `anyrank< T, offset_t, _meta_view< offset_t > >` | [`as_anyrank`](#as_anyrank)  | Build an `anyrank` that **wraps** the caller's shape/stride arrays with **no copy** (the default) — e.g. |
| `anyrank< T, offset_t, _meta_store< offset_t, MaxRank > >` | [`as_anyrank`](#as_anyrank-1)  | `as_anyrank(data, shape, stride, ndim, copy_meta)` — COPY shape/stride into an inline store, so the carrier is trivially copyable and can be passed into a CUDA kernel by value (peel on device). |
| `bool` | [`dispatch_rank`](#dispatch_rank)  | Call `f` with a fixed-rank view of `t` chosen by its runtime `ndim`. |
| `bool` | [`dispatch_value`](#dispatch_value)  | Turn a runtime value into a compile-time one from a candidate list. |
| `auto` | [`slice`](#slice)  | A python-like slice `[start : stop : step)` for `operator()` / `slice_along`. |
| `auto` | [`slice`](#slice-1)  |  |
| `auto` | [`slice`](#slice-2)  |  |
| `auto` | [`peel_at`](#peel_at)  | The `i`-th sub-view obtained by peeling `Axes...` (0 <= i < product of the peeled extents). |
| `auto` | [`peel_at`](#peel_at-1)  |  |
| `auto` | [`peel_at`](#peel_at-2)  |  |
| `auto` | [`peel`](#peel)  | Build a range of sub-views by peeling `Axes...` of `t`. |
| `auto` | [`peel`](#peel-1)  |  |
| `peel_range< MD, own::view, Axes... >` | [`peel_of`](#peel_of)  | Build a range of sub-views over a raw mdspan. |
| `auto` | [`peel_front`](#peel_front)  | Peel the FIRST `N` axes -> a range of sub-views over the rest — the runtime-batch-rank half of `(*batch, *spatial, C)`. |
| `auto` | [`peel_front`](#peel_front-1)  |  |
| `auto` | [`peel_front_at`](#peel_front_at)  | The `i`-th sub-view obtained by peeling the first `N` axes (grid-stride style). |
| `auto` | [`peel_front_at`](#peel_front_at-1)  |  |
| `auto` | [`operator+`](#operator)  |  |
| `auto` | [`operator*`](#operator-1)  |  |
| `auto` | [`operator-`](#operator-2)  |  |
| `auto` | [`operator/`](#operator-3)  |  |
| `auto` | [`operator-`](#operator-4)  |  |
| `auto` | [`operator~`](#operator-5)  |  |
| `auto` | [`sum`](#sum)  | Sum of all elements (empty -> 0). |
| `auto` | [`prod`](#prod)  | Product of all elements (empty -> 1). |
| `auto` | [`max`](#max)  | Maximum element. |
| `auto` | [`min`](#min)  | Minimum element. |
| `auto` | [`mean`](#mean)  | Mean over the named axes -> a lower-rank tensor (sum / reduced count). |
| `auto` | [`dot`](#dot)  | Inner product over matching extents. |
| `bool` | [`allclose`](#allclose)  | True if every element satisfies `\|a-b\| <= atol + rtol*\|b\|` (numpy `allclose`; broadcasts, computes in the compute type). |
| `auto` | [`minimum`](#minimum)  |  |
| `auto` | [`maximum`](#maximum)  |  |
| `auto` | [`minimum`](#minimum-1)  |  |
| `auto` | [`maximum`](#maximum-1)  |  |
| `auto` | [`clamp`](#clamp)  | `clamp(a, lo, hi)` -> a new tensor with each element clamped. |
| `auto` | [`mean`](#mean-1)  | Arithmetic mean of all elements. |
| `constexpr bool` | [`own_is_owning`](#own_is_owning) `constexpr` `noexcept` | Whether the mode owns (and therefore allocates) its storage. |
| `constexpr bool` | [`own_is_view`](#own_is_view) `constexpr` `noexcept` | Whether the mode is a non-owning view (host `view` or `gpu_view`) — the pointer-wrapping modes (as opposed to `stack`'s inline array). |
| `constexpr bool` | [`own_is_device`](#own_is_device) `constexpr` `noexcept` | Whether the storage lives in device (GPU) memory (owning or view). |
| `constexpr bool` | [`own_is_host_accessible`](#own_is_host_accessible) `constexpr` `noexcept` | Whether the storage is dereferenceable from the host. |
| `constexpr own` | [`own_view_of`](#own_view_of) `constexpr` `noexcept` | The non-owning VIEW kind that preserves a source's memory space: a device source (`gpu`/`gpu_view`) -> `gpu_view`, anything else -> `view`. |
| `tensor< typename MD::element_type, typename MD::extents_type, typename MD::layout_type, OW >` | [`as_tensor`](#as_tensor)  | Wrap any `cuda::std::mdspan` (e.g. |
| `void` | [`fetch_add`](#fetch_add) `noexcept` | Accumulate `v` into `*p`, atomic **on the device only**. |
| `tensor< T, Shape, Layout, own::view >` | [`wrap`](#wrap)  | Wrap `p` as a non-owning view with a contiguous layout (default C-order). |
| `tensor< T, Shape, layout_stride, own::view >` | [`wrap`](#wrap-1)  | Wrap `p` as a non-owning view with explicit **runtime strides** (a `layout_stride` view). |
| `tensor< T, Shape, strides< Strides... >, own::view >` | [`wrap`](#wrap-2)  | Wrap `p` as a non-owning view with per-dimension **compile-time strides** (may be negative): pass a `strides<S...>{}` as the third argument. |
| `tensor< T, Shape, strides< S0, Srest... >, own::view >` | [`wrap`](#wrap-3)  | Wrap `p` with a **mix of static and runtime strides** — the exact analogue of `shape<-1,2,3,-1>{d0,d1}` for strides. |
| `auto` | [`make_view`](#make_view)  | `make_view<L>(ptr, extents)` — a non-owning view (alias of `wrap`). |
| `auto` | [`make_local`](#make_local)  | `make_local<T>(extents)` — a stack-owned tensor (static shape). |
| `auto` | [`make_heap`](#make_heap)  | `make_heap<T>(extents)` — a heap-owned tensor (host, move-only). |
| `auto` | [`full`](#full)  | `full(extents, v)` — a new tensor filled with `v`. |
| `auto` | [`zeros`](#zeros)  | `zeros<T>(extents)` / `ones<T>(extents)` — a new tensor of 0s / 1s. |
| `auto` | [`ones`](#ones)  |  |
| `auto` | [`arange`](#arange)  | `arange<T>(n)` — a 1-D tensor `[0, 1, ..., n-1]` (heap, host). |
| `auto` | [`arange`](#arange-1)  | Static `arange<T, N>()` — a stack `[0..N-1]` (host+device, folds). |
| `auto` | [`arange`](#arange-2)  | `arange<T>(Int<N>())` — the static form spelled with a static integer. |

---

### make_gpu

```cpp
template<class T = float, class Layout = layout_right, class Shape> auto make_gpu(Shape e)
```

---

### make_pinned

```cpp
template<class T = float, class Layout = layout_right, class Shape> auto make_pinned(Shape e)
```

---

### make_mapped

```cpp
template<class T = float, class Layout = layout_right, class Shape> auto make_mapped(Shape e)
```

---

### to

```cpp
template<own Space, class ET = void, bool Force = false, class T, class Shape, class Layout, own O> auto to(const tensor< T, Shape, Layout, O > & x)
```

Move `x` to memory space `Space` (`[own::gpu](#namespacetny_1a6a432f80fb491dbcb5d4b0692616b918a0aa0be2a866411d9ff03515227454947)`/`pinned`/`mapped`/`heap`), optionally converting the element type to `ET` — the memory-backend half of pytorch's `.to`.

`ET` defaults to the source type. 
```
auto d = to<own::gpu>(h);           // upload host -> device
auto e = to<own::gpu, half>(h);     // convert to half AND upload
auto c = to<own::heap>(d);          // download device -> host
```
**No-op when already there:** if the source is already an owning tensor of element type `ET` in space `Space`, and `Force` is false, this returns a *view* of it (no copy, borrows the source) — like any teeny view op. Pass `Force = true` to always materialise a fresh owning copy (a force-clone into a space): 
```
       auto v = to<own::gpu>(g);           // g is already gpu<T> -> a view, no copy
       auto k = to<own::gpu, void, true>(g);  // forced: a fresh gpu copy
```
 Any **device** source — an owning `gpu` OR a `gpu_view` (a slice/permute/peel of a gpu tensor) — is downloaded via `cudaMemcpy` (any layout, C/F/strided, is preserved and densified on the host); a host-accessible source is read directly, gathering only the viewed extent. `Space == stack` needs a static shape. Since #15, a device view is correctly *typed* (`gpu_view`), so it takes the download path instead of being host-dereferenced — the hazard the earlier version warned about is closed. NB a **contiguous** device view downloads exactly its `numel` elements; a **strided** device view currently downloads its full span (over-copies — a run-wise gather is a tracked follow-up, #50).

:::note
The no-copy branch returns a **borrow** of `x` (a `gpu_view` when `x` is a gpu tensor, else a host `view`), so it must outlive the result — same lifetime rule as `[view()](#view)`/`permute()`/slicing. Calling it on a temporary lvalue would dangle; the rvalue overload below forces a copy instead. 

:::

---

### to

```cpp
template<own Space, class ET = void, bool Force = false, class T, class Shape, class Layout, own O> auto to(tensor< T, Shape, Layout, O > && x)
```

Rvalue overload of `to<Space>`: a temporary source cannot be borrowed (the no-copy branch would dangle — and for a `gpu` temporary would point at freed device memory), so this always **forces a fresh owning copy**.

---

### to_dlpack

```cpp
template<class T, class Shape, class Layout, own O, enable_if_t< own_is_view(O), int > = 0> DLManagedTensor * to_dlpack(const tensor< T, Shape, Layout, O > & t, DLDevice dev = { _dl::device_of< O >(), 0 })
```

Export a **view** (host `view` or device `gpu_view`) to a `DLManagedTensor` (borrows the data — the caller must keep the underlying memory alive; only the metadata is owned by the capsule).

The device defaults to the tensor's memory space (`kDLCPU` for a host view, `kDLCUDA` for a `gpu_view`; pass `dev` to override). The consumer owns the returned pointer and MUST call `m->deleter(m)` exactly once.

---

### to_dlpack

```cpp
template<class T, class Shape, class Layout, own O, enable_if_t< own_is_owning(O), int > = 0> DLManagedTensor * to_dlpack(tensor< T, Shape, Layout, O > && t)
```

Export an **owning** tensor, TRANSFERRING ownership of the buffer into the capsule (the tensor is moved-from; the capsule's `deleter` frees the buffer).

Device is taken from the tensor's memory space.

---

### from_dlpack

```cpp
template<class T> anyrank< T, int64_t > from_dlpack(const DLManagedTensor * m)
```

Import a `DLManagedTensor` of known element type `T` as an `anyrank` (runtime rank).

The shape/stride METADATA is copied into the carrier (so it is self-contained), while the DATA is BORROWED — the caller keeps `m` alive while the view is used, then calls `m->deleter(m)`. A null `strides` (DLPack's C-contiguous shorthand) is expanded to row-major. `byte_offset` is folded into the data pointer. NB the `device` field is not inspected: the imported carrier is currently HOST-tagged (`anyrank`/`dyn_tensor` have no memory-space parameter yet), so a `kDLCUDA` capsule yields a host-tagged view over a device pointer — consult `m->dl_tensor.device` yourself before dereferencing on the host. (Tagging the rank-erased boundary with its space is tracked as follow-up on #15.)

---

### from_dlpack

```cpp
template<class T, size_t R> dyn_tensor< T, int64_t, R > from_dlpack(const DLManagedTensor * m)
```

Import as a **fixed-rank** view (requires `m->dl_tensor.ndim == R`).

Returns a `layout_stride` tensor view borrowing the data; the caller owns `m`'s lifetime.

---

### dispatch_dlpack

```cpp
template<class F> bool dispatch_dlpack(const DLManagedTensor * m, F && f)
```

Import + dispatch: read the dtype/rank from the `DLManagedTensor` and call `f` with a fixed-rank typed view (one instantiation per (dtype, rank)).

Returns false if the dtype/rank is outside the supported set. Data borrowed; caller owns `m`.

---

### as_anyrank

```cpp
template<class T, class offset_t> anyrank< T, offset_t, _meta_view< offset_t > > as_anyrank(T * data, offset_t * shape, offset_t * stride, int ndim)
```

Build an `anyrank` that **wraps** the caller's shape/stride arrays with **no copy** (the default) — e.g.

straight off a DLPack tensor. The arrays must outlive the carrier. HOST only: the pointers are not valid inside a device kernel, so peel/dispatch on the host and pass the resulting fixed-rank views to the device. To instead copy into an inline, device-passable store, pass the `copy_meta` tag (overload below). DLPack strides are in ELEMENTS; numpy `__array_interface__` in BYTES (divide by the itemsize first).

---

### as_anyrank

```cpp
template<size_t MaxRank = TNY_MAX_RANK, class T, class offset_t> anyrank< T, offset_t, _meta_store< offset_t, MaxRank > > as_anyrank(T * data, const offset_t * shape, const offset_t * stride, int ndim, copy_meta_t)
```

`as_anyrank(data, shape, stride, ndim, copy_meta)` — COPY shape/stride into an inline store, so the carrier is trivially copyable and can be passed into a CUDA kernel by value (peel on device).

`MaxRank` sets the inline capacity (default `TNY_MAX_RANK`); pass it as `as_anyrank<64>(..., copy_meta)`. Accepts `const` arrays (it copies).

---

### dispatch_rank

```cpp
template<class T, class offset_t, class Meta, class F> bool dispatch_rank(const anyrank< T, offset_t, Meta > & t, F && f)
```

Call `f` with a fixed-rank view of `t` chosen by its runtime `ndim`.

`f` is a generic callable instantiated once per possible rank; the kernel it launches is fully static. Returns false if `ndim` exceeds `max_rank`. Prefer `peel_front<Sr>` when only the trailing dims need to be static — one instantiation instead of one per total rank. 
```
dispatch_rank(as_anyrank(data, size, stride, ndim), [&](auto v){ kernel(v); });
```

---

### dispatch_value

```cpp
template<int... Vs, class F> bool dispatch_value(int v, F && f)
```

Turn a runtime value into a compile-time one from a candidate list.

`dispatch_value<1,2,3>(D, f)` calls `f(Int<k>{})` for the matching candidate `k == D` (so `f` receives a static `integral_constant` it can use as a template argument), and returns whether any matched. 
```
dispatch_value<1,2,3>(ndim_spatial, [&](auto d){ kernel<d.value>(view); });
```

---

### slice

```cpp
template<class A, class B> auto slice(A start, B stop)
```

A python-like slice `[start : stop : step)` for `operator()` / `slice_along`.

`none` marks an open end; negative bounds wrap (count from the back); `step` defaults to 1 and may exceed 1.

`slice(1, 4)` = `[1,4)`; `slice(none, 4)` = `[0,4)`; `slice(2, none)` = `[2,end)`; `slice(0, none, 2)` = every other element; `slice(none, none)` keeps the whole axis (== `all`, which is preferable when you want the axis kept — it folds and preserves static extents). A ranged axis is resolved at run time (its extent becomes dynamic); axes kept with `all` stay static.

---

### slice

```cpp
template<class A, class B, class S> auto slice(A start, B stop, S step)
```

---

### slice

```cpp
template<long Start, long Stop, long Step = 1> auto slice()
```

---

### peel_at

```cpp
template<size_t... Axes, class MD> auto peel_at(const MD & src, typename MD::index_type i)
```

The `i`-th sub-view obtained by peeling `Axes...` (0 <= i < product of the peeled extents).

Peeled axes vary in row-major order (the last listed axis fastest). Returns a `md::tensor` view. A raw mdspan carries no memory space, so this tags the result as a host `view`; the `md::tensor` overloads below preserve the source's space.

---

### peel_at

```cpp
template<long... Axes, class T, class E, class L, own O> auto peel_at(tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i)
```

---

### peel_at

```cpp
template<long... Axes, class T, class E, class L, own O> auto peel_at(const tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i)
```

---

### peel

```cpp
template<long... Axes, class T, class E, class L, own O> auto peel(tensor< T, E, L, O > & t)
```

Build a range of sub-views by peeling `Axes...` of `t`.

Non-const `t` yields mutable peel; const `t` yields read-only peel.

---

### peel

```cpp
template<long... Axes, class T, class E, class L, own O> auto peel(const tensor< T, E, L, O > & t)
```

---

### peel_of

```cpp
template<size_t... Axes, class MD> peel_range< MD, own::view, Axes... > peel_of(const MD & m)
```

Build a range of sub-views over a raw mdspan.

---

### peel_front

```cpp
template<long N, class T, class E, class L, own O> auto peel_front(tensor< T, E, L, O > & t)
```

Peel the FIRST `N` axes -> a range of sub-views over the rest — the runtime-batch-rank half of `(*batch, *spatial, C)`.

`N` is **signed**: `peel_front<3>` peels 3 leading dims; `peel_front<-1>` keeps the last axis (peels all but it), so negative = "keep the last |N|".

---

### peel_front

```cpp
template<long N, class T, class E, class L, own O> auto peel_front(const tensor< T, E, L, O > & t)
```

---

### peel_front_at

```cpp
template<long N, class T, class E, class L, own O> auto peel_front_at(tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i)
```

The `i`-th sub-view obtained by peeling the first `N` axes (grid-stride style).

---

### peel_front_at

```cpp
template<long N, class T, class E, class L, own O> auto peel_front_at(const tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i)
```

---

### operator+

```cpp
template<class S, class T, class E, class L, own O, enable_if_t< is_arithmetic< S >::value, int > = 0> auto operator+(S s, const tensor< T, E, L, O > & a)
```

---

### operator*

```cpp
template<class S, class T, class E, class L, own O, enable_if_t< is_arithmetic< S >::value, int > = 0> auto operator*(S s, const tensor< T, E, L, O > & a)
```

---

### operator-

```cpp
template<class S, class T, class E, class L, own O, enable_if_t< is_arithmetic< S >::value, int > = 0> auto operator-(S s, const tensor< T, E, L, O > & a)
```

---

### operator/

```cpp
template<class S, class T, class E, class L, own O, enable_if_t< is_arithmetic< S >::value, int > = 0> auto operator/(S s, const tensor< T, E, L, O > & a)
```

---

### operator-

```cpp
template<class T, class E, class L, own O> auto operator-(const tensor< T, E, L, O > & a)
```

---

### operator~

```cpp
template<class T, class E, class L, own O, enable_if_t< is_integral< T >::value, int > = 0> auto operator~(const tensor< T, E, L, O > & a)
```

---

### sum

```cpp
template<class Acc = void, class T, class E, class L, own O> auto sum(const tensor< T, E, L, O > & a)
```

Sum of all elements (empty -> 0).

Accumulates in the reduce type (`double` for small floats), result cast to `T`; `sum<Acc>(a)` returns `Acc`.

---

### prod

```cpp
template<class Acc = void, class T, class E, class L, own O> auto prod(const tensor< T, E, L, O > & a)
```

Product of all elements (empty -> 1).

Accumulates in the reduce type, result cast to `T`; `prod<Acc>(a)` returns `Acc`.

---

### max

```cpp
template<class Acc = void, class T, class E, class L, own O> auto max(const tensor< T, E, L, O > & a)
```

Maximum element.

Requires a non-empty tensor. Result type `T` (`max<Acc>(a)` returns `Acc`).

---

### min

```cpp
template<class Acc = void, class T, class E, class L, own O> auto min(const tensor< T, E, L, O > & a)
```

Minimum element.

Requires a non-empty tensor. Result type `T` (`min<Acc>(a)` returns `Acc`).

---

### mean

```cpp
template<long... Axes, class T, class E, class L, own O, class R = reduce_type_t<T>, enable_if_t<(sizeof...(Axes) > 0) &&_md::reduced_extents< E, Axes... >::rank_dynamic()==0, int > = 0> auto mean(const tensor< T, E, L, O > & a)
```

Mean over the named axes -> a lower-rank tensor (sum / reduced count).

Accumulates in the reduce type, result cast to `T`; `mean<Acc, Axes...>(a)` makes `Acc` both the accumulator and result type.

---

### dot

```cpp
template<class Acc = void, class Ta, class Ea, class La, own Oa, class Tb, class Eb, class Lb, own Ob> auto dot(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b)
```

Inner product over matching extents.

Accumulates in the reduce type of the promoted element type (`double` for small floats), result cast to `promote(Ta,Tb)`; `dot<Acc>(a, b)` returns `Acc`.

---

### allclose

```cpp
template<class Ta, class Ea, class La, own Oa, class Tb, class Eb, class Lb, own Ob> bool allclose(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b, double rtol = 1e-5, double atol = 1e-8)
```

True if every element satisfies `|a-b| <= atol + rtol*|b|` (numpy `allclose`; broadcasts, computes in the compute type).

---

### minimum

```cpp
template<class Ta, class Ea, class La, own Oa, class Tb, class Eb, class Lb, own Ob> auto minimum(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b)
```

---

### maximum

```cpp
template<class Ta, class Ea, class La, own Oa, class Tb, class Eb, class Lb, own Ob> auto maximum(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b)
```

---

### minimum

```cpp
template<class T, class E, class L, own O, class S, enable_if_t< is_arithmetic< S >::value, int > = 0> auto minimum(const tensor< T, E, L, O > & a, S s)
```

---

### maximum

```cpp
template<class T, class E, class L, own O, class S, enable_if_t< is_arithmetic< S >::value, int > = 0> auto maximum(const tensor< T, E, L, O > & a, S s)
```

---

### clamp

```cpp
template<class T, class E, class L, own O> auto clamp(const tensor< T, E, L, O > & a, T lo, T hi)
```

`clamp(a, lo, hi)` -> a new tensor with each element clamped.

---

### mean

```cpp
template<class Acc = void, class T, class E, class L, own O> auto mean(const tensor< T, E, L, O > & a)
```

Arithmetic mean of all elements.

Accumulates in the reduce type (`double` for small floats), result cast to `T`; `mean<Acc>(a)` makes `Acc` both the accumulator and the result type.

---

### own_is_owning

`constexpr` `noexcept`

```cpp
constexpr constexpr bool own_is_owning(own o) noexcept
```

Whether the mode owns (and therefore allocates) its storage.

---

### own_is_view

`constexpr` `noexcept`

```cpp
constexpr constexpr bool own_is_view(own o) noexcept
```

Whether the mode is a non-owning view (host `view` or `gpu_view`) — the pointer-wrapping modes (as opposed to `stack`'s inline array).

---

### own_is_device

`constexpr` `noexcept`

```cpp
constexpr constexpr bool own_is_device(own o) noexcept
```

Whether the storage lives in device (GPU) memory (owning or view).

---

### own_is_host_accessible

`constexpr` `noexcept`

```cpp
constexpr constexpr bool own_is_host_accessible(own o) noexcept
```

Whether the storage is dereferenceable from the host.

---

### own_view_of

`constexpr` `noexcept`

```cpp
constexpr constexpr own own_view_of(own o) noexcept
```

The non-owning VIEW kind that preserves a source's memory space: a device source (`gpu`/`gpu_view`) -> `gpu_view`, anything else -> `view`.

Every view-producing op (slice / permute / peel / reshape / at) tags its result with this so a device view is never mistaken for a host one.

---

### as_tensor

```cpp
template<own OW = own::view, class MD> tensor< typename MD::element_type, typename MD::extents_type, typename MD::layout_type, OW > as_tensor(const MD & m)
```

Wrap any `cuda::std::mdspan` (e.g.

a `submdspan` result) as a non-owning `md::tensor` view, so the tensor API applies to it.

---

### fetch_add

`noexcept`

```cpp
template<class T> void fetch_add(T * p, T v) noexcept
```

Accumulate `v` into `*p`, atomic **on the device only**.

The scatter/"push" write: on the device many threads accumulate into overlapping outputs, which a plain `+=` would race. Device -> `atomicAdd` (`double` needs sm_60+, `__half` sm_70+; not all integer widths have an overload — that surfaces as an nvcc error at instantiation). Use via `t.add_at(v, i...)` / `t.add_<true>(...)`.

WARNING: on the **host** this is a plain `*p += v` — NOT atomic. A push kernel parallelised with std::thread over overlapping outputs races; guard those writes yourself (per-thread partials, a mutex, or std::atomic_ref).

---

### wrap

```cpp
template<class Layout = layout_right, class T, class Shape> tensor< T, Shape, Layout, own::view > wrap(T * p, Shape e)
```

Wrap `p` as a non-owning view with a contiguous layout (default C-order).

This is the factory; the `view<T,E>` alias is the type it produces, and the member `t.view()` re-views an existing tensor.

---

### wrap

```cpp
template<class T, class Shape> tensor< T, Shape, layout_stride, own::view > wrap(T * p, Shape e, array< typename Shape::index_type, Shape::rank()> st)
```

Wrap `p` as a non-owning view with explicit **runtime strides** (a `layout_stride` view).

Pass one stride per dimension — an `array` or a braced list — in ELEMENTS; strides may be negative (a reversed view).

`wrap(p, shape<2,3>{}, {3, 1})` is the row-major view; `{1, 2}` the column-major one. For strides known at compile time pass a `strides<S...>{}` instead (overload below) so they fold into the type.

---

### wrap

```cpp
template<int64_t... Strides, class T, class Shape> tensor< T, Shape, strides< Strides... >, own::view > wrap(T * p, Shape e, strides< Strides... >)
```

Wrap `p` as a non-owning view with per-dimension **compile-time strides** (may be negative): pass a `strides<S...>{}` as the third argument.

`wrap(p, shape<3,3>{}, strides<4,1>{})` folds the strides into the type (`strides<S...>` layout, EBO). Every stride must be a compile-time value — a `strides<...>` tag is a *stateless* layout, so it cannot carry runtime strides. For a **mix** of static and runtime strides, use the template form below; for all-runtime strides the `{s...}` overload above (a `layout_stride` view) is simplest.

---

### wrap

```cpp
template<int64_t S0, int64_t... Srest, class T, class Shape> tensor< T, Shape, strides< S0, Srest... >, own::view > wrap(T * p, Shape e, array< typename Shape::index_type, strides< S0, Srest... >::ndyn()> dyn)
```

Wrap `p` with a **mix of static and runtime strides** — the exact analogue of `shape<-1,2,3,-1>{d0,d1}` for strides.

Give the per-dim pattern as template args (a compile-time stride, or `dynamic_stride` for a runtime one) and the runtime strides for the `dynamic_stride` slots as a braced list, in order: 
```
wrap<dynamic_stride, 1>(ptr, shape<3,3>{}, {4});   // outer=4 (runtime), inner=1 (folds)
wrap<dynamic_stride, dynamic_stride>(ptr, sh, {4,1}); // both runtime (a strides<> layout)
```
 The static slots fold into the type; only the runtime ones are stored.

---

### make_view

```cpp
template<class Layout = layout_right, class T, class Shape> auto make_view(T * p, Shape e)
```

`make_view<L>(ptr, extents)` — a non-owning view (alias of `wrap`).

---

### make_local

```cpp
template<class T = float, class Layout = layout_right, class Shape> auto make_local(Shape = Shape{})
```

`make_local<T>(extents)` — a stack-owned tensor (static shape).

`T` defaults to `float` (numpy's default float dtype).

---

### make_heap

```cpp
template<class T = float, class Layout = layout_right, class Shape> auto make_heap(Shape e)
```

`make_heap<T>(extents)` — a heap-owned tensor (host, move-only).

`T` defaults to `float`.

---

### full

```cpp
template<class T = void, class Layout = layout_right, class Shape, class V, class ET = conditional_t<is_same<T, void>::value, V, T>, enable_if_t< Shape::rank_dynamic()==0, int > = 0> auto full(Shape, V v)
```

`full(extents, v)` — a new tensor filled with `v`.

The element type defaults to the **value's** type (numpy/pytorch: `full(s, 3)` is int, `full(s, 3.0)` is float); pass `full<T>(...)` to override. Unlike the value-less `zeros`/`ones` (which default to `float`), there is a value here to infer from, so we do.

---

### zeros

```cpp
template<class T = float, class Layout = layout_right, class Shape, enable_if_t< Shape::rank_dynamic()==0, int > = 0> auto zeros(Shape e)
```

`zeros<T>(extents)` / `ones<T>(extents)` — a new tensor of 0s / 1s.

`T` defaults to `float`. Static shape -> stack (host+device); dynamic -> heap (host only). The annotation is split so it matches the overload `full` resolves to.

---

### ones

```cpp
template<class T = float, class Layout = layout_right, class Shape, enable_if_t< Shape::rank_dynamic()==0, int > = 0> auto ones(Shape e)
```

---

### arange

```cpp
template<class T = int64_t> auto arange(long n)
```

`arange<T>(n)` — a 1-D tensor `[0, 1, ..., n-1]` (heap, host).

`T` defaults to `int64_t` (an integer range, like numpy `[arange(n)](#arange)`).

---

### arange

```cpp
template<class T = int64_t, long N> auto arange()
```

Static `arange<T, N>()` — a stack `[0..N-1]` (host+device, folds).

---

### arange

```cpp
template<class T = int64_t, class V, V N> auto arange(integral_constant< V, N >)
```

`arange<T>(Int<N>())` — the static form spelled with a static integer.

## Variables

| Return | Name | Description |
|--------|------|-------------|
| `constexpr full_extent_t` | [`all`](#all) `constexpr` | Keep-this-axis marker for slicing (an alias of `full_extent`). |
| `constexpr copy_meta_t` | [`copy_meta`](#copy_meta) `constexpr` |  |
| `constexpr none_t` | [`none`](#none) `constexpr` |  |
| `constexpr ellipsis_t` | [`ellipsis`](#ellipsis) `constexpr` |  |
| `constexpr int64_t` | [`dynamic_stride`](#dynamic_stride) `constexpr` | Per-dimension dynamic-stride sentinel. |
| `constexpr bool` | [`is_view_v`](#is_view_v) `constexpr` | Compile-time memory-space traits (SFINAE-friendly free forms of the tensor's `is_view`/`is_device`/… members): `is_view_v<decltype(x)>`. |
| `constexpr bool` | [`is_owning_v`](#is_owning_v) `constexpr` |  |
| `constexpr bool` | [`is_device_v`](#is_device_v) `constexpr` |  |
| `constexpr bool` | [`is_host_accessible_v`](#is_host_accessible_v) `constexpr` |  |

---

### all

`constexpr`

```cpp
constexpr full_extent_t all {}
```

Keep-this-axis marker for slicing (an alias of `full_extent`).

---

### copy_meta

`constexpr`

```cpp
constexpr copy_meta_t copy_meta {}
```

---

### none

`constexpr`

```cpp
constexpr none_t none {}
```

---

### ellipsis

`constexpr`

```cpp
constexpr ellipsis_t ellipsis {}
```

---

### dynamic_stride

`constexpr`

```cpp
constexpr int64_t dynamic_stride = numeric_limits<int64_t>()
```

Per-dimension dynamic-stride sentinel.

Strides are **signed**: a negative stride is a legitimate value (reversed / flipped views, and DLPack tensors carry them). So — unlike `shape<...>`, where `-1` marks a dynamic extent — we cannot use `-1` to mean "runtime" for a stride. Instead a reserved out-of-band value (`INT64_MIN`) marks a dynamic stride, leaving every ordinary stride (including negatives) expressible.

---

### is_view_v

`constexpr`

```cpp
constexpr bool is_view_v = Tn::is_view
```

Compile-time memory-space traits (SFINAE-friendly free forms of the tensor's `is_view`/`is_device`/… members): `is_view_v<decltype(x)>`.

---

### is_owning_v

`constexpr`

```cpp
constexpr bool is_owning_v = Tn::is_owning
```

---

### is_device_v

`constexpr`

```cpp
constexpr bool is_device_v = Tn::is_device
```

---

### is_host_accessible_v

`constexpr`

```cpp
constexpr bool is_host_accessible_v = Tn::is_host_accessible
```



## anyrank

```cpp
#include <dynamic.h>
```

```cpp
template<class T, class offset_t = int64_t, class Meta = _meta_store<offset_t, TNY_MAX_RANK>>
struct anyrank
```

Defined in include/teeny/dynamic.h:61

A rank-erased tensor for the host/ndarray dispatch boundary.

Holds a data pointer, a runtime `ndim`, and 1-D `shape`/`stride` tensors (`Meta`). `as_anyrank(...)`**wraps** the caller's arrays with no copy (a `_meta_view` store, HOST only) — the default; `as_anyrank(..., copy_meta)` COPIES them into an INLINE `TNY_MAX_RANK` store, so the carrier is trivially copyable and passes into a CUDA kernel by value (`device_passable == true`).

You do NOT compute on it — it is a *doorway*, not a room. Turn it into a statically-typed view at the boundary and compute on that:

* `fixed<R>()` — force a known total rank R.

* `dispatch_rank(...)` — pick R from the runtime `ndim`.

* `peel_front<Sr>()` — the batch idiom: peel the runtime number of leading batch dims, keep the trailing `Sr` "interesting" dims STATIC. One kernel per Sr.

Deliberately no `add_`/`mul_`/etc.: a runtime-rank arithmetic path would loop over `ndim` (killing folding) or dispatch to every rank (the bloat `peel_front<Sr>` avoids). Do host-side math on a `fixed<R>()`/`peel_front<Sr>()` view instead.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`data`](#data) | `variable` | Declared here |
| [`shape`](#shape-1) | `variable` | Declared here |
| [`stride`](#stride) | `variable` | Declared here |
| [`ndim`](#ndim) | `variable` | Declared here |
| [`size`](#size-1) | `function` | Declared here |
| [`step`](#step) | `function` | Declared here |
| [`fixed`](#fixed) | `function` | Declared here |
| [`peel_front_at`](#peel_front_at-2) | `function` | Declared here |
| [`peel_front`](#peel_front-2) | `function` | Declared here |
| [`max_rank`](#max_rank) | `variable` | Declared here |
| [`device_passable`](#device_passable) | `variable` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `T *` | [`data`](#data)  |  |
| `Meta` | [`shape`](#shape-1)  |  |
| `Meta` | [`stride`](#stride)  |  |
| `int` | [`ndim`](#ndim)  |  |

---

#### data

```cpp
T * data = nullptr
```

Defined in include/teeny/dynamic.h:62

---

#### shape

```cpp
Meta shape {}
```

Defined in include/teeny/dynamic.h:63

---

#### stride

```cpp
Meta stride {}
```

Defined in include/teeny/dynamic.h:64

---

#### ndim

```cpp
int ndim = 0
```

Defined in include/teeny/dynamic.h:65

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `offset_t` | [`size`](#size-1) `const` `inline` `noexcept` |  |
| `offset_t` | [`step`](#step) `const` `inline` `noexcept` |  |
| `dyn_tensor< T, offset_t, R >` | [`fixed`](#fixed) `const` `inline` | View this tensor as a fixed rank `R` (requires `ndim == R`). |
| `auto` | [`peel_front_at`](#peel_front_at-2) `const` `inline` | The `lin`-th sub-view keeping the last `\|N\|` axes static (grid-stride style). |
| `anyrank_front< T, offset_t, Meta, static_cast< size_t >(N< 0 ? -N :0)>` | [`peel_front`](#peel_front-2) `const` `inline` | Peel the leading batch axes -> an iterable of fixed-rank-`\|N\|` sub-views (range-for, `[size()](#size-1)`, `operator[]`). |

---

#### size

`const` `inline` `noexcept`

```cpp
inline offset_t size(int i) const noexcept
```

Defined in include/teeny/dynamic.h:89

---

#### step

`const` `inline` `noexcept`

```cpp
inline offset_t step(int i) const noexcept
```

Defined in include/teeny/dynamic.h:90

---

#### fixed

`const` `inline`

```cpp
template<size_t R> inline dyn_tensor< T, offset_t, R > fixed() const
```

Defined in include/teeny/dynamic.h:94

View this tensor as a fixed rank `R` (requires `ndim == R`).

---

#### peel_front_at

`const` `inline`

```cpp
template<long N> inline auto peel_front_at(offset_t lin) const
```

Defined in include/teeny/dynamic.h:125

The `lin`-th sub-view keeping the last `|N|` axes static (grid-stride style).

`N` is **negative** — matching the tensor's `peel_front`, negative means "keep the last |N| dims". (A positive front-count would leave a runtime rank, which can't be a static view — hence the assert.) Follow with `recast<shape<-1,...>>()`.

---

#### peel_front

`const` `inline`

```cpp
template<long N> inline anyrank_front< T, offset_t, Meta, static_cast< size_t >(N< 0 ? -N :0)> peel_front() const
```

Defined in include/teeny/dynamic.h:136

Peel the leading batch axes -> an iterable of fixed-rank-`|N|` sub-views (range-for, `[size()](#size-1)`, `operator[]`).

The `(*batch, *spatial, C)` boundary with `|N| = spatial + channels`: one kernel instantiation for `|N|`, not one per total rank. `N` is negative (keep the last |N| dims), as on the tensor.

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr size_t` | [`max_rank`](#max_rank) `static` `constexpr` |  |
| `constexpr bool` | [`device_passable`](#device_passable) `static` `constexpr` |  |

---

#### max_rank

`static` `constexpr`

```cpp
constexpr size_t max_rank =
        Meta::extents_type::static_extent(0) != dynamic_extent
            ? Meta::extents_type::static_extent(0) : size_t()
```

Defined in include/teeny/dynamic.h:69

---

#### device_passable

`static` `constexpr`

```cpp
constexpr bool device_passable =
        (Meta::extents_type::static_extent(0) != dynamic_extent)
```

Defined in include/teeny/dynamic.h:75



## anyrank_front

```cpp
#include <dynamic.h>
```

```cpp
template<class T, class offset_t, class Meta, size_t Sr>
struct anyrank_front
```

Defined in include/teeny/dynamic.h:144

A range of fixed-rank-`Sr` sub-views over an `anyrank`'s batch axes.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`src`](#src) | `variable` | Declared here |
| [`size`](#size-2) | `function` | Declared here |
| [`operator[]`](#operator-6) | `function` | Declared here |
| [`begin`](#begin) | `function` | Declared here |
| [`end`](#end) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `anyrank< T, offset_t, Meta >` | [`src`](#src)  |  |

---

#### src

```cpp
anyrank< T, offset_t, Meta > src
```

Defined in include/teeny/dynamic.h:145

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `offset_t` | [`size`](#size-2) `const` `inline` `noexcept` |  |
| `auto` | [`operator[]`](#operator-6) `const` `inline` |  |
| `iterator` | [`begin`](#begin) `const` `inline` |  |
| `iterator` | [`end`](#end) `const` `inline` |  |

---

#### size

`const` `inline` `noexcept`

```cpp
inline offset_t size() const noexcept
```

Defined in include/teeny/dynamic.h:147

---

#### operator[]

`const` `inline`

```cpp
inline auto operator[](offset_t i) const
```

Defined in include/teeny/dynamic.h:152

---

#### begin

`const` `inline`

```cpp
inline iterator begin() const
```

Defined in include/teeny/dynamic.h:160

---

#### end

`const` `inline`

```cpp
inline iterator end() const
```

Defined in include/teeny/dynamic.h:161



## iterator

```cpp
#include <dynamic.h>
```

```cpp
struct iterator
```

Defined in include/teeny/dynamic.h:154

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`r`](#r) | `variable` | Declared here |
| [`i`](#i) | `variable` | Declared here |
| [`operator*`](#operator-7) | `function` | Declared here |
| [`operator++`](#operator-8) | `function` | Declared here |
| [`operator!=`](#operator-9) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `anyrank_front` | [`r`](#r)  |  |
| `offset_t` | [`i`](#i)  |  |

---

#### r

```cpp
anyrank_front r
```

Defined in include/teeny/dynamic.h:155

---

#### i

```cpp
offset_t i
```

Defined in include/teeny/dynamic.h:155

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `auto` | [`operator*`](#operator-7) `const` `inline` |  |
| `iterator &` | [`operator++`](#operator-8) `inline` |  |
| `bool` | [`operator!=`](#operator-9) `const` `inline` |  |

---

#### operator*

`const` `inline`

```cpp
inline auto operator*() const
```

Defined in include/teeny/dynamic.h:156

---

#### operator++

`inline`

```cpp
inline iterator & operator++()
```

Defined in include/teeny/dynamic.h:157

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const iterator & o) const
```

Defined in include/teeny/dynamic.h:158



## compute_type

```cpp
#include <half.h>
```

```cpp
template<class T>
struct compute_type
```

Defined in include/teeny/half.h:136

The type math should ACCUMULATE / compute in for element type `T`.

Half types compute in `float` (16-bit accumulation loses precision fast, the usual mixed-precision rule: accumulate wider than you store — and it lets the engines avoid depending on native half host operators). Everything else computes in itself.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`type`](#type) | `typedef` | Declared here |

### Public Types

| Name | Description |
|------|-------------|
| [`type`](#type)  |  |

---

#### type

```cpp
using type = T
```

Defined in include/teeny/half.h:136



## compute_type< bfloat16 >

```cpp
#include <half.h>
```

```cpp
struct compute_type< bfloat16 >
```

Defined in include/teeny/half.h:138

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`type`](#type-1) | `typedef` | Declared here |

### Public Types

| Name | Description |
|------|-------------|
| [`type`](#type-1)  |  |

---

#### type

```cpp
using type = float
```

Defined in include/teeny/half.h:138



## compute_type< half >

```cpp
#include <half.h>
```

```cpp
struct compute_type< half >
```

Defined in include/teeny/half.h:137

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`type`](#type-2) | `typedef` | Declared here |

### Public Types

| Name | Description |
|------|-------------|
| [`type`](#type-2)  |  |

---

#### type

```cpp
using type = float
```

Defined in include/teeny/half.h:137



## copy_meta_t

```cpp
#include <dynamic.h>
```

```cpp
struct copy_meta_t
```

Defined in include/teeny/dynamic.h:35

Tag for `as_anyrank(..., copy_meta)`: COPY shape/stride into an inline, device-passable store instead of wrapping the caller's arrays.

Named `copy_meta`, not `copy`: a bare `copy` variable in `tny` would, under `using namespace tny`, shadow an unqualified `std::copy(...)` call (finding a variable suppresses ADL) — a nasty surprise.



## cpp_alloc

```cpp
#include <storage.h>
```

```cpp
struct cpp_alloc
```

Defined in include/teeny/storage.h:59

Host allocator using C++ `new[]` / `delete[]`.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`allocate`](#allocate) | `function` | Declared here |
| [`deallocate`](#deallocate) | `function` | Declared here |

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `T *` | [`allocate`](#allocate) `static` `inline` |  |
| `void` | [`deallocate`](#deallocate) `static` `inline` |  |

---

#### allocate

`static` `inline`

```cpp
template<class T> static inline T * allocate(size_t n)
```

Defined in include/teeny/storage.h:60

---

#### deallocate

`static` `inline`

```cpp
template<class T> static inline void deallocate(T * p)
```

Defined in include/teeny/storage.h:61



## cuda_gpu_alloc

```cpp
#include <cuda.h>
```

```cpp
struct cuda_gpu_alloc
```

Defined in include/teeny/cuda.h:29

Device (GPU) memory (`cudaMalloc`).

Not host-dereferenceable.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`allocate`](#allocate-1) | `function` | Declared here |
| [`deallocate`](#deallocate-1) | `function` | Declared here |

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `T *` | [`allocate`](#allocate-1) `static` `inline` |  |
| `void` | [`deallocate`](#deallocate-1) `static` `inline` |  |

---

#### allocate

`static` `inline`

```cpp
template<class T> static inline T * allocate(size_t n)
```

Defined in include/teeny/cuda.h:30

---

#### deallocate

`static` `inline`

```cpp
template<class T> static inline void deallocate(T * p)
```

Defined in include/teeny/cuda.h:33



## cuda_mapped_alloc

```cpp
#include <cuda.h>
```

```cpp
struct cuda_mapped_alloc
```

Defined in include/teeny/cuda.h:45

Page-locked + device-mapped (zero-copy) host memory (`cudaHostAlloc`).

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`allocate`](#allocate-2) | `function` | Declared here |
| [`deallocate`](#deallocate-2) | `function` | Declared here |

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `T *` | [`allocate`](#allocate-2) `static` `inline` |  |
| `void` | [`deallocate`](#deallocate-2) `static` `inline` |  |

---

#### allocate

`static` `inline`

```cpp
template<class T> static inline T * allocate(size_t n)
```

Defined in include/teeny/cuda.h:46

---

#### deallocate

`static` `inline`

```cpp
template<class T> static inline void deallocate(T * p)
```

Defined in include/teeny/cuda.h:49



## cuda_pinned_alloc

```cpp
#include <cuda.h>
```

```cpp
struct cuda_pinned_alloc
```

Defined in include/teeny/cuda.h:37

Page-locked ("pinned") host memory (`cudaMallocHost`).

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`allocate`](#allocate-3) | `function` | Declared here |
| [`deallocate`](#deallocate-3) | `function` | Declared here |

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `T *` | [`allocate`](#allocate-3) `static` `inline` |  |
| `void` | [`deallocate`](#deallocate-3) `static` `inline` |  |

---

#### allocate

`static` `inline`

```cpp
template<class T> static inline T * allocate(size_t n)
```

Defined in include/teeny/cuda.h:38

---

#### deallocate

`static` `inline`

```cpp
template<class T> static inline void deallocate(T * p)
```

Defined in include/teeny/cuda.h:41



## ellipsis_t

```cpp
#include <indexing.h>
```

```cpp
struct ellipsis_t
```

Defined in include/teeny/indexing.h:50

Ellipsis sentinel — teeny's `...` (python `a[..., 0]` / numpy `Ellipsis`).

In an index expression it stands for "as many `all` as it takes to fill the
rank": `t(1, ellipsis, 2)` on a rank-5 tensor is `t(1, all, all, all, 2)`. At most one ellipsis per call. It expands to `rank - (#other args)` copies of `all` (which may be zero), then the call proceeds as usual — so if what remains is all integers you get an element `T&`, otherwise a view.



## none_t

```cpp
#include <indexing.h>
```

```cpp
struct none_t
```

Defined in include/teeny/indexing.h:40

Open-ended slice sentinel — teeny's `None` (python `a[:n]` / `a[m:]`).

`slice(none, n)` starts at 0, `slice(m, none)` runs to the end, and `slice(none, none)`**folds** to `full_extent` — so `all == slice(none, none)`, keeping the axis and its static extent (`all` is built from it). Combined with runtime bounds it resolves at run time, so the one sentinel covers both.



## owning_storage

```cpp
#include <storage.h>
```

```cpp
template<class T, class Alloc>
struct owning_storage
```

Defined in include/teeny/storage.h:69

Generic owning storage (move-only, no ref-counting), parameterised by an allocator policy.

Shared by all owning `own` modes.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p) | `variable` | Declared here |
| [`owning_storage`](#owning_storage-1) | `function` | Declared here |
| [`owning_storage`](#owning_storage-2) | `function` | Declared here |
| [`owning_storage`](#owning_storage-3) | `function` | Declared here |
| [`owning_storage`](#owning_storage-4) | `function` | Declared here |
| [`data`](#data-1) | `function` | Declared here |
| [`data`](#data-2) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `T *` | [`p`](#p)  |  |

---

#### p

```cpp
T * p = nullptr
```

Defined in include/teeny/storage.h:70

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
|  | [`owning_storage`](#owning_storage-1)  | Defaulted constructor. |
|  | [`owning_storage`](#owning_storage-2) `inline` `explicit` |  |
|  | [`owning_storage`](#owning_storage-3)  | Deleted constructor. |
|  | [`owning_storage`](#owning_storage-4) `inline` `noexcept` |  |
| `T *` | [`data`](#data-1) `inline` `noexcept` |  |
| `const T *` | [`data`](#data-2) `const` `inline` `noexcept` |  |

---

#### owning_storage

```cpp
owning_storage() = default
```

Defined in include/teeny/storage.h:71

Defaulted constructor.

---

#### owning_storage

`inline` `explicit`

```cpp
inline explicit owning_storage(size_t n)
```

Defined in include/teeny/storage.h:72

---

#### owning_storage

```cpp
owning_storage(const owning_storage &) = delete
```

Defined in include/teeny/storage.h:73

Deleted constructor.

---

#### owning_storage

`inline` `noexcept`

```cpp
inline owning_storage(owning_storage && o) noexcept
```

Defined in include/teeny/storage.h:75

---

#### data

`inline` `noexcept`

```cpp
inline T * data() noexcept
```

Defined in include/teeny/storage.h:81

---

#### data

`const` `inline` `noexcept`

```cpp
inline const T * data() const noexcept
```

Defined in include/teeny/storage.h:82



## peel_range

```cpp
#include <iterate.h>
```

```cpp
template<class MD, own OW, size_t... Axes>
struct peel_range
```

Defined in include/teeny/iterate.h:111

A range of sub-views obtained by peeling `Axes...`.

Supports `[size()](#size-3)`, `operator[]`, and range-for.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`src`](#src-1) | `variable` | Declared here |
| [`size`](#size-3) | `function` | Declared here |
| [`operator[]`](#operator-10) | `function` | Declared here |
| [`begin`](#begin-1) | `function` | Declared here |
| [`end`](#end-1) | `function` | Declared here |
| [`index_type`](#index_type) | `typedef` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `MD` | [`src`](#src-1)  |  |

---

#### src

```cpp
MD src
```

Defined in include/teeny/iterate.h:113

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `index_type` | [`size`](#size-3) `const` `inline` `noexcept` |  |
| `auto` | [`operator[]`](#operator-10) `const` `inline` |  |
| `iterator` | [`begin`](#begin-1) `const` `inline` |  |
| `iterator` | [`end`](#end-1) `const` `inline` |  |

---

#### size

`const` `inline` `noexcept`

```cpp
inline index_type size() const noexcept
```

Defined in include/teeny/iterate.h:115

---

#### operator[]

`const` `inline`

```cpp
inline auto operator[](index_type i) const
```

Defined in include/teeny/iterate.h:121

---

#### begin

`const` `inline`

```cpp
inline iterator begin() const
```

Defined in include/teeny/iterate.h:131

---

#### end

`const` `inline`

```cpp
inline iterator end() const
```

Defined in include/teeny/iterate.h:132

### Public Types

| Name | Description |
|------|-------------|
| [`index_type`](#index_type)  |  |

---

#### index_type

```cpp
using index_type = typename MD::index_type
```

Defined in include/teeny/iterate.h:112



## iterator

```cpp
#include <iterate.h>
```

```cpp
struct iterator
```

Defined in include/teeny/iterate.h:123

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`r`](#r-1) | `variable` | Declared here |
| [`i`](#i-1) | `variable` | Declared here |
| [`operator*`](#operator-11) | `function` | Declared here |
| [`operator++`](#operator-12) | `function` | Declared here |
| [`operator!=`](#operator-13) | `function` | Declared here |
| [`operator==`](#operator-14) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `peel_range` | [`r`](#r-1)  |  |
| `index_type` | [`i`](#i-1)  |  |

---

#### r

```cpp
peel_range r
```

Defined in include/teeny/iterate.h:124

---

#### i

```cpp
index_type i
```

Defined in include/teeny/iterate.h:125

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `auto` | [`operator*`](#operator-11) `const` `inline` |  |
| `iterator &` | [`operator++`](#operator-12) `inline` |  |
| `bool` | [`operator!=`](#operator-13) `const` `inline` |  |
| `bool` | [`operator==`](#operator-14) `const` `inline` |  |

---

#### operator*

`const` `inline`

```cpp
inline auto operator*() const
```

Defined in include/teeny/iterate.h:126

---

#### operator++

`inline`

```cpp
inline iterator & operator++()
```

Defined in include/teeny/iterate.h:127

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const iterator & o) const
```

Defined in include/teeny/iterate.h:128

---

#### operator==

`const` `inline`

```cpp
inline bool operator==(const iterator & o) const
```

Defined in include/teeny/iterate.h:129



## storage

```cpp
template<class T, own O, size_t N>
struct storage
```

Defined in include/teeny/storage.h:90



## gpu, N >

```cpp
#include <cuda.h>
```

```cpp
template<class T, size_t N>
struct gpu, N >
```

Defined in include/teeny/cuda.h:57

> **Inherits:** [`owning_storage< T, cuda_gpu_alloc >`](#owning_storage)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p) | `variable` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-3) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-4) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |

### Inherited from [`owning_storage`](#owning_storage)

| Kind | Name | Description |
|------|------|-------------|
| `variable` | [`p`](#p)  |  |
| `function` | [`owning_storage`](#owning_storage-1)  | Defaulted constructor. |
| `function` | [`owning_storage`](#owning_storage-2) `inline` `explicit` |  |
| `function` | [`owning_storage`](#owning_storage-3)  | Deleted constructor. |
| `function` | [`owning_storage`](#owning_storage-4) `inline` `noexcept` |  |
| `function` | [`data`](#data-1) `inline` `noexcept` |  |
| `function` | [`data`](#data-2) `const` `inline` `noexcept` |  |



## gpu_view, N >

```cpp
#include <storage.h>
```

```cpp
template<class T, size_t N>
struct gpu_view, N >
```

Defined in include/teeny/storage.h:104

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p-1) | `variable` | Declared here |
| [`storage`](#storage-1) | `function` | Declared here |
| [`storage`](#storage-2) | `function` | Declared here |
| [`data`](#data-3) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `T *` | [`p`](#p-1)  |  |

---

#### p

```cpp
T * p = nullptr
```

Defined in include/teeny/storage.h:105

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
|  | [`storage`](#storage-1)  | Defaulted constructor. |
| `constexpr` | [`storage`](#storage-2) `inline` `constexpr` `noexcept` |  |
| `constexpr T *` | [`data`](#data-3) `const` `inline` `constexpr` `noexcept` |  |

---

#### storage

```cpp
storage() = default
```

Defined in include/teeny/storage.h:106

Defaulted constructor.

---

#### storage

`inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr storage(T * q) noexcept
```

Defined in include/teeny/storage.h:107

---

#### data

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr T * data() const noexcept
```

Defined in include/teeny/storage.h:108



## heap, N >

```cpp
#include <storage.h>
```

```cpp
template<class T, size_t N>
struct heap, N >
```

Defined in include/teeny/storage.h:121

> **Inherits:** [`owning_storage< T, cpp_alloc >`](#owning_storage)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p) | `variable` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-3) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-4) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |

### Inherited from [`owning_storage`](#owning_storage)

| Kind | Name | Description |
|------|------|-------------|
| `variable` | [`p`](#p)  |  |
| `function` | [`owning_storage`](#owning_storage-1)  | Defaulted constructor. |
| `function` | [`owning_storage`](#owning_storage-2) `inline` `explicit` |  |
| `function` | [`owning_storage`](#owning_storage-3)  | Deleted constructor. |
| `function` | [`owning_storage`](#owning_storage-4) `inline` `noexcept` |  |
| `function` | [`data`](#data-1) `inline` `noexcept` |  |
| `function` | [`data`](#data-2) `const` `inline` `noexcept` |  |



## mapped, N >

```cpp
#include <cuda.h>
```

```cpp
template<class T, size_t N>
struct mapped, N >
```

Defined in include/teeny/cuda.h:65

> **Inherits:** [`owning_storage< T, cuda_mapped_alloc >`](#owning_storage)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p) | `variable` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-3) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-4) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |

### Inherited from [`owning_storage`](#owning_storage)

| Kind | Name | Description |
|------|------|-------------|
| `variable` | [`p`](#p)  |  |
| `function` | [`owning_storage`](#owning_storage-1)  | Defaulted constructor. |
| `function` | [`owning_storage`](#owning_storage-2) `inline` `explicit` |  |
| `function` | [`owning_storage`](#owning_storage-3)  | Deleted constructor. |
| `function` | [`owning_storage`](#owning_storage-4) `inline` `noexcept` |  |
| `function` | [`data`](#data-1) `inline` `noexcept` |  |
| `function` | [`data`](#data-2) `const` `inline` `noexcept` |  |



## pinned, N >

```cpp
#include <cuda.h>
```

```cpp
template<class T, size_t N>
struct pinned, N >
```

Defined in include/teeny/cuda.h:61

> **Inherits:** [`owning_storage< T, cuda_pinned_alloc >`](#owning_storage)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p) | `variable` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-3) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-4) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |

### Inherited from [`owning_storage`](#owning_storage)

| Kind | Name | Description |
|------|------|-------------|
| `variable` | [`p`](#p)  |  |
| `function` | [`owning_storage`](#owning_storage-1)  | Defaulted constructor. |
| `function` | [`owning_storage`](#owning_storage-2) `inline` `explicit` |  |
| `function` | [`owning_storage`](#owning_storage-3)  | Deleted constructor. |
| `function` | [`owning_storage`](#owning_storage-4) `inline` `noexcept` |  |
| `function` | [`data`](#data-1) `inline` `noexcept` |  |
| `function` | [`data`](#data-2) `const` `inline` `noexcept` |  |



## stack, N >

```cpp
#include <storage.h>
```

```cpp
template<class T, size_t N>
struct stack, N >
```

Defined in include/teeny/storage.h:113

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`a`](#a) | `variable` | Declared here |
| [`data`](#data-4) | `function` | Declared here |
| [`data`](#data-5) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `array< T, N >` | [`a`](#a)  |  |

---

#### a

```cpp
array< T, N > a {}
```

Defined in include/teeny/storage.h:114

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `constexpr T *` | [`data`](#data-4) `inline` `constexpr` `noexcept` |  |
| `constexpr const T *` | [`data`](#data-5) `const` `inline` `constexpr` `noexcept` |  |

---

#### data

`inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr T * data() noexcept
```

Defined in include/teeny/storage.h:115

---

#### data

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const T * data() const noexcept
```

Defined in include/teeny/storage.h:116



## view, N >

```cpp
#include <storage.h>
```

```cpp
template<class T, size_t N>
struct view, N >
```

Defined in include/teeny/storage.h:94

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p-2) | `variable` | Declared here |
| [`storage`](#storage-3) | `function` | Declared here |
| [`storage`](#storage-4) | `function` | Declared here |
| [`data`](#data-6) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `T *` | [`p`](#p-2)  |  |

---

#### p

```cpp
T * p = nullptr
```

Defined in include/teeny/storage.h:95

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
|  | [`storage`](#storage-3)  | Defaulted constructor. |
| `constexpr` | [`storage`](#storage-4) `inline` `constexpr` `noexcept` |  |
| `constexpr T *` | [`data`](#data-6) `const` `inline` `constexpr` `noexcept` |  |

---

#### storage

```cpp
storage() = default
```

Defined in include/teeny/storage.h:96

Defaulted constructor.

---

#### storage

`inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr storage(T * q) noexcept
```

Defined in include/teeny/storage.h:97

---

#### data

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr T * data() const noexcept
```

Defined in include/teeny/storage.h:98



## storage_size

```cpp
#include <storage.h>
```

```cpp
template<class Mapping, bool Stack>
struct storage_size
```

Defined in include/teeny/storage.h:127

Storage element count for a stack tensor (0 for view/owning).

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`value`](#value) | `variable` | Declared here |

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr size_t` | [`value`](#value) `static` `constexpr` |  |

---

#### value

`static` `constexpr`

```cpp
constexpr size_t value = 0
```

Defined in include/teeny/storage.h:127



## storage_size< Mapping, true >

```cpp
#include <storage.h>
```

```cpp
template<class Mapping>
struct storage_size< Mapping, true >
```

Defined in include/teeny/storage.h:129

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`value`](#value-1) | `variable` | Declared here |

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr size_t` | [`value`](#value-1) `static` `constexpr` |  |

---

#### value

`static` `constexpr`

```cpp
constexpr size_t value =
        static_cast<size_t>(Mapping().required_span_size())
```

Defined in include/teeny/storage.h:130



## strides

```cpp
#include <layout.h>
```

```cpp
template<int64_t... S>
struct strides
```

Defined in include/teeny/layout.h:66

An mdspan layout policy with **per-dimension static or dynamic strides** — the stride analogue of `extents`/`shape`.

`layout_right`/`layout_left` give contiguous (extent-derived) strides; `layout_stride` stores every stride at run time. `strides<S...>` bakes the KNOWN strides into the type (folding to immediates) — **including negative strides** — while any dimension marked `dynamic_stride` is supplied at run time: 
```
tensor<float, shape<3,4>, strides<4,1>>(ptr);                    // static, folds
tensor<float, shape<3,4>, strides<-4,1>>(ptr);                   // reversed rows
tensor<float, shape<-1,4>, strides<dynamic_stride,1>>(ptr, {n}); // outer stride runtime
```
 When every stride is static the mapping is empty (EBO), so a stack tensor is still exactly `sizeof` its data. Only the *dynamic* strides are stored.

Note: CCCL's `submdspan` is only defined for the standard layouts, so it does not apply here — but teeny's own slicing/`slice_along`/`permute`/`flip`/ `peel` build their views by hand (no submdspan), so they all work on a strides<...> source and in fact fold their output strides the same way. And `required_span_size` assumes non-negative strides — negative strides are for VIEWS into existing storage, not owning allocation.

#### Template Parameters
* `S` One stride per dimension: a compile-time value (may be negative), or `dynamic_stride` for a runtime stride.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`N`](#n) | `variable` | Declared here |
| [`S_`](#s_) | `variable` | Declared here |
| [`ndyn`](#ndyn) | `function` | Declared here |
| [`all_static`](#all_static) | `function` | Declared here |
| [`slot`](#slot) | `function` | Declared here |

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr size_t` | [`N`](#n) `static` `constexpr` |  |
| `constexpr int64_t` | [`S_`](#s_) `static` `constexpr` |  |

---

#### N

`static` `constexpr`

```cpp
constexpr size_t N = sizeof...(S)
```

Defined in include/teeny/layout.h:67

---

#### S_

`static` `constexpr`

```cpp
constexpr int64_t S_ = { S... }
```

Defined in include/teeny/layout.h:68

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `constexpr size_t` | [`ndyn`](#ndyn) `static` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`all_static`](#all_static) `static` `inline` `constexpr` `noexcept` |  |
| `constexpr size_t` | [`slot`](#slot) `static` `inline` `constexpr` `noexcept` |  |

---

#### ndyn

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr size_t ndyn() noexcept
```

Defined in include/teeny/layout.h:70

---

#### all_static

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr bool all_static() noexcept
```

Defined in include/teeny/layout.h:73

---

#### slot

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr size_t slot(size_t r) noexcept
```

Defined in include/teeny/layout.h:75



## mapping

```cpp
#include <layout.h>
```

```cpp
template<class Shape>
struct mapping
```

Defined in include/teeny/layout.h:82

> **Inherits:** `ndyn()>`, `Shape`

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`mapping`](#mapping-1) | `function` | Declared here |
| [`mapping`](#mapping-2) | `function` | Declared here |
| [`mapping`](#mapping-3) | `function` | Declared here |
| [`extents`](#extents) | `function` | Declared here |
| [`stride`](#stride-1) | `function` | Declared here |
| [`operator()`](#operator-15) | `function` | Declared here |
| [`required_span_size`](#required_span_size) | `function` | Declared here |
| [`is_unique`](#is_unique) | `function` | Declared here |
| [`is_exhaustive`](#is_exhaustive) | `function` | Declared here |
| [`is_strided`](#is_strided) | `function` | Declared here |
| [`is_always_unique`](#is_always_unique) | `function` | Declared here |
| [`is_always_exhaustive`](#is_always_exhaustive) | `function` | Declared here |
| [`is_always_strided`](#is_always_strided) | `function` | Declared here |
| [`extents_type`](#extents_type) | `typedef` | Declared here |
| [`index_type`](#index_type-1) | `typedef` | Declared here |
| [`rank_type`](#rank_type) | `typedef` | Declared here |
| [`layout_type`](#layout_type) | `typedef` | Declared here |

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
|  | [`mapping`](#mapping-1)  | Defaulted constructor. |
| `constexpr` | [`mapping`](#mapping-2) `inline` `constexpr` | Fully-static strides: construct from extents only. |
| `constexpr` | [`mapping`](#mapping-3) `inline` `constexpr` | Mixed strides: extents + the runtime strides (dim order, dynamic ones only). |
| `constexpr const Shape &` | [`extents`](#extents) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr index_type` | [`stride`](#stride-1) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr index_type` | [`operator()`](#operator-15) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr index_type` | [`required_span_size`](#required_span_size) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_unique`](#is_unique) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_exhaustive`](#is_exhaustive) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_strided`](#is_strided) `const` `inline` `constexpr` `noexcept` |  |

---

#### mapping

```cpp
mapping() = default
```

Defined in include/teeny/layout.h:90

Defaulted constructor.

---

#### mapping

`inline` `constexpr`

```cpp
template<size_t M = strides::ndyn(), enable_if_t< M==0, int > = 0> constexpr inline constexpr mapping(const Shape & e)
```

Defined in include/teeny/layout.h:94

Fully-static strides: construct from extents only.

---

#### mapping

`inline` `constexpr`

```cpp
constexpr inline constexpr mapping(const Shape & e, const array< index_type, strides::ndyn()> & dyn)
```

Defined in include/teeny/layout.h:97

Mixed strides: extents + the runtime strides (dim order, dynamic ones only).

---

#### extents

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const Shape & extents() const noexcept
```

Defined in include/teeny/layout.h:100

---

#### stride

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr index_type stride(rank_type r) const noexcept
```

Defined in include/teeny/layout.h:101

---

#### operator()

`const` `inline` `constexpr` `noexcept`

```cpp
template<class... I> constexpr inline constexpr index_type operator()(I... i) const noexcept
```

Defined in include/teeny/layout.h:105

---

#### required_span_size

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr index_type required_span_size() const noexcept
```

Defined in include/teeny/layout.h:111

---

#### is_unique

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_unique() const noexcept
```

Defined in include/teeny/layout.h:122

---

#### is_exhaustive

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_exhaustive() const noexcept
```

Defined in include/teeny/layout.h:123

---

#### is_strided

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_strided() const noexcept
```

Defined in include/teeny/layout.h:124

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `constexpr bool` | [`is_always_unique`](#is_always_unique) `static` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_always_exhaustive`](#is_always_exhaustive) `static` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_always_strided`](#is_always_strided) `static` `inline` `constexpr` `noexcept` |  |

---

#### is_always_unique

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr bool is_always_unique() noexcept
```

Defined in include/teeny/layout.h:119

---

#### is_always_exhaustive

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr bool is_always_exhaustive() noexcept
```

Defined in include/teeny/layout.h:120

---

#### is_always_strided

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr bool is_always_strided() noexcept
```

Defined in include/teeny/layout.h:121

### Public Types

| Name | Description |
|------|-------------|
| [`extents_type`](#extents_type)  |  |
| [`index_type`](#index_type-1)  |  |
| [`rank_type`](#rank_type)  |  |
| [`layout_type`](#layout_type)  |  |

---

#### extents_type

```cpp
using extents_type = Shape
```

Defined in include/teeny/layout.h:83

---

#### index_type

```cpp
using index_type = typename Shape::index_type
```

Defined in include/teeny/layout.h:84

---

#### rank_type

```cpp
using rank_type = typename Shape::rank_type
```

Defined in include/teeny/layout.h:85

---

#### layout_type

```cpp
using layout_type = strides
```

Defined in include/teeny/layout.h:86



## tensor

```cpp
#include <tensor.h>
```

```cpp
template<class T, class Shape, class Layout, own O>
struct tensor
```

Defined in include/teeny/tensor.h:66

> **Inherits:** `template mapping< Shape >`

One N-dimensional tensor, parameterised by ownership.

The layout / extents / offset mapping is delegated to `cuda::std::mdspan` (the mapping lives in an empty base, so a fully-static tensor is exactly the size of its data). Ownership is a policy: `[own::view](#namespacetny_1a6a432f80fb491dbcb5d4b0692616b918a1bda80f2be4d3658e0baa43fbe7ae8c1)` (non-owning, trivially copyable, kernel-passable), `[own::stack](#namespacetny_1a6a432f80fb491dbcb5d4b0692616b918afac2a47adace059aff113283a03f6760)` (inline storage, static shape), or `[own::heap](#namespacetny_1a6a432f80fb491dbcb5d4b0692616b918a4d4a9aa362b6ffe089fd2e992ccf4f5f)` (host-only, move-only). The tensor's copy/move semantics are induced by the storage member, not hand-written.

#### Template Parameters
* `T` Element type. 

* `Shape` The shape: any `cuda::std::extents<Idx, E...>` (static or dynamic per dim). Spell it with the `shape<...>` alias. 

* `Layout` mdspan layout policy (default `layout_right`). 

* `O` Ownership kind (default `[own::view](#namespacetny_1a6a432f80fb491dbcb5d4b0692616b918a1bda80f2be4d3658e0baa43fbe7ae8c1)`).

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`store_`](#store_) | `variable` | Declared here |
| [`tensor`](#tensor-1) | `function` | Declared here |
| [`tensor`](#tensor-2) | `function` | Declared here |
| [`tensor`](#tensor-3) | `function` | Declared here |
| [`tensor`](#tensor-4) | `function` | Declared here |
| [`tensor`](#tensor-5) | `function` | Declared here |
| [`tensor`](#tensor-6) | `function` | Declared here |
| [`mapping`](#mapping-4) | `function` | Declared here |
| [`extents`](#extents-1) | `function` | Declared here |
| [`extent`](#extent) | `function` | Declared here |
| [`extent`](#extent-1) | `function` | Declared here |
| [`shape`](#shape-2) | `function` | Declared here |
| [`shape`](#shape-3) | `function` | Declared here |
| [`stride`](#stride-2) | `function` | Declared here |
| [`stride`](#stride-3) | `function` | Declared here |
| [`numel`](#numel) | `function` | Declared here |
| [`is_contiguous`](#is_contiguous) | `function` | Declared here |
| [`is_contiguous`](#is_contiguous-1) | `function` | Declared here |
| [`is_contiguous`](#is_contiguous-2) | `function` | Declared here |
| [`data`](#data-7) | `function` | Declared here |
| [`data`](#data-8) | `function` | Declared here |
| [`mdspan`](#mdspan) | `function` | Declared here |
| [`mdspan`](#mdspan-1) | `function` | Declared here |
| [`view`](#view-1) | `function` | Declared here |
| [`view`](#view-2) | `function` | Declared here |
| [`operator()`](#operator-16) | `function` | Declared here |
| [`operator()`](#operator-17) | `function` | Declared here |
| [`at`](#at) | `function` | Declared here |
| [`at`](#at-1) | `function` | Declared here |
| [`add_at`](#add_at) | `function` | Declared here |
| [`operator()`](#operator-18) | `function` | Declared here |
| [`operator()`](#operator-19) | `function` | Declared here |
| [`operator()`](#operator-20) | `function` | Declared here |
| [`operator()`](#operator-21) | `function` | Declared here |
| [`operator T`](#operatort) | `function` | Declared here |
| [`item`](#item) | `function` | Declared here |
| [`slice_along`](#slice_along) | `function` | Declared here |
| [`slice_along`](#slice_along-1) | `function` | Declared here |
| [`permute`](#permute) | `function` | Declared here |
| [`permute`](#permute-1) | `function` | Declared here |
| [`flip`](#flip) | `function` | Declared here |
| [`flip`](#flip-1) | `function` | Declared here |
| [`clone`](#clone) | `function` | Declared here |
| [`to`](#to-2) | `function` | Declared here |
| [`to`](#to-3) | `function` | Declared here |
| [`reshape`](#reshape) | `function` | Declared here |
| [`reshape`](#reshape-1) | `function` | Declared here |
| [`recast`](#recast) | `function` | Declared here |
| [`recast`](#recast-1) | `function` | Declared here |
| [`flatten`](#flatten) | `function` | Declared here |
| [`flatten`](#flatten-1) | `function` | Declared here |
| [`unsqueeze`](#unsqueeze) | `function` | Declared here |
| [`unsqueeze`](#unsqueeze-1) | `function` | Declared here |
| [`squeeze`](#squeeze) | `function` | Declared here |
| [`squeeze`](#squeeze-1) | `function` | Declared here |
| [`flip`](#flip-2) | `function` | Declared here |
| [`flip`](#flip-3) | `function` | Declared here |
| [`squeeze`](#squeeze-2) | `function` | Declared here |
| [`squeeze`](#squeeze-3) | `function` | Declared here |
| [`unsqueeze`](#unsqueeze-2) | `function` | Declared here |
| [`unsqueeze`](#unsqueeze-3) | `function` | Declared here |
| [`permute`](#permute-2) | `function` | Declared here |
| [`permute`](#permute-3) | `function` | Declared here |
| [`reshape`](#reshape-2) | `function` | Declared here |
| [`reshape`](#reshape-3) | `function` | Declared here |
| [`recast`](#recast-2) | `function` | Declared here |
| [`recast`](#recast-3) | `function` | Declared here |
| [`add_`](#add_) | `function` | Declared here |
| [`sub_`](#sub_) | `function` | Declared here |
| [`mul_`](#mul_) | `function` | Declared here |
| [`div_`](#div_) | `function` | Declared here |
| [`add_`](#add_-1) | `function` | Declared here |
| [`sub_`](#sub_-1) | `function` | Declared here |
| [`mul_`](#mul_-1) | `function` | Declared here |
| [`div_`](#div_-1) | `function` | Declared here |
| [`operator+=`](#operator-22) | `function` | Declared here |
| [`operator-=`](#operator-23) | `function` | Declared here |
| [`operator*=`](#operator-24) | `function` | Declared here |
| [`operator/=`](#operator-25) | `function` | Declared here |
| [`copy_`](#copy_) | `function` | Declared here |
| [`fill_`](#fill_) | `function` | Declared here |
| [`zero_`](#zero_) | `function` | Declared here |
| [`iota_`](#iota_) | `function` | Declared here |
| [`add`](#add) | `function` | Declared here |
| [`sub`](#sub) | `function` | Declared here |
| [`mul`](#mul) | `function` | Declared here |
| [`div`](#div) | `function` | Declared here |
| [`pow`](#pow) | `function` | Declared here |
| [`map_`](#map_) | `function` | Declared here |
| [`zip_with_`](#zip_with_) | `function` | Declared here |
| [`map`](#map) | `function` | Declared here |
| [`all`](#all-1) | `function` | Declared here |
| [`any`](#any) | `function` | Declared here |
| [`neg_`](#neg_) | `function` | Declared here |
| [`abs_`](#abs_) | `function` | Declared here |
| [`exp_`](#exp_) | `function` | Declared here |
| [`log_`](#log_) | `function` | Declared here |
| [`sin_`](#sin_) | `function` | Declared here |
| [`cos_`](#cos_) | `function` | Declared here |
| [`sqrt_`](#sqrt_) | `function` | Declared here |
| [`tanh_`](#tanh_) | `function` | Declared here |
| [`floor_`](#floor_) | `function` | Declared here |
| [`ceil_`](#ceil_) | `function` | Declared here |
| [`round_`](#round_) | `function` | Declared here |
| [`trunc_`](#trunc_) | `function` | Declared here |
| [`sign_`](#sign_) | `function` | Declared here |
| [`pow_`](#pow_) | `function` | Declared here |
| [`clamp_`](#clamp_) | `function` | Declared here |
| [`operator++`](#operator-26) | `function` | Declared here |
| [`operator--`](#operator-27) | `function` | Declared here |
| [`operator++`](#operator-28) | `function` | Declared here |
| [`operator--`](#operator-29) | `function` | Declared here |
| [`add_`](#add_-2) | `function` | Declared here |
| [`sub_`](#sub_-2) | `function` | Declared here |
| [`mul_`](#mul_-2) | `function` | Declared here |
| [`div_`](#div_-2) | `function` | Declared here |
| [`add_`](#add_-3) | `function` | Declared here |
| [`sub_`](#sub_-3) | `function` | Declared here |
| [`copy_`](#copy_-1) | `function` | Declared here |
| [`map_`](#map_-1) | `function` | Declared here |
| [`zip_with_`](#zip_with_-1) | `function` | Declared here |
| [`ownership`](#ownership) | `variable` | Declared here |
| [`is_static`](#is_static) | `variable` | Declared here |
| [`is_view`](#is_view) | `variable` | Declared here |
| [`is_owning`](#is_owning) | `variable` | Declared here |
| [`is_device`](#is_device) | `variable` | Declared here |
| [`is_host_accessible`](#is_host_accessible) | `variable` | Declared here |
| [`buffer_size`](#buffer_size) | `variable` | Declared here |
| [`is_strides_layout`](#is_strides_layout) | `variable` | Declared here |
| [`is_contiguous_layout`](#is_contiguous_layout) | `variable` | Declared here |
| [`rank`](#rank-1) | `function` | Declared here |
| [`element_type`](#element_type) | `typedef` | Declared here |
| [`extents_type`](#extents_type-1) | `typedef` | Declared here |
| [`shape_type`](#shape_type) | `typedef` | Declared here |
| [`layout_type`](#layout_type-1) | `typedef` | Declared here |
| [`index_type`](#index_type-2) | `typedef` | Declared here |
| [`mapping_type`](#mapping_type) | `typedef` | Declared here |
| [`view_type`](#view_type) | `typedef` | Declared here |
| [`const_view_type`](#const_view_type) | `typedef` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `storage< T, O, buffer_size >` | [`store_`](#store_)  |  |

---

#### store_

```cpp
storage< T, O, buffer_size > store_ {}
```

Defined in include/teeny/tensor.h:86

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
|  | [`tensor`](#tensor-1)  | Defaulted constructor. |
|  | [`tensor`](#tensor-2) `inline` | View constructor: wrap `p` with the given mapping. |
|  | [`tensor`](#tensor-3) `inline` `explicit` | View constructor from a pointer alone — for a fully-static geometry (static extents AND a fully determined layout: contiguous, or an all-static `strides<...>`). |
|  | [`tensor`](#tensor-4) `inline` | View constructor from a pointer + extents (contiguous / static-stride layouts). |
|  | [`tensor`](#tensor-5) `inline` `explicit` | Owning constructor: allocate storage for `m` (heap/device/host/pinned). |
|  | [`tensor`](#tensor-6) `inline` `explicit` | Owning constructor from extents (contiguous / static-stride layouts). |
| `constexpr const mapping_type &` | [`mapping`](#mapping-4) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr const Shape &` | [`extents`](#extents-1) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr auto` | [`extent`](#extent) `const` `inline` `constexpr` `noexcept` | Extent of an axis given by a STATIC index (`extent(Int<0>())`): a compile-time `integral_constant` when that extent is static, else a runtime `index_type`. |
| `constexpr index_type` | [`extent`](#extent-1) `const` `inline` `constexpr` `noexcept` | Extent of an axis given by a RUNTIME index (`extent(0)`). |
| `constexpr const Shape &` | [`shape`](#shape-2) `const` `inline` `constexpr` `noexcept` | `[shape()](#shape)` / `[shape(d)](#shape)` — python-friendly aliases of `extents()` / `extent(d)` (static index -> integral_constant, runtime -> value). |
| `constexpr auto` | [`shape`](#shape-3) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr auto` | [`stride`](#stride-2) `const` `inline` `constexpr` `noexcept` | Stride of an axis given by a STATIC index (`stride(Int<0>())`): a compile-time `integral_constant` when known statically (static- stride layout; a contiguous layout over static extents; or the always-unit stride of a contiguous layout even for dynamic shapes). |
| `constexpr index_type` | [`stride`](#stride-3) `const` `inline` `constexpr` `noexcept` | Stride of an axis given by a RUNTIME index (`stride(0)`). |
| `constexpr auto` | [`numel`](#numel) `const` `inline` `constexpr` `noexcept` | Number of elements. |
| `constexpr bool` | [`is_contiguous`](#is_contiguous) `const` `inline` `constexpr` `noexcept` | Whether the elements occupy a **dense block of memory**, in *some* axis order — true for a C- or F-contiguous tensor, and also for a permuted one (a permuted C-contiguous view still packs the same memory densely). |
| `bool` | [`is_contiguous`](#is_contiguous-1) `const` `inline` `noexcept` | Exact contiguity in layout `L` (e.g. |
| `bool` | [`is_contiguous`](#is_contiguous-2) `const` `inline` `noexcept` |  |
| `T *` | [`data`](#data-7) `inline` `noexcept` |  |
| `const T *` | [`data`](#data-8) `const` `inline` `noexcept` |  |
| `view_type` | [`mdspan`](#mdspan) `inline` `noexcept` | The raw `cuda::std::mdspan` over this tensor's storage. |
| `const_view_type` | [`mdspan`](#mdspan-1) `const` `inline` `noexcept` |  |
| `auto` | [`view`](#view-1) `inline` `noexcept` | A non-owning teeny **view** of this tensor's storage — a `view` (or `gpu_view`, for a device tensor) that aliases the same memory (no copy), keeping the source layout. |
| `auto` | [`view`](#view-2) `const` `inline` `noexcept` |  |
| `T &` | [`operator()`](#operator-16) `inline` `noexcept` | Element access when every argument is an integer (negatives wrap). |
| `const T &` | [`operator()`](#operator-17) `const` `inline` `noexcept` |  |
| `auto` | [`at`](#at) `inline` `noexcept` | `at(i...)` — a single element as a **rank-0 VIEW** (all-integer args; negatives wrap). |
| `auto` | [`at`](#at-1) `const` `inline` `noexcept` |  |
| `void` | [`add_at`](#add_at) `inline` `noexcept` | Scatter-accumulate: `(*this)(i...) += v`, atomic on the device — the write half of a "push"/splat kernel. |
| `auto` | [`operator()`](#operator-18) `inline` `noexcept` | Sub-view when any argument is a slice (`all`, `slice(a,b[,step])`). |
| `auto` | [`operator()`](#operator-19) `const` `inline` `noexcept` |  |
| `decltype(auto)` | [`operator()`](#operator-20) `inline` `noexcept` | Ellipsis form: exactly one `ellipsis` in the args expands to `rank - (#other args)` copies of `all`, then the call re-runs — so `t(1, ellipsis, 2)` on rank 5 is `t(1, all, all, all, 2)`. |
| `decltype(auto)` | [`operator()`](#operator-21) `const` `inline` `noexcept` |  |
|  | [`operator T`](#operatort) `const` `inline` `noexcept` |  |
| `T` | [`item`](#item) `const` `inline` `noexcept` | The single element of a rank-0 tensor (explicit reader). |
| `auto` | [`slice_along`](#slice_along) `inline` `noexcept` | Index/slice one or more named axes; other axes are kept. |
| `auto` | [`slice_along`](#slice_along-1) `const` `inline` `noexcept` |  |
| `auto` | [`permute`](#permute) `inline` `noexcept` | Reorder the axes (a permutation of 0..N-1; negatives wrap) -> a rank-N view. |
| `auto` | [`permute`](#permute-1) `const` `inline` `noexcept` |  |
| `auto` | [`flip`](#flip) `inline` `noexcept` | Reverse axis `Ax` (negatives wrap) -> a view (numpy `flip`). |
| `auto` | [`flip`](#flip-1) `const` `inline` `noexcept` |  |
| `auto` | [`clone`](#clone) `const` `inline` |  |
| `auto` | [`to`](#to-2) `const` `inline` | pytorch-like `.to<T2>()`: convert the element type to `T2`. |
| `auto` | [`to`](#to-3) `const` `inline` |  |
| `auto` | [`reshape`](#reshape) `inline` `noexcept` | View this tensor as a new shape — requires it be C-contiguous (`[clone()](#clone)` first otherwise) and the element count to match. |
| `auto` | [`reshape`](#reshape-1) `const` `inline` `noexcept` |  |
| `auto` | [`recast`](#recast) `inline` | Reinterpret with a MORE-STATIC extents type of the same rank — recover statically-known inner dims at the dynamic (ndarray) boundary: a runtime `(n,3,3)` view -> `.recast<shape<-1,3,3>>()` so the `3`s fold. |
| `auto` | [`recast`](#recast-1) `const` `inline` |  |
| `auto` | [`flatten`](#flatten) `inline` `noexcept` | View as 1-D (`ravel`) — requires C-contiguous (`[clone()](#clone)` first). |
| `auto` | [`flatten`](#flatten-1) `const` `inline` `noexcept` |  |
| `auto` | [`unsqueeze`](#unsqueeze) `inline` `noexcept` | Insert a size-1 axis at position `Ax` (numpy `newaxis`/`unsqueeze`) -> a rank-(N+1) view. |
| `auto` | [`unsqueeze`](#unsqueeze-1) `const` `inline` `noexcept` |  |
| `auto` | [`squeeze`](#squeeze) `inline` `noexcept` | Drop a size-1 axis `Ax` (negatives wrap) -> a rank-(N-1) view. |
| `auto` | [`squeeze`](#squeeze-1) `const` `inline` `noexcept` |  |
| `auto` | [`flip`](#flip-2) `inline` `noexcept` |  |
| `auto` | [`flip`](#flip-3) `const` `inline` `noexcept` |  |
| `auto` | [`squeeze`](#squeeze-2) `inline` `noexcept` |  |
| `auto` | [`squeeze`](#squeeze-3) `const` `inline` `noexcept` |  |
| `auto` | [`unsqueeze`](#unsqueeze-2) `inline` `noexcept` |  |
| `auto` | [`unsqueeze`](#unsqueeze-3) `const` `inline` `noexcept` |  |
| `auto` | [`permute`](#permute-2) `inline` `noexcept` |  |
| `auto` | [`permute`](#permute-3) `const` `inline` `noexcept` |  |
| `auto` | [`reshape`](#reshape-2) `inline` `noexcept` |  |
| `auto` | [`reshape`](#reshape-3) `const` `inline` `noexcept` |  |
| `auto` | [`recast`](#recast-2) `inline` |  |
| `auto` | [`recast`](#recast-3) `const` `inline` |  |
| `tensor &` | [`add_`](#add_)  |  |
| `tensor &` | [`sub_`](#sub_)  |  |
| `tensor &` | [`mul_`](#mul_)  |  |
| `tensor &` | [`div_`](#div_)  |  |
| `tensor &` | [`add_`](#add_-1)  |  |
| `tensor &` | [`sub_`](#sub_-1)  |  |
| `tensor &` | [`mul_`](#mul_-1)  |  |
| `tensor &` | [`div_`](#div_-1)  |  |
| `tensor &` | [`operator+=`](#operator-22) `inline` |  |
| `tensor &` | [`operator-=`](#operator-23) `inline` |  |
| `tensor &` | [`operator*=`](#operator-24) `inline` |  |
| `tensor &` | [`operator/=`](#operator-25) `inline` |  |
| `tensor &` | [`copy_`](#copy_)  |  |
| `tensor &` | [`fill_`](#fill_)  |  |
| `tensor &` | [`zero_`](#zero_)  |  |
| `tensor &` | [`iota_`](#iota_)  |  |
| `auto` | [`add`](#add) `const` |  |
| `auto` | [`sub`](#sub) `const` |  |
| `auto` | [`mul`](#mul) `const` |  |
| `auto` | [`div`](#div) `const` |  |
| `auto` | [`pow`](#pow) `const` |  |
| `tensor &` | [`map_`](#map_)  |  |
| `tensor &` | [`zip_with_`](#zip_with_)  |  |
| `auto` | [`map`](#map) `const` |  |
| `bool` | [`all`](#all-1) `const` |  |
| `bool` | [`any`](#any) `const` |  |
| `tensor &` | [`neg_`](#neg_)  |  |
| `tensor &` | [`abs_`](#abs_)  |  |
| `tensor &` | [`exp_`](#exp_)  |  |
| `tensor &` | [`log_`](#log_)  |  |
| `tensor &` | [`sin_`](#sin_)  |  |
| `tensor &` | [`cos_`](#cos_)  |  |
| `tensor &` | [`sqrt_`](#sqrt_)  |  |
| `tensor &` | [`tanh_`](#tanh_)  |  |
| `tensor &` | [`floor_`](#floor_)  |  |
| `tensor &` | [`ceil_`](#ceil_)  |  |
| `tensor &` | [`round_`](#round_)  |  |
| `tensor &` | [`trunc_`](#trunc_)  |  |
| `tensor &` | [`sign_`](#sign_)  |  |
| `tensor &` | [`pow_`](#pow_)  |  |
| `tensor &` | [`clamp_`](#clamp_)  |  |
| `tensor &` | [`operator++`](#operator-26) `inline` |  |
| `tensor &` | [`operator--`](#operator-27) `inline` |  |
| `tensor< T, Shape, layout_right, own::stack >` | [`operator++`](#operator-28) `inline` |  |
| `tensor< T, Shape, layout_right, own::stack >` | [`operator--`](#operator-29) `inline` |  |
| `tensor< T, E, L, O > &` | [`add_`](#add_-2)  |  |
| `tensor< T, E, L, O > &` | [`sub_`](#sub_-2)  |  |
| `tensor< T, E, L, O > &` | [`mul_`](#mul_-2)  |  |
| `tensor< T, E, L, O > &` | [`div_`](#div_-2)  |  |
| `tensor< T, E, L, O > &` | [`add_`](#add_-3)  |  |
| `tensor< T, E, L, O > &` | [`sub_`](#sub_-3)  |  |
| `tensor< T, E, L, O > &` | [`copy_`](#copy_-1)  |  |
| `tensor< T, E, L, O > &` | [`map_`](#map_-1)  |  |
| `tensor< T, E, L, O > &` | [`zip_with_`](#zip_with_-1)  |  |

---

#### tensor

```cpp
tensor() = default
```

Defined in include/teeny/tensor.h:89

Defaulted constructor.

---

#### tensor

`inline`

```cpp
template<own OO = O, enable_if_t< own_is_view(OO), int > = 0> inline tensor(T * p, mapping_type m)
```

Defined in include/teeny/tensor.h:93

View constructor: wrap `p` with the given mapping.

---

#### tensor

`inline` `explicit`

```cpp
template<own OO = O, enable_if_t< own_is_view(OO) &&is_static &&(_contiguous_layout< Layout >::value||_strides_all_static< Layout >::value), int > = 0> inline explicit tensor(T * p)
```

Defined in include/teeny/tensor.h:100

View constructor from a pointer alone — for a fully-static geometry (static extents AND a fully determined layout: contiguous, or an all-static `strides<...>`).

e.g. `tensor<float, shape<3,4>, strides<4,1>>(ptr)`.

---

#### tensor

`inline`

```cpp
template<own OO = O, enable_if_t< own_is_view(OO) &&is_constructible< mapping_type, Shape >::value, int > = 0> inline tensor(T * p, Shape e)
```

Defined in include/teeny/tensor.h:104

View constructor from a pointer + extents (contiguous / static-stride layouts).

---

#### tensor

`inline` `explicit`

```cpp
template<own OO = O, enable_if_t< own_is_owning(OO), int > = 0> inline explicit tensor(mapping_type m)
```

Defined in include/teeny/tensor.h:116

Owning constructor: allocate storage for `m` (heap/device/host/pinned).

---

#### tensor

`inline` `explicit`

```cpp
template<own OO = O, enable_if_t< own_is_owning(OO) &&is_constructible< mapping_type, Shape >::value, int > = 0> inline explicit tensor(Shape e)
```

Defined in include/teeny/tensor.h:121

Owning constructor from extents (contiguous / static-stride layouts).

---

#### mapping

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const mapping_type & mapping() const noexcept
```

Defined in include/teeny/tensor.h:126

---

#### extents

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const Shape & extents() const noexcept
```

Defined in include/teeny/tensor.h:127

---

#### extent

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx, enable_if_t< _is_ic< Idx >::value, int > = 0> constexpr inline constexpr auto extent(Idx) const noexcept
```

Defined in include/teeny/tensor.h:135

Extent of an axis given by a STATIC index (`extent(Int<0>())`): a compile-time `integral_constant` when that extent is static, else a runtime `index_type`.

---

#### extent

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx, enable_if_t<!_is_ic< Idx >::value, int > = 0> constexpr inline constexpr index_type extent(Idx d) const noexcept
```

Defined in include/teeny/tensor.h:144

Extent of an axis given by a RUNTIME index (`extent(0)`).

---

#### shape

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const Shape & shape() const noexcept
```

Defined in include/teeny/tensor.h:149

`[shape()](#shape)` / `[shape(d)](#shape)` — python-friendly aliases of `extents()` / `extent(d)` (static index -> integral_constant, runtime -> value).

---

#### shape

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx> constexpr inline constexpr auto shape(Idx d) const noexcept
```

Defined in include/teeny/tensor.h:150

---

#### stride

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx, enable_if_t< _is_ic< Idx >::value, int > = 0> constexpr inline constexpr auto stride(Idx) const noexcept
```

Defined in include/teeny/tensor.h:157

Stride of an axis given by a STATIC index (`stride(Int<0>())`): a compile-time `integral_constant` when known statically (static- stride layout; a contiguous layout over static extents; or the always-unit stride of a contiguous layout even for dynamic shapes).

---

#### stride

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx, enable_if_t<!_is_ic< Idx >::value, int > = 0> constexpr inline constexpr index_type stride(Idx d) const noexcept
```

Defined in include/teeny/tensor.h:175

Stride of an axis given by a RUNTIME index (`stride(0)`).

---

#### numel

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr auto numel() const noexcept
```

Defined in include/teeny/tensor.h:188

Number of elements.

A **fully static** shape folds to an `integral_constant` (so it propagates into later compile-time arithmetic, like `extent(Int<k>())`); any dynamic dim -> a runtime `index_type`.

---

#### is_contiguous

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_contiguous() const noexcept
```

Defined in include/teeny/tensor.h:209

Whether the elements occupy a **dense block of memory**, in *some* axis order — true for a C- or F-contiguous tensor, and also for a permuted one (a permuted C-contiguous view still packs the same memory densely).

Formally: the strides are a permutation of a dense nested packing (`1, e0, e0·e1, ...`). Size-1 axes are ignored (their stride is unconstrained); an empty tensor is trivially contiguous. Negative strides (flips) are *not* dense in this sense -> false.

Pass a layout for an **exact** check: `is_contiguous<layout_right>()` / `is_contiguous<layout_left>()` (aka `corder`/`forder`) test C- / F-contiguity specifically — or any layout whose mapping is derivable from the extents.

---

#### is_contiguous

`const` `inline` `noexcept`

```cpp
template<class L> inline bool is_contiguous() const noexcept
```

Defined in include/teeny/tensor.h:232

Exact contiguity in layout `L` (e.g.

`corder`/`forder`): the actual strides equal what `L` produces for these extents. Two spellings — `t.is_contiguous<corder>()` (type form) and `t.is_contiguous(corder())` (value form, layout deduced from the argument).

---

#### is_contiguous

`const` `inline` `noexcept`

```cpp
template<class L> inline bool is_contiguous(L) const noexcept
```

Defined in include/teeny/tensor.h:239

---

#### data

`inline` `noexcept`

```cpp
inline T * data() noexcept
```

Defined in include/teeny/tensor.h:242

---

#### data

`const` `inline` `noexcept`

```cpp
inline const T * data() const noexcept
```

Defined in include/teeny/tensor.h:243

---

#### mdspan

`inline` `noexcept`

```cpp
inline view_type mdspan() noexcept
```

Defined in include/teeny/tensor.h:245

The raw `cuda::std::mdspan` over this tensor's storage.

---

#### mdspan

`const` `inline` `noexcept`

```cpp
inline const_view_type mdspan() const noexcept
```

Defined in include/teeny/tensor.h:246

---

#### view

`inline` `noexcept`

```cpp
inline auto view() noexcept
```

Defined in include/teeny/tensor.h:253

A non-owning teeny **view** of this tensor's storage — a `view` (or `gpu_view`, for a device tensor) that aliases the same memory (no copy), keeping the source layout.

On an already-non-owning tensor it re-wraps the same pointer (an equivalent view). For the raw mdspan, use `[mdspan()](#mdspan)`.

---

#### view

`const` `inline` `noexcept`

```cpp
inline auto view() const noexcept
```

Defined in include/teeny/tensor.h:254

---

#### operator()

`inline` `noexcept`

```cpp
template<class... Args, enable_if_t<(_is_index< Args >::value &&...), int > = 0> inline T & operator()(Args... a) noexcept
```

Defined in include/teeny/tensor.h:373

Element access when every argument is an integer (negatives wrap).

---

#### operator()

`const` `inline` `noexcept`

```cpp
template<class... Args, enable_if_t<(_is_index< Args >::value &&...), int > = 0> inline const T & operator()(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:376

---

#### at

`inline` `noexcept`

```cpp
template<class... Args, enable_if_t<(_is_index< Args >::value &&...), int > = 0> inline auto at(Args... a) noexcept
```

Defined in include/teeny/tensor.h:386

`at(i...)` — a single element as a **rank-0 VIEW** (all-integer args; negatives wrap).

Unlike `operator()`, which returns a plain `T&`, this is a view, so the whole tensor API applies to one element: `x.at(i,j) = 3` writes it, `float v = x.at(i,j)` reads it (rank-0 tensors convert to/from `T`), and `x.at(i,j).add_<true>(v)` is an atomic scatter.

---

#### at

`const` `inline` `noexcept`

```cpp
template<class... Args, enable_if_t<(_is_index< Args >::value &&...), int > = 0> inline auto at(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:391

---

#### add_at

`inline` `noexcept`

```cpp
template<class... Args, enable_if_t<(_is_index< Args >::value &&...), int > = 0> inline void add_at(T v, Args... a) noexcept
```

Defined in include/teeny/tensor.h:400

Scatter-accumulate: `(*this)(i...) += v`, atomic on the device — the write half of a "push"/splat kernel.

Shorthand for `at(i...).add_<true>(v)` (integer indices only; negatives wrap).

---

#### operator()

`inline` `noexcept`

```cpp
template<class... Args, enable_if_t<!(_is_index< Args >::value &&...) &&!_has_ellipsis< Args... >::value, int > = 0> inline auto operator()(Args... a) noexcept
```

Defined in include/teeny/tensor.h:408

Sub-view when any argument is a slice (`all`, `slice(a,b[,step])`).

Integer args drop their axis, `all` keeps it, a range keeps a strided window — all via the one gather (folds static strides into `strides<...>`; works on any source layout).

---

#### operator()

`const` `inline` `noexcept`

```cpp
template<class... Args, enable_if_t<!(_is_index< Args >::value &&...) &&!_has_ellipsis< Args... >::value, int > = 0> inline auto operator()(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:411

---

#### operator()

`inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _has_ellipsis< Args... >::value, int > = 0> inline decltype(auto) operator()(Args... a) noexcept
```

Defined in include/teeny/tensor.h:419

Ellipsis form: exactly one `ellipsis` in the args expands to `rank - (#other args)` copies of `all`, then the call re-runs — so `t(1, ellipsis, 2)` on rank 5 is `t(1, all, all, all, 2)`.

What remains decides the result (all integers -> element, else view).

---

#### operator()

`const` `inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _has_ellipsis< Args... >::value, int > = 0> inline decltype(auto) operator()(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:422

---

#### operator T

`const` `inline` `noexcept`

```cpp
template<size_t R = rank(), enable_if_t< R==0, int > = 0> inline operator T() const noexcept
```

Defined in include/teeny/tensor.h:430

---

#### item

`const` `inline` `noexcept`

```cpp
template<size_t R = rank(), enable_if_t< R==0, int > = 0> inline T item() const noexcept
```

Defined in include/teeny/tensor.h:435

The single element of a rank-0 tensor (explicit reader).

---

#### slice_along

`inline` `noexcept`

```cpp
template<long... Axes, class... Args> inline auto slice_along(Args... args) noexcept
```

Defined in include/teeny/tensor.h:473

Index/slice one or more named axes; other axes are kept.

`slice_along<Axes...>(args...)` applies `args[k]` to axis `Axes[k]` (each an integer &ndash; negatives wrap &ndash; or a slice `all`/`rng`) and keeps every other axis, returning a view. e.g. `t.slice_along<1>(2)` drops axis 1 at index 2; `t.slice_along<0,2>(i, rng(1,4))` binds axes 0 and 2 at once.

---

#### slice_along

`const` `inline` `noexcept`

```cpp
template<long... Axes, class... Args> inline auto slice_along(Args... args) const noexcept
```

Defined in include/teeny/tensor.h:478

---

#### permute

`inline` `noexcept`

```cpp
template<long... Perm> inline auto permute() noexcept
```

Defined in include/teeny/tensor.h:485

Reorder the axes (a permutation of 0..N-1; negatives wrap) -> a rank-N view.

---

#### permute

`const` `inline` `noexcept`

```cpp
template<long... Perm> inline auto permute() const noexcept
```

Defined in include/teeny/tensor.h:488

---

#### flip

`inline` `noexcept`

```cpp
template<long Ax = 0> inline auto flip() noexcept
```

Defined in include/teeny/tensor.h:494

Reverse axis `Ax` (negatives wrap) -> a view (numpy `flip`).

Uses a negative stride, so the index type must be signed (`shape<...>` is).

---

#### flip

`const` `inline` `noexcept`

```cpp
template<long Ax = 0> inline auto flip() const noexcept
```

Defined in include/teeny/tensor.h:497

---

#### clone

`const` `inline`

```cpp
template<bool S = is_static, enable_if_t<!S, int > = 0> inline auto clone() const
```

Defined in include/teeny/tensor.h:506

---

#### to

`const` `inline`

```cpp
template<class T2 = element_type, bool Force = false, enable_if_t<!Force &&is_same< T2, element_type >::value, int > = 0> inline auto to() const
```

Defined in include/teeny/tensor.h:527

pytorch-like `.to<T2>()`: convert the element type to `T2`.

**No copy when it already matches** — if `T2` is the current element type and `Force` is false, this returns a (read-only) *view* of `*this`, no allocation, keeping the source layout. So `x.to<>()` is a zero-cost borrow, not a clone. Because it borrows, the result must not outlive `x` — the same lifetime rule as `[view()](#view-1)`/`[permute()](#permute)`/slicing (don't bind `.to<>()` of a temporary to a longer-lived name). Pass `Force = true` to always materialise a fresh owning copy even when the dtype already matches (`x.to<float, true>()` force-clones a `float` tensor); `x.clone()` is the unconditional-copy spelling.

When a conversion IS needed (`T2` differs, or `Force`), the result is a dense, row-major OWNING copy cast elementwise (via `copy_`): static shape -> stack (host+device), dynamic -> heap (host only). To also move across memory spaces (host <-> CUDA) use the `to<[own::gpu](#namespacetny_1a6a432f80fb491dbcb5d4b0692616b918a0aa0be2a866411d9ff03515227454947), T2, Force>(x)` free functions from `<[teeny/cuda.h](#cudah)>`.

---

#### to

`const` `inline`

```cpp
template<class T2 = element_type, bool Force = false, bool S = is_static, enable_if_t<(Force||!is_same< T2, element_type >::value) &&!S, int > = 0> inline auto to() const
```

Defined in include/teeny/tensor.h:535

---

#### reshape

`inline` `noexcept`

```cpp
template<long... NewExt> inline auto reshape() noexcept
```

Defined in include/teeny/tensor.h:555

View this tensor as a new shape — requires it be C-contiguous (`[clone()](#clone)` first otherwise) and the element count to match.

One extent may be **`-1`** (numpy-style), inferred from the total size: `t.reshape<6,-1>()`.

---

#### reshape

`const` `inline` `noexcept`

```cpp
template<long... NewExt> inline auto reshape() const noexcept
```

Defined in include/teeny/tensor.h:556

---

#### recast

`inline`

```cpp
template<class NewE> inline auto recast()
```

Defined in include/teeny/tensor.h:580

Reinterpret with a MORE-STATIC extents type of the same rank — recover statically-known inner dims at the dynamic (ndarray) boundary: a runtime `(n,3,3)` view -> `.recast<shape<-1,3,3>>()` so the `3`s fold.

**Requires a C-contiguous tensor** (`[clone()](#clone)` first otherwise); each static dim of `NewE` is validated against the actual extent.

---

#### recast

`const` `inline`

```cpp
template<class NewE> inline auto recast() const
```

Defined in include/teeny/tensor.h:581

---

#### flatten

`inline` `noexcept`

```cpp
inline auto flatten() noexcept
```

Defined in include/teeny/tensor.h:584

View as 1-D (`ravel`) — requires C-contiguous (`[clone()](#clone)` first).

---

#### flatten

`const` `inline` `noexcept`

```cpp
inline auto flatten() const noexcept
```

Defined in include/teeny/tensor.h:589

---

#### unsqueeze

`inline` `noexcept`

```cpp
template<long Ax = 0> inline auto unsqueeze() noexcept
```

Defined in include/teeny/tensor.h:599

Insert a size-1 axis at position `Ax` (numpy `newaxis`/`unsqueeze`) -> a rank-(N+1) view.

Negative `Ax` counts from the back, so `.unsqueeze<-1>()` appends a trailing axis: `(H,W)` -> `(H,W,1)`.

---

#### unsqueeze

`const` `inline` `noexcept`

```cpp
template<long Ax = 0> inline auto unsqueeze() const noexcept
```

Defined in include/teeny/tensor.h:602

---

#### squeeze

`inline` `noexcept`

```cpp
template<long Ax = _ax_all> inline auto squeeze() noexcept
```

Defined in include/teeny/tensor.h:621

Drop a size-1 axis `Ax` (negatives wrap) -> a rank-(N-1) view.

`[squeeze()](#squeeze)` (no axis) drops EVERY statically-size-1 axis.

---

#### squeeze

`const` `inline` `noexcept`

```cpp
template<long Ax = _ax_all> inline auto squeeze() const noexcept
```

Defined in include/teeny/tensor.h:627

---

#### flip

`inline` `noexcept`

```cpp
template<class I, enable_if_t< _is_ic< I >::value, int > = 0> inline auto flip(I) noexcept
```

Defined in include/teeny/tensor.h:637

---

#### flip

`const` `inline` `noexcept`

```cpp
template<class I, enable_if_t< _is_ic< I >::value, int > = 0> inline auto flip(I) const noexcept
```

Defined in include/teeny/tensor.h:638

---

#### squeeze

`inline` `noexcept`

```cpp
template<class I, enable_if_t< _is_ic< I >::value, int > = 0> inline auto squeeze(I) noexcept
```

Defined in include/teeny/tensor.h:639

---

#### squeeze

`const` `inline` `noexcept`

```cpp
template<class I, enable_if_t< _is_ic< I >::value, int > = 0> inline auto squeeze(I) const noexcept
```

Defined in include/teeny/tensor.h:640

---

#### unsqueeze

`inline` `noexcept`

```cpp
template<class I, enable_if_t< _is_ic< I >::value, int > = 0> inline auto unsqueeze(I) noexcept
```

Defined in include/teeny/tensor.h:641

---

#### unsqueeze

`const` `inline` `noexcept`

```cpp
template<class I, enable_if_t< _is_ic< I >::value, int > = 0> inline auto unsqueeze(I) const noexcept
```

Defined in include/teeny/tensor.h:642

---

#### permute

`inline` `noexcept`

```cpp
template<class... I, enable_if_t<(sizeof...(I) > 0) &&(_is_ic< I >::value &&...), int > = 0> inline auto permute(I...) noexcept
```

Defined in include/teeny/tensor.h:643

---

#### permute

`const` `inline` `noexcept`

```cpp
template<class... I, enable_if_t<(sizeof...(I) > 0) &&(_is_ic< I >::value &&...), int > = 0> inline auto permute(I...) const noexcept
```

Defined in include/teeny/tensor.h:644

---

#### reshape

`inline` `noexcept`

```cpp
template<class... I, enable_if_t<(sizeof...(I) > 0) &&(_is_ic< I >::value &&...), int > = 0> inline auto reshape(I...) noexcept
```

Defined in include/teeny/tensor.h:645

---

#### reshape

`const` `inline` `noexcept`

```cpp
template<class... I, enable_if_t<(sizeof...(I) > 0) &&(_is_ic< I >::value &&...), int > = 0> inline auto reshape(I...) const noexcept
```

Defined in include/teeny/tensor.h:646

---

#### recast

`inline`

```cpp
template<class NewE> inline auto recast(NewE)
```

Defined in include/teeny/tensor.h:647

---

#### recast

`const` `inline`

```cpp
template<class NewE> inline auto recast(NewE) const
```

Defined in include/teeny/tensor.h:648

---

#### add_

```cpp
template<bool Atomic = false, class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & add_(const B & b)
```

Defined in include/teeny/tensor.h:654

---

#### sub_

```cpp
template<bool Atomic = false, class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & sub_(const B & b)
```

Defined in include/teeny/tensor.h:655

---

#### mul_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & mul_(const B & b)
```

Defined in include/teeny/tensor.h:656

---

#### div_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & div_(const B & b)
```

Defined in include/teeny/tensor.h:657

---

#### add_

```cpp
template<bool Atomic = false> tensor & add_(T s)
```

Defined in include/teeny/tensor.h:658

---

#### sub_

```cpp
template<bool Atomic = false> tensor & sub_(T s)
```

Defined in include/teeny/tensor.h:659

---

#### mul_

```cpp
tensor & mul_(T s)
```

Defined in include/teeny/tensor.h:660

---

#### div_

```cpp
tensor & div_(T s)
```

Defined in include/teeny/tensor.h:661

---

#### operator+=

`inline`

```cpp
template<class B> inline tensor & operator+=(const B & b)
```

Defined in include/teeny/tensor.h:665

---

#### operator-=

`inline`

```cpp
template<class B> inline tensor & operator-=(const B & b)
```

Defined in include/teeny/tensor.h:666

---

#### operator*=

`inline`

```cpp
template<class B> inline tensor & operator*=(const B & b)
```

Defined in include/teeny/tensor.h:667

---

#### operator/=

`inline`

```cpp
template<class B> inline tensor & operator/=(const B & b)
```

Defined in include/teeny/tensor.h:668

---

#### copy_

```cpp
template<class B> tensor & copy_(const B & b)
```

Defined in include/teeny/tensor.h:671

---

#### fill_

```cpp
tensor & fill_(T s)
```

Defined in include/teeny/tensor.h:672

---

#### zero_

```cpp
tensor & zero_()
```

Defined in include/teeny/tensor.h:673

---

#### iota_

```cpp
tensor & iota_(T start = T(0), T step = T(1))
```

Defined in include/teeny/tensor.h:674

---

#### add

`const`

```cpp
template<class B> auto add(const B & b) const
```

Defined in include/teeny/tensor.h:677

---

#### sub

`const`

```cpp
template<class B> auto sub(const B & b) const
```

Defined in include/teeny/tensor.h:678

---

#### mul

`const`

```cpp
template<class B> auto mul(const B & b) const
```

Defined in include/teeny/tensor.h:679

---

#### div

`const`

```cpp
template<class B> auto div(const B & b) const
```

Defined in include/teeny/tensor.h:680

---

#### pow

`const`

```cpp
template<class B> auto pow(const B & b) const
```

Defined in include/teeny/tensor.h:681

---

#### map_

```cpp
template<class F> tensor & map_(F f)
```

Defined in include/teeny/tensor.h:684

---

#### zip_with_

```cpp
template<class G, class B> tensor & zip_with_(G g, const B & b)
```

Defined in include/teeny/tensor.h:685

---

#### map

`const`

```cpp
template<class F> auto map(F f) const
```

Defined in include/teeny/tensor.h:686

---

#### all

`const`

```cpp
bool all() const
```

Defined in include/teeny/tensor.h:690

---

#### any

`const`

```cpp
bool any() const
```

Defined in include/teeny/tensor.h:691

---

#### neg_

```cpp
tensor & neg_()
```

Defined in include/teeny/tensor.h:694

---

#### abs_

```cpp
tensor & abs_()
```

Defined in include/teeny/tensor.h:695

---

#### exp_

```cpp
tensor & exp_()
```

Defined in include/teeny/tensor.h:696

---

#### log_

```cpp
tensor & log_()
```

Defined in include/teeny/tensor.h:697

---

#### sin_

```cpp
tensor & sin_()
```

Defined in include/teeny/tensor.h:698

---

#### cos_

```cpp
tensor & cos_()
```

Defined in include/teeny/tensor.h:699

---

#### sqrt_

```cpp
tensor & sqrt_()
```

Defined in include/teeny/tensor.h:700

---

#### tanh_

```cpp
tensor & tanh_()
```

Defined in include/teeny/tensor.h:701

---

#### floor_

```cpp
tensor & floor_()
```

Defined in include/teeny/tensor.h:702

---

#### ceil_

```cpp
tensor & ceil_()
```

Defined in include/teeny/tensor.h:703

---

#### round_

```cpp
tensor & round_()
```

Defined in include/teeny/tensor.h:704

---

#### trunc_

```cpp
tensor & trunc_()
```

Defined in include/teeny/tensor.h:705

---

#### sign_

```cpp
tensor & sign_()
```

Defined in include/teeny/tensor.h:706

---

#### pow_

```cpp
tensor & pow_(T e)
```

Defined in include/teeny/tensor.h:707

---

#### clamp_

```cpp
tensor & clamp_(T lo, T hi)
```

Defined in include/teeny/tensor.h:708

---

#### operator++

`inline`

```cpp
inline tensor & operator++()
```

Defined in include/teeny/tensor.h:714

---

#### operator--

`inline`

```cpp
inline tensor & operator--()
```

Defined in include/teeny/tensor.h:715

---

#### operator++

`inline`

```cpp
template<bool S = is_static, enable_if_t< S, int > = 0> inline tensor< T, Shape, layout_right, own::stack > operator++(int)
```

Defined in include/teeny/tensor.h:717

---

#### operator--

`inline`

```cpp
template<bool S = is_static, enable_if_t< S, int > = 0> inline tensor< T, Shape, layout_right, own::stack > operator--(int)
```

Defined in include/teeny/tensor.h:719

---

#### add_

```cpp
template<bool Atomic, class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & add_(const B & b)
```

Defined in include/teeny/math.h:544

---

#### sub_

```cpp
template<bool Atomic, class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & sub_(const B & b)
```

Defined in include/teeny/math.h:550

---

#### mul_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & mul_(const B & b)
```

Defined in include/teeny/math.h:556

---

#### div_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & div_(const B & b)
```

Defined in include/teeny/math.h:558

---

#### add_

```cpp
template<bool Atomic> tensor< T, E, L, O > & add_(T s)
```

Defined in include/teeny/math.h:560

---

#### sub_

```cpp
template<bool Atomic> tensor< T, E, L, O > & sub_(T s)
```

Defined in include/teeny/math.h:566

---

#### copy_

```cpp
template<class B> tensor< T, E, L, O > & copy_(const B & b)
```

Defined in include/teeny/math.h:574

---

#### map_

```cpp
template<class F> tensor< T, E, L, O > & map_(F f)
```

Defined in include/teeny/math.h:707

---

#### zip_with_

```cpp
template<class G, class B> tensor< T, E, L, O > & zip_with_(G g, const B & b)
```

Defined in include/teeny/math.h:709

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr own` | [`ownership`](#ownership) `static` `constexpr` |  |
| `constexpr bool` | [`is_static`](#is_static) `static` `constexpr` |  |
| `constexpr bool` | [`is_view`](#is_view) `static` `constexpr` |  |
| `constexpr bool` | [`is_owning`](#is_owning) `static` `constexpr` |  |
| `constexpr bool` | [`is_device`](#is_device) `static` `constexpr` |  |
| `constexpr bool` | [`is_host_accessible`](#is_host_accessible) `static` `constexpr` |  |
| `constexpr size_t` | [`buffer_size`](#buffer_size) `static` `constexpr` |  |
| `constexpr bool` | [`is_strides_layout`](#is_strides_layout) `static` `constexpr` |  |
| `constexpr bool` | [`is_contiguous_layout`](#is_contiguous_layout) `static` `constexpr` |  |

---

#### ownership

`static` `constexpr`

```cpp
constexpr own ownership = O
```

Defined in include/teeny/tensor.h:76

---

#### is_static

`static` `constexpr`

```cpp
constexpr bool is_static = (Shape::rank_dynamic() == 0)
```

Defined in include/teeny/tensor.h:77

---

#### is_view

`static` `constexpr`

```cpp
constexpr bool is_view = (O)
```

Defined in include/teeny/tensor.h:79

---

#### is_owning

`static` `constexpr`

```cpp
constexpr bool is_owning = (O)
```

Defined in include/teeny/tensor.h:80

---

#### is_device

`static` `constexpr`

```cpp
constexpr bool is_device = (O)
```

Defined in include/teeny/tensor.h:81

---

#### is_host_accessible

`static` `constexpr`

```cpp
constexpr bool is_host_accessible = (O)
```

Defined in include/teeny/tensor.h:82

---

#### buffer_size

`static` `constexpr`

```cpp
constexpr size_t buffer_size = <, O == >::value
```

Defined in include/teeny/tensor.h:83

---

#### is_strides_layout

`static` `constexpr`

```cpp
constexpr bool is_strides_layout = _is_strides<Layout>::value
```

Defined in include/teeny/tensor.h:128

---

#### is_contiguous_layout

`static` `constexpr`

```cpp
constexpr bool is_contiguous_layout = _contiguous_layout<Layout>::value
```

Defined in include/teeny/tensor.h:129

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `constexpr size_t` | [`rank`](#rank-1) `static` `inline` `constexpr` `noexcept` |  |

---

#### rank

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr size_t rank() noexcept
```

Defined in include/teeny/tensor.h:125

### Public Types

| Name | Description |
|------|-------------|
| [`element_type`](#element_type)  |  |
| [`extents_type`](#extents_type-1)  |  |
| [`shape_type`](#shape_type)  |  |
| [`layout_type`](#layout_type-1)  |  |
| [`index_type`](#index_type-2)  |  |
| [`mapping_type`](#mapping_type)  |  |
| [`view_type`](#view_type)  |  |
| [`const_view_type`](#const_view_type)  |  |

---

#### element_type

```cpp
using element_type = T
```

Defined in include/teeny/tensor.h:67

---

#### extents_type

```cpp
using extents_type = Shape
```

Defined in include/teeny/tensor.h:68

---

#### shape_type

```cpp
using shape_type = Shape
```

Defined in include/teeny/tensor.h:69

---

#### layout_type

```cpp
using layout_type = Layout
```

Defined in include/teeny/tensor.h:70

---

#### index_type

```cpp
using index_type = typename Shape::index_type
```

Defined in include/teeny/tensor.h:71

---

#### mapping_type

```cpp
using mapping_type = typename Layout::template mapping< Shape >
```

Defined in include/teeny/tensor.h:72

---

#### view_type

```cpp
using view_type = mdspan< T, Shape, Layout >
```

Defined in include/teeny/tensor.h:73

---

#### const_view_type

```cpp
using const_view_type = mdspan< const T, Shape, Layout >
```

Defined in include/teeny/tensor.h:74



## mapping

```cpp
#include <layout.h>
```

```cpp
template<class Shape>
struct mapping
```

Defined in include/teeny/layout.h:82

> **Inherits:** `ndyn()>`, `Shape`

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`mapping`](#mapping-1) | `function` | Declared here |
| [`mapping`](#mapping-2) | `function` | Declared here |
| [`mapping`](#mapping-3) | `function` | Declared here |
| [`extents`](#extents) | `function` | Declared here |
| [`stride`](#stride-1) | `function` | Declared here |
| [`operator()`](#operator-15) | `function` | Declared here |
| [`required_span_size`](#required_span_size) | `function` | Declared here |
| [`is_unique`](#is_unique) | `function` | Declared here |
| [`is_exhaustive`](#is_exhaustive) | `function` | Declared here |
| [`is_strided`](#is_strided) | `function` | Declared here |
| [`is_always_unique`](#is_always_unique) | `function` | Declared here |
| [`is_always_exhaustive`](#is_always_exhaustive) | `function` | Declared here |
| [`is_always_strided`](#is_always_strided) | `function` | Declared here |
| [`extents_type`](#extents_type) | `typedef` | Declared here |
| [`index_type`](#index_type-1) | `typedef` | Declared here |
| [`rank_type`](#rank_type) | `typedef` | Declared here |
| [`layout_type`](#layout_type) | `typedef` | Declared here |

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
|  | [`mapping`](#mapping-1)  | Defaulted constructor. |
| `constexpr` | [`mapping`](#mapping-2) `inline` `constexpr` | Fully-static strides: construct from extents only. |
| `constexpr` | [`mapping`](#mapping-3) `inline` `constexpr` | Mixed strides: extents + the runtime strides (dim order, dynamic ones only). |
| `constexpr const Shape &` | [`extents`](#extents) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr index_type` | [`stride`](#stride-1) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr index_type` | [`operator()`](#operator-15) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr index_type` | [`required_span_size`](#required_span_size) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_unique`](#is_unique) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_exhaustive`](#is_exhaustive) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_strided`](#is_strided) `const` `inline` `constexpr` `noexcept` |  |

---

#### mapping

```cpp
mapping() = default
```

Defined in include/teeny/layout.h:90

Defaulted constructor.

---

#### mapping

`inline` `constexpr`

```cpp
template<size_t M = strides::ndyn(), enable_if_t< M==0, int > = 0> constexpr inline constexpr mapping(const Shape & e)
```

Defined in include/teeny/layout.h:94

Fully-static strides: construct from extents only.

---

#### mapping

`inline` `constexpr`

```cpp
constexpr inline constexpr mapping(const Shape & e, const array< index_type, strides::ndyn()> & dyn)
```

Defined in include/teeny/layout.h:97

Mixed strides: extents + the runtime strides (dim order, dynamic ones only).

---

#### extents

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const Shape & extents() const noexcept
```

Defined in include/teeny/layout.h:100

---

#### stride

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr index_type stride(rank_type r) const noexcept
```

Defined in include/teeny/layout.h:101

---

#### operator()

`const` `inline` `constexpr` `noexcept`

```cpp
template<class... I> constexpr inline constexpr index_type operator()(I... i) const noexcept
```

Defined in include/teeny/layout.h:105

---

#### required_span_size

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr index_type required_span_size() const noexcept
```

Defined in include/teeny/layout.h:111

---

#### is_unique

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_unique() const noexcept
```

Defined in include/teeny/layout.h:122

---

#### is_exhaustive

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_exhaustive() const noexcept
```

Defined in include/teeny/layout.h:123

---

#### is_strided

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_strided() const noexcept
```

Defined in include/teeny/layout.h:124

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `constexpr bool` | [`is_always_unique`](#is_always_unique) `static` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_always_exhaustive`](#is_always_exhaustive) `static` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_always_strided`](#is_always_strided) `static` `inline` `constexpr` `noexcept` |  |

---

#### is_always_unique

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr bool is_always_unique() noexcept
```

Defined in include/teeny/layout.h:119

---

#### is_always_exhaustive

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr bool is_always_exhaustive() noexcept
```

Defined in include/teeny/layout.h:120

---

#### is_always_strided

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr bool is_always_strided() noexcept
```

Defined in include/teeny/layout.h:121

### Public Types

| Name | Description |
|------|-------------|
| [`extents_type`](#extents_type)  |  |
| [`index_type`](#index_type-1)  |  |
| [`rank_type`](#rank_type)  |  |
| [`layout_type`](#layout_type)  |  |

---

#### extents_type

```cpp
using extents_type = Shape
```

Defined in include/teeny/layout.h:83

---

#### index_type

```cpp
using index_type = typename Shape::index_type
```

Defined in include/teeny/layout.h:84

---

#### rank_type

```cpp
using rank_type = typename Shape::rank_type
```

Defined in include/teeny/layout.h:85

---

#### layout_type

```cpp
using layout_type = strides
```

Defined in include/teeny/layout.h:86



## iterator

```cpp
#include <iterate.h>
```

```cpp
struct iterator
```

Defined in include/teeny/iterate.h:123

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`r`](#r-1) | `variable` | Declared here |
| [`i`](#i-1) | `variable` | Declared here |
| [`operator*`](#operator-11) | `function` | Declared here |
| [`operator++`](#operator-12) | `function` | Declared here |
| [`operator!=`](#operator-13) | `function` | Declared here |
| [`operator==`](#operator-14) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `peel_range` | [`r`](#r-1)  |  |
| `index_type` | [`i`](#i-1)  |  |

---

#### r

```cpp
peel_range r
```

Defined in include/teeny/iterate.h:124

---

#### i

```cpp
index_type i
```

Defined in include/teeny/iterate.h:125

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `auto` | [`operator*`](#operator-11) `const` `inline` |  |
| `iterator &` | [`operator++`](#operator-12) `inline` |  |
| `bool` | [`operator!=`](#operator-13) `const` `inline` |  |
| `bool` | [`operator==`](#operator-14) `const` `inline` |  |

---

#### operator*

`const` `inline`

```cpp
inline auto operator*() const
```

Defined in include/teeny/iterate.h:126

---

#### operator++

`inline`

```cpp
inline iterator & operator++()
```

Defined in include/teeny/iterate.h:127

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const iterator & o) const
```

Defined in include/teeny/iterate.h:128

---

#### operator==

`const` `inline`

```cpp
inline bool operator==(const iterator & o) const
```

Defined in include/teeny/iterate.h:129



## iterator

```cpp
#include <dynamic.h>
```

```cpp
struct iterator
```

Defined in include/teeny/dynamic.h:154

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`r`](#r) | `variable` | Declared here |
| [`i`](#i) | `variable` | Declared here |
| [`operator*`](#operator-7) | `function` | Declared here |
| [`operator++`](#operator-8) | `function` | Declared here |
| [`operator!=`](#operator-9) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `anyrank_front` | [`r`](#r)  |  |
| `offset_t` | [`i`](#i)  |  |

---

#### r

```cpp
anyrank_front r
```

Defined in include/teeny/dynamic.h:155

---

#### i

```cpp
offset_t i
```

Defined in include/teeny/dynamic.h:155

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `auto` | [`operator*`](#operator-7) `const` `inline` |  |
| `iterator &` | [`operator++`](#operator-8) `inline` |  |
| `bool` | [`operator!=`](#operator-9) `const` `inline` |  |

---

#### operator*

`const` `inline`

```cpp
inline auto operator*() const
```

Defined in include/teeny/dynamic.h:156

---

#### operator++

`inline`

```cpp
inline iterator & operator++()
```

Defined in include/teeny/dynamic.h:157

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const iterator & o) const
```

Defined in include/teeny/dynamic.h:158

Generated by [Moxygen](https://0state.com/moxygen)