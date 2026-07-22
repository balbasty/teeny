# Autodoc

Generated from the header Doxygen comments (`doxygen` + `moxygen`). For a
curated view see the [Reference](../reference.md) and [Cheat sheet](../cheatsheet.md).

# API Reference

## Namespaces

| Name | Description |
|------|-------------|
| [`tny`](#tny) |  |

## Classes

| Name | Description |
|------|-------------|
| [`mapping`](#mapping) |  |
| [`iterator`](#iterator-1) |  |
| [`iterator`](#iterator) |  |



## tny

### Classes

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
| [`heap, N >`](#heapn) |  |
| [`mapped, N >`](#mappedn) |  |
| [`pinned, N >`](#pinnedn) |  |
| [`stack, N >`](#stackn) |  |
| [`view, N >`](#viewn) |  |
| [`storage_size`](#storage_size) | Storage element count for a stack tensor (0 for view/owning). |
| [`storage_size< Mapping, true >`](#storage_sizemappingtrue) |  |
| [`strides`](#strides) | An mdspan layout policy with **per-dimension static or dynamic strides** — the stride analogue of `extents`/`shape`. |
| [`tensor`](#tensor) | One N-dimensional tensor, parameterised by ownership. |

### Enumerations

| Name | Description |
|------|-------------|
| [`own`](#own)  | Ownership / memory-space of a tensor's storage. |

---

#### own

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

### Typedefs

| Return | Name | Description |
|--------|------|-------------|
| `cs::integral_constant< int, V >` | [`Int`](#int)  |  |
| `cs::integral_constant< long, V >` | [`Long`](#long)  |  |
| `cs::integral_constant< cs::size_t, V >` | [`Size`](#size)  |  |
| `cs::integral_constant< unsigned, V >` | [`Uint`](#uint)  |  |
| `cs::integral_constant< cs::ptrdiff_t, V >` | [`Diff`](#diff)  |  |
| `cs::integral_constant< bool, V >` | [`Bool`](#bool)  |  |
| `cs::integral_constant< cs::int8_t, V >` | [`Int8`](#int8)  |  |
| `cs::integral_constant< cs::int16_t, V >` | [`Int16`](#int16)  |  |
| `cs::integral_constant< cs::int32_t, V >` | [`Int32`](#int32)  |  |
| `cs::integral_constant< cs::int64_t, V >` | [`Int64`](#int64)  |  |
| `cs::integral_constant< cs::uint8_t, V >` | [`Uint8`](#uint8)  |  |
| `cs::integral_constant< cs::uint16_t, V >` | [`Uint16`](#uint16)  |  |
| `cs::integral_constant< cs::uint32_t, V >` | [`Uint32`](#uint32)  |  |
| `cs::integral_constant< cs::uint64_t, V >` | [`Uint64`](#uint64)  |  |
| `Int8< V >` | [`I1`](#i1)  |  |
| `Int16< V >` | [`I2`](#i2)  |  |
| `Int32< V >` | [`I4`](#i4)  |  |
| `Int64< V >` | [`I8`](#i8)  |  |
| `Uint8< V >` | [`U1`](#u1)  |  |
| `Uint16< V >` | [`U2`](#u2)  |  |
| `Uint32< V >` | [`U4`](#u4)  |  |
| `Uint64< V >` | [`U8`](#u8)  |  |
| `cs::integral_constant< long, V >` | [`ic`](#ic)  | Alias of `Long`; a compile-time index value. |
| `cs::int8_t` | [`i1`](#i1-1)  |  |
| `cs::int16_t` | [`i2`](#i2-1)  |  |
| `cs::int32_t` | [`i4`](#i4-1)  |  |
| `cs::int64_t` | [`i8`](#i8-1)  |  |
| `cs::uint8_t` | [`u1`](#u1-1)  |  |
| `cs::uint16_t` | [`u2`](#u2-1)  |  |
| `cs::uint32_t` | [`u4`](#u4-1)  |  |
| `cs::uint64_t` | [`u8`](#u8-1)  |  |
| `float` | [`f4`](#f4)  |  |
| `double` | [`f8`](#f8)  |  |
| `cs::extents< cs::int64_t, _dyn_extent(E)... >` | [`shape`](#shape)  | User-friendly shape type: `shape<2,3,4>` == `extents<int64_t, 2,3,4>`. |
| `tensor< T, Extents, Layout, own::gpu >` | [`gpu`](#gpu)  | Owning tensor in device (GPU) memory (move-only). |
| `tensor< T, Extents, Layout, own::pinned >` | [`pinned`](#pinned)  | Owning tensor in page-locked ("pinned") host memory (move-only). |
| `tensor< T, Extents, Layout, own::mapped >` | [`mapped`](#mapped)  | Owning tensor in mapped (zero-copy) host memory (move-only). |
| `tensor< T, cs::dextents< offset_t, R >, cs::layout_stride, own::view >` | [`dyn_tensor`](#dyn_tensor)  | A fixed-rank, fully-dynamic, arbitrarily-strided tensor view. |
| `__half` | [`half`](#half)  | IEEE binary16 — the native CUDA `__half` under nvcc. |
| `__nv_bfloat16` | [`bfloat16`](#bfloat16)  | bfloat16 — the native CUDA `__nv_bfloat16` under nvcc. |
| `typename compute_type< T >::type` | [`compute_type_t`](#compute_type_t)  |  |
| `half` | [`f2`](#f2)  |  |
| `bfloat16` | [`bf16`](#bf16)  |  |
| `strides< S... >` | [`layout_static_stride`](#layout_static_stride)  | Back-compat alias: the original all-static-stride layout name. |
| `typename _promote< A, B, true >::type` | [`promote_t`](#promote_t)  |  |
| `cs::conditional_t<(cs::is_floating_point< T >::value||!cs::is_same< compute_type_t< T >, T >::value), cs::conditional_t<(sizeof(T) > 8), T, double >, T >` | [`reduce_type_t`](#reduce_type_t)  | Default accumulator type for a reduction over element type `T`. |
| `tensor< T, Extents, Layout, own::view >` | [`view_t`](#view_t)  | A non-owning view type. |
| `tensor< T, Extents, Layout, own::stack >` | [`local`](#local)  | Stack-owned tensor (fully static shape). |
| `tensor< T, Extents, Layout, own::heap >` | [`owned`](#owned)  | Heap-owned tensor (host only, move-only). |

---

#### Int

```cpp
using Int = cs::integral_constant< int, V >
```

---

#### Long

```cpp
using Long = cs::integral_constant< long, V >
```

---

#### Size

```cpp
using Size = cs::integral_constant< cs::size_t, V >
```

---

#### Uint

```cpp
using Uint = cs::integral_constant< unsigned, V >
```

---

#### Diff

```cpp
using Diff = cs::integral_constant< cs::ptrdiff_t, V >
```

---

#### Bool

```cpp
using Bool = cs::integral_constant< bool, V >
```

---

#### Int8

```cpp
using Int8 = cs::integral_constant< cs::int8_t, V >
```

---

#### Int16

```cpp
using Int16 = cs::integral_constant< cs::int16_t, V >
```

---

#### Int32

```cpp
using Int32 = cs::integral_constant< cs::int32_t, V >
```

---

#### Int64

```cpp
using Int64 = cs::integral_constant< cs::int64_t, V >
```

---

#### Uint8

```cpp
using Uint8 = cs::integral_constant< cs::uint8_t, V >
```

---

#### Uint16

```cpp
using Uint16 = cs::integral_constant< cs::uint16_t, V >
```

---

#### Uint32

```cpp
using Uint32 = cs::integral_constant< cs::uint32_t, V >
```

---

#### Uint64

```cpp
using Uint64 = cs::integral_constant< cs::uint64_t, V >
```

---

#### I1

```cpp
using I1 = Int8< V >
```

---

#### I2

```cpp
using I2 = Int16< V >
```

---

#### I4

```cpp
using I4 = Int32< V >
```

---

#### I8

```cpp
using I8 = Int64< V >
```

---

#### U1

```cpp
using U1 = Uint8< V >
```

---

#### U2

```cpp
using U2 = Uint16< V >
```

---

#### U4

```cpp
using U4 = Uint32< V >
```

---

#### U8

```cpp
using U8 = Uint64< V >
```

---

#### ic

```cpp
using ic = cs::integral_constant< long, V >
```

Alias of `Long`; a compile-time index value.

---

#### i1

```cpp
using i1 = cs::int8_t
```

---

#### i2

```cpp
using i2 = cs::int16_t
```

---

#### i4

```cpp
using i4 = cs::int32_t
```

---

#### i8

```cpp
using i8 = cs::int64_t
```

---

#### u1

```cpp
using u1 = cs::uint8_t
```

---

#### u2

```cpp
using u2 = cs::uint16_t
```

---

#### u4

```cpp
using u4 = cs::uint32_t
```

---

#### u8

```cpp
using u8 = cs::uint64_t
```

---

#### f4

```cpp
using f4 = float
```

---

#### f8

```cpp
using f8 = double
```

---

#### shape

```cpp
using shape = cs::extents< cs::int64_t, _dyn_extent(E)... >
```

User-friendly shape type: `shape<2,3,4>` == `extents<int64_t, 2,3,4>`.

The fixed-size `int64_t` index type matches DLPack's `shape` exactly, so it drops straight onto ndarray bindings. A dynamic dimension can be spelled either `dynamic_extent` or, numpy-style, **`-1`** — so `shape<-1,2,3>` == `shape<dynamic_extent,2,3>` == `extents<int64_t, dynamic_extent, 2, 3>`. Use it in place of `extents<...>`: `local<double, shape<3,3>>`, `owned<float, shape<-1,4>>`.

---

#### gpu

```cpp
using gpu = tensor< T, Extents, Layout, own::gpu >
```

Owning tensor in device (GPU) memory (move-only).

`gpu<T,E>(extents)`.

---

#### pinned

```cpp
using pinned = tensor< T, Extents, Layout, own::pinned >
```

Owning tensor in page-locked ("pinned") host memory (move-only).

`pinned<T,E>(extents)` — pytorch's `pin_memory`.

---

#### mapped

```cpp
using mapped = tensor< T, Extents, Layout, own::mapped >
```

Owning tensor in mapped (zero-copy) host memory (move-only).

`mapped<T,E>(extents)`.

---

#### dyn_tensor

```cpp
using dyn_tensor = tensor< T, cs::dextents< offset_t, R >, cs::layout_stride, own::view >
```

A fixed-rank, fully-dynamic, arbitrarily-strided tensor view.

---

#### half

```cpp
using half = __half
```

IEEE binary16 — the native CUDA `__half` under nvcc.

---

#### bfloat16

```cpp
using bfloat16 = __nv_bfloat16
```

bfloat16 — the native CUDA `__nv_bfloat16` under nvcc.

---

#### compute_type_t

```cpp
using compute_type_t = typename compute_type< T >::type
```

---

#### f2

```cpp
using f2 = half
```

---

#### bf16

```cpp
using bf16 = bfloat16
```

---

#### layout_static_stride

```cpp
using layout_static_stride = strides< S... >
```

Back-compat alias: the original all-static-stride layout name.

---

#### promote_t

```cpp
using promote_t = typename _promote< A, B, true >::type
```

---

#### reduce_type_t

```cpp
using reduce_type_t = cs::conditional_t<(cs::is_floating_point< T >::value||!cs::is_same< compute_type_t< T >, T >::value), cs::conditional_t<(sizeof(T) > 8), T, double >, T >
```

Default accumulator type for a reduction over element type `T`.

`double` for floating-point types of at most 8 bytes (`float`, `double`, `half`, `bfloat16`) — enough headroom that summing many low-precision values doesn't lose catastrophically; a *wider* floating type (`long double`) keeps itself; every other type (integers, ...) accumulates in its own item type. Half types are spotted via `[compute_type](#compute_type)` (the only `T` whose compute type differs from itself). Override per call, e.g. `sum<float>(a)`.

---

#### view_t

```cpp
using view_t = tensor< T, Extents, Layout, own::view >
```

A non-owning view type.

Construct as `view_t<T,E>(ptr, extents)`.

---

#### local

```cpp
using local = tensor< T, Extents, Layout, own::stack >
```

Stack-owned tensor (fully static shape).

Use `local<T,E>{}`.

---

#### owned

```cpp
using owned = tensor< T, Extents, Layout, own::heap >
```

Heap-owned tensor (host only, move-only).

Use `owned<T,E>(extents)`.

### Functions

| Return | Name | Description |
|--------|------|-------------|
| `auto` | [`make_gpu`](#make_gpu)  |  |
| `auto` | [`make_pinned`](#make_pinned)  |  |
| `auto` | [`make_mapped`](#make_mapped)  |  |
| `anyrank< T, offset_t, _meta_view< offset_t > >` | [`as_anyrank`](#as_anyrank)  | Build an `anyrank` that **wraps** the caller's shape/stride arrays with **no copy** (the default) — e.g. |
| `anyrank< T, offset_t, _meta_store< offset_t, MaxRank > >` | [`as_anyrank`](#as_anyrank-1)  | `as_anyrank(data, shape, stride, ndim, copy_meta)` — COPY shape/stride into an inline store, so the carrier is trivially copyable and can be passed into a CUDA kernel by value (peel on device). |
| `bool` | [`dispatch_rank`](#dispatch_rank)  | Call `f` with a fixed-rank view of `t` chosen by its runtime `ndim`. |
| `bool` | [`dispatch_value`](#dispatch_value)  | Turn a runtime value into a compile-time one from a candidate list. |
| `auto` | [`slice`](#slice)  | A python-like slice `[start : stop : step)` for `operator()` / `take_along`. |
| `auto` | [`slice`](#slice-1)  |  |
| `auto` | [`slice`](#slice-2)  |  |
| `auto` | [`peel_at`](#peel_at)  | The `i`-th sub-view obtained by peeling `Axes...` (0 <= i < product of the peeled extents). |
| `auto` | [`peel_at`](#peel_at-1)  |  |
| `auto` | [`peel_at`](#peel_at-2)  |  |
| `auto` | [`peel`](#peel)  | Build a range of sub-views by peeling `Axes...` of `t`. |
| `auto` | [`peel`](#peel-1)  |  |
| `peel_range< MD, Axes... >` | [`peel_of`](#peel_of)  | Build a range of sub-views over a raw mdspan. |
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
| `constexpr bool` | [`own_is_host_accessible`](#own_is_host_accessible) `constexpr` `noexcept` | Whether the storage is dereferenceable from the host. |
| `tensor< typename MD::element_type, typename MD::extents_type, typename MD::layout_type, own::view >` | [`as_tensor`](#as_tensor)  | Wrap any `cuda::std::mdspan` (e.g. |
| `void` | [`fetch_add`](#fetch_add) `noexcept` | Accumulate `v` into `*p`, atomic **on the device only**. |
| `tensor< T, Extents, Layout, own::view >` | [`wrap`](#wrap)  | Wrap `p` as a non-owning view with a contiguous layout (default C-order). |
| `tensor< T, Extents, cs::layout_stride, own::view >` | [`wrap`](#wrap-1)  | Wrap `p` as a non-owning view with explicit **runtime strides** (a `layout_stride` view). |
| `tensor< T, Extents, strides< Strides... >, own::view >` | [`wrap_strided`](#wrap_strided)  | Wrap `p` as a non-owning view with per-dimension compile-time strides (may be negative). |
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

#### make_gpu

```cpp
template<class T = float, class Layout = cs::layout_right, class Extents> auto make_gpu(Extents e)
```

---

#### make_pinned

```cpp
template<class T = float, class Layout = cs::layout_right, class Extents> auto make_pinned(Extents e)
```

---

#### make_mapped

```cpp
template<class T = float, class Layout = cs::layout_right, class Extents> auto make_mapped(Extents e)
```

---

#### as_anyrank

```cpp
template<class T, class offset_t> anyrank< T, offset_t, _meta_view< offset_t > > as_anyrank(T * data, offset_t * shape, offset_t * stride, int ndim)
```

Build an `anyrank` that **wraps** the caller's shape/stride arrays with **no copy** (the default) — e.g.

straight off a DLPack tensor. The arrays must outlive the carrier. HOST only: the pointers are not valid inside a device kernel, so peel/dispatch on the host and pass the resulting fixed-rank views to the device. To instead copy into an inline, device-passable store, pass the `copy_meta` tag (overload below). DLPack strides are in ELEMENTS; numpy `__array_interface__` in BYTES (divide by the itemsize first).

---

#### as_anyrank

```cpp
template<cs::size_t MaxRank = TNY_MAX_RANK, class T, class offset_t> anyrank< T, offset_t, _meta_store< offset_t, MaxRank > > as_anyrank(T * data, const offset_t * shape, const offset_t * stride, int ndim, copy_meta_t)
```

`as_anyrank(data, shape, stride, ndim, copy_meta)` — COPY shape/stride into an inline store, so the carrier is trivially copyable and can be passed into a CUDA kernel by value (peel on device).

`MaxRank` sets the inline capacity (default `TNY_MAX_RANK`); pass it as `as_anyrank<64>(..., copy_meta)`. Accepts `const` arrays (it copies).

---

#### dispatch_rank

```cpp
template<class T, class offset_t, class Meta, class F> bool dispatch_rank(const anyrank< T, offset_t, Meta > & t, F && f)
```

Call `f` with a fixed-rank view of `t` chosen by its runtime `ndim`.

`f` is a generic callable instantiated once per possible rank; the kernel it launches is fully static. Returns false if `ndim` exceeds `max_rank`. Prefer `peel_front<Sr>` when only the trailing dims need to be static — one instantiation instead of one per total rank. 
```
dispatch_rank(as_anyrank(data, size, stride, ndim), [&](auto v){ kernel(v); });
```

---

#### dispatch_value

```cpp
template<int... Vs, class F> bool dispatch_value(int v, F && f)
```

Turn a runtime value into a compile-time one from a candidate list.

`dispatch_value<1,2,3>(D, f)` calls `f(Int<k>{})` for the matching candidate `k == D` (so `f` receives a static `integral_constant` it can use as a template argument), and returns whether any matched. 
```
dispatch_value<1,2,3>(ndim_spatial, [&](auto d){ kernel<d.value>(view); });
```

---

#### slice

```cpp
template<class A, class B> auto slice(A start, B stop)
```

A python-like slice `[start : stop : step)` for `operator()` / `take_along`.

`none` marks an open end; negative bounds wrap (count from the back); `step` defaults to 1 and may exceed 1.

`slice(1, 4)` = `[1,4)`; `slice(none, 4)` = `[0,4)`; `slice(2, none)` = `[2,end)`; `slice(0, none, 2)` = every other element; `slice(none, none)` keeps the whole axis (== `all`, which is preferable when you want the axis kept — it folds and preserves static extents). A ranged axis is resolved at run time (its extent becomes dynamic); axes kept with `all` stay static.

---

#### slice

```cpp
template<class A, class B, class S> auto slice(A start, B stop, S step)
```

---

#### slice

```cpp
template<long Start, long Stop, long Step = 1> auto slice()
```

---

#### peel_at

```cpp
template<cs::size_t... Axes, class MD> auto peel_at(const MD & src, typename MD::index_type i)
```

The `i`-th sub-view obtained by peeling `Axes...` (0 <= i < product of the peeled extents).

Peeled axes vary in row-major order (the last listed axis fastest). Returns a `md::tensor` view.

---

#### peel_at

```cpp
template<long... Axes, class T, class E, class L, own O> auto peel_at(tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i)
```

---

#### peel_at

```cpp
template<long... Axes, class T, class E, class L, own O> auto peel_at(const tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i)
```

---

#### peel

```cpp
template<long... Axes, class T, class E, class L, own O> auto peel(tensor< T, E, L, O > & t)
```

Build a range of sub-views by peeling `Axes...` of `t`.

Non-const `t` yields mutable peel; const `t` yields read-only peel.

---

#### peel

```cpp
template<long... Axes, class T, class E, class L, own O> auto peel(const tensor< T, E, L, O > & t)
```

---

#### peel_of

```cpp
template<cs::size_t... Axes, class MD> peel_range< MD, Axes... > peel_of(const MD & m)
```

Build a range of sub-views over a raw mdspan.

---

#### peel_front

```cpp
template<long N, class T, class E, class L, own O> auto peel_front(tensor< T, E, L, O > & t)
```

Peel the FIRST `N` axes -> a range of sub-views over the rest — the runtime-batch-rank half of `(*batch, *spatial, C)`.

`N` is **signed**: `peel_front<3>` peels 3 leading dims; `peel_front<-1>` keeps the last axis (peels all but it), so negative = "keep the last |N|".

---

#### peel_front

```cpp
template<long N, class T, class E, class L, own O> auto peel_front(const tensor< T, E, L, O > & t)
```

---

#### peel_front_at

```cpp
template<long N, class T, class E, class L, own O> auto peel_front_at(tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i)
```

The `i`-th sub-view obtained by peeling the first `N` axes (grid-stride style).

---

#### peel_front_at

```cpp
template<long N, class T, class E, class L, own O> auto peel_front_at(const tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i)
```

---

#### operator+

```cpp
template<class S, class T, class E, class L, own O, cs::enable_if_t< cs::is_arithmetic< S >::value, int > = 0> auto operator+(S s, const tensor< T, E, L, O > & a)
```

---

#### operator*

```cpp
template<class S, class T, class E, class L, own O, cs::enable_if_t< cs::is_arithmetic< S >::value, int > = 0> auto operator*(S s, const tensor< T, E, L, O > & a)
```

---

#### operator-

```cpp
template<class S, class T, class E, class L, own O, cs::enable_if_t< cs::is_arithmetic< S >::value, int > = 0> auto operator-(S s, const tensor< T, E, L, O > & a)
```

---

#### operator/

```cpp
template<class S, class T, class E, class L, own O, cs::enable_if_t< cs::is_arithmetic< S >::value, int > = 0> auto operator/(S s, const tensor< T, E, L, O > & a)
```

---

#### operator-

```cpp
template<class T, class E, class L, own O> auto operator-(const tensor< T, E, L, O > & a)
```

---

#### operator~

```cpp
template<class T, class E, class L, own O, cs::enable_if_t< cs::is_integral< T >::value, int > = 0> auto operator~(const tensor< T, E, L, O > & a)
```

---

#### sum

```cpp
template<class Acc = void, class T, class E, class L, own O> auto sum(const tensor< T, E, L, O > & a)
```

Sum of all elements (empty -> 0).

Accumulates in `Acc` (default: the reduce type — `double` for small floats).

---

#### prod

```cpp
template<class Acc = void, class T, class E, class L, own O> auto prod(const tensor< T, E, L, O > & a)
```

Product of all elements (empty -> 1).

Accumulates in `Acc`.

---

#### max

```cpp
template<class Acc = void, class T, class E, class L, own O> auto max(const tensor< T, E, L, O > & a)
```

Maximum element.

Requires a non-empty tensor. Computed in `Acc`.

---

#### min

```cpp
template<class Acc = void, class T, class E, class L, own O> auto min(const tensor< T, E, L, O > & a)
```

Minimum element.

Requires a non-empty tensor. Computed in `Acc`.

---

#### mean

```cpp
template<long... Axes, class T, class E, class L, own O, class R = reduce_type_t<T>, cs::enable_if_t<(sizeof...(Axes) > 0) &&_md::reduced_extents< E, Axes... >::rank_dynamic()==0, int > = 0> auto mean(const tensor< T, E, L, O > & a)
```

Mean over the named axes -> a lower-rank tensor (sum / reduced count).

Accumulates in the reduce type by default; `mean<Acc, Axes...>(a)` overrides.

---

#### dot

```cpp
template<class Acc = void, class Ta, class Ea, class La, own Oa, class Tb, class Eb, class Lb, own Ob> auto dot(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b)
```

Inner product over matching extents.

Accumulates in `Acc` (default: the reduce type of the promoted element type — `double` for small floats); `dot<float>(a, b)` overrides.

---

#### allclose

```cpp
template<class Ta, class Ea, class La, own Oa, class Tb, class Eb, class Lb, own Ob> bool allclose(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b, double rtol = 1e-5, double atol = 1e-8)
```

True if every element satisfies `|a-b| <= atol + rtol*|b|` (numpy `allclose`; broadcasts, computes in the compute type).

---

#### minimum

```cpp
template<class Ta, class Ea, class La, own Oa, class Tb, class Eb, class Lb, own Ob> auto minimum(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b)
```

---

#### maximum

```cpp
template<class Ta, class Ea, class La, own Oa, class Tb, class Eb, class Lb, own Ob> auto maximum(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b)
```

---

#### minimum

```cpp
template<class T, class E, class L, own O, class S, cs::enable_if_t< cs::is_arithmetic< S >::value, int > = 0> auto minimum(const tensor< T, E, L, O > & a, S s)
```

---

#### maximum

```cpp
template<class T, class E, class L, own O, class S, cs::enable_if_t< cs::is_arithmetic< S >::value, int > = 0> auto maximum(const tensor< T, E, L, O > & a, S s)
```

---

#### clamp

```cpp
template<class T, class E, class L, own O> auto clamp(const tensor< T, E, L, O > & a, T lo, T hi)
```

`clamp(a, lo, hi)` -> a new tensor with each element clamped.

---

#### mean

```cpp
template<class Acc = void, class T, class E, class L, own O> auto mean(const tensor< T, E, L, O > & a)
```

Arithmetic mean of all elements.

Accumulates in `Acc` (default: the reduce type — `double` for small floats); `mean<float>(a)` overrides.

---

#### own_is_owning

`constexpr` `noexcept`

```cpp
constexpr constexpr bool own_is_owning(own o) noexcept
```

Whether the mode owns (and therefore allocates) its storage.

---

#### own_is_host_accessible

`constexpr` `noexcept`

```cpp
constexpr constexpr bool own_is_host_accessible(own o) noexcept
```

Whether the storage is dereferenceable from the host.

---

#### as_tensor

```cpp
template<class MD> tensor< typename MD::element_type, typename MD::extents_type, typename MD::layout_type, own::view > as_tensor(const MD & m)
```

Wrap any `cuda::std::mdspan` (e.g.

a `submdspan` result) as a non-owning `md::tensor` view, so the tensor API applies to it.

---

#### fetch_add

`noexcept`

```cpp
template<class T> void fetch_add(T * p, T v) noexcept
```

Accumulate `v` into `*p`, atomic **on the device only**.

The scatter/"push" write: on the device many threads accumulate into overlapping outputs, which a plain `+=` would race. Device -> `atomicAdd` (`double` needs sm_60+, `__half` sm_70+; not all integer widths have an overload — that surfaces as an nvcc error at instantiation). Use via `t.add_at(v, i...)` / `t.add_<true>(...)`.

WARNING: on the **host** this is a plain `*p += v` — NOT atomic. A push kernel parallelised with std::thread over overlapping outputs races; guard those writes yourself (per-thread partials, a mutex, or std::atomic_ref).

---

#### wrap

```cpp
template<class Layout = cs::layout_right, class T, class Extents> tensor< T, Extents, Layout, own::view > wrap(T * p, Extents e)
```

Wrap `p` as a non-owning view with a contiguous layout (default C-order).

Named `wrap` (not `view`) so it never collides with the member `t.view()` that returns a raw mdspan.

---

#### wrap

```cpp
template<class T, class Extents> tensor< T, Extents, cs::layout_stride, own::view > wrap(T * p, Extents e, cs::array< typename Extents::index_type, Extents::rank()> st)
```

Wrap `p` as a non-owning view with explicit **runtime strides** (a `layout_stride` view).

Pass one stride per dimension — an `array` or a braced list — in ELEMENTS; strides may be negative (a reversed view).

`wrap(p, shape<2,3>{}, {3, 1})` is the row-major view; `{1, 2}` the column-major one. For strides known at compile time prefer `wrap_strided<S...>` (they fold into the type).

---

#### wrap_strided

```cpp
template<cs::int64_t... Strides, class T, class Extents> tensor< T, Extents, strides< Strides... >, own::view > wrap_strided(T * p, Extents e)
```

Wrap `p` as a non-owning view with per-dimension compile-time strides (may be negative).

---

#### make_view

```cpp
template<class Layout = cs::layout_right, class T, class Extents> auto make_view(T * p, Extents e)
```

`make_view<L>(ptr, extents)` — a non-owning view (alias of `wrap`).

---

#### make_local

```cpp
template<class T = float, class Layout = cs::layout_right, class Extents> auto make_local(Extents = Extents{})
```

`make_local<T>(extents)` — a stack-owned tensor (static shape).

`T` defaults to `float` (numpy's default float dtype).

---

#### make_heap

```cpp
template<class T = float, class Layout = cs::layout_right, class Extents> auto make_heap(Extents e)
```

`make_heap<T>(extents)` — a heap-owned tensor (host, move-only).

`T` defaults to `float`.

---

#### full

```cpp
template<class T = void, class Layout = cs::layout_right, class Extents, class V, class ET = cs::conditional_t<cs::is_same<T, void>::value, V, T>, cs::enable_if_t< Extents::rank_dynamic()==0, int > = 0> auto full(Extents, V v)
```

`full(extents, v)` — a new tensor filled with `v`.

The element type defaults to the **value's** type (numpy/pytorch: `full(s, 3)` is int, `full(s, 3.0)` is float); pass `full<T>(...)` to override. Unlike the value-less `zeros`/`ones` (which default to `float`), there is a value here to infer from, so we do.

---

#### zeros

```cpp
template<class T = float, class Layout = cs::layout_right, class Extents, cs::enable_if_t< Extents::rank_dynamic()==0, int > = 0> auto zeros(Extents e)
```

`zeros<T>(extents)` / `ones<T>(extents)` — a new tensor of 0s / 1s.

`T` defaults to `float`. Static shape -> stack (host+device); dynamic -> heap (host only). The annotation is split so it matches the overload `full` resolves to.

---

#### ones

```cpp
template<class T = float, class Layout = cs::layout_right, class Extents, cs::enable_if_t< Extents::rank_dynamic()==0, int > = 0> auto ones(Extents e)
```

---

#### arange

```cpp
template<class T = cs::int64_t> auto arange(long n)
```

`arange<T>(n)` — a 1-D tensor `[0, 1, ..., n-1]` (heap, host).

`T` defaults to `int64_t` (an integer range, like numpy `[arange(n)](#arange)`).

---

#### arange

```cpp
template<class T = cs::int64_t, long N> auto arange()
```

Static `arange<T, N>()` — a stack `[0..N-1]` (host+device, folds).

---

#### arange

```cpp
template<class T = cs::int64_t, class V, V N> auto arange(cs::integral_constant< V, N >)
```

`arange<T>(Int<N>())` — the static form spelled with a static integer.

### Variables

| Return | Name | Description |
|--------|------|-------------|
| `constexpr cs::full_extent_t` | [`all`](#all) `constexpr` | Keep-this-axis marker for slicing (an alias of `full_extent`). |
| `constexpr copy_meta_t` | [`copy_meta`](#copy_meta) `constexpr` |  |
| `constexpr none_t` | [`none`](#none) `constexpr` |  |
| `constexpr ellipsis_t` | [`ellipsis`](#ellipsis) `constexpr` |  |
| `constexpr cs::int64_t` | [`dynamic_stride`](#dynamic_stride) `constexpr` | Per-dimension dynamic-stride sentinel. |

---

#### all

`constexpr`

```cpp
constexpr cs::full_extent_t all {}
```

Keep-this-axis marker for slicing (an alias of `full_extent`).

---

#### copy_meta

`constexpr`

```cpp
constexpr copy_meta_t copy_meta {}
```

---

#### none

`constexpr`

```cpp
constexpr none_t none {}
```

---

#### ellipsis

`constexpr`

```cpp
constexpr ellipsis_t ellipsis {}
```

---

#### dynamic_stride

`constexpr`

```cpp
constexpr cs::int64_t dynamic_stride = cs::numeric_limits<cs::int64_t>()
```

Per-dimension dynamic-stride sentinel.

Strides are **signed**: a negative stride is a legitimate value (reversed / flipped views, and DLPack tensors carry them). So — unlike `shape<...>`, where `-1` marks a dynamic extent — we cannot use `-1` to mean "runtime" for a stride. Instead a reserved out-of-band value (`INT64_MIN`) marks a dynamic stride, leaving every ordinary stride (including negatives) expressible.



## anyrank

```cpp
#include <dynamic.h>
```

```cpp
template<class T, class offset_t = cs::int64_t, class Meta = _meta_store<offset_t, TNY_MAX_RANK>>
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
| `anyrank_front< T, offset_t, Meta, static_cast< cs::size_t >(N< 0 ? -N :0)>` | [`peel_front`](#peel_front-2) `const` `inline` | Peel the leading batch axes -> an iterable of fixed-rank-`\|N\|` sub-views (range-for, `[size()](#size-1)`, `operator[]`). |

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
template<cs::size_t R> inline dyn_tensor< T, offset_t, R > fixed() const
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
template<long N> inline anyrank_front< T, offset_t, Meta, static_cast< cs::size_t >(N< 0 ? -N :0)> peel_front() const
```

Defined in include/teeny/dynamic.h:136

Peel the leading batch axes -> an iterable of fixed-rank-`|N|` sub-views (range-for, `[size()](#size-1)`, `operator[]`).

The `(*batch, *spatial, C)` boundary with `|N| = spatial + channels`: one kernel instantiation for `|N|`, not one per total rank. `N` is negative (keep the last |N| dims), as on the tensor.

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr cs::size_t` | [`max_rank`](#max_rank) `static` `constexpr` |  |
| `constexpr bool` | [`device_passable`](#device_passable) `static` `constexpr` |  |

---

#### max_rank

`static` `constexpr`

```cpp
constexpr cs::size_t max_rank =
        Meta::extents_type::static_extent(0) != cs::dynamic_extent
            ? Meta::extents_type::static_extent(0) : cs::size_t()
```

Defined in include/teeny/dynamic.h:69

---

#### device_passable

`static` `constexpr`

```cpp
constexpr bool device_passable =
        (Meta::extents_type::static_extent(0) != cs::dynamic_extent)
```

Defined in include/teeny/dynamic.h:75



## anyrank_front

```cpp
#include <dynamic.h>
```

```cpp
template<class T, class offset_t, class Meta, cs::size_t Sr>
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

Half types compute in `float` (16-bit accumulation loses precision fast — jitfields' `reduce_t` pattern — and it lets the engines avoid depending on native half host operators). Everything else computes in itself.

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

Defined in include/teeny/storage.h:39

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
template<class T> static inline T * allocate(cs::size_t n)
```

Defined in include/teeny/storage.h:40

---

#### deallocate

`static` `inline`

```cpp
template<class T> static inline void deallocate(T * p)
```

Defined in include/teeny/storage.h:41



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
template<class T> static inline T * allocate(cs::size_t n)
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
template<class T> static inline T * allocate(cs::size_t n)
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
template<class T> static inline T * allocate(cs::size_t n)
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

Defined in include/teeny/storage.h:49

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

Defined in include/teeny/storage.h:50

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

Defined in include/teeny/storage.h:51

Defaulted constructor.

---

#### owning_storage

`inline` `explicit`

```cpp
inline explicit owning_storage(cs::size_t n)
```

Defined in include/teeny/storage.h:52

---

#### owning_storage

```cpp
owning_storage(const owning_storage &) = delete
```

Defined in include/teeny/storage.h:53

Deleted constructor.

---

#### owning_storage

`inline` `noexcept`

```cpp
inline owning_storage(owning_storage && o) noexcept
```

Defined in include/teeny/storage.h:55

---

#### data

`inline` `noexcept`

```cpp
inline T * data() noexcept
```

Defined in include/teeny/storage.h:61

---

#### data

`const` `inline` `noexcept`

```cpp
inline const T * data() const noexcept
```

Defined in include/teeny/storage.h:62



## peel_range

```cpp
#include <iterate.h>
```

```cpp
template<class MD, cs::size_t... Axes>
struct peel_range
```

Defined in include/teeny/iterate.h:102

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

Defined in include/teeny/iterate.h:104

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

Defined in include/teeny/iterate.h:106

---

#### operator[]

`const` `inline`

```cpp
inline auto operator[](index_type i) const
```

Defined in include/teeny/iterate.h:112

---

#### begin

`const` `inline`

```cpp
inline iterator begin() const
```

Defined in include/teeny/iterate.h:122

---

#### end

`const` `inline`

```cpp
inline iterator end() const
```

Defined in include/teeny/iterate.h:123

### Public Types

| Name | Description |
|------|-------------|
| [`index_type`](#index_type)  |  |

---

#### index_type

```cpp
using index_type = typename MD::index_type
```

Defined in include/teeny/iterate.h:103



## iterator

```cpp
#include <iterate.h>
```

```cpp
struct iterator
```

Defined in include/teeny/iterate.h:114

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

Defined in include/teeny/iterate.h:115

---

#### i

```cpp
index_type i
```

Defined in include/teeny/iterate.h:116

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

Defined in include/teeny/iterate.h:117

---

#### operator++

`inline`

```cpp
inline iterator & operator++()
```

Defined in include/teeny/iterate.h:118

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const iterator & o) const
```

Defined in include/teeny/iterate.h:119

---

#### operator==

`const` `inline`

```cpp
inline bool operator==(const iterator & o) const
```

Defined in include/teeny/iterate.h:120



## storage

```cpp
template<class T, own O, cs::size_t N>
struct storage
```

Defined in include/teeny/storage.h:70



## gpu, N >

```cpp
#include <cuda.h>
```

```cpp
template<class T, cs::size_t N>
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



## heap, N >

```cpp
#include <storage.h>
```

```cpp
template<class T, cs::size_t N>
struct heap, N >
```

Defined in include/teeny/storage.h:91

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
template<class T, cs::size_t N>
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
template<class T, cs::size_t N>
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
template<class T, cs::size_t N>
struct stack, N >
```

Defined in include/teeny/storage.h:83

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`a`](#a) | `variable` | Declared here |
| [`data`](#data-3) | `function` | Declared here |
| [`data`](#data-4) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `cs::array< T, N >` | [`a`](#a)  |  |

---

#### a

```cpp
cs::array< T, N > a {}
```

Defined in include/teeny/storage.h:84

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `constexpr T *` | [`data`](#data-3) `inline` `constexpr` `noexcept` |  |
| `constexpr const T *` | [`data`](#data-4) `const` `inline` `constexpr` `noexcept` |  |

---

#### data

`inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr T * data() noexcept
```

Defined in include/teeny/storage.h:85

---

#### data

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const T * data() const noexcept
```

Defined in include/teeny/storage.h:86



## view, N >

```cpp
#include <storage.h>
```

```cpp
template<class T, cs::size_t N>
struct view, N >
```

Defined in include/teeny/storage.h:74

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p-1) | `variable` | Declared here |
| [`storage`](#storage-1) | `function` | Declared here |
| [`storage`](#storage-2) | `function` | Declared here |
| [`data`](#data-5) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `T *` | [`p`](#p-1)  |  |

---

#### p

```cpp
T * p = nullptr
```

Defined in include/teeny/storage.h:75

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
|  | [`storage`](#storage-1)  | Defaulted constructor. |
| `constexpr` | [`storage`](#storage-2) `inline` `constexpr` `noexcept` |  |
| `constexpr T *` | [`data`](#data-5) `const` `inline` `constexpr` `noexcept` |  |

---

#### storage

```cpp
storage() = default
```

Defined in include/teeny/storage.h:76

Defaulted constructor.

---

#### storage

`inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr storage(T * q) noexcept
```

Defined in include/teeny/storage.h:77

---

#### data

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr T * data() const noexcept
```

Defined in include/teeny/storage.h:78



## storage_size

```cpp
#include <storage.h>
```

```cpp
template<class Mapping, bool Stack>
struct storage_size
```

Defined in include/teeny/storage.h:97

Storage element count for a stack tensor (0 for view/owning).

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`value`](#value) | `variable` | Declared here |

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr cs::size_t` | [`value`](#value) `static` `constexpr` |  |

---

#### value

`static` `constexpr`

```cpp
constexpr cs::size_t value = 0
```

Defined in include/teeny/storage.h:97



## storage_size< Mapping, true >

```cpp
#include <storage.h>
```

```cpp
template<class Mapping>
struct storage_size< Mapping, true >
```

Defined in include/teeny/storage.h:99

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`value`](#value-1) | `variable` | Declared here |

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr cs::size_t` | [`value`](#value-1) `static` `constexpr` |  |

---

#### value

`static` `constexpr`

```cpp
constexpr cs::size_t value =
        static_cast<cs::size_t>(Mapping().required_span_size())
```

Defined in include/teeny/storage.h:100



## strides

```cpp
#include <layout.h>
```

```cpp
template<cs::int64_t... S>
struct strides
```

Defined in include/teeny/layout.h:66

An mdspan layout policy with **per-dimension static or dynamic strides** — the stride analogue of `extents`/`shape`.

`layout_right`/`layout_left` give contiguous (extent-derived) strides; `layout_stride` stores every stride at run time. `strides<S...>` bakes the KNOWN strides into the type (folding to immediates, like jitfields' posdef `Pointer<T,S>`) — **including negative strides** — while any dimension marked `dynamic_stride` is supplied at run time: 
```
tensor<float, shape<3,4>, strides<4,1>>(ptr);                    // static, folds
tensor<float, shape<3,4>, strides<-4,1>>(ptr);                   // reversed rows
tensor<float, shape<-1,4>, strides<dynamic_stride,1>>(ptr, {n}); // outer stride runtime
```
 When every stride is static the mapping is empty (EBO), so a stack tensor is still exactly `sizeof` its data. Only the *dynamic* strides are stored.

Note: CCCL's `cs::submdspan` is only defined for the standard layouts, so it does not apply here — but teeny's own slicing/`take_along`/`permute`/`flip`/ `peel` build their views by hand (no submdspan), so they all work on a strides<...> source and in fact fold their output strides the same way. And `required_span_size` assumes non-negative strides — negative strides are for VIEWS into existing storage, not owning allocation.

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
| `constexpr cs::size_t` | [`N`](#n) `static` `constexpr` |  |
| `constexpr cs::int64_t` | [`S_`](#s_) `static` `constexpr` |  |

---

#### N

`static` `constexpr`

```cpp
constexpr cs::size_t N = sizeof...(S)
```

Defined in include/teeny/layout.h:67

---

#### S_

`static` `constexpr`

```cpp
constexpr cs::int64_t S_ = { S... }
```

Defined in include/teeny/layout.h:68

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `constexpr cs::size_t` | [`ndyn`](#ndyn) `static` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`all_static`](#all_static) `static` `inline` `constexpr` `noexcept` |  |
| `constexpr cs::size_t` | [`slot`](#slot) `static` `inline` `constexpr` `noexcept` |  |

---

#### ndyn

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr cs::size_t ndyn() noexcept
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
constexpr static inline constexpr cs::size_t slot(cs::size_t r) noexcept
```

Defined in include/teeny/layout.h:75



## mapping

```cpp
#include <layout.h>
```

```cpp
template<class Extents>
struct mapping
```

Defined in include/teeny/layout.h:82

> **Inherits:** `ndyn()>`, `Extents`

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
| `constexpr const Extents &` | [`extents`](#extents) `const` `inline` `constexpr` `noexcept` |  |
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
template<cs::size_t M = strides::ndyn(), cs::enable_if_t< M==0, int > = 0> constexpr inline constexpr mapping(const Extents & e)
```

Defined in include/teeny/layout.h:94

Fully-static strides: construct from extents only.

---

#### mapping

`inline` `constexpr`

```cpp
constexpr inline constexpr mapping(const Extents & e, const cs::array< index_type, strides::ndyn()> & dyn)
```

Defined in include/teeny/layout.h:97

Mixed strides: extents + the runtime strides (dim order, dynamic ones only).

---

#### extents

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const Extents & extents() const noexcept
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
using extents_type = Extents
```

Defined in include/teeny/layout.h:83

---

#### index_type

```cpp
using index_type = typename Extents::index_type
```

Defined in include/teeny/layout.h:84

---

#### rank_type

```cpp
using rank_type = typename Extents::rank_type
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
template<class T, class Extents, class Layout, own O>
struct tensor
```

Defined in include/teeny/tensor.h:65

> **Inherits:** `template mapping< Extents >`

One N-dimensional tensor, parameterised by ownership.

The layout / extents / offset mapping is delegated to `cuda::std::mdspan` (the mapping lives in an empty base, so a fully-static tensor is exactly the size of its data). Ownership is a policy: `[own::view](#namespacetny_1a6a432f80fb491dbcb5d4b0692616b918a1bda80f2be4d3658e0baa43fbe7ae8c1)` (non-owning, trivially copyable, kernel-passable), `[own::stack](#namespacetny_1a6a432f80fb491dbcb5d4b0692616b918afac2a47adace059aff113283a03f6760)` (inline storage, static shape), or `[own::heap](#namespacetny_1a6a432f80fb491dbcb5d4b0692616b918a4d4a9aa362b6ffe089fd2e992ccf4f5f)` (host-only, move-only). The tensor's copy/move semantics are induced by the storage member, not hand-written.

#### Template Parameters
* `T` Element type. 

* `Extents` `cuda::std::extents<Idx, E...>` (static or dynamic per dim). 

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
| [`data`](#data-6) | `function` | Declared here |
| [`data`](#data-7) | `function` | Declared here |
| [`view`](#view) | `function` | Declared here |
| [`view`](#view-1) | `function` | Declared here |
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
| [`take_along`](#take_along) | `function` | Declared here |
| [`take_along`](#take_along-1) | `function` | Declared here |
| [`permute`](#permute) | `function` | Declared here |
| [`permute`](#permute-1) | `function` | Declared here |
| [`flip`](#flip) | `function` | Declared here |
| [`flip`](#flip-1) | `function` | Declared here |
| [`clone`](#clone) | `function` | Declared here |
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
| [`buffer_size`](#buffer_size) | `variable` | Declared here |
| [`is_strides_layout`](#is_strides_layout) | `variable` | Declared here |
| [`is_contiguous_layout`](#is_contiguous_layout) | `variable` | Declared here |
| [`rank`](#rank) | `function` | Declared here |
| [`element_type`](#element_type) | `typedef` | Declared here |
| [`extents_type`](#extents_type-1) | `typedef` | Declared here |
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

Defined in include/teeny/tensor.h:79

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
| `constexpr const Extents &` | [`extents`](#extents-1) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr auto` | [`extent`](#extent) `const` `inline` `constexpr` `noexcept` | Extent of an axis given by a STATIC index (`extent(Int<0>())`): a compile-time `integral_constant` when that extent is static, else a runtime `index_type`. |
| `constexpr index_type` | [`extent`](#extent-1) `const` `inline` `constexpr` `noexcept` | Extent of an axis given by a RUNTIME index (`extent(0)`). |
| `constexpr const Extents &` | [`shape`](#shape-2) `const` `inline` `constexpr` `noexcept` | `[shape()](#shape)` / `[shape(d)](#shape)` — python-friendly aliases of `extents()` / `extent(d)` (static index -> integral_constant, runtime -> value). |
| `constexpr auto` | [`shape`](#shape-3) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr auto` | [`stride`](#stride-2) `const` `inline` `constexpr` `noexcept` | Stride of an axis given by a STATIC index (`stride(Int<0>())`): a compile-time `integral_constant` when known statically (static- stride layout; a contiguous layout over static extents; or the always-unit stride of a contiguous layout even for dynamic shapes). |
| `constexpr index_type` | [`stride`](#stride-3) `const` `inline` `constexpr` `noexcept` | Stride of an axis given by a RUNTIME index (`stride(0)`). |
| `constexpr index_type` | [`numel`](#numel) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_contiguous`](#is_contiguous) `const` `inline` `constexpr` `noexcept` | Whether the strides are dense row-major (C-contiguous). |
| `T *` | [`data`](#data-6) `inline` `noexcept` |  |
| `const T *` | [`data`](#data-7) `const` `inline` `noexcept` |  |
| `view_type` | [`view`](#view) `inline` `noexcept` |  |
| `const_view_type` | [`view`](#view-1) `const` `inline` `noexcept` |  |
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
| `auto` | [`take_along`](#take_along) `inline` `noexcept` | Index/slice one or more named axes; other axes are kept. |
| `auto` | [`take_along`](#take_along-1) `const` `inline` `noexcept` |  |
| `auto` | [`permute`](#permute) `inline` `noexcept` | Reorder the axes (a permutation of 0..N-1; negatives wrap) -> a rank-N view. |
| `auto` | [`permute`](#permute-1) `const` `inline` `noexcept` |  |
| `auto` | [`flip`](#flip) `inline` `noexcept` | Reverse axis `Ax` (negatives wrap) -> a view (numpy `flip`). |
| `auto` | [`flip`](#flip-1) `const` `inline` `noexcept` |  |
| `auto` | [`clone`](#clone) `const` `inline` |  |
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
| `tensor< T, Extents, cs::layout_right, own::stack >` | [`operator++`](#operator-28) `inline` |  |
| `tensor< T, Extents, cs::layout_right, own::stack >` | [`operator--`](#operator-29) `inline` |  |
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

Defined in include/teeny/tensor.h:82

Defaulted constructor.

---

#### tensor

`inline`

```cpp
template<own OO = O, cs::enable_if_t< OO==own::view, int > = 0> inline tensor(T * p, mapping_type m)
```

Defined in include/teeny/tensor.h:86

View constructor: wrap `p` with the given mapping.

---

#### tensor

`inline` `explicit`

```cpp
template<own OO = O, cs::enable_if_t< OO==own::view &&is_static &&(_contiguous_layout< Layout >::value||_strides_all_static< Layout >::value), int > = 0> inline explicit tensor(T * p)
```

Defined in include/teeny/tensor.h:93

View constructor from a pointer alone — for a fully-static geometry (static extents AND a fully determined layout: contiguous, or an all-static `strides<...>`).

e.g. `tensor<float, shape<3,4>, strides<4,1>>(ptr)`.

---

#### tensor

`inline`

```cpp
template<own OO = O, cs::enable_if_t< OO==own::view &&cs::is_constructible< mapping_type, Extents >::value, int > = 0> inline tensor(T * p, Extents e)
```

Defined in include/teeny/tensor.h:97

View constructor from a pointer + extents (contiguous / static-stride layouts).

---

#### tensor

`inline` `explicit`

```cpp
template<own OO = O, cs::enable_if_t< own_is_owning(OO), int > = 0> inline explicit tensor(mapping_type m)
```

Defined in include/teeny/tensor.h:109

Owning constructor: allocate storage for `m` (heap/device/host/pinned).

---

#### tensor

`inline` `explicit`

```cpp
template<own OO = O, cs::enable_if_t< own_is_owning(OO) &&cs::is_constructible< mapping_type, Extents >::value, int > = 0> inline explicit tensor(Extents e)
```

Defined in include/teeny/tensor.h:114

Owning constructor from extents (contiguous / static-stride layouts).

---

#### mapping

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const mapping_type & mapping() const noexcept
```

Defined in include/teeny/tensor.h:119

---

#### extents

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const Extents & extents() const noexcept
```

Defined in include/teeny/tensor.h:120

---

#### extent

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx, cs::enable_if_t< _is_ic< Idx >::value, int > = 0> constexpr inline constexpr auto extent(Idx) const noexcept
```

Defined in include/teeny/tensor.h:128

Extent of an axis given by a STATIC index (`extent(Int<0>())`): a compile-time `integral_constant` when that extent is static, else a runtime `index_type`.

---

#### extent

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx, cs::enable_if_t<!_is_ic< Idx >::value, int > = 0> constexpr inline constexpr index_type extent(Idx d) const noexcept
```

Defined in include/teeny/tensor.h:137

Extent of an axis given by a RUNTIME index (`extent(0)`).

---

#### shape

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const Extents & shape() const noexcept
```

Defined in include/teeny/tensor.h:142

`[shape()](#shape)` / `[shape(d)](#shape)` — python-friendly aliases of `extents()` / `extent(d)` (static index -> integral_constant, runtime -> value).

---

#### shape

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx> constexpr inline constexpr auto shape(Idx d) const noexcept
```

Defined in include/teeny/tensor.h:143

---

#### stride

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx, cs::enable_if_t< _is_ic< Idx >::value, int > = 0> constexpr inline constexpr auto stride(Idx) const noexcept
```

Defined in include/teeny/tensor.h:150

Stride of an axis given by a STATIC index (`stride(Int<0>())`): a compile-time `integral_constant` when known statically (static- stride layout; a contiguous layout over static extents; or the always-unit stride of a contiguous layout even for dynamic shapes).

---

#### stride

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx, cs::enable_if_t<!_is_ic< Idx >::value, int > = 0> constexpr inline constexpr index_type stride(Idx d) const noexcept
```

Defined in include/teeny/tensor.h:168

Stride of an axis given by a RUNTIME index (`stride(0)`).

---

#### numel

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr index_type numel() const noexcept
```

Defined in include/teeny/tensor.h:170

---

#### is_contiguous

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_contiguous() const noexcept
```

Defined in include/teeny/tensor.h:176

Whether the strides are dense row-major (C-contiguous).

---

#### data

`inline` `noexcept`

```cpp
inline T * data() noexcept
```

Defined in include/teeny/tensor.h:186

---

#### data

`const` `inline` `noexcept`

```cpp
inline const T * data() const noexcept
```

Defined in include/teeny/tensor.h:187

---

#### view

`inline` `noexcept`

```cpp
inline view_type view() noexcept
```

Defined in include/teeny/tensor.h:188

---

#### view

`const` `inline` `noexcept`

```cpp
inline const_view_type view() const noexcept
```

Defined in include/teeny/tensor.h:189

---

#### operator()

`inline` `noexcept`

```cpp
template<class... Args, cs::enable_if_t<(_is_index< Args >::value &&...), int > = 0> inline T & operator()(Args... a) noexcept
```

Defined in include/teeny/tensor.h:308

Element access when every argument is an integer (negatives wrap).

---

#### operator()

`const` `inline` `noexcept`

```cpp
template<class... Args, cs::enable_if_t<(_is_index< Args >::value &&...), int > = 0> inline const T & operator()(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:311

---

#### at

`inline` `noexcept`

```cpp
template<class... Args, cs::enable_if_t<(_is_index< Args >::value &&...), int > = 0> inline auto at(Args... a) noexcept
```

Defined in include/teeny/tensor.h:321

`at(i...)` — a single element as a **rank-0 VIEW** (all-integer args; negatives wrap).

Unlike `operator()`, which returns a plain `T&`, this is a view, so the whole tensor API applies to one element: `x.at(i,j) = 3` writes it, `float v = x.at(i,j)` reads it (rank-0 tensors convert to/from `T`), and `x.at(i,j).add_<true>(v)` is an atomic scatter.

---

#### at

`const` `inline` `noexcept`

```cpp
template<class... Args, cs::enable_if_t<(_is_index< Args >::value &&...), int > = 0> inline auto at(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:326

---

#### add_at

`inline` `noexcept`

```cpp
template<class... Args, cs::enable_if_t<(_is_index< Args >::value &&...), int > = 0> inline void add_at(T v, Args... a) noexcept
```

Defined in include/teeny/tensor.h:335

Scatter-accumulate: `(*this)(i...) += v`, atomic on the device — the write half of a "push"/splat kernel.

Shorthand for `at(i...).add_<true>(v)` (integer indices only; negatives wrap).

---

#### operator()

`inline` `noexcept`

```cpp
template<class... Args, cs::enable_if_t<!(_is_index< Args >::value &&...) &&!_has_ellipsis< Args... >::value, int > = 0> inline auto operator()(Args... a) noexcept
```

Defined in include/teeny/tensor.h:343

Sub-view when any argument is a slice (`all`, `slice(a,b[,step])`).

Integer args drop their axis, `all` keeps it, a range keeps a strided window — all via the one gather (folds static strides into `strides<...>`; works on any source layout).

---

#### operator()

`const` `inline` `noexcept`

```cpp
template<class... Args, cs::enable_if_t<!(_is_index< Args >::value &&...) &&!_has_ellipsis< Args... >::value, int > = 0> inline auto operator()(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:346

---

#### operator()

`inline` `noexcept`

```cpp
template<class... Args, cs::enable_if_t< _has_ellipsis< Args... >::value, int > = 0> inline decltype(auto) operator()(Args... a) noexcept
```

Defined in include/teeny/tensor.h:354

Ellipsis form: exactly one `ellipsis` in the args expands to `rank - (#other args)` copies of `all`, then the call re-runs — so `t(1, ellipsis, 2)` on rank 5 is `t(1, all, all, all, 2)`.

What remains decides the result (all integers -> element, else view).

---

#### operator()

`const` `inline` `noexcept`

```cpp
template<class... Args, cs::enable_if_t< _has_ellipsis< Args... >::value, int > = 0> inline decltype(auto) operator()(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:357

---

#### operator T

`const` `inline` `noexcept`

```cpp
template<cs::size_t R = rank(), cs::enable_if_t< R==0, int > = 0> inline operator T() const noexcept
```

Defined in include/teeny/tensor.h:365

---

#### item

`const` `inline` `noexcept`

```cpp
template<cs::size_t R = rank(), cs::enable_if_t< R==0, int > = 0> inline T item() const noexcept
```

Defined in include/teeny/tensor.h:370

The single element of a rank-0 tensor (explicit reader).

---

#### take_along

`inline` `noexcept`

```cpp
template<long... Axes, class... Args> inline auto take_along(Args... args) noexcept
```

Defined in include/teeny/tensor.h:408

Index/slice one or more named axes; other axes are kept.

`take_along<Axes...>(args...)` applies `args[k]` to axis `Axes[k]` (each an integer &ndash; negatives wrap &ndash; or a slice `all`/`rng`) and keeps every other axis, returning a view. e.g. `t.take_along<1>(2)` drops axis 1 at index 2; `t.take_along<0,2>(i, rng(1,4))` binds axes 0 and 2 at once.

---

#### take_along

`const` `inline` `noexcept`

```cpp
template<long... Axes, class... Args> inline auto take_along(Args... args) const noexcept
```

Defined in include/teeny/tensor.h:413

---

#### permute

`inline` `noexcept`

```cpp
template<long... Perm> inline auto permute() noexcept
```

Defined in include/teeny/tensor.h:420

Reorder the axes (a permutation of 0..N-1; negatives wrap) -> a rank-N view.

---

#### permute

`const` `inline` `noexcept`

```cpp
template<long... Perm> inline auto permute() const noexcept
```

Defined in include/teeny/tensor.h:423

---

#### flip

`inline` `noexcept`

```cpp
template<long Ax = 0> inline auto flip() noexcept
```

Defined in include/teeny/tensor.h:429

Reverse axis `Ax` (negatives wrap) -> a view (numpy `flip`).

Uses a negative stride, so the index type must be signed (`shape<...>` is).

---

#### flip

`const` `inline` `noexcept`

```cpp
template<long Ax = 0> inline auto flip() const noexcept
```

Defined in include/teeny/tensor.h:432

---

#### clone

`const` `inline`

```cpp
template<bool S = is_static, cs::enable_if_t<!S, int > = 0> inline auto clone() const
```

Defined in include/teeny/tensor.h:441

---

#### reshape

`inline` `noexcept`

```cpp
template<long... NewExt> inline auto reshape() noexcept
```

Defined in include/teeny/tensor.h:461

View this tensor as a new shape — requires it be C-contiguous (`[clone()](#clone)` first otherwise) and the element count to match.

One extent may be **`-1`** (numpy-style), inferred from the total size: `t.reshape<6,-1>()`.

---

#### reshape

`const` `inline` `noexcept`

```cpp
template<long... NewExt> inline auto reshape() const noexcept
```

Defined in include/teeny/tensor.h:462

---

#### recast

`inline`

```cpp
template<class NewE> inline auto recast()
```

Defined in include/teeny/tensor.h:486

Reinterpret with a MORE-STATIC extents type of the same rank — recover statically-known inner dims at the dynamic (ndarray) boundary: a runtime `(n,3,3)` view -> `.recast<shape<-1,3,3>>()` so the `3`s fold.

**Requires a C-contiguous tensor** (`[clone()](#clone)` first otherwise); each static dim of `NewE` is validated against the actual extent.

---

#### recast

`const` `inline`

```cpp
template<class NewE> inline auto recast() const
```

Defined in include/teeny/tensor.h:487

---

#### flatten

`inline` `noexcept`

```cpp
inline auto flatten() noexcept
```

Defined in include/teeny/tensor.h:490

View as 1-D (`ravel`) — requires C-contiguous (`[clone()](#clone)` first).

---

#### flatten

`const` `inline` `noexcept`

```cpp
inline auto flatten() const noexcept
```

Defined in include/teeny/tensor.h:495

---

#### unsqueeze

`inline` `noexcept`

```cpp
template<long Ax = 0> inline auto unsqueeze() noexcept
```

Defined in include/teeny/tensor.h:505

Insert a size-1 axis at position `Ax` (numpy `newaxis`/`unsqueeze`) -> a rank-(N+1) view.

Negative `Ax` counts from the back, so `.unsqueeze<-1>()` appends a trailing axis: `(H,W)` -> `(H,W,1)`.

---

#### unsqueeze

`const` `inline` `noexcept`

```cpp
template<long Ax = 0> inline auto unsqueeze() const noexcept
```

Defined in include/teeny/tensor.h:508

---

#### squeeze

`inline` `noexcept`

```cpp
template<long Ax = _ax_all> inline auto squeeze() noexcept
```

Defined in include/teeny/tensor.h:527

Drop a size-1 axis `Ax` (negatives wrap) -> a rank-(N-1) view.

`[squeeze()](#squeeze)` (no axis) drops EVERY statically-size-1 axis.

---

#### squeeze

`const` `inline` `noexcept`

```cpp
template<long Ax = _ax_all> inline auto squeeze() const noexcept
```

Defined in include/teeny/tensor.h:533

---

#### flip

`inline` `noexcept`

```cpp
template<class I, cs::enable_if_t< _is_ic< I >::value, int > = 0> inline auto flip(I) noexcept
```

Defined in include/teeny/tensor.h:543

---

#### flip

`const` `inline` `noexcept`

```cpp
template<class I, cs::enable_if_t< _is_ic< I >::value, int > = 0> inline auto flip(I) const noexcept
```

Defined in include/teeny/tensor.h:544

---

#### squeeze

`inline` `noexcept`

```cpp
template<class I, cs::enable_if_t< _is_ic< I >::value, int > = 0> inline auto squeeze(I) noexcept
```

Defined in include/teeny/tensor.h:545

---

#### squeeze

`const` `inline` `noexcept`

```cpp
template<class I, cs::enable_if_t< _is_ic< I >::value, int > = 0> inline auto squeeze(I) const noexcept
```

Defined in include/teeny/tensor.h:546

---

#### unsqueeze

`inline` `noexcept`

```cpp
template<class I, cs::enable_if_t< _is_ic< I >::value, int > = 0> inline auto unsqueeze(I) noexcept
```

Defined in include/teeny/tensor.h:547

---

#### unsqueeze

`const` `inline` `noexcept`

```cpp
template<class I, cs::enable_if_t< _is_ic< I >::value, int > = 0> inline auto unsqueeze(I) const noexcept
```

Defined in include/teeny/tensor.h:548

---

#### permute

`inline` `noexcept`

```cpp
template<class... I, cs::enable_if_t<(sizeof...(I) > 0) &&(_is_ic< I >::value &&...), int > = 0> inline auto permute(I...) noexcept
```

Defined in include/teeny/tensor.h:549

---

#### permute

`const` `inline` `noexcept`

```cpp
template<class... I, cs::enable_if_t<(sizeof...(I) > 0) &&(_is_ic< I >::value &&...), int > = 0> inline auto permute(I...) const noexcept
```

Defined in include/teeny/tensor.h:550

---

#### reshape

`inline` `noexcept`

```cpp
template<class... I, cs::enable_if_t<(sizeof...(I) > 0) &&(_is_ic< I >::value &&...), int > = 0> inline auto reshape(I...) noexcept
```

Defined in include/teeny/tensor.h:551

---

#### reshape

`const` `inline` `noexcept`

```cpp
template<class... I, cs::enable_if_t<(sizeof...(I) > 0) &&(_is_ic< I >::value &&...), int > = 0> inline auto reshape(I...) const noexcept
```

Defined in include/teeny/tensor.h:552

---

#### recast

`inline`

```cpp
template<class NewE> inline auto recast(NewE)
```

Defined in include/teeny/tensor.h:553

---

#### recast

`const` `inline`

```cpp
template<class NewE> inline auto recast(NewE) const
```

Defined in include/teeny/tensor.h:554

---

#### add_

```cpp
template<bool Atomic = false, class B, cs::enable_if_t<!cs::is_arithmetic< B >::value, int > = 0> tensor & add_(const B & b)
```

Defined in include/teeny/tensor.h:560

---

#### sub_

```cpp
template<bool Atomic = false, class B, cs::enable_if_t<!cs::is_arithmetic< B >::value, int > = 0> tensor & sub_(const B & b)
```

Defined in include/teeny/tensor.h:561

---

#### mul_

```cpp
template<class B, cs::enable_if_t<!cs::is_arithmetic< B >::value, int > = 0> tensor & mul_(const B & b)
```

Defined in include/teeny/tensor.h:562

---

#### div_

```cpp
template<class B, cs::enable_if_t<!cs::is_arithmetic< B >::value, int > = 0> tensor & div_(const B & b)
```

Defined in include/teeny/tensor.h:563

---

#### add_

```cpp
template<bool Atomic = false> tensor & add_(T s)
```

Defined in include/teeny/tensor.h:564

---

#### sub_

```cpp
template<bool Atomic = false> tensor & sub_(T s)
```

Defined in include/teeny/tensor.h:565

---

#### mul_

```cpp
tensor & mul_(T s)
```

Defined in include/teeny/tensor.h:566

---

#### div_

```cpp
tensor & div_(T s)
```

Defined in include/teeny/tensor.h:567

---

#### operator+=

`inline`

```cpp
template<class B> inline tensor & operator+=(const B & b)
```

Defined in include/teeny/tensor.h:571

---

#### operator-=

`inline`

```cpp
template<class B> inline tensor & operator-=(const B & b)
```

Defined in include/teeny/tensor.h:572

---

#### operator*=

`inline`

```cpp
template<class B> inline tensor & operator*=(const B & b)
```

Defined in include/teeny/tensor.h:573

---

#### operator/=

`inline`

```cpp
template<class B> inline tensor & operator/=(const B & b)
```

Defined in include/teeny/tensor.h:574

---

#### copy_

```cpp
template<class B> tensor & copy_(const B & b)
```

Defined in include/teeny/tensor.h:577

---

#### fill_

```cpp
tensor & fill_(T s)
```

Defined in include/teeny/tensor.h:578

---

#### zero_

```cpp
tensor & zero_()
```

Defined in include/teeny/tensor.h:579

---

#### iota_

```cpp
tensor & iota_(T start = T(0), T step = T(1))
```

Defined in include/teeny/tensor.h:580

---

#### add

`const`

```cpp
template<class B> auto add(const B & b) const
```

Defined in include/teeny/tensor.h:583

---

#### sub

`const`

```cpp
template<class B> auto sub(const B & b) const
```

Defined in include/teeny/tensor.h:584

---

#### mul

`const`

```cpp
template<class B> auto mul(const B & b) const
```

Defined in include/teeny/tensor.h:585

---

#### div

`const`

```cpp
template<class B> auto div(const B & b) const
```

Defined in include/teeny/tensor.h:586

---

#### pow

`const`

```cpp
template<class B> auto pow(const B & b) const
```

Defined in include/teeny/tensor.h:587

---

#### map_

```cpp
template<class F> tensor & map_(F f)
```

Defined in include/teeny/tensor.h:590

---

#### zip_with_

```cpp
template<class G, class B> tensor & zip_with_(G g, const B & b)
```

Defined in include/teeny/tensor.h:591

---

#### map

`const`

```cpp
template<class F> auto map(F f) const
```

Defined in include/teeny/tensor.h:592

---

#### all

`const`

```cpp
bool all() const
```

Defined in include/teeny/tensor.h:596

---

#### any

`const`

```cpp
bool any() const
```

Defined in include/teeny/tensor.h:597

---

#### neg_

```cpp
tensor & neg_()
```

Defined in include/teeny/tensor.h:600

---

#### abs_

```cpp
tensor & abs_()
```

Defined in include/teeny/tensor.h:601

---

#### exp_

```cpp
tensor & exp_()
```

Defined in include/teeny/tensor.h:602

---

#### log_

```cpp
tensor & log_()
```

Defined in include/teeny/tensor.h:603

---

#### sin_

```cpp
tensor & sin_()
```

Defined in include/teeny/tensor.h:604

---

#### cos_

```cpp
tensor & cos_()
```

Defined in include/teeny/tensor.h:605

---

#### sqrt_

```cpp
tensor & sqrt_()
```

Defined in include/teeny/tensor.h:606

---

#### tanh_

```cpp
tensor & tanh_()
```

Defined in include/teeny/tensor.h:607

---

#### floor_

```cpp
tensor & floor_()
```

Defined in include/teeny/tensor.h:608

---

#### ceil_

```cpp
tensor & ceil_()
```

Defined in include/teeny/tensor.h:609

---

#### round_

```cpp
tensor & round_()
```

Defined in include/teeny/tensor.h:610

---

#### trunc_

```cpp
tensor & trunc_()
```

Defined in include/teeny/tensor.h:611

---

#### sign_

```cpp
tensor & sign_()
```

Defined in include/teeny/tensor.h:612

---

#### pow_

```cpp
tensor & pow_(T e)
```

Defined in include/teeny/tensor.h:613

---

#### clamp_

```cpp
tensor & clamp_(T lo, T hi)
```

Defined in include/teeny/tensor.h:614

---

#### operator++

`inline`

```cpp
inline tensor & operator++()
```

Defined in include/teeny/tensor.h:620

---

#### operator--

`inline`

```cpp
inline tensor & operator--()
```

Defined in include/teeny/tensor.h:621

---

#### operator++

`inline`

```cpp
template<bool S = is_static, cs::enable_if_t< S, int > = 0> inline tensor< T, Extents, cs::layout_right, own::stack > operator++(int)
```

Defined in include/teeny/tensor.h:623

---

#### operator--

`inline`

```cpp
template<bool S = is_static, cs::enable_if_t< S, int > = 0> inline tensor< T, Extents, cs::layout_right, own::stack > operator--(int)
```

Defined in include/teeny/tensor.h:625

---

#### add_

```cpp
template<bool Atomic, class B, cs::enable_if_t<!cs::is_arithmetic< B >::value, int >> tensor< T, E, L, O > & add_(const B & b)
```

Defined in include/teeny/math.h:528

---

#### sub_

```cpp
template<bool Atomic, class B, cs::enable_if_t<!cs::is_arithmetic< B >::value, int >> tensor< T, E, L, O > & sub_(const B & b)
```

Defined in include/teeny/math.h:534

---

#### mul_

```cpp
template<class B, cs::enable_if_t<!cs::is_arithmetic< B >::value, int >> tensor< T, E, L, O > & mul_(const B & b)
```

Defined in include/teeny/math.h:540

---

#### div_

```cpp
template<class B, cs::enable_if_t<!cs::is_arithmetic< B >::value, int >> tensor< T, E, L, O > & div_(const B & b)
```

Defined in include/teeny/math.h:542

---

#### add_

```cpp
template<bool Atomic> tensor< T, E, L, O > & add_(T s)
```

Defined in include/teeny/math.h:544

---

#### sub_

```cpp
template<bool Atomic> tensor< T, E, L, O > & sub_(T s)
```

Defined in include/teeny/math.h:550

---

#### copy_

```cpp
template<class B> tensor< T, E, L, O > & copy_(const B & b)
```

Defined in include/teeny/math.h:558

---

#### map_

```cpp
template<class F> tensor< T, E, L, O > & map_(F f)
```

Defined in include/teeny/math.h:691

---

#### zip_with_

```cpp
template<class G, class B> tensor< T, E, L, O > & zip_with_(G g, const B & b)
```

Defined in include/teeny/math.h:693

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr own` | [`ownership`](#ownership) `static` `constexpr` |  |
| `constexpr bool` | [`is_static`](#is_static) `static` `constexpr` |  |
| `constexpr cs::size_t` | [`buffer_size`](#buffer_size) `static` `constexpr` |  |
| `constexpr bool` | [`is_strides_layout`](#is_strides_layout) `static` `constexpr` |  |
| `constexpr bool` | [`is_contiguous_layout`](#is_contiguous_layout) `static` `constexpr` |  |

---

#### ownership

`static` `constexpr`

```cpp
constexpr own ownership = O
```

Defined in include/teeny/tensor.h:74

---

#### is_static

`static` `constexpr`

```cpp
constexpr bool is_static = (Extents::rank_dynamic() == 0)
```

Defined in include/teeny/tensor.h:75

---

#### buffer_size

`static` `constexpr`

```cpp
constexpr cs::size_t buffer_size = <, O == >::value
```

Defined in include/teeny/tensor.h:76

---

#### is_strides_layout

`static` `constexpr`

```cpp
constexpr bool is_strides_layout = _is_strides<Layout>::value
```

Defined in include/teeny/tensor.h:121

---

#### is_contiguous_layout

`static` `constexpr`

```cpp
constexpr bool is_contiguous_layout = _contiguous_layout<Layout>::value
```

Defined in include/teeny/tensor.h:122

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `constexpr cs::size_t` | [`rank`](#rank) `static` `inline` `constexpr` `noexcept` |  |

---

#### rank

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr cs::size_t rank() noexcept
```

Defined in include/teeny/tensor.h:118

### Public Types

| Name | Description |
|------|-------------|
| [`element_type`](#element_type)  |  |
| [`extents_type`](#extents_type-1)  |  |
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

Defined in include/teeny/tensor.h:66

---

#### extents_type

```cpp
using extents_type = Extents
```

Defined in include/teeny/tensor.h:67

---

#### layout_type

```cpp
using layout_type = Layout
```

Defined in include/teeny/tensor.h:68

---

#### index_type

```cpp
using index_type = typename Extents::index_type
```

Defined in include/teeny/tensor.h:69

---

#### mapping_type

```cpp
using mapping_type = typename Layout::template mapping< Extents >
```

Defined in include/teeny/tensor.h:70

---

#### view_type

```cpp
using view_type = cs::mdspan< T, Extents, Layout >
```

Defined in include/teeny/tensor.h:71

---

#### const_view_type

```cpp
using const_view_type = cs::mdspan< const T, Extents, Layout >
```

Defined in include/teeny/tensor.h:72



## mapping

```cpp
#include <layout.h>
```

```cpp
template<class Extents>
struct mapping
```

Defined in include/teeny/layout.h:82

> **Inherits:** `ndyn()>`, `Extents`

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
| `constexpr const Extents &` | [`extents`](#extents) `const` `inline` `constexpr` `noexcept` |  |
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
template<cs::size_t M = strides::ndyn(), cs::enable_if_t< M==0, int > = 0> constexpr inline constexpr mapping(const Extents & e)
```

Defined in include/teeny/layout.h:94

Fully-static strides: construct from extents only.

---

#### mapping

`inline` `constexpr`

```cpp
constexpr inline constexpr mapping(const Extents & e, const cs::array< index_type, strides::ndyn()> & dyn)
```

Defined in include/teeny/layout.h:97

Mixed strides: extents + the runtime strides (dim order, dynamic ones only).

---

#### extents

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const Extents & extents() const noexcept
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
using extents_type = Extents
```

Defined in include/teeny/layout.h:83

---

#### index_type

```cpp
using index_type = typename Extents::index_type
```

Defined in include/teeny/layout.h:84

---

#### rank_type

```cpp
using rank_type = typename Extents::rank_type
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

Defined in include/teeny/iterate.h:114

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

Defined in include/teeny/iterate.h:115

---

#### i

```cpp
index_type i
```

Defined in include/teeny/iterate.h:116

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

Defined in include/teeny/iterate.h:117

---

#### operator++

`inline`

```cpp
inline iterator & operator++()
```

Defined in include/teeny/iterate.h:118

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const iterator & o) const
```

Defined in include/teeny/iterate.h:119

---

#### operator==

`const` `inline`

```cpp
inline bool operator==(const iterator & o) const
```

Defined in include/teeny/iterate.h:120



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