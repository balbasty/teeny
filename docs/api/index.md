# Autodoc

Generated from the header Doxygen comments (`doxygen` + `moxygen`). For a curated view see the [Reference](../reference.md) and [Cheat sheet](../cheatsheet.md).

## Classes

| Name | Description |
|------|-------------|
| [`anyrank`](#anyrank) | A rank-erased tensor for the host/ndarray dispatch boundary. |
| [`anyrank_front`](#anyrank_front) | A range of fixed-rank-`Sr` sub-views over an `anyrank`'s batch axes. |
| [`anyshape`](#anyshape) | The shape spelling for the rank-erased `anyrank` boundary: exactly one `etc` marks the dynamic-rank region, the dims AFTER it are the static **Tail** (anchored at `ndim`), the dims BEFORE it are the static **Head** (anchored at 0). |
| [`axis`](#axis) | Compile-time **axis selector** — a value tag carrying a list of axes, the sibling of `shape<...>` for axis arguments. |
| [`compute_type`](#compute_type) | The type math should ACCUMULATE / compute in for element type `T`. |
| [`compute_type< bfloat16 >`](#compute_typebfloat16) |  |
| [`compute_type< half >`](#compute_typehalf) |  |
| [`copy_meta_t`](#copy_meta_t) | Tag for `as_anyrank(..., copy_meta)`: COPY shape/stride into an inline, device-passable store instead of wrapping the caller's arrays. |
| [`cpp_alloc`](#cpp_alloc) | Host allocator using C++ `new[]` / `delete[]`. |
| [`cuda_gpu_alloc`](#cuda_gpu_alloc) | Device (GPU) memory (`cudaMalloc`). |
| [`cuda_mapped_alloc`](#cuda_mapped_alloc) | Page-locked + device-mapped (zero-copy) host memory (`cudaHostAlloc`). |
| [`cuda_pinned_alloc`](#cuda_pinned_alloc) | Page-locked ("pinned") host memory (`cudaMallocHost`). |
| [`dtype`](#dtype) | Compile-time **element-type tag** — a value carrier for `T`, the sibling of `axis<...>` for the dtype argument. |
| [`into_t`](#into_t) |  |
| [`keep_strides`](#keep_strides) | Sentinel `Layout` selector for `recast<NewShape, [keep_strides](#keep_strides)>()` (the default): PRESERVE the source strides (fold where the source layout makes them derivable, keep runtime otherwise). |
| [`keepdims_t`](#keepdims_t) | numpy/pytorch `keepdims=True` tag for axis reductions — pass as any trailing keyword (composes with `dtype<...>`/`axis<...>`/`into(dest)` in any order) to keep the reduced axes as size-1 instead of removing them, so the result broadcasts back against the input: `sum<0>(a, keepdims)`, `sum(a, axis<0,2>{}, keepdims)`. |
| [`none_t`](#none_t) | Open-ended slice sentinel — teeny's `None` (python `a[:n]` / `a[m:]`). |
| [`owning_storage`](#owning_storage) | Generic owning storage (move-only, no ref-counting), parameterised by an allocator policy. |
| [`peel_range`](#peel_range) | A range of sub-views obtained by peeling `Axes...`. |
| [`ptr_storage`](#ptr_storage) |  |
| [`storage_policy`](#storage_policy) |  |
| [`gpu, N >`](#gpun) |  |
| [`gpu_view, N >`](#gpu_viewn) |  |
| [`heap, N >`](#heapn) |  |
| [`mapped, N >`](#mappedn) |  |
| [`mapped_view, N >`](#mapped_viewn) |  |
| [`pinned, N >`](#pinnedn) |  |
| [`pinned_view, N >`](#pinned_viewn) |  |
| [`stack, N >`](#stackn) |  |
| [`view, N >`](#viewn) |  |
| [`storage_size`](#storage_size) | Storage element count for a stack tensor (0 for view/owning). |
| [`storage_size< Mapping, true >`](#storage_sizemappingtrue) |  |
| [`strides`](#strides) | An mdspan layout policy with **per-dimension static or dynamic strides** — the stride analogue of `extents`/`shape`. |
| [`tensor`](#tensor) | One N-dimensional tensor, parameterised by ownership. |

## Enumerations

| Name | Description |
|------|-------------|
| [`ellipsis_t`](#ellipsis_t)  | The **ellipsis** marker (numpy `...`) for the unspecified middle axes. |
| [`storage`](#storage)  | Ownership / memory-space of a tensor's storage. |

---

### ellipsis_t

```cpp
enum ellipsis_t
```

The **ellipsis** marker (numpy `...`) for the unspecified middle axes.

Its own type (an empty enum) keeps it distinct from any real extent value, and being an enum it is a valid non-type template argument for `anyshape<...>`.

It has two roles that never overlap, so one marker serves both:

* when **indexing**, `t(1, ellipsis, 2)` stands for as many `all` as fill the rank;

* in an `anyshape<...>` boundary tag it marks the rank-erased region — there it is conventionally spelled **`etc`** ("and so on"), an alias of `ellipsis`. So `ellipsis`/`ellipsis_t` is the primary name and `etc`/`etc_t` the alias: `t(1, etc, 2)` == `t(1, ellipsis, 2)` and `anyshape<ellipsis, 3>` == `anyshape<etc, 3>`. (The `_is_ellipsis` indexing trait lives in `[indexing.h](#indexingh)`.)

---

### storage

```cpp
enum storage
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
| `pinned_view` |  |
| `mapped_view` |  |

## Typedefs

| Return | Name | Description |
|--------|------|-------------|
| `layout_right` | [`ccontiguous`](#ccontiguous)  | Names for the two contiguous layouts: `ccontiguous` is C-contiguous (row-major, `layout_right`), `fcontiguous` is Fortran-contiguous (column-major, `layout_left`). |
| `layout_left` | [`fcontiguous`](#fcontiguous)  |  |
| `ccontiguous` | [`corder`](#corder)  | Legacy aliases (`corder`/`forder`); prefer `ccontiguous`/`fcontiguous`. |
| `fcontiguous` | [`forder`](#forder)  |  |
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
| `extents< Idx, _dyn_extent(E)... >` | [`shape_as`](#shape_as)  | `shape<...>` with an explicit index type: `shape_as<int32_t, -1,3,3>`. |
| `shape_as< int32_t, E... >` | [`shape32`](#shape32)  |  |
| `dextents< int64_t, N >` | [`rank`](#rank)  | Fully-dynamic shape of a given rank: `rank<3>` == `shape<-1,-1,-1>` == `extents<int64_t, dynamic_extent, dynamic_extent, dynamic_extent>`. |
| `ellipsis_t` | [`etc_t`](#etc_t)  | `etc` — the `anyshape<...>` spelling of `ellipsis` (same marker, same value). |
| `_kw::resolve_t< _dtype_arg, Expl, void, _is_dtype, Dflt, Tags... >` | [`dtype_arg_t`](#dtype_arg_t)  |  |
| `tensor< T, Shape, Layout, storage::gpu >` | [`gpu`](#gpu)  | Owning tensor in device (GPU) memory (move-only). |
| `tensor< T, Shape, Layout, storage::pinned >` | [`pinned`](#pinned)  | Owning tensor in page-locked ("pinned") host memory (move-only). |
| `tensor< T, Shape, Layout, storage::mapped >` | [`mapped`](#mapped)  | Owning tensor in mapped (zero-copy) host memory (move-only). |
| `tensor< T, dextents< offset_t, R >, layout_stride, O >` | [`dyn_tensor`](#dyn_tensor)  | A fixed-rank, fully-dynamic, arbitrarily-strided tensor view. |
| `__half` | [`half`](#half)  | IEEE binary16 — the native CUDA `__half` under nvcc. |
| `__nv_bfloat16` | [`bfloat16`](#bfloat16)  | bfloat16 — the native CUDA `__nv_bfloat16` under nvcc. |
| `typename compute_type< T >::type` | [`compute_type_t`](#compute_type_t)  |  |
| `half` | [`f2`](#f2)  |  |
| `bfloat16` | [`bf16`](#bf16)  |  |
| `strides< S... >` | [`layout_static_stride`](#layout_static_stride)  | Back-compat alias: the original all-static-stride layout name. |
| `_kw::resolve_t< _kw::keep_tag, Expl, void, _is_layout_tag, Dflt, Tags... >` | [`layout_arg_t`](#layout_arg_t)  | layout_arg_t<Expl, Dflt, Tags...>: the Layout a call site should use &ndash; an explicit template argument (Expl != void) wins, else a bare ccontiguous{}/fcontiguous{} tag found in Tags..., else the library default Dflt; supplying BOTH an explicit Expl and a tag is a static_assert. |
| `typename _promote< A, B, true >::type` | [`promote_t`](#promote_t)  |  |
| `conditional_t<(is_floating_point< T >::value||!is_same< compute_type_t< T >, T >::value), conditional_t<(sizeof(T) >=sizeof(double)), T, double >, conditional_t<(sizeof(T) >=8), T, conditional_t< is_signed< T >::value, int64_t, uint64_t > > >` | [`reduce_type_t`](#reduce_type_t)  | Default accumulator type for a reduction over element type `T`. |
| `integral_constant< storage, O >` | [`storage_c`](#storage_c)  | Value-tag carrier for an ownership mode, for the factories' value-tag backend form, e.g. |
| `tensor< T, Shape, Layout, storage::view >` | [`view`](#view)  | A non-owning view type. |
| `tensor< T, Shape, Layout, storage::stack >` | [`local`](#local)  | Stack-owned tensor (fully static shape). |
| `tensor< T, Shape, Layout, storage::heap >` | [`owned`](#owned)  | Heap-owned tensor (host only, move-only). |

---

### ccontiguous

```cpp
using ccontiguous = layout_right
```

Names for the two contiguous layouts: `ccontiguous` is C-contiguous (row-major, `layout_right`), `fcontiguous` is Fortran-contiguous (column-major, `layout_left`).

Use wherever a `Layout` is expected — this is teeny's default and preferred spelling.

---

### fcontiguous

```cpp
using fcontiguous = layout_left
```

---

### corder

```cpp
using corder = ccontiguous
```

Legacy aliases (`corder`/`forder`); prefer `ccontiguous`/`fcontiguous`.

---

### forder

```cpp
using forder = fcontiguous
```

---

### dynamic_strides

```cpp
using dynamic_strides = layout_stride
```

`dynamic_strides` — the all-runtime strided layout (`layout_stride`, a full runtime stride array).

Prefer teeny's `strides<S...>` (folds known strides to immediates, and is what slicing produces); `dynamic_strides` is `strides<>` with every stride runtime.

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

### shape_as

```cpp
using shape_as = extents< Idx, _dyn_extent(E)... >
```

`shape<...>` with an explicit index type: `shape_as<int32_t, -1,3,3>`.

`shape<>` is the int64 default (DLPack's index type); `shape32<...>` narrows the offset math to **int32** for the kernel-boundary view (see `reindex`). The `-1` == dynamic rule is the same.

---

### shape32

```cpp
using shape32 = shape_as< int32_t, E... >
```

---

### rank

```cpp
using rank = dextents< int64_t, N >
```

Fully-dynamic shape of a given rank: `rank<3>` == `shape<-1,-1,-1>` == `extents<int64_t, dynamic_extent, dynamic_extent, dynamic_extent>`.

Handy for a rank-N view whose sizes are all runtime: `view<float, rank<3>>`. `rank<0>` is the rank-0 (scalar) shape.

---

### etc_t

```cpp
using etc_t = ellipsis_t
```

`etc` — the `anyshape<...>` spelling of `ellipsis` (same marker, same value).

---

### dtype_arg_t

```cpp
using dtype_arg_t = _kw::resolve_t< _dtype_arg, Expl, void, _is_dtype, Dflt, Tags... >
```

---

### gpu

```cpp
using gpu = tensor< T, Shape, Layout, storage::gpu >
```

Owning tensor in device (GPU) memory (move-only).

`gpu<T,E>(extents)`.

---

### pinned

```cpp
using pinned = tensor< T, Shape, Layout, storage::pinned >
```

Owning tensor in page-locked ("pinned") host memory (move-only).

`pinned<T,E>(extents)` — pytorch's `pin_memory`.

---

### mapped

```cpp
using mapped = tensor< T, Shape, Layout, storage::mapped >
```

Owning tensor in mapped (zero-copy) host memory (move-only).

`mapped<T,E>(extents)`.

---

### dyn_tensor

```cpp
using dyn_tensor = tensor< T, dextents< offset_t, R >, layout_stride, O >
```

A fixed-rank, fully-dynamic, arbitrarily-strided tensor view.

`O` is the memory space of the view — `[storage::view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a1bda80f2be4d3658e0baa43fbe7ae8c1)` (host) by default, `[storage::gpu_view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a716b431c57855c3a30f4c286ad4f0299)` when the pointer lives in device memory (see `anyrank`'s `Space`).

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

### layout_arg_t

```cpp
using layout_arg_t = _kw::resolve_t< _kw::keep_tag, Expl, void, _is_layout_tag, Dflt, Tags... >
```

layout_arg_t<Expl, Dflt, Tags...>: the Layout a call site should use &ndash; an explicit template argument (Expl != void) wins, else a bare ccontiguous{}/fcontiguous{} tag found in Tags..., else the library default Dflt; supplying BOTH an explicit Expl and a tag is a static_assert.

That precedence rule (and its wording) lives ONCE, in `_kw::resolve` ([kwargs.h](#kwargsh)) &ndash; shared with `dtype_arg_t`/`storage_arg`. Unlike dtype_arg_t, no unwrapping is needed here: the tag itself IS the layout type, so the answer is the tag that was found (`keep_tag`, which also supplies Dflt when there was none).

---

### promote_t

```cpp
using promote_t = typename _promote< A, B, true >::type
```

---

### reduce_type_t

```cpp
using reduce_type_t = conditional_t<(is_floating_point< T >::value||!is_same< compute_type_t< T >, T >::value), conditional_t<(sizeof(T) >=sizeof(double)), T, double >, conditional_t<(sizeof(T) >=8), T, conditional_t< is_signed< T >::value, int64_t, uint64_t > > >
```

Default accumulator type for a reduction over element type `T`.

`double` for floating-point types narrower than `double` (`float`, `half`, `bfloat16`) — enough headroom that summing many low-precision values doesn't lose catastrophically; a floating type already **at least** as wide as `double` (`double` itself, or `long double`) keeps itself — `long double` is `double`-sized on some ABIs (e.g. MSVC, arm64 macOS), where widening it would be a no-op precision-wise but a needless type change. Integer types narrower than 8 bytes accumulate in 64-bit (`int64_t` if signed, `uint64_t` if unsigned — `bool` counts as unsigned) so that summing / multiplying many small integers can't overflow mid-accumulation (signed overflow is UB); integers already ≥8 bytes keep their own type. The RESULT is still cast back to the element type `T` (accumulate wide, cast down); a caller who wants the untruncated wide value uses the explicit accumulator (`sum<int64_t>(a)`). Half types are spotted via `[compute_type](#compute_type)` (the only `T` whose compute type differs from itself). Override per call, e.g. `sum<float>(a)`.

---

### storage_c

```cpp
using storage_c = integral_constant< storage, O >
```

Value-tag carrier for an ownership mode, for the factories' value-tag backend form, e.g.

`empty<T>(shape, storage_c<[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)>{})`.

---

### view

```cpp
using view = tensor< T, Shape, Layout, storage::view >
```

A non-owning view type.

Construct as `view<T,E>(ptr, extents)`.

---

### local

```cpp
using local = tensor< T, Shape, Layout, storage::stack >
```

Stack-owned tensor (fully static shape).

Use `local<T,E>{}`.

---

### owned

```cpp
using owned = tensor< T, Shape, Layout, storage::heap >
```

Heap-owned tensor (host only, move-only).

Use `owned<T,E>(extents)`.

## Functions

| Return | Name | Description |
|--------|------|-------------|
| `auto` | [`make_gpu`](#make_gpu)  |  |
| `auto` | [`make_pinned`](#make_pinned)  |  |
| `auto` | [`make_mapped`](#make_mapped)  |  |
| `auto` | [`to`](#to)  | Move `x` to memory space `Space` (`[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)`/`pinned`/`mapped`/`heap`), optionally converting the element type to `ET` — the memory-backend half of pytorch's `.to`. |
| `auto` | [`to`](#to-1)  | Rvalue overload of `to<Space>`: a temporary source cannot be borrowed (the no-copy branch would dangle — and for a device temporary would point at freed device memory). |
| `DLManagedTensor *` | [`to_dlpack`](#to_dlpack)  | Export a **view** (`view` / `gpu_view` / `pinned_view` / `mapped_view`) to a `DLManagedTensor` (borrows the data — the caller must keep the underlying memory alive; only the metadata is owned by the capsule). |
| `DLManagedTensor *` | [`to_dlpack`](#to_dlpack-1)  | Export an **owning** tensor, TRANSFERRING ownership of the buffer into the capsule (the tensor is moved-from; the capsule's `deleter` frees the buffer). |
| `DLTensor` | [`to_dltensor`](#to_dltensor)  | Export to a **bare `DLTensor`** (unmanaged — no capsule, no deleter, no allocation). |
| `auto` | [`from_dlpack`](#from_dlpack)  | Import a `DLManagedTensor` of known element type `T` as an `anyrank` (runtime rank). |
| `auto` | [`from_dlpack`](#from_dlpack-1)  | Import a **bare `DLTensor`** (unmanaged — no deleter). |
| `auto` | [`from_dlpack`](#from_dlpack-2)  | Import a **versioned** managed tensor (DLPack 1.0+, what a modern `__dlpack__(max_version=…)` emits). |
| `auto` | [`from_dlpack`](#from_dlpack-3)  | Import with a STATIC TRAILING shape baked into the carrier's type — `from_dlpack<float, anyshape<etc,-1,-1,3>>(m)` for a `(*batch, *spatial, C)` tensor with a static channel count. |
| `auto` | [`from_dlpack`](#from_dlpack-4)  |  |
| `auto` | [`from_dlpack`](#from_dlpack-5)  |  |
| `auto` | [`from_dlpack`](#from_dlpack-6)  | Import as a **fixed-rank** view (requires the payload's `ndim == R`). |
| `auto` | [`from_dlpack`](#from_dlpack-7)  |  |
| `auto` | [`from_dlpack`](#from_dlpack-8)  |  |
| `bool` | [`dispatch_dlpack`](#dispatch_dlpack)  | Import + dispatch: read the dtype/rank from the `DLManagedTensor` and call `f` with a fixed-rank typed view (one instantiation per (dtype, rank)). |
| `bool` | [`dispatch_dlpack_dtype`](#dispatch_dlpack_dtype)  | Import + **dtype-only** dispatch that PRESERVES the rank: read the dtype from the capsule and call `f` with the **typed `anyrank`** (rank still dynamic), instead of collapsing to a fixed rank like `dispatch_dlpack`. |
| `anyrank< T, offset_t, _meta_view< offset_t >, Space >` | [`as_anyrank`](#as_anyrank)  | Build an `anyrank` that **wraps** the caller's shape/stride arrays with **no copy** (the default) — e.g. |
| `anyrank< T, offset_t, _meta_store< offset_t, MaxRank >, Space >` | [`as_anyrank`](#as_anyrank-1)  | `as_anyrank(data, shape, stride, ndim, copy_meta)` — COPY shape/stride into an inline store, so the carrier is trivially copyable and can be passed into a CUDA kernel by value (peel on device). |
| `auto` | [`as_anyrank`](#as_anyrank-2)  | `as_anyrank(..., anyshape<etc,c,c>{}[, layout])` — carry a STATIC TRAILING shape (and, with a layout tag, static trailing STRIDES) in the carrier's type, so `fixed`/`peel_front` hand out cells with those inner extents/strides already folded (no per-call `recast`). |
| `auto` | [`as_anyrank`](#as_anyrank-3)  | `as_anyrank(..., copy_meta, anyshape<etc,c,c>{}[, layout])` — the static-tail carrier over an INLINE (device-passable) meta store. |
| `void` | [`dispatch_index`](#dispatch_index)  | Narrow a fixed-rank view's OFFSET INDEX WIDTH to `Idx2` (default `int32_t`) when its element span fits, then call `f` — else call `f` with the view as-is. |
| `void` | [`dispatch_layout`](#dispatch_layout)  | Runtime-classify a DYNAMIC-strided view's contiguity and hand `f` a view whose LAYOUT is baked into the type — `ccontiguous` (C-order) or `fcontiguous` (F-order) when the runtime strides match, else the original `dynamic_strides`. |
| `bool` | [`dispatch_rank`](#dispatch_rank)  | Call `f` with a fixed-rank view of `t` chosen by its runtime `ndim`. |
| `bool` | [`dispatch_value`](#dispatch_value)  | Turn a runtime value into a compile-time one from a candidate list. |
| `auto` | [`slice`](#slice)  | A python-like slice `[start : stop : step)` for `operator()` / `slice_along`. |
| `auto` | [`slice`](#slice-1)  |  |
| `auto` | [`slice`](#slice-2)  |  |
| `auto` | [`peel_at`](#peel_at)  | The `i`-th sub-view obtained by peeling `Axes...` (0 <= i < product of the peeled extents). |
| `auto` | [`peel_at`](#peel_at-1)  |  |
| `auto` | [`peel_at`](#peel_at-2)  |  |
| `auto` | [`peel_at`](#peel_at-3)  |  |
| `auto` | [`peel_at`](#peel_at-4)  |  |
| `auto` | [`peel`](#peel)  | Build a range of sub-views by peeling `Axes...` of `t`. |
| `auto` | [`peel`](#peel-1)  |  |
| `auto` | [`peel`](#peel-2)  |  |
| `auto` | [`peel`](#peel-3)  |  |
| `peel_range< MD, storage::view, Axes... >` | [`peel_of`](#peel_of)  | Build a range of sub-views over a raw mdspan. |
| `auto` | [`peel_front`](#peel_front)  | Peel the FIRST `N` axes -> a range of sub-views over the rest — the runtime-batch-rank half of `(*batch, *spatial, C)`. |
| `auto` | [`peel_front`](#peel_front-1)  |  |
| `auto` | [`peel_front_at`](#peel_front_at)  | The `i`-th sub-view obtained by peeling the first `N` axes (grid-stride style). |
| `auto` | [`peel_front_at`](#peel_front_at-1)  |  |
| `tensor< T, E, L, O >::index_type` | [`size_front`](#size_front)  | The number of sub-views `peel_front<N>(t)` would yield — the product of the peeled leading extents — computed directly, without materialising the range. |
| `auto` | [`peel_zip`](#peel_zip)  | Zip-peel 2 tensors' `Axes...` in lock-step -> a range of `tuple<ViewA,ViewB>` (numpy-style broadcast: shapes may differ as long as they're broadcast-compatible; `Axes...` name axes in the BROADCAST rank's numbering — the larger of the two operands' own ranks — negatives wrap against it). |
| `auto` | [`peel_zip`](#peel_zip-1)  |  |
| `auto` | [`peel_zip`](#peel_zip-2)  |  |
| `auto` | [`peel_zip`](#peel_zip-3)  |  |
| `auto` | [`peel_zip`](#peel_zip-4)  | Zip-peel 3 tensors' `Axes...` in lock-step -> a range of `tuple<ViewA,ViewB,ViewC>` (same broadcast/axis-numbering rule as the 2-tensor form). |
| `auto` | [`peel_zip`](#peel_zip-5)  |  |
| `auto` | [`peel_zip`](#peel_zip-6)  |  |
| `auto` | [`peel_zip`](#peel_zip-7)  |  |
| `void` | [`scan_`](#scan_)  | In-place sequential fold ("scan") along axis `Axis`, batched over every other axis: `carry = init`, then for each element along `Axis` (in increasing order) `carry = f(carry, x)`, `x = carry`&ndash; the new carry doubles as the new element. |
| `void` | [`scan_`](#scan_-1)  |  |
| `void` | [`scan_`](#scan_-2)  |  |
| `void` | [`scan_`](#scan_-3)  |  |
| `auto` | [`scan`](#scan)  | Out-of-place twin of `scan_`: a fresh dense copy of `t`, scanned. |
| `auto` | [`scan`](#scan-1)  | Value form: `scan(t, axis<Axis>{}, init, f)` == `scan<Axis>(t, init, f)`. |
| `auto &` | [`scan`](#scan-2)  | `into(dest)` form: write the scanned result into a preallocated `dest` (a shape matching `t`'s EXACTLY, checked &ndash; a `static_assert` when both are static, `_TNY_CHECK` otherwise; unlike `copy_`'s own numpy-style broadcast, `dest` must match rather than merely receive a broadcast copy, since `scan_` then walks `dest`'s own axis numbering) &ndash; one copy, no fresh allocation beyond that; device-safe. |
| `auto &` | [`scan`](#scan-3)  |  |
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
|  | [`reduce_to< _reduce_result_t< Acc, _mean_result_t< T > > >`](#reduce_to_reduce_result_tacc_mean_result_tt)  |  |
| `_acc_t< Acc, T > auto` | [`dot`](#dot)  | Inner product over matching extents. |
| `auto` | [`sqnorm`](#sqnorm)  | Squared Euclidean norm — the sum of squares `Σ aᵢ²`, over ALL axes. |
| `auto` | [`norm`](#norm)  | Euclidean (L2) norm `√Σ aᵢ²`, over ALL axes. |
|  | [`reduce_to< _reduce_result_t< Acc, _mean_result_t< T > > >`](#reduce_to_reduce_result_tacc_mean_result_tt-1)  |  |
| `conditional_t< is_void< Acc >::value, _norm_root_t< T >, Acc > auto` | [`sqdist`](#sqdist)  | Squared Euclidean distance `Σ(aᵢ-bᵢ)²` between two same-shape tensors — mathematically `sqnorm(a-b)`, computed as one fused pass with no `a-b` intermediate (mirrors `dot`'s convenience-wrapper status over a manual `sum(a*b)`). |
| `auto` | [`dist`](#dist)  | Euclidean distance `√Σ(aᵢ-bᵢ)²` — mathematically `norm(a-b)`, one fused pass (see `sqdist`'s doc comment for the accuracy note). |
| `auto` | [`normalize`](#normalize)  | Out-of-place unit vector `a / norm(a)` -> a NEW dense tensor (static shape -> stack, dynamic -> heap). |
| `auto &` | [`normalize`](#normalize-1)  | `normalize(a, into(y))` — the unit vector into a caller buffer `y`. |
| `auto` | [`normalize`](#normalize-2)  | `normalize<Axes...>(a)` — unit vectors along the named axes: each element divided by the L2 norm over those axes (keepdim broadcast). |
| `auto` | [`normalize`](#normalize-3)  |  |
| `auto &` | [`normalize`](#normalize-4)  | `normalize<Axes...>(a, into(y))` — the axis-scoped unit vectors into a caller buffer `y` (same shape as `a`, since only the DIVISOR is reduced). |
| `auto &` | [`normalize`](#normalize-5)  |  |
| `auto` | [`cross`](#cross)  | 3D cross product `a × b` -> a NEW stack 3-vector of `promote(Ta,Tb)`. |
| `auto &` | [`cross`](#cross-1)  | `cross(a, b, into(y))` — the cross product into a caller buffer `y` (rank-1, length 3); `y` may alias `a` or `b`. |
| `bool` | [`allclose`](#allclose)  | True if every element satisfies `\|a-b\| <= atol + rtol*\|b\|` (numpy `allclose`; broadcasts, computes in the compute type of the promoted element type). |
| `decltype(auto)` | [`allclose`](#allclose-1)  | Generic trailing keyword bag for `allclose` — `dot`/`sqdist`/`dist`'s binary (no axis concept) shape, with numpy's OPTIONAL `rtol`/`atol` positionals kept ahead of the bag: `allclose(a, b, dtype<double>{})`, `allclose(a, b, into(cell))`, `allclose(a, b, rtol, into(cell))`, `allclose(a, b, rtol, atol, dtype<double>{}, into(cell))`. |
| `decltype(auto)` | [`allclose`](#allclose-2)  | `allclose(a, b, tags...)` — the keyword bag with both tolerances defaulted. |
| `decltype(auto)` | [`allclose`](#allclose-3)  | `allclose(a, b, rtol, tags...)` — the keyword bag with `atol` defaulted. |
| `auto` | [`minimum`](#minimum)  |  |
| `auto` | [`maximum`](#maximum)  |  |
| `auto` | [`minimum`](#minimum-1)  |  |
| `auto` | [`maximum`](#maximum-1)  |  |
| `auto &` | [`minimum`](#minimum-2)  |  |
| `auto &` | [`maximum`](#maximum-2)  |  |
| `auto &` | [`minimum`](#minimum-3)  |  |
| `auto &` | [`maximum`](#maximum-3)  |  |
| `auto` | [`clamp`](#clamp)  | `clamp(a, lo, hi)` -> a new tensor with each element clamped; `clamp(a, lo, hi, into(y))` writes into `y`. |
| `auto &` | [`clamp`](#clamp-1)  |  |
| `auto` | [`mean`](#mean)  | Arithmetic mean of all elements. |
| `constexpr bool` | [`storage_is_owning`](#storage_is_owning) `constexpr` `noexcept` | Whether the mode owns (and therefore allocates) its storage. |
| `constexpr bool` | [`storage_is_view`](#storage_is_view) `constexpr` `noexcept` | Whether the mode is a non-owning view (`view`/`gpu_view`/`pinned_view`/ `mapped_view`) — the pointer-wrapping modes (vs `stack`'s inline array). |
| `constexpr bool` | [`storage_is_device`](#storage_is_device) `constexpr` `noexcept` | Whether the storage lives in device (GPU) memory (owning or view). |
| `constexpr bool` | [`storage_is_host_accessible`](#storage_is_host_accessible) `constexpr` `noexcept` | Whether the storage is dereferenceable from the host. |
| `constexpr storage` | [`storage_view_of`](#storage_view_of) `constexpr` `noexcept` | The non-owning VIEW kind that preserves a source's memory space: a device source (`gpu`/`gpu_view`) -> `gpu_view`, a `pinned`/`mapped` source -> `pinned_view`/`mapped_view`, anything else -> `view`. |
| `constexpr storage` | [`storage_arg`](#storage_arg) `constexpr` | [storage_arg<Oexpl, Dflt, Tags...>()](#storage_arg): the backend a call site should use &ndash; an explicit template argument (Oexpl != storage_deduce) wins, else a storage_c<O>{} tag found in Tags..., else the library default Dflt (typically storage_deduce itself, resolved later from the shape by storage_resolve); supplying BOTH an explicit Oexpl and a tag is a static_assert. |
| `constexpr storage` | [`storage_resolve`](#storage_resolve) `constexpr` `noexcept` | Resolve a factory's ownership: an explicitly named mode passes through, `storage_deduce` becomes `stack` for a static shape / `heap` for a dynamic one. |
| `tensor< typename MD::element_type, typename MD::extents_type, typename MD::layout_type, OW >` | [`as_tensor`](#as_tensor)  | Wrap any `cuda::std::mdspan` (e.g. |
| `void` | [`fetch_add`](#fetch_add) `noexcept` | Accumulate `v` into `*p`, atomic on both host and device (#257). |
| `into_t< tensor< T, E, L, O > >` | [`into`](#into) `noexcept` | `into(y)` — the output-destination tag: pass it as the last argument to an out-of-place math producer (`a.add(b, into(y))`, `cross(a,b,into(y))`, `exp(a, into(y))`, …) to write the result into `y` (one fused pass, no allocation) and get `y&` back, instead of a freshly allocated result. |
| `into_t< tensor< T, E, L, O > >` | [`into`](#into-1) `noexcept` | `into(y)` over a TEMPORARY **view** — the destination may be written straight out of a view-producing op, with no named intermediate: `cross(a, b, into(N(i, all)))`, `sum(a, into(cells.at(i, j)))`, `x.add(y, into(z.permute<1,0>()))`. |
| `auto` | [`reindex`](#reindex)  | Free forms of `reindex`/`index_fits` — deduce the tensor, so a type-dependent receiver avoids `.template`: `reindex<int32_t>(t)`, `index_fits<int32_t>(t)`. |
| `auto` | [`reindex`](#reindex-1)  |  |
| `bool` | [`index_fits`](#index_fits)  |  |
| `auto` | [`wrap`](#wrap)  | Wrap `p` as a non-owning view with a contiguous layout (default C-order). |
| `auto` | [`wrap`](#wrap-1)  | Value-tag layout form: `wrap(p, e, fcontiguous{})` == `wrap<fcontiguous>(p, e)` — deduces the layout from a bare `ccontiguous{}`/`fcontiguous{}` argument instead of an explicit `<Layout>` template argument, so a type-dependent receiver needs no `.template`. |
| `auto` | [`wrap`](#wrap-2)  | `wrap(mdspan)` — a spelling of `as_tensor(mdspan)` under the one factory name users already reach for. |
| `auto` | [`wrap`](#wrap-3)  | Wrap `p` as a non-owning view with explicit **runtime strides** (a `layout_stride` view). |
| `auto` | [`wrap`](#wrap-4)  | Wrap `p` as a non-owning view with per-dimension **compile-time strides** (may be negative): pass a `strides<S...>{}` as the third argument. |
| `auto` | [`wrap`](#wrap-5)  | Wrap `p` with a **mix of static and runtime strides** — the exact analogue of `shape<-1,2,3,-1>{d0,d1}` for strides. |
| `auto` | [`make_view`](#make_view)  | `make_view<L>(ptr, extents)` — a non-owning view (alias of `wrap`). |
| `auto` | [`make_view`](#make_view-1)  | Value-tag layout form, mirroring `wrap`'s (#374): `make_view(p, e, fcontiguous{})` == `make_view<fcontiguous>(p, e)`, deduced from a bare `ccontiguous{}`/`fcontiguous{}` argument so a type-dependent receiver needs no `.template`. |
| `auto` | [`empty`](#empty)  | `empty<T>(extents)` — a new UNINITIALISED tensor. |
| `auto` | [`empty`](#empty-1)  | BACKEND-LED entry point — the one spelling the `T`-led entry point above cannot cover: a LEADING explicit template argument that names the BACKEND rather than the element type (`empty<[storage::pinned](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a8a4f12ef77f9e30413cabd15cf16c913)>(e, dtype<double>{}, fcontiguous{})`), because a value can never bind the `class T` of the entry point above. |
| `auto` | [`make_local`](#make_local)  | `make_local<T>(extents)` — a stack-owned tensor (static shape). |
| `auto` | [`make_heap`](#make_heap)  | `make_heap<T>(extents)` — a heap-owned tensor (host, move-only). |
| `auto` | [`full`](#full)  | `full(extents, v)` — a new tensor filled with `v`. |
| `auto` | [`full`](#full-1)  | BACKEND-LED entry point — a LEADING explicit template argument that names the BACKEND rather than the element type: `full<[storage::pinned](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a8a4f12ef77f9e30413cabd15cf16c913)>(e, v, dtype<double>{}, fcontiguous{})`. |
| `auto` | [`zeros`](#zeros)  | `zeros<T>(extents)` / `ones<T>(extents)` — a new tensor of 0s / 1s. |
| `auto` | [`zeros`](#zeros-1)  | BACKEND-LED entry point — see `[empty()](#empty)`'s twin above (#373). |
| `auto` | [`ones`](#ones)  |  |
| `auto` | [`ones`](#ones-1)  | BACKEND-LED entry point — see `[empty()](#empty)`'s twin above (#373). |
| `auto` | [`arange`](#arange)  | `arange<T>(n)` — a 1-D tensor `[0, 1, ..., n-1]` (heap, host). |
| `auto` | [`arange`](#arange-1)  | BACKEND-LED entry point — see `[empty()](#empty)`'s twin above (#373). |
| `auto` | [`arange`](#arange-2)  | Static `arange<T, N>()` — a stack `[0..N-1]` (host+device, folds). |
| `auto` | [`arange`](#arange-3)  | `arange<T>(Int<N>())` — the static form spelled with a static integer. |

---

### make_gpu

```cpp
template<class T = void, class Layout = void, class Shape, class... Tags> auto make_gpu(Shape e, Tags... tags)
```

---

### make_pinned

```cpp
template<class T = void, class Layout = void, class Shape, class... Tags> auto make_pinned(Shape e, Tags... tags)
```

---

### make_mapped

```cpp
template<class T = void, class Layout = void, class Shape, class... Tags> auto make_mapped(Shape e, Tags... tags)
```

---

### to

```cpp
template<storage Space, class ET = void, bool Force = false, class T, class Shape, class Layout, storage O> auto to(const tensor< T, Shape, Layout, O > & x)
```

Move `x` to memory space `Space` (`[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)`/`pinned`/`mapped`/`heap`), optionally converting the element type to `ET` — the memory-backend half of pytorch's `.to`.

`ET` defaults to the source type. 
```
auto d = to<storage::gpu>(h);           // upload host -> device
auto e = to<storage::gpu, half>(h);     // convert to half AND upload
auto c = to<storage::heap>(d);          // download device -> host
```
**Stays put when already there (#58):** with `Force` false and no dtype change, a source already in a **compatible** space borrows instead of copying — the exact same space (`heap`->`heap`, `gpu`->`gpu`), OR a device source moving to a device space (a `gpu` OR a `gpu_view` slice -> `gpu`). So the common "send data that's
already on the device to the device" returns a `gpu_view`, NOT a host round-trip. Pass `Force = true` for a fresh owning copy: 
```
       auto v = to<storage::gpu>(g);              // g is gpu (or a gpu_view slice) -> a view, no copy
       auto k = to<storage::gpu, void, true>(g);  // forced: a fresh gpu copy
```
 When a copy IS made: a **device -> device** copy (Force, same dtype) of a dense row-major source is a single **device-to-device**`cudaMemcpy` — no host hop; a strided device source falls back to the host densify (a device gather kernel is the #50 follow-up). A **device -> host** copy downloads via `cudaMemcpy` (any layout C/F/strided preserved, densified on the host). A **host -> device** copy reads the source directly (gathering only the viewed extent) and uploads. `Space == stack` needs a static shape.

:::note
The no-copy branch returns a **borrow** of `x` (a `gpu_view` for a device source, else a host `view`), so it must outlive the result — same lifetime rule as `[view()](#view)`/`permute()`/slicing. On a temporary the rvalue overload below instead **moves** a same-space dense owning source (steals its buffer) or forces a copy, so nothing dangles. NB a **contiguous** device download copies exactly `numel`; a **strided** device download still copies its full span (over-copies — #50). 

:::

---

### to

```cpp
template<storage Space, class ET = void, bool Force = false, class T, class Shape, class Layout, storage O> auto to(tensor< T, Shape, Layout, O > && x)
```

Rvalue overload of `to<Space>`: a temporary source cannot be borrowed (the no-copy branch would dangle — and for a device temporary would point at freed device memory).

A same-space, same-dtype, dense OWNING temporary is **moved** (its buffer stolen — no copy, no round-trip); otherwise this forces a fresh owning copy (which, for a device->device contiguous source, is the device-to-device path above, not a host round-trip).

---

### to_dlpack

```cpp
template<class T, class Shape, class Layout, storage O, enable_if_t< storage_is_view(O), int > = 0> DLManagedTensor * to_dlpack(const tensor< T, Shape, Layout, O > & t, DLDevice dev = { _dl::device_of< O >(), 0 })
```

Export a **view** (`view` / `gpu_view` / `pinned_view` / `mapped_view`) to a `DLManagedTensor` (borrows the data — the caller must keep the underlying memory alive; only the metadata is owned by the capsule).

The device defaults to the tensor's memory space (`kDLCPU` for a host view, `kDLCUDA` for a `gpu_view`, `kDLCUDAHost` for a `pinned_view`/ `mapped_view`; pass `dev` to override). The consumer owns the returned pointer and MUST call `m->deleter(m)` exactly once.

---

### to_dlpack

```cpp
template<class T, class Shape, class Layout, storage O, enable_if_t< storage_is_owning(O), int > = 0> DLManagedTensor * to_dlpack(tensor< T, Shape, Layout, O > && t)
```

Export an **owning** tensor, TRANSFERRING ownership of the buffer into the capsule (the tensor is moved-from; the capsule's `deleter` frees the buffer).

Device is taken from the tensor's memory space.

---

### to_dltensor

```cpp
template<class T, class Shape, class Layout, storage O> DLTensor to_dltensor(const tensor< T, Shape, Layout, O > & t, int64_t * shape_out, int64_t * strides_out, DLDevice dev = { _dl::device_of< O >(), 0 })
```

Export to a **bare `DLTensor`** (unmanaged — no capsule, no deleter, no allocation).

Borrows both the data AND the shape/stride arrays: the caller supplies `shape_out`/`strides_out` (each ≥ `t.rank()``int64_t`s), which this fills, and the returned `DLTensor` points at them + `t.data()`. The caller must keep the tensor's memory *and* those two buffers alive for as long as the `DLTensor` is used. Use for a consumer that takes a plain `DLTensor` rather than a managed capsule. Device defaults to the tensor's memory space (override with `dev`). Works for any storage (a pure borrow).

---

### from_dlpack

```cpp
template<class T, storage Space = storage::view> auto from_dlpack(const DLManagedTensor * m)
```

Import a `DLManagedTensor` of known element type `T` as an `anyrank` (runtime rank).

The shape/stride METADATA is copied into the carrier (so it is self-contained), while the DATA is BORROWED — the caller keeps `m` alive while the view is used, then calls `m->deleter(m)`. A null `strides` (DLPack's C-contiguous shorthand) is expanded to row-major. `byte_offset` is folded into the data pointer.

`Space` is the memory space to tag the carrier with (default `[storage::view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a1bda80f2be4d3658e0baa43fbe7ae8c1)` = host); every view peeled off it inherits it. It is **checked against `m->dl_tensor.device`**: importing a `kDLCUDA` capsule as the default host `Space` trips `_TNY_CHECK` — spell `from_dlpack<T, [storage::gpu_view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a716b431c57855c3a30f4c286ad4f0299)>(m)` so `fixed()`/`peel_front` yield device-tagged views (no host deref of a device pointer). (Closes the #38 hole where the device field was ignored and a device capsule silently became a host view.)

---

### from_dlpack

```cpp
template<class T, storage Space = storage::view> auto from_dlpack(const DLTensor * dt)
```

Import a **bare `DLTensor`** (unmanaged — no deleter).

Borrows the data, copies the metadata; the CALLER owns the whole lifetime (there is nothing to free). Use when a producer hands you a plain `DLTensor*` rather than a capsule. Same `Space`/device check as the managed overload.

---

### from_dlpack

```cpp
template<class T, storage Space = storage::view> auto from_dlpack(const DLManagedTensorVersioned * m)
```

Import a **versioned** managed tensor (DLPack 1.0+, what a modern `__dlpack__(max_version=…)` emits).

Reads its `dl_tensor` payload; as with the classic capsule the caller keeps `m` alive and calls `m->deleter(m)`.

---

### from_dlpack

```cpp
template<class T, class S, storage Space = storage::view, class Layout = keep_strides, enable_if_t< _is_anyshape< S >::value, int > = 0> auto from_dlpack(const DLManagedTensor * m, Layout = {})
```

Import with a STATIC TRAILING shape baked into the carrier's type — `from_dlpack<float, anyshape<etc,-1,-1,3>>(m)` for a `(*batch, *spatial, C)` tensor with a static channel count.

The payload's trailing dims are debug-checked against the tag once, here at the import boundary (next to the producer), then folded into every `fixed`/`peel_front` cell — no per-call `recast`. `etc` = the erased batch (see `anyshape`); the `Space` device check is the same as the tag-less overloads. Accepts all three carriers.

Pass a **layout tag by value** to also fold the trailing STRIDES: `from_dlpack<float, anyshape<etc,-1,-1,3>>(m, ccontiguous{})` bakes a C-contiguous inner block (checked vs the payload's strides here — the "input
       is contiguous" precondition, asserted at the boundary once instead of a per-call `recast`/`dispatch_layout`). Default `[keep_strides](#keep_strides)` keeps them runtime.

---

### from_dlpack

```cpp
template<class T, class S, storage Space = storage::view, class Layout = keep_strides, enable_if_t< _is_anyshape< S >::value, int > = 0> auto from_dlpack(const DLTensor * dt, Layout = {})
```

---

### from_dlpack

```cpp
template<class T, class S, storage Space = storage::view, class Layout = keep_strides, enable_if_t< _is_anyshape< S >::value, int > = 0> auto from_dlpack(const DLManagedTensorVersioned * m, Layout = {})
```

---

### from_dlpack

```cpp
template<class T, size_t R, storage Space = storage::view> auto from_dlpack(const DLManagedTensor * m)
```

Import as a **fixed-rank** view (requires the payload's `ndim == R`).

Returns a `layout_stride` tensor view borrowing the data. `Space` (default host `[storage::view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a1bda80f2be4d3658e0baa43fbe7ae8c1)`) tags the view and is checked against the device, as in the `anyrank` overloads — `from_dlpack<T, R, [storage::gpu_view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a716b431c57855c3a30f4c286ad4f0299)>(m)` for a device tensor. Accepts all three carriers (managed / bare / versioned).

---

### from_dlpack

```cpp
template<class T, size_t R, storage Space = storage::view> auto from_dlpack(const DLTensor * dt)
```

---

### from_dlpack

```cpp
template<class T, size_t R, storage Space = storage::view> auto from_dlpack(const DLManagedTensorVersioned * m)
```

---

### dispatch_dlpack

```cpp
template<storage Space = storage::view, class Carrier, class F> bool dispatch_dlpack(const Carrier * m, F && f)
```

Import + dispatch: read the dtype/rank from the `DLManagedTensor` and call `f` with a fixed-rank typed view (one instantiation per (dtype, rank)).

Returns false if the dtype/rank is outside the supported set. Data borrowed; caller owns `m`. `Space` (default host `[storage::view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a1bda80f2be4d3658e0baa43fbe7ae8c1)`) tags the views and is checked against the capsule's device — dispatch a device capsule with `dispatch_dlpack<[storage::gpu_view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a716b431c57855c3a30f4c286ad4f0299)>(m, f)`.

---

### dispatch_dlpack_dtype

```cpp
template<storage Space = storage::view, class Carrier, class F> bool dispatch_dlpack_dtype(const Carrier * m, F && f)
```

Import + **dtype-only** dispatch that PRESERVES the rank: read the dtype from the capsule and call `f` with the **typed `anyrank`** (rank still dynamic), instead of collapsing to a fixed rank like `dispatch_dlpack`.

The caller then peels its own axes — the `(*batch, *spatial, C)` batch idiom `for (auto cell : at.peel_front<-Sr>()) …`, which instantiates the kernel **once per `Sr`**, not once per total rank. Returns false for an unsupported dtype. Data borrowed; caller owns `m`. `Space` tags the carrier and is checked against the capsule's device (see `from_dlpack`).

---

### as_anyrank

```cpp
template<storage Space = storage::view, class T, class offset_t> anyrank< T, offset_t, _meta_view< offset_t >, Space > as_anyrank(T * data, offset_t * shape, offset_t * stride, int ndim)
```

Build an `anyrank` that **wraps** the caller's shape/stride arrays with **no copy** (the default) — e.g.

straight off a DLPack tensor. The arrays must outlive the carrier. HOST only: the pointers are not valid inside a device kernel, so peel/dispatch on the host and pass the resulting fixed-rank views to the device. To instead copy into an inline, device-passable store, pass the `copy_meta` tag (overload below). DLPack strides are in ELEMENTS; numpy `__array_interface__` in BYTES (divide by the itemsize first).

`Space` is the memory space of `data` (default `[storage::view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a1bda80f2be4d3658e0baa43fbe7ae8c1)` = host); pass `as_anyrank<[storage::gpu_view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a716b431c57855c3a30f4c286ad4f0299)>(...)` for a device pointer so the views peeled off it are `gpu_view`-tagged. (The shape/stride metadata arrays are host either way — `Space` labels the DATA, not the metadata store.)

---

### as_anyrank

```cpp
template<size_t MaxRank = TNY_MAX_RANK, storage Space = storage::view, class T, class offset_t> anyrank< T, offset_t, _meta_store< offset_t, MaxRank >, Space > as_anyrank(T * data, const offset_t * shape, const offset_t * stride, int ndim, copy_meta_t)
```

`as_anyrank(data, shape, stride, ndim, copy_meta)` — COPY shape/stride into an inline store, so the carrier is trivially copyable and can be passed into a CUDA kernel by value (peel on device).

`MaxRank` sets the inline capacity (default `TNY_MAX_RANK`); pass it as `as_anyrank<64>(..., copy_meta)`. Accepts `const` arrays (it copies).

---

### as_anyrank

```cpp
template<storage Space = storage::view, class T, class offset_t, class S, class Layout = keep_strides, enable_if_t< _is_anyshape< S >::value, int > = 0> auto as_anyrank(T * data, offset_t * shape, offset_t * stride, int ndim, S, Layout = {})
```

`as_anyrank(..., anyshape<etc,c,c>{}[, layout])` — carry a STATIC TRAILING shape (and, with a layout tag, static trailing STRIDES) in the carrier's type, so `fixed`/`peel_front` hand out cells with those inner extents/strides already folded (no per-call `recast`).

The runtime shape/strides' trailing dims are debug-checked against the tag once, here, then trusted. `etc` = the erased batch; the dims after it are the static tail (see `anyshape`). The optional layout tag chooses the trailing strides (like `recast`'s 2nd arg): `[keep_strides](#keep_strides)` (default — strides stay runtime), `ccontiguous`/`fcontiguous` (fold the contiguous inner block — the "input is contiguous" precondition, checked here), or `strides<S...>` (impose them). Wraps the caller's arrays (no copy).

---

### as_anyrank

```cpp
template<size_t MaxRank = TNY_MAX_RANK, storage Space = storage::view, class T, class offset_t, class S, class Layout = keep_strides, enable_if_t< _is_anyshape< S >::value, int > = 0> auto as_anyrank(T * data, const offset_t * shape, const offset_t * stride, int ndim, copy_meta_t, S, Layout = {})
```

`as_anyrank(..., copy_meta, anyshape<etc,c,c>{}[, layout])` — the static-tail carrier over an INLINE (device-passable) meta store.

---

### dispatch_index

```cpp
template<class Idx2 = int32_t, class V, class F> void dispatch_index(V && v, F && f)
```

Narrow a fixed-rank view's OFFSET INDEX WIDTH to `Idx2` (default `int32_t`) when its element span fits, then call `f` — else call `f` with the view as-is.

The kernel-boundary primitive behind the int32 fast path (#115): it instantiates `f` for BOTH widths and picks at run time via `index_fits`/`reindex`, so a genuinely dynamic view runs its offset math in 32-bit (half the by-value footprint, fewer device registers) exactly when that is lossless. `_TNY_HOST`; preserves the view's mutability. Use it standalone on a known-rank view (or a `peel_front` batch cell), or via `dispatch_rank<narrow_index>` to fuse it with the rank dispatch. 
```
for (auto cell : at.peel_front<-Sr>()) dispatch_index(cell, [&](auto c){ kernel<Sr>(c); });
```

---

### dispatch_layout

```cpp
template<class T, class E, storage O, class F> void dispatch_layout(tensor< T, E, dynamic_strides, O > v, F && f)
```

Runtime-classify a DYNAMIC-strided view's contiguity and hand `f` a view whose LAYOUT is baked into the type — `ccontiguous` (C-order) or `fcontiguous` (F-order) when the runtime strides match, else the original `dynamic_strides`.

The layout counterpart of `dispatch_index`. An `anyrank` boundary erases the producer's contiguity into `layout_stride`, so a later `recast<shape<…>>` can only KEEP runtime strides. `dispatch_layout` cheaply checks (`is_dense<ccontiguous>()` / `<fcontiguous>()` — a stride compare, no data touched) and, in the contiguous arms, hands `f` a view whose strides are EXTENT-DERIVED — so `recast<shape<-1,c,c>>()` then folds the inner strides to immediates SAFELY (no "I promise it's contiguous" — the runtime check already proved it). `f` is instantiated up to 3× (only the matching arm runs), so make it generic over the view type.

OPT-IN per call site (like `dispatch_index`): do NOT wrap `from_dlpack` in it by default — it triples instantiations and composes multiplicatively with the rank/width dispatchers. Reach for it when the inner block's folded strides actually matter (a small static-`C` kernel; see the efficient-kernels guide). 
```
for (auto cell : at.peel_front<-Sr>())
    dispatch_layout(cell, [&](auto v){ kernel<Sr>(v.recast(shape<-1,c,c>{})); });
```

---

### dispatch_rank

```cpp
template<bool Narrow = false, class T, class offset_t, class Meta, storage Space, class Tail, class TailS, class Head, class HeadS, class F> bool dispatch_rank(const anyrank< T, offset_t, Meta, Space, Tail, TailS, Head, HeadS > & t, F && f)
```

Call `f` with a fixed-rank view of `t` chosen by its runtime `ndim`.

`f` is a generic callable instantiated once per possible rank; the kernel it launches is fully static. Returns false if `ndim` exceeds `max_rank`. Prefer `peel_front<-Sr>` when only the trailing dims need to be static — one instantiation instead of one per total rank. 
```
dispatch_rank(as_anyrank(data, size, stride, ndim), [&](auto v){ kernel(v); });
```
 Opt into the int32 fast path with the compile-time `narrow_index` flag: each fixed cell is then also `dispatch_index`-narrowed (rank OUTER, width INNER — only the leaf doubles). `Narrow = false` (the default) is exactly the plain rank dispatch — no extra instantiation. 
```
dispatch_rank<narrow_index>(at, [&](auto v){ kernel(v); });   // int32 cells when they fit
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

Peeled axes vary in row-major order (the last listed axis fastest). Returns a `[tny::tensor](#tensor)` view. A raw mdspan carries no memory space, so this tags the result as a host `view`; the `[tny::tensor](#tensor)` overloads below preserve the source's space.

---

### peel_at

```cpp
template<long... Axes, class T, class E, class L, storage O> auto peel_at(tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i)
```

---

### peel_at

```cpp
template<long... Axes, class T, class E, class L, storage O> auto peel_at(const tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i)
```

---

### peel_at

```cpp
template<long... Axes, class T, class E, class L, storage O> auto peel_at(tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i, axis< Axes... >)
```

---

### peel_at

```cpp
template<long... Axes, class T, class E, class L, storage O> auto peel_at(const tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i, axis< Axes... >)
```

---

### peel

```cpp
template<long... Axes, class T, class E, class L, storage O> auto peel(tensor< T, E, L, O > & t)
```

Build a range of sub-views by peeling `Axes...` of `t`.

Non-const `t` yields mutable peel; const `t` yields read-only peel.

---

### peel

```cpp
template<long... Axes, class T, class E, class L, storage O> auto peel(const tensor< T, E, L, O > & t)
```

---

### peel

```cpp
template<long... Axes, class T, class E, class L, storage O> auto peel(tensor< T, E, L, O > & t, axis< Axes... >)
```

---

### peel

```cpp
template<long... Axes, class T, class E, class L, storage O> auto peel(const tensor< T, E, L, O > & t, axis< Axes... >)
```

---

### peel_of

```cpp
template<size_t... Axes, class MD> peel_range< MD, storage::view, Axes... > peel_of(const MD & m)
```

Build a range of sub-views over a raw mdspan.

---

### peel_front

```cpp
template<long N, class T, class E, class L, storage O> auto peel_front(tensor< T, E, L, O > & t)
```

Peel the FIRST `N` axes -> a range of sub-views over the rest — the runtime-batch-rank half of `(*batch, *spatial, C)`.

`N` is **signed**: `peel_front<3>` peels 3 leading dims; `peel_front<-1>` keeps the last axis (peels all but it), so negative = "keep the last |N|".

---

### peel_front

```cpp
template<long N, class T, class E, class L, storage O> auto peel_front(const tensor< T, E, L, O > & t)
```

---

### peel_front_at

```cpp
template<long N, class T, class E, class L, storage O> auto peel_front_at(tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i)
```

The `i`-th sub-view obtained by peeling the first `N` axes (grid-stride style).

---

### peel_front_at

```cpp
template<long N, class T, class E, class L, storage O> auto peel_front_at(const tensor< T, E, L, O > & t, typename tensor< T, E, L, O >::index_type i)
```

---

### size_front

```cpp
template<long N, class T, class E, class L, storage O> tensor< T, E, L, O >::index_type size_front(const tensor< T, E, L, O > & t)
```

The number of sub-views `peel_front<N>(t)` would yield — the product of the peeled leading extents — computed directly, without materialising the range.

Same signed `N` as `peel_front`: `size_front<3>(t)` multiplies the first 3 extents; `size_front<-2>(t)` the all-but-last-two (the flattened batch count of a `(*batch, C, C)` tensor).

---

### peel_zip

```cpp
template<long... Axes, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob> auto peel_zip(tensor< Ta, Ea, La, Oa > & a, tensor< Tb, Eb, Lb, Ob > & b)
```

Zip-peel 2 tensors' `Axes...` in lock-step -> a range of `tuple<ViewA,ViewB>` (numpy-style broadcast: shapes may differ as long as they're broadcast-compatible; `Axes...` name axes in the BROADCAST rank's numbering — the larger of the two operands' own ranks — negatives wrap against it).

A distinct name from `peel` (see the design note above `peel_zip_range`), not an overload.

The operands need not share an INDEX TYPE: the cells carry one wide enough — and, where the operands disagree on signedness, signed enough — to address every operand exactly, so a reversed view (`flip`, or a negative slice step) zipped against an unsigned-indexed tensor steps backwards instead of wrapping to a huge positive offset (#362).

---

### peel_zip

```cpp
template<long... Axes, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob> auto peel_zip(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b)
```

---

### peel_zip

```cpp
template<long... Axes, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob> auto peel_zip(tensor< Ta, Ea, La, Oa > & a, tensor< Tb, Eb, Lb, Ob > & b, axis< Axes... >)
```

---

### peel_zip

```cpp
template<long... Axes, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob> auto peel_zip(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b, axis< Axes... >)
```

---

### peel_zip

```cpp
template<long... Axes, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob, class Tc, class Ec, class Lc, storage Oc> auto peel_zip(tensor< Ta, Ea, La, Oa > & a, tensor< Tb, Eb, Lb, Ob > & b, tensor< Tc, Ec, Lc, Oc > & c)
```

Zip-peel 3 tensors' `Axes...` in lock-step -> a range of `tuple<ViewA,ViewB,ViewC>` (same broadcast/axis-numbering rule as the 2-tensor form).

The "triangle's three vertex tensors" idiom.

---

### peel_zip

```cpp
template<long... Axes, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob, class Tc, class Ec, class Lc, storage Oc> auto peel_zip(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b, const tensor< Tc, Ec, Lc, Oc > & c)
```

---

### peel_zip

```cpp
template<long... Axes, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob, class Tc, class Ec, class Lc, storage Oc> auto peel_zip(tensor< Ta, Ea, La, Oa > & a, tensor< Tb, Eb, Lb, Ob > & b, tensor< Tc, Ec, Lc, Oc > & c, axis< Axes... >)
```

---

### peel_zip

```cpp
template<long... Axes, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob, class Tc, class Ec, class Lc, storage Oc> auto peel_zip(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b, const tensor< Tc, Ec, Lc, Oc > & c, axis< Axes... >)
```

---

### scan_

```cpp
template<long Axis, class T, class E, class L, storage O, class Carry, class F> void scan_(tensor< T, E, L, O > & t, Carry init, F f)
```

In-place sequential fold ("scan") along axis `Axis`, batched over every other axis: `carry = init`, then for each element along `Axis` (in increasing order) `carry = f(carry, x)`, `x = carry`&ndash; the new carry doubles as the new element.

`f` is a device-safe functor (lambda-free engines, like `map_`/`zip_with_`): `Carry operator()(Carry carry, T x) const`. A reverse sweep composes with the existing negative-stride view, no separate "direction" flag: `scan_<Axis>(t.flip<Axis>(), init, f)` (an rvalue view binds fine &ndash;`scan_` has both lvalue and rvalue overloads, unlike `peel` this doesn't need a named temporary first). `scan_<Axis>(t, init, f)` == `scan_(t, axis<Axis>{}, init, f)`.

---

### scan_

```cpp
template<long Axis, class T, class E, class L, storage O, class Carry, class F> void scan_(tensor< T, E, L, O > && t, Carry init, F f)
```

---

### scan_

```cpp
template<long Axis, class T, class E, class L, storage O, class Carry, class F> void scan_(tensor< T, E, L, O > & t, axis< Axis >, Carry init, F f)
```

---

### scan_

```cpp
template<long Axis, class T, class E, class L, storage O, class Carry, class F> void scan_(tensor< T, E, L, O > && t, axis< Axis >, Carry init, F f)
```

---

### scan

```cpp
template<long Axis, class T, class E, class L, storage O, class Carry, class F, enable_if_t< tensor< T, E, L, O >::is_static, int > = 0> auto scan(const tensor< T, E, L, O > & t, Carry init, F f)
```

Out-of-place twin of `scan_`: a fresh dense copy of `t`, scanned.

Static shape -> stack (host+device); dynamic -> heap (host only, like `clone()`, which this is built on).

---

### scan

```cpp
template<long Axis, class T, class E, class L, storage O, class Carry, class F, enable_if_t< tensor< T, E, L, O >::is_static, int > = 0> auto scan(const tensor< T, E, L, O > & t, axis< Axis >, Carry init, F f)
```

Value form: `scan(t, axis<Axis>{}, init, f)` == `scan<Axis>(t, init, f)`.

SPLIT IN TWO on the same `is_static` key as the `<Axis>` pair it forwards to (#375): a static shape yields a stack result (host+device) so the forwarder is `_TNY_API`; a dynamic shape yields a heap result (host only, via `clone()`) so it is `_TNY_HOST` — else nvcc's device pass would see a `_TNY_API` forwarder call a `__host__` allocator.

---

### scan

```cpp
template<long Axis, class T, class E, class L, storage O, class Carry, class F, class D> auto & scan(const tensor< T, E, L, O > & t, Carry init, F f, into_t< D > out)
```

`into(dest)` form: write the scanned result into a preallocated `dest` (a shape matching `t`'s EXACTLY, checked &ndash; a `static_assert` when both are static, `_TNY_CHECK` otherwise; unlike `copy_`'s own numpy-style broadcast, `dest` must match rather than merely receive a broadcast copy, since `scan_` then walks `dest`'s own axis numbering) &ndash; one copy, no fresh allocation beyond that; device-safe.

`copy_` casts INTO `dest`'s element type FIRST, so if `dest`'s dtype differs from `t`'s the whole recurrence then runs in `dest`'s own precision. `scan` is the ONE producer in the library that does this &ndash; every other `into(dest)` (the elementwise/unary/scalar/axpy family, `index_select`, the reductions) computes in the SOURCE's precision and casts only the final result (#379 made that true of the elementwise family, which used to take its compute type from `dest` too &ndash; silently, and wrongly). Here it is deliberate: `scan_`'s carry is sequential and stateful, so the precision the recurrence runs in IS the precision of every intermediate carry, and there is no single "final result" to cast (see `docs/api-ux-review.md`'s F4-e). Returns `dest&`.

---

### scan

```cpp
template<long Axis, class T, class E, class L, storage O, class Carry, class F, class D> auto & scan(const tensor< T, E, L, O > & t, axis< Axis >, Carry init, F f, into_t< D > out)
```

---

### operator+

```cpp
template<class S, class T, class E, class L, storage O, enable_if_t< is_arithmetic< S >::value, int > = 0> auto operator+(S s, const tensor< T, E, L, O > & a)
```

---

### operator*

```cpp
template<class S, class T, class E, class L, storage O, enable_if_t< is_arithmetic< S >::value, int > = 0> auto operator*(S s, const tensor< T, E, L, O > & a)
```

---

### operator-

```cpp
template<class S, class T, class E, class L, storage O, enable_if_t< is_arithmetic< S >::value, int > = 0> auto operator-(S s, const tensor< T, E, L, O > & a)
```

---

### operator/

```cpp
template<class S, class T, class E, class L, storage O, enable_if_t< is_arithmetic< S >::value, int > = 0> auto operator/(S s, const tensor< T, E, L, O > & a)
```

---

### operator-

```cpp
template<class T, class E, class L, storage O> auto operator-(const tensor< T, E, L, O > & a)
```

---

### operator~

```cpp
template<class T, class E, class L, storage O, enable_if_t< is_integral< T >::value, int > = 0> auto operator~(const tensor< T, E, L, O > & a)
```

---

### sum

```cpp
template<class Acc = void, class T, class E, class L, storage O> auto sum(const tensor< T, E, L, O > & a)
```

Sum of all elements (empty -> 0).

Accumulates in the reduce type (`double` for small floats), result cast to `T`; `sum<Acc>(a)` returns `Acc`.

---

### prod

```cpp
template<class Acc = void, class T, class E, class L, storage O> auto prod(const tensor< T, E, L, O > & a)
```

Product of all elements (empty -> 1).

Accumulates in the reduce type, result cast to `T`; `prod<Acc>(a)` returns `Acc`.

---

### max

```cpp
template<class Acc = void, class T, class E, class L, storage O> auto max(const tensor< T, E, L, O > & a)
```

Maximum element.

Requires a non-empty tensor. Result type `T` (`max<Acc>(a)` returns `Acc`).

---

### min

```cpp
template<class Acc = void, class T, class E, class L, storage O> auto min(const tensor< T, E, L, O > & a)
```

Minimum element.

Requires a non-empty tensor. Result type `T` (`min<Acc>(a)` returns `Acc`).

---

### reduce_to< _reduce_result_t< Acc, _mean_result_t< T > > >

```cpp
reduce_to< _reduce_result_t< Acc, _mean_result_t< T > > >(axreduce<>)
```

---

### dot

```cpp
template<class Acc = void, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob> _acc_t< Acc, T > auto dot(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b)
```

Inner product over matching extents.

Accumulates in the reduce type of the promoted element type (`double` for small floats), result cast to `promote(Ta,Tb)`; `dot<Acc>(a, b)` returns `Acc`.

---

### sqnorm

```cpp
template<class Acc = void, class T, class E, class L, storage O> auto sqnorm(const tensor< T, E, L, O > & a)
```

Squared Euclidean norm — the sum of squares `Σ aᵢ²`, over ALL axes.

Just `dot(a, a)`: accumulates in the reduce type, result cast to the element type (`sqnorm<Acc>(a)` accumulates AND returns `Acc`). The value-tag/axis/`keepdims`/`into` composition (`sqnorm(a, dtype<Acc>{})`, `sqnorm(a, axis<0>{})`, ...) is handled generically by `_TNY_RED_TAGGED` (invoked further below, right after `sqnorm`'s own axis core — see `_TNY_RED_AXIS_CORE(sqnorm, ...)` above).

---

### norm

```cpp
template<class Acc = void, class T, class E, class L, storage O> auto norm(const tensor< T, E, L, O > & a)
```

Euclidean (L2) norm `√Σ aᵢ²`, over ALL axes.

Accumulates the squares in the reduce type and takes the root there, then casts to the result type: a floating element type keeps its type, an INTEGER one yields `double` (numpy/`mean` rule). `norm<Acc>(a)` makes `Acc` accumulator AND result.

---

### reduce_to< _reduce_result_t< Acc, _mean_result_t< T > > >

```cpp
reduce_to< _reduce_result_t< Acc, _mean_result_t< T > > >(_red_sqrt(axreduce<>(a, R(0), _md::r_addsq{})))
```

---

### sqdist

```cpp
template<class Acc = void, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob> conditional_t< is_void< Acc >::value, _norm_root_t< T >, Acc > auto sqdist(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b)
```

Squared Euclidean distance `Σ(aᵢ-bᵢ)²` between two same-shape tensors — mathematically `sqnorm(a-b)`, computed as one fused pass with no `a-b` intermediate (mirrors `dot`'s convenience-wrapper status over a manual `sum(a*b)`).

Each difference is formed and squared directly in the accumulator type, so the result can be MORE accurate than the un-fused `sqnorm(a-b)` spelling for a narrow element type (`a-b` there rounds to the operands' own type before `sqnorm` widens it) — not necessarily bit-identical, only for `double` operands are the two guaranteed equal. Binary only (no axis-list form, like `dot`); `sqdist<Acc>(a,b)` makes `Acc` accumulator AND result.

---

### dist

```cpp
template<class Acc = void, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob> auto dist(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b)
```

Euclidean distance `√Σ(aᵢ-bᵢ)²` — mathematically `norm(a-b)`, one fused pass (see `sqdist`'s doc comment for the accuracy note).

Floating result (integer operands -> `double`, the `norm`/`mean` rule); `dist<Acc>(a,b)` makes `Acc` accumulator AND result.

---

### normalize

```cpp
template<class T, class E, class L, storage O> auto normalize(const tensor< T, E, L, O > & a)
```

Out-of-place unit vector `a / norm(a)` -> a NEW dense tensor (static shape -> stack, dynamic -> heap).

The result element type is floating (integer input -> `double`, like `norm`). A zero vector yields NaNs (no epsilon — exact math; add one at the call site if you need it).

---

### normalize

```cpp
template<class T, class E, class L, storage O, class D> auto & normalize(const tensor< T, E, L, O > & a, into_t< D > out)
```

`normalize(a, into(y))` — the unit vector into a caller buffer `y`.

---

### normalize

```cpp
template<long... Axes, class T, class E, class L, storage O, enable_if_t<(sizeof...(Axes) > 0) &&_md::_nrm_out_api< E, Axes... >::value, int > = 0> auto normalize(const tensor< T, E, L, O > & a)
```

`normalize<Axes...>(a)` — unit vectors along the named axes: each element divided by the L2 norm over those axes (keepdim broadcast).

Floating result (integer -> double). Axes distinct, in any order (numpy-normalised). Static shape -> a stack result (host+device); dynamic -> heap (host only).

---

### normalize

```cpp
template<long... Axes, class T, class E, class L, storage O, enable_if_t<(sizeof...(Axes) > 0) &&_md::_nrm_out_api< E, Axes... >::value, int > = 0> auto normalize(const tensor< T, E, L, O > & a, axis< Axes... >)
```

---

### normalize

```cpp
template<long... Axes, class T, class E, class L, storage O, class D, enable_if_t<(sizeof...(Axes) > 0) &&_md::_nrm_kept_api< E, Axes... >::value, int > = 0> auto & normalize(const tensor< T, E, L, O > & a, into_t< D > out)
```

`normalize<Axes...>(a, into(y))` — the axis-scoped unit vectors into a caller buffer `y` (same shape as `a`, since only the DIVISOR is reduced).

Same one-line forward to `.div(..., out)` as the full-tensor form; the reduced norm itself is still materialised (it is a tensor, not a scalar) — which is the ONLY allocation here, hence the weaker `_nrm_kept_*` key: `normalize<0>(a, into(y))` on a `shape<-1,3>` source reduces to a `shape<3>` stack norm and stays device-callable. Axes distinct, in any order — same rule as the allocating form (`_keepdims` asserts distinctness and sorts).

---

### normalize

```cpp
template<long... Axes, class T, class E, class L, storage O, class D, enable_if_t<(sizeof...(Axes) > 0) &&_md::_nrm_kept_api< E, Axes... >::value, int > = 0> auto & normalize(const tensor< T, E, L, O > & a, axis< Axes... >, into_t< D > out)
```

---

### cross

```cpp
template<class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob> auto cross(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b)
```

3D cross product `a × b` -> a NEW stack 3-vector of `promote(Ta,Tb)`.

Both operands are rank-1, length 3. In place: the member `a.cross_(b)` (`a` becomes `a × b`). Into a preallocated slot: `cross(a, b, into(y))`.

---

### cross

```cpp
template<class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob, class D> auto & cross(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b, into_t< D > out)
```

`cross(a, b, into(y))` — the cross product into a caller buffer `y` (rank-1, length 3); `y` may alias `a` or `b`.

This is ff's "crossto". `y` may be a SLICE of a bigger output, written with no named intermediate: `cross(a, b, into(N(i, all)))` fills row `i` of a matrix of 3-vectors (`[into()](#into)` binds such a temporary view — [tensor.h](#tensorh)).

---

### allclose

```cpp
template<class Acc = void, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob> bool allclose(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b, double rtol = _allclose_rtol(), double atol = _allclose_atol())
```

True if every element satisfies `|a-b| <= atol + rtol*|b|` (numpy `allclose`; broadcasts, computes in the compute type of the promoted element type).

`allclose<Acc>(a, b)` forces that comparison to be carried out in `Acc` instead (the `dot`/`sqdist` accumulator convention — the answer is a `bool` either way).

---

### allclose

```cpp
template<class Acc = void, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob, class Tag0, class... Tags, enable_if_t< _kw::is_keyword< Tag0 >::value, int > = 0> decltype(auto) allclose(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b, double rtol, double atol, Tag0 tag0, Tags... tags)
```

Generic trailing keyword bag for `allclose` — `dot`/`sqdist`/`dist`'s binary (no axis concept) shape, with numpy's OPTIONAL `rtol`/`atol` positionals kept ahead of the bag: `allclose(a, b, dtype<double>{})`, `allclose(a, b, into(cell))`, `allclose(a, b, rtol, into(cell))`, `allclose(a, b, rtol, atol, dtype<double>{}, into(cell))`.

`dtype<Acc>{}` picks the comparison's compute type (== `allclose<Acc>`); `into(dest)` writes the answer into a RANK-0 destination (cast to its element type — a `bool` cell keeps it exactly) and returns `dest&`, allocating nothing.

Not an invocation of `_TNY_RED_BINARY_TAGGED`: that macro's wrappers forward `(a, b)` only, and `allclose` has the two tolerance positionals in between, which C++17 cannot default ahead of a trailing pack. Hence one bag overload per tolerance arity (0/1/2 given), the shorter two delegating. `Tag0` is constrained to a keyword, so a tolerance can never be swallowed as a tag nor a tag be read as a tolerance, and the plain form above is never in competition.

---

### allclose

```cpp
template<class Acc = void, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob, class Tag0, class... Tags, enable_if_t< _kw::is_keyword< Tag0 >::value, int > = 0> decltype(auto) allclose(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b, Tag0 tag0, Tags... tags)
```

`allclose(a, b, tags...)` — the keyword bag with both tolerances defaulted.

---

### allclose

```cpp
template<class Acc = void, class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob, class Tag0, class... Tags, enable_if_t< _kw::is_keyword< Tag0 >::value, int > = 0> decltype(auto) allclose(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b, double rtol, Tag0 tag0, Tags... tags)
```

`allclose(a, b, rtol, tags...)` — the keyword bag with `atol` defaulted.

---

### minimum

```cpp
template<class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob> auto minimum(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b)
```

---

### maximum

```cpp
template<class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob> auto maximum(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b)
```

---

### minimum

```cpp
template<class T, class E, class L, storage O, class S, enable_if_t< is_arithmetic< S >::value, int > = 0> auto minimum(const tensor< T, E, L, O > & a, S s)
```

---

### maximum

```cpp
template<class T, class E, class L, storage O, class S, enable_if_t< is_arithmetic< S >::value, int > = 0> auto maximum(const tensor< T, E, L, O > & a, S s)
```

---

### minimum

```cpp
template<class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob, class D> auto & minimum(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b, into_t< D > out)
```

---

### maximum

```cpp
template<class Ta, class Ea, class La, storage Oa, class Tb, class Eb, class Lb, storage Ob, class D> auto & maximum(const tensor< Ta, Ea, La, Oa > & a, const tensor< Tb, Eb, Lb, Ob > & b, into_t< D > out)
```

---

### minimum

```cpp
template<class T, class E, class L, storage O, class S, class D, enable_if_t< is_arithmetic< S >::value, int > = 0> auto & minimum(const tensor< T, E, L, O > & a, S s, into_t< D > out)
```

---

### maximum

```cpp
template<class T, class E, class L, storage O, class S, class D, enable_if_t< is_arithmetic< S >::value, int > = 0> auto & maximum(const tensor< T, E, L, O > & a, S s, into_t< D > out)
```

---

### clamp

```cpp
template<class T, class E, class L, storage O> auto clamp(const tensor< T, E, L, O > & a, T lo, T hi)
```

`clamp(a, lo, hi)` -> a new tensor with each element clamped; `clamp(a,
       lo, hi, into(y))` writes into `y`.

---

### clamp

```cpp
template<class T, class E, class L, storage O, class D> auto & clamp(const tensor< T, E, L, O > & a, T lo, T hi, into_t< D > out)
```

---

### mean

```cpp
template<class Acc = void, class T, class E, class L, storage O> auto mean(const tensor< T, E, L, O > & a)
```

Arithmetic mean of all elements.

For a floating `T`, accumulates in the reduce type (`double` for small floats) and the result is cast to `T`. For an INTEGER `T` the result is `double` (numpy: integer mean is float64; the division runs in `double`, not truncating integer division). `mean<Acc>(a)` makes `Acc` both the accumulator and the result type.

---

### storage_is_owning

`constexpr` `noexcept`

```cpp
constexpr constexpr bool storage_is_owning(storage o) noexcept
```

Whether the mode owns (and therefore allocates) its storage.

---

### storage_is_view

`constexpr` `noexcept`

```cpp
constexpr constexpr bool storage_is_view(storage o) noexcept
```

Whether the mode is a non-owning view (`view`/`gpu_view`/`pinned_view`/ `mapped_view`) — the pointer-wrapping modes (vs `stack`'s inline array).

---

### storage_is_device

`constexpr` `noexcept`

```cpp
constexpr constexpr bool storage_is_device(storage o) noexcept
```

Whether the storage lives in device (GPU) memory (owning or view).

---

### storage_is_host_accessible

`constexpr` `noexcept`

```cpp
constexpr constexpr bool storage_is_host_accessible(storage o) noexcept
```

Whether the storage is dereferenceable from the host.

---

### storage_view_of

`constexpr` `noexcept`

```cpp
constexpr constexpr storage storage_view_of(storage o) noexcept
```

The non-owning VIEW kind that preserves a source's memory space: a device source (`gpu`/`gpu_view`) -> `gpu_view`, a `pinned`/`mapped` source -> `pinned_view`/`mapped_view`, anything else -> `view`.

Every view-producing op (slice / permute / peel / reshape / at) tags its result with this so a view never loses (or misreports) its space.

---

### storage_arg

`constexpr`

```cpp
template<storage Oexpl, storage Dflt, class... Tags> constexpr constexpr storage storage_arg()
```

[storage_arg<Oexpl, Dflt, Tags...>()](#storage_arg): the backend a call site should use &ndash; an explicit template argument (Oexpl != storage_deduce) wins, else a storage_c<O>{} tag found in Tags..., else the library default Dflt (typically storage_deduce itself, resolved later from the shape by storage_resolve); supplying BOTH an explicit Oexpl and a tag is a static_assert.

That precedence rule (and its wording) lives ONCE, in `_kw::resolve` ([kwargs.h](#kwargsh)) &ndash; shared with `dtype_arg_t`/`layout_arg_t`. The only storage-specific part is the currency: this keyword's explicit form and its answer are a `storage` VALUE, not a type, so both travel through `resolve` inside their own `storage_c<O>` carrier &ndash; which IS the value tag, hence `keep_tag` as the unwrap step &ndash; and are read back out with `::value` here.

---

### storage_resolve

`constexpr` `noexcept`

```cpp
constexpr constexpr storage storage_resolve(storage o, bool static_shape) noexcept
```

Resolve a factory's ownership: an explicitly named mode passes through, `storage_deduce` becomes `stack` for a static shape / `heap` for a dynamic one.

---

### as_tensor

```cpp
template<storage OW = storage::view, class MD> tensor< typename MD::element_type, typename MD::extents_type, typename MD::layout_type, OW > as_tensor(const MD & m)
```

Wrap any `cuda::std::mdspan` (e.g.

a `submdspan` result) as a non-owning `[tny::tensor](#tensor)` view, so the tensor API applies to it.

---

### fetch_add

`noexcept`

```cpp
template<class T> void fetch_add(T * p, T v) noexcept
```

Accumulate `v` into `*p`, atomic on both host and device (#257).

INTERNAL primitive behind the atomic accumulate ops — prefer `a.atomic_add_(x)` / `t.at(i...).atomic_add_(v)` in user code.

The scatter/"push" write: many threads may accumulate into overlapping outputs, which a plain `+=` would race. Device -> `atomicAdd` (`double` needs sm_60+, `__half` sm_70+; not all integer widths have an overload — that surfaces as an nvcc error at instantiation). Host, arithmetic `T` EXCLUDING `bool`/`long double` -> `cuda::std::atomic_ref<T>` (libcu++'s C++17-usable backport of `std::atomic_ref`) so a push kernel parallelised with `std::thread`/OpenMP over overlapping outputs is genuinely race-free, matching the device semantics instead of merely documenting the caller must work around it.

The remaining element types keep the old plain `*p += v` (still not thread-safe there — same as before this fix, not a regression): `bool` (libcu++'s `atomic_ref<bool>` has no `fetch_add`) and `long double` (`atomic_ref<long double>` needs a 16-byte atomic RMW, which pulls in `libatomic` and fails to LINK on common toolchains that don't provide it — a working build must not start failing to link just because a caller touches `atomic_add_` on a `long double` tensor). Non-arithmetic `T` (a portable software `half`/`bfloat16` struct, OR the native `__half`/ `__nv_bfloat16` CUDA types under `__CUDACC__` on a host translation unit) has no atomic representation to route through `atomic_ref` either way.

---

### into

`noexcept`

```cpp
template<class T, class E, class L, storage O> into_t< tensor< T, E, L, O > > into(tensor< T, E, L, O > & d) noexcept
```

`into(y)` — the output-destination tag: pass it as the last argument to an out-of-place math producer (`a.add(b, into(y))`, `cross(a,b,into(y))`, `exp(a, into(y))`, …) to write the result into `y` (one fused pass, no allocation) and get `y&` back, instead of a freshly allocated result.

`y`'s dtype need not match: the arithmetic runs in the OPERANDS' own precision (a scalar rhs and the fused `alpha` included) and only the RESULT is cast to `y`, so `a.op(b, into(y))` gives exactly the numbers `y.copy_(a.op(b))` would (#379) — including for a `half`/`bfloat16` operand, where `into(y)` rounds through the twin's own `promote_t` (a 16-bit float there) before casting to `y`, not straight from the float compute value. `scan(t, init, f, into(y))` is the one deliberate exception — see its own doc-comment in [iterate.h](#iterateh).

---

### into

`noexcept`

```cpp
template<class T, class E, class L, storage O> into_t< tensor< T, E, L, O > > into(tensor< T, E, L, O > && d) noexcept
```

`into(y)` over a TEMPORARY **view** — the destination may be written straight out of a view-producing op, with no named intermediate: `cross(a, b, into(N(i, all)))`, `sum(a, into(cells.at(i, j)))`, `x.add(y, into(z.permute<1,0>()))`.

Every view-producing op (slicing, `at`, `permute`, `unsqueeze`, `slice_along`, `peel_at`, …) returns its view BY VALUE, so without this overload the most natural destination there is — a slot of a bigger output — had to be given a name first, which is exactly the boilerplate `into(dest)` exists to remove.

Restricted to the non-owning VIEW storages (`view`/`gpu_view`/ `pinned_view`/`mapped_view`): a temporary view aliases backing storage the caller owns elsewhere, so the write outlives the call, and the view itself lives to the end of the full expression that contains the producer. A temporary OWNING tensor (`into(zeros<double>(shape<3>{}))`, `into(local<double,shape<3>>{})`) is rejected instead: its storage dies with the expression, so the result would be computed and thrown away.

The one sharp edge: use the call for its EFFECT, don't keep the `dest&` it returns — `auto & r = cross(a, b, into(N(i, all)))` dangles once the temporary view goes away (same rule as `for (auto v : peel<0>(t))` and the other temporaries in the library).

---

### reindex

```cpp
template<class Idx2, class T, class E, class L, storage O> auto reindex(tensor< T, E, L, O > & t)
```

Free forms of `reindex`/`index_fits` — deduce the tensor, so a type-dependent receiver avoids `.template`: `reindex<int32_t>(t)`, `index_fits<int32_t>(t)`.

(`Idx2` is a TYPE, so there is no value form.)

---

### reindex

```cpp
template<class Idx2, class T, class E, class L, storage O> auto reindex(const tensor< T, E, L, O > & t)
```

---

### index_fits

```cpp
template<class Idx2, class T, class E, class L, storage O> bool index_fits(const tensor< T, E, L, O > & t)
```

---

### wrap

```cpp
template<class Layout = ccontiguous, storage Space = storage_deduce, class T, class Shape, class... Tags> auto wrap(T * p, Shape e, Tags...)
```

Wrap `p` as a non-owning view with a contiguous layout (default C-order).

This is the factory; the `view<T,E>` alias is the type it produces, and the member `t.view()` re-views an existing tensor.

MEMORY SPACE: `p` is a **host** pointer unless a trailing `storage_c<Space>{}` (or `storage_v<Space>`) tag names where it lives — pass the plain BACKEND the memory is in (`[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)` for a device pointer, `[storage::pinned](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a8a4f12ef77f9e30413cabd15cf16c913)`/`[storage::mapped](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a55c0eb766bdf4045fa0997d162971e31)` for page-locked host memory). Since `wrap` always yields a VIEW, the space folds to its view kind (`gpu -> gpu_view`, …) via `storage_view_of` — you never spell the `_view` kinds. Symmetric with `as_anyrank<Space>` / `from_dlpack<T,Space>`.

`[storage::heap](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a4d4a9aa362b6ffe089fd2e992ccf4f5f)`/`[storage::stack](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508afac2a47adace059aff113283a03f6760)` name no distinct memory space (they are *ownership* kinds, not backends), so passing one here just folds to a plain `[storage::view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a1bda80f2be4d3658e0baa43fbe7ae8c1)`, same as leaving the tag off — it does NOT make `wrap` return an owning tensor. For an owning heap/stack tensor, copy into one with `empty<T, [storage::heap](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a4d4a9aa362b6ffe089fd2e992ccf4f5f)>(e)`/`make_heap<T>(e)` instead.

The trailing argument is a keyword-tag bag (#277/#282), not a fixed `storage_c<Space>` parameter — today the only recognised keyword is `storage_c`/`storage_v`, but a future keyword (e.g. a `stream` tag) lands on all four `wrap` positional forms without touching any of them again.

---

### wrap

```cpp
template<class Layout, storage Space = storage_deduce, class T, class Shape, class... Tags, enable_if_t< is_same< Layout, ccontiguous >::value||is_same< Layout, fcontiguous >::value, int > = 0> auto wrap(T * p, Shape e, Layout, Tags...)
```

Value-tag layout form: `wrap(p, e, fcontiguous{})` == `wrap<fcontiguous>(p, e)` — deduces the layout from a bare `ccontiguous{}`/`fcontiguous{}` argument instead of an explicit `<Layout>` template argument, so a type-dependent receiver needs no `.template`.

Composes with a trailing `storage_c<Space>{}` exactly like the template form. (`strides<S...>{}` keeps its own dedicated overload above — it carries the static strides themselves, not just a layout kind, so it is not a `Layout` here.) A SECOND layout tag after this one — `wrap(p, e, fcontiguous{}, fcontiguous{})`, agreeing or not — is a `static_assert` (#394): `Layout` is a single positional slot, not a composable keyword, so it can only be given once per call.

---

### wrap

```cpp
template<storage Space = storage_deduce, class MD, class... Tags, enable_if_t< _is_mdspan_like< MD >::value, int > = 0> auto wrap(const MD & md, Tags...)
```

`wrap(mdspan)` — a spelling of `as_tensor(mdspan)` under the one factory name users already reach for.

Wrap any `cuda::std::mdspan`/`submdspan` result as a non-owning view; the element type, extents and layout all come from the mdspan.

MEMORY SPACE: same contract as the pointer forms — the mdspan wraps a **host** pointer unless a trailing `storage_c<Space>{}` (or `storage_v<Space>`) tag names where it lives, and since `wrap` always yields a VIEW the plain backend folds to its view kind (`[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947) -> gpu_view`, …) via `storage_view_of`, so you never spell the `_view` kinds: `wrap(md, storage_v<[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)>)` is a `gpu_view`. Takes the same trailing keyword-tag bag as the four positional forms (#282/#370).

NB the explicit template argument of THIS overload is the memory SPACE (an `storage` value: `wrap<[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)>(md)`), not a layout **type** as in `wrap<fcontiguous>(p, e)` — the layout is already carried by the mdspan, so there is nothing to name. Prefer the value-tag spelling above, which reads the same on every `wrap` form.

`as_tensor` stays public alongside this overload ON PURPOSE — two layers, not an accidental duplicate (#351): `wrap` is the caller-facing factory name for viewing existing memory whatever the carrier (pointer+shape, pointer+strides, an mdspan), with the family's keyword bag and backend->view-kind space folding; `as_tensor` is the mdspan-adaptation primitive underneath (no keyword bag, its `<OW>` is the already-folded view kind) — what teeny's own view-producing ops (`permute`/`flip`/`squeeze`/…) call with a pre-folded space, and the spelling the mdspan-interop docs teach.

---

### wrap

```cpp
template<storage Space = storage_deduce, class T, class Shape, class... Tags> auto wrap(T * p, Shape e, array< typename Shape::index_type, _shape_rank< Shape >()> st, Tags...)
```

Wrap `p` as a non-owning view with explicit **runtime strides** (a `layout_stride` view).

Pass one stride per dimension — an `array` or a braced list — in ELEMENTS; strides may be negative (a reversed view).

`wrap(p, shape<2,3>{}, {3, 1})` is the row-major view; `{1, 2}` the column-major one. For strides known at compile time pass a `strides<S...>{}` instead (overload below) so they fold into the type. A trailing `storage_c<Space>{}` tags the memory space (default host; the plain backend folds to its view kind, e.g. `[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947) -> gpu_view`).

`wrap` TRUSTS the strides you give it: a **stride 0** (or a stride smaller than an inner extent) makes a SELF-OVERLAPPING view where several indices alias one element. Reading such a view is fine (that is how a broadcast works), but an **in-place write** into it (`v.add_(b)`, `v.iota_(...)`) applies the update to the same element repeatedly — a host-debug check rejects an in-place write whose destination has an `extent > 1` axis with stride 0. `clone()` to a dense tensor first if you need to write.

---

### wrap

```cpp
template<int64_t... Strides, storage Space = storage_deduce, class T, class Shape, class... Tags> auto wrap(T * p, Shape e, strides< Strides... >, Tags...)
```

Wrap `p` as a non-owning view with per-dimension **compile-time strides** (may be negative): pass a `strides<S...>{}` as the third argument.

`wrap(p, shape<3,3>{}, strides<4,1>{})` folds the strides into the type (`strides<S...>` layout, EBO). Every stride must be a compile-time value — a `strides<...>` tag is a *stateless* layout, so it cannot carry runtime strides. For a **mix** of static and runtime strides, use the template form below; for all-runtime strides the `{s...}` overload above (a `layout_stride` view) is simplest.

---

### wrap

```cpp
template<int64_t S0, int64_t... Srest, storage Space = storage_deduce, class T, class Shape, class... Tags> auto wrap(T * p, Shape e, array< typename Shape::index_type, strides< S0, Srest... >::ndyn()> dyn, Tags...)
```

Wrap `p` with a **mix of static and runtime strides** — the exact analogue of `shape<-1,2,3,-1>{d0,d1}` for strides.

Give the per-dim pattern as template args (a compile-time stride, or `dynamic_stride` for a runtime one) and the runtime strides for the `dynamic_stride` slots as a braced list, in order: 
```
wrap<dynamic_stride, 1>(ptr, shape<3,3>{}, {4});   // outer=4 (runtime), inner=1 (folds)
wrap<dynamic_stride, dynamic_stride>(ptr, sh, {4,1}); // both runtime (a strides<> layout)
```
 The static slots fold into the type; only the runtime ones are stored. A trailing `storage_c<Space>{}` tags the memory space (default host; the plain backend folds to its view kind, e.g. `[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947) -> gpu_view`).

---

### make_view

```cpp
template<class Layout = ccontiguous, storage Space = storage_deduce, class T, class Shape, class... Tags> auto make_view(T * p, Shape e, Tags... tags)
```

`make_view<L>(ptr, extents)` — a non-owning view (alias of `wrap`).

The layout may be an explicit `<L>` template argument or, like `wrap`, a positional value tag (`make_view(p, e, fcontiguous{})`) — see the overload below. Takes the same optional trailing keyword-tag bag as `wrap` (#282) — today just `storage_c<Space>{}`/`storage_v<Space>`.

---

### make_view

```cpp
template<class Layout, storage Space = storage_deduce, class T, class Shape, class... Tags, enable_if_t< is_same< Layout, ccontiguous >::value||is_same< Layout, fcontiguous >::value, int > = 0> auto make_view(T * p, Shape e, Layout, Tags... tags)
```

Value-tag layout form, mirroring `wrap`'s (#374): `make_view(p, e, fcontiguous{})` == `make_view<fcontiguous>(p, e)`, deduced from a bare `ccontiguous{}`/`fcontiguous{}` argument so a type-dependent receiver needs no `.template`.

Composes with a trailing `storage_c<Space>{}` exactly like the template form. Without this overload only `ccontiguous{}` would work — it would reach `wrap`'s own positional layout overload by accident, because `make_view`'s `Layout`*defaults* to `ccontiguous` — while `fcontiguous{}` fell through to the keyword bag and was rejected as an unrecognised trailing argument. A SECOND layout tag — `make_view(p, e, fcontiguous{}, fcontiguous{})` — is a `static_assert` (#394), same as `wrap`'s.

---

### empty

```cpp
template<class T = void, storage O = storage_deduce, class Layout = void, class Shape, class... Tags, enable_if_t< _fac_on_stack< O, Shape, Tags... >::value, int > = 0> auto empty(Shape = Shape{}, Tags...)
```

`empty<T>(extents)` — a new UNINITIALISED tensor.

The one factory the `make_*` family fuses into: ownership is **deduced** from the shape (fully static -> `stack` (host+device); any dynamic extent -> `heap` (host)) unless a backend is named — `empty<T, [storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)>(extents)`, or the value-tag spelling `empty<T>(extents, storage_c<[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)>{})`. `gpu`/`pinned`/`mapped` require `<[teeny/cuda.h](#cudah)>` (their storage lives there). `T` defaults to `float`. Split by the resolved ownership so the `stack` case stays `_TNY_API` (host+device) while the allocating cases are `_TNY_HOST`.

Element type, backend, and layout may each be given as a leading explicit template argument OR as a trailing value tag (`dtype<T>{}`/`storage_c<O>{}`/ a layout tag), in ANY order and ANY subset: `empty<double>(e)`, `empty(e, dtype<double>{})`, `empty(e, storage_c<[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)>{}, dtype<double>{})` all work. `_TNY_KW_CHECK`/`dtype_arg_t`/`storage_arg`/`layout_arg_t` (`[kwargs.h](#kwargsh)` and each tag's own header) validate and resolve the trailing bag; an unrecognised or duplicated keyword fails on one clean `static_assert` instead of an overload-resolution wall (#279/#280).

---

### empty

```cpp
template<storage O, class Layout = void, class Shape, class... Tags, enable_if_t< _fac_on_stack< O, Shape, Tags... >::value, int > = 0> auto empty(Shape e, Tags... tags)
```

BACKEND-LED entry point — the one spelling the `T`-led entry point above cannot cover: a LEADING explicit template argument that names the BACKEND rather than the element type (`empty<[storage::pinned](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a8a4f12ef77f9e30413cabd15cf16c913)>(e, dtype<double>{}, fcontiguous{})`), because a value can never bind the `class T` of the entry point above.

`storage O` has **no default** here, so this overload is viable only when the backend is actually named: with no explicit template argument `O` is neither deducible nor defaulted and the candidate simply drops out, leaving the `T`-led entry point alone. Past that leading argument it takes the very SAME keyword bag, so every keyword still composes in ANY subset and ANY order (#373) — a leading backend argument is no longer a "one `dtype{}` tag and nothing else" dead end. A `storage_c<...>{}` tag on top of the explicit backend is the one thing it rejects, on `storage_arg`'s named "pick one" `static_assert`.

---

### make_local

```cpp
template<class T = void, class Layout = void, class Shape, class... Tags> auto make_local(Shape e = Shape{}, Tags... tags)
```

`make_local<T>(extents)` — a stack-owned tensor (static shape).

`T` defaults to `float` (numpy's default float dtype). Thin spelling of `empty<T, [storage::stack](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508afac2a47adace059aff113283a03f6760)>`. Takes the same trailing `dtype`/layout keyword-tag bag as `empty` (#282; no `storage_c` — the backend is fixed).

---

### make_heap

```cpp
template<class T = void, class Layout = void, class Shape, class... Tags> auto make_heap(Shape e, Tags... tags)
```

`make_heap<T>(extents)` — a heap-owned tensor (host, move-only).

`T` defaults to `float`. Thin spelling of `empty<T, [storage::heap](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a4d4a9aa362b6ffe089fd2e992ccf4f5f)>`. Takes the same trailing `dtype`/layout keyword-tag bag as `empty` (#282).

---

### full

```cpp
template<class T = void, storage O = storage_deduce, class Layout = void, class Shape, class V, class... Tags, enable_if_t< _fac_on_stack< O, Shape, Tags... >::value, int > = 0> auto full(Shape e, V v, Tags...)
```

`full(extents, v)` — a new tensor filled with `v`.

The element type defaults to the **value's** type (numpy/pytorch: `full(s, 3)` is int, `full(s, 3.0)` is float); pass `full<T>(...)` to override. Unlike the value-less `zeros`/`ones` (which default to `float`), there is a value here to infer from, so we do.

Ownership is deduced from the shape (static -> stack, dynamic -> heap) unless a **backend** is named — `full<T, [storage::pinned](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a8a4f12ef77f9e30413cabd15cf16c913)>(s, v)` or the value-tag `full<T>(s, v, storage_c<[storage::pinned](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a8a4f12ef77f9e30413cabd15cf16c913)>{})`. Because it fills host-side, only host-accessible backends (stack/heap/pinned/mapped) are allowed; a device (`gpu`) fill needs a kernel launch, so it is a `static_assert` steering you to `to<[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)>(full<T>(s, v))`. Split by resolved ownership for the `_TNY_API`/`_TNY_HOST` annotation.

Element type, backend, and layout may each be given as a leading explicit template argument OR as a trailing value tag, in ANY order and ANY subset, same as `empty` (#280/#281): `full(e, v, fcontiguous{})`, `full(e, v, storage_c<[storage::heap](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a4d4a9aa362b6ffe089fd2e992ccf4f5f)>{}, dtype<double>{})`.

---

### full

```cpp
template<storage O, class Layout = void, class Shape, class V, class... Tags, enable_if_t< _fac_on_stack< O, Shape, Tags... >::value, int > = 0> auto full(Shape e, V v, Tags... tags)
```

BACKEND-LED entry point — a LEADING explicit template argument that names the BACKEND rather than the element type: `full<[storage::pinned](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a8a4f12ef77f9e30413cabd15cf16c913)>(e, v, dtype<double>{}, fcontiguous{})`.

See `[empty()](#empty)`'s twin above for why `storage O` carries no default here and how the keyword bag composes (#373).

---

### zeros

```cpp
template<class T = void, storage O = storage_deduce, class Layout = void, class Shape, class... Tags, enable_if_t< _fac_on_stack< O, Shape, Tags... >::value, int > = 0> auto zeros(Shape e, Tags...)
```

`zeros<T>(extents)` / `ones<T>(extents)` — a new tensor of 0s / 1s.

`T` defaults to `float`. Same ownership deduction, backend selector, and `_TNY_API`/`_TNY_HOST` split as `full`; also composes `dtype`/`storage_c`/a layout tag in ANY order/subset, same as `empty` (#280/#281): `zeros(e, dtype<double>{})`, `zeros(e, fcontiguous{}, storage_c<[storage::heap](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a4d4a9aa362b6ffe089fd2e992ccf4f5f)>{})`. A device backend `static_assert`s — fill via `to<[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)>(zeros<T>(shape))`.

---

### zeros

```cpp
template<storage O, class Layout = void, class Shape, class... Tags, enable_if_t< _fac_on_stack< O, Shape, Tags... >::value, int > = 0> auto zeros(Shape e, Tags... tags)
```

BACKEND-LED entry point — see `[empty()](#empty)`'s twin above (#373).

---

### ones

```cpp
template<class T = void, storage O = storage_deduce, class Layout = void, class Shape, class... Tags, enable_if_t< _fac_on_stack< O, Shape, Tags... >::value, int > = 0> auto ones(Shape e, Tags...)
```

---

### ones

```cpp
template<storage O, class Layout = void, class Shape, class... Tags, enable_if_t< _fac_on_stack< O, Shape, Tags... >::value, int > = 0> auto ones(Shape e, Tags... tags)
```

BACKEND-LED entry point — see `[empty()](#empty)`'s twin above (#373).

---

### arange

```cpp
template<class T = void, storage O = storage_deduce, class... Tags> auto arange(long n, Tags...)
```

`arange<T>(n)` — a 1-D tensor `[0, 1, ..., n-1]` (heap, host).

`T` defaults to `int64_t` (an integer range, like numpy `arange(n)`). A host-accessible backend may be named — `arange<T, [storage::pinned](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a8a4f12ef77f9e30413cabd15cf16c913)>(n)` or `arange<T>(n, storage_c<[storage::pinned](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a8a4f12ef77f9e30413cabd15cf16c913)>{})`; a device backend `static_assert`s (use `to<[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)>(arange<T>(n))`). The static-N forms below stay stack. `T`/backend compose via the generic keyword mechanism too (#280/#281): `arange(n, dtype<double>{}, storage_c<[storage::pinned](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a8a4f12ef77f9e30413cabd15cf16c913)>{})`, either order. No layout keyword — a 1-D tensor has no C/F distinction.

---

### arange

```cpp
template<storage O, class... Tags> auto arange(long n, Tags... tags)
```

BACKEND-LED entry point — see `[empty()](#empty)`'s twin above (#373).

The analogous "leading explicit O" spelling: `arange<[storage::pinned](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a8a4f12ef77f9e30413cabd15cf16c913)>(n, dtype<double>{})`. A `storage_c<...>{}` tag on top of the explicit backend names the duplicate on `storage_arg`'s "pick one" `static_assert` rather than falling off the overload set.

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
| `constexpr ellipsis_t` | [`ellipsis`](#ellipsis) `constexpr` |  |
| `constexpr ellipsis_t` | [`etc`](#etc) `constexpr` |  |
| `constexpr full_extent_t` | [`all`](#all) `constexpr` | Keep-this-axis marker for slicing (an alias of `full_extent`). |
| `constexpr keepdims_t` | [`keepdims`](#keepdims) `constexpr` |  |
| `constexpr copy_meta_t` | [`copy_meta`](#copy_meta) `constexpr` |  |
| `constexpr bool` | [`narrow_index`](#narrow_index) `constexpr` | The spelling for `dispatch_rank`'s opt-in flag: `dispatch_rank<narrow_index>(at, f)`. |
| `constexpr none_t` | [`none`](#none) `constexpr` |  |
| `constexpr none_t` | [`newaxis`](#newaxis) `constexpr` |  |
| `constexpr int64_t` | [`dynamic_stride`](#dynamic_stride) `constexpr` | Per-dimension dynamic-stride sentinel. |
| `constexpr storage` | [`storage_deduce`](#storage_deduce) `constexpr` | Factory sentinel meaning "deduce the ownership from the shape" — a fully static shape -> `stack` (host+device), any dynamic extent -> `heap` (host). |
| `constexpr storage_c< O >` | [`storage_v`](#storage_v) `constexpr` | A ready-made `storage_c<O>` VALUE — the no-braces spelling of the value tag: `wrap(p, e, storage_v<[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)>)` instead of `storage_c<[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)>{}`. |
| `constexpr bool` | [`is_view_v`](#is_view_v) `constexpr` | Compile-time memory-space traits (SFINAE-friendly free forms of the tensor's `is_view`/`is_device`/… members): `is_view_v<decltype(x)>`. |
| `constexpr bool` | [`is_owning_v`](#is_owning_v) `constexpr` |  |
| `constexpr bool` | [`is_device_v`](#is_device_v) `constexpr` |  |
| `constexpr bool` | [`is_host_accessible_v`](#is_host_accessible_v) `constexpr` |  |

---

### ellipsis

`constexpr`

```cpp
constexpr ellipsis_t ellipsis {}
```

---

### etc

`constexpr`

```cpp
constexpr ellipsis_t etc = 
```

---

### all

`constexpr`

```cpp
constexpr full_extent_t all {}
```

Keep-this-axis marker for slicing (an alias of `full_extent`).

---

### keepdims

`constexpr`

```cpp
constexpr keepdims_t keepdims {}
```

---

### copy_meta

`constexpr`

```cpp
constexpr copy_meta_t copy_meta {}
```

---

### narrow_index

`constexpr`

```cpp
constexpr bool narrow_index = true
```

The spelling for `dispatch_rank`'s opt-in flag: `dispatch_rank<narrow_index>(at, f)`.

---

### none

`constexpr`

```cpp
constexpr none_t none {}
```

---

### newaxis

`constexpr`

```cpp
constexpr none_t newaxis = 
```

---

### dynamic_stride

`constexpr`

```cpp
constexpr int64_t dynamic_stride = (numeric_limits<int64_t>)()
```

Per-dimension dynamic-stride sentinel.

Strides are **signed**: a negative stride is a legitimate value (reversed / flipped views, and DLPack tensors carry them). So — unlike `shape<...>`, where `-1` marks a dynamic extent — we cannot use `-1` to mean "runtime" for a stride. Instead a reserved out-of-band value (`INT64_MIN`) marks a dynamic stride, leaving every ordinary stride (including negatives) expressible.

---

### storage_deduce

`constexpr`

```cpp
constexpr storage storage_deduce = static_cast<>(-1)
```

Factory sentinel meaning "deduce the ownership from the shape" — a fully static shape -> `stack` (host+device), any dynamic extent -> `heap` (host).

It is the default backend of `empty` (and the creation factories), out of the enum's normal range so it never names storage.

---

### storage_v

`constexpr`

```cpp
constexpr storage_c< O > storage_v {}
```

A ready-made `storage_c<O>` VALUE — the no-braces spelling of the value tag: `wrap(p, e, storage_v<[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)>)` instead of `storage_c<[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)>{}`.

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
template<class T, class offset_t = int64_t, class Meta = _meta_store<offset_t, TNY_MAX_RANK>, storage Space = storage::view, class Tail = shape<>, class TailS = _runtime_strides_t<Tail::rank()>, class Head = shape<>, class HeadS = _runtime_strides_t<Head::rank()>>
struct anyrank
```

Defined in include/teeny/dynamic.h:241

A rank-erased tensor for the host/ndarray dispatch boundary.

Holds a data pointer, a runtime `ndim`, and 1-D `shape`/`stride` tensors (`Meta`). `as_anyrank(...)`**wraps** the caller's arrays with no copy (a `_meta_view` store, HOST only) — the default; `as_anyrank(..., copy_meta)` COPIES them into an INLINE `TNY_MAX_RANK` store, so the carrier is trivially copyable and passes into a CUDA kernel by value (`device_passable == true`).

You do NOT compute on it — it is a *doorway*, not a room. Turn it into a statically-typed view at the boundary and compute on that:

* `fixed<R>()` — force a known total rank R.

* `dispatch_rank(...)` — pick R from the runtime `ndim`.

* `peel_front<-Sr>()` — the batch idiom: peel the runtime number of leading batch dims, keep the trailing `Sr` "interesting" dims STATIC. One kernel per Sr. NB the template arg is NEGATIVE: pass `-Sr` (`peel_front<-2>()` keeps the last two dims), matching the tensor's `peel_front` sign rule — a positive front-count would leave a runtime rank, which can't be a static view (asserted).

Deliberately no `add_`/`mul_`/etc.: a runtime-rank arithmetic path would loop over `ndim` (killing folding) or dispatch to every rank (the bloat `peel_front<-Sr>` avoids). Do host-side math on a `fixed<R>()`/`peel_front<-Sr>()` view instead.

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
| [`peel_front_at`](#peel_front_at-3) | `function` | Declared here |
| [`peel_front_at`](#peel_front_at-4) | `function` | Declared here |
| [`peel_front_at`](#peel_front_at-5) | `function` | Declared here |
| [`peel_front`](#peel_front-2) | `function` | Declared here |
| [`size_front`](#size_front-1) | `function` | Declared here |
| [`tail_rank`](#tail_rank) | `variable` | Declared here |
| [`head_rank`](#head_rank) | `variable` | Declared here |
| [`ends_rank`](#ends_rank) | `variable` | Declared here |
| [`space`](#space) | `variable` | Declared here |
| [`is_device`](#is_device) | `variable` | Declared here |
| [`view_space`](#view_space) | `variable` | Declared here |
| [`max_rank`](#max_rank) | `variable` | Declared here |
| [`device_passable`](#device_passable) | `variable` | Declared here |
| [`tail_type`](#tail_type) | `typedef` | Declared here |
| [`tail_stride_type`](#tail_stride_type) | `typedef` | Declared here |
| [`head_type`](#head_type) | `typedef` | Declared here |
| [`head_stride_type`](#head_stride_type) | `typedef` | Declared here |

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

Defined in include/teeny/dynamic.h:242

---

#### shape

```cpp
Meta shape {}
```

Defined in include/teeny/dynamic.h:243

---

#### stride

```cpp
Meta stride {}
```

Defined in include/teeny/dynamic.h:244

---

#### ndim

```cpp
int ndim = 0
```

Defined in include/teeny/dynamic.h:245

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `offset_t` | [`size`](#size-1) `const` `inline` `noexcept` |  |
| `offset_t` | [`step`](#step) `const` `inline` `noexcept` |  |
| `_ends_cell< T, offset_t, R, Head, HeadS, Tail, TailS, view_space >` | [`fixed`](#fixed) `const` `inline` | View this tensor as a fixed rank `R` (requires `ndim == R`). |
| `auto` | [`peel_front_at`](#peel_front_at-2) `const` `inline` | The `lin`-th sub-view keeping the last `\|N\|` axes static (grid-stride style). |
| `auto` | [`peel_front_at`](#peel_front_at-3) `const` `inline` | The `lin`-th cell peeled DIRECTLY to a target trailing shape — fuses `peel_front_at<-NewE::rank()>(lin).recast<NewE, NewL>()` into one call, so no separate `recast` in the caller. |
| `auto` | [`peel_front_at`](#peel_front_at-4) `const` `inline` | Value-form twins (no `.template` on a dependent receiver): pass the target shape (and optional layout) as a tag — `at.peel_front_at(i, shape<-1,c,c>{})` / `at.peel_front_at(i, shape<-1,c,c>{}, ccontiguous{})`. |
| `auto` | [`peel_front_at`](#peel_front_at-5) `const` `inline` |  |
| `anyrank_front< T, offset_t, Meta, Space, Tail, TailS, static_cast< size_t >(N< 0 ? -N :0)>` | [`peel_front`](#peel_front-2) `const` `inline` | Peel the leading batch axes -> an iterable of fixed-rank-`\|N\|` sub-views (range-for, `[size()](#size-1)`, `operator[]`). |
| `offset_t` | [`size_front`](#size_front-1) `const` `inline` `noexcept` | The number of cells `peel_front<N>()` would yield — the product of the peeled leading (batch) extents — computed directly, without building the range. |

---

#### size

`const` `inline` `noexcept`

```cpp
inline offset_t size(int i) const noexcept
```

Defined in include/teeny/dynamic.h:297

---

#### step

`const` `inline` `noexcept`

```cpp
inline offset_t step(int i) const noexcept
```

Defined in include/teeny/dynamic.h:298

---

#### fixed

`const` `inline`

```cpp
template<size_t R> inline _ends_cell< T, offset_t, R, Head, HeadS, Tail, TailS, view_space > fixed() const
```

Defined in include/teeny/dynamic.h:335

View this tensor as a fixed rank `R` (requires `ndim == R`).

BOTH the static `Head` (first `head_rank` dims) and `Tail` (last `tail_rank`) fold — the full-rank window has a compile-time left edge, so the Head anchors.

---

#### peel_front_at

`const` `inline`

```cpp
template<long N> inline auto peel_front_at(offset_t lin) const
```

Defined in include/teeny/dynamic.h:361

The `lin`-th sub-view keeping the last `|N|` axes static (grid-stride style).

`N` is **negative** — matching the tensor's `peel_front`, negative means "keep the last |N| dims". (A positive front-count would leave a runtime rank, which can't be a static view — hence the assert.) Follow with `recast<shape<-1,...>>()`.

---

#### peel_front_at

`const` `inline`

```cpp
template<class NewE, class NewL = keep_strides, enable_if_t< _is_extents< NewE >::value, int > = 0> inline auto peel_front_at(offset_t lin) const
```

Defined in include/teeny/dynamic.h:381

The `lin`-th cell peeled DIRECTLY to a target trailing shape — fuses `peel_front_at<-NewE::rank()>(lin).recast<NewE, NewL>()` into one call, so no separate `recast` in the caller.

`NewE`'s rank = the number of KEPT trailing dims (the batch is the leading `ndim - rank` dims, decoded into the pointer); a static extent in `NewE` folds, a `-1` extent stays dynamic (read from the carrier). `(*batch, *spatial, C)` -> 2-D pull with C=3 is `peel_front_at<shape<-1,-1,3>>(i)`. Removes the hand-kept `Sr == recast-shape rank` invariant. STRIDES: `NewL` defaults to `[keep_strides](#keep_strides)` so the cell keeps the carrier's RUNTIME strides (`layout_stride`) — an anyrank has no compile-time stride info to fold. To fold the inner strides, either pass a layout (`peel_front_at<shape<-1,c,c>, ccontiguous>` — a debug-checked "I promise it's contiguous") or use the runtime-proven `dispatch_layout` on the result. UB if a baked static extent doesn't match the carrier (debug-checked in `recast`, same contract).

---

#### peel_front_at

`const` `inline`

```cpp
template<class NewE, enable_if_t< _is_extents< NewE >::value, int > = 0> inline auto peel_front_at(offset_t lin, NewE) const
```

Defined in include/teeny/dynamic.h:388

Value-form twins (no `.template` on a dependent receiver): pass the target shape (and optional layout) as a tag — `at.peel_front_at(i, shape<-1,c,c>{})` / `at.peel_front_at(i, shape<-1,c,c>{}, ccontiguous{})`.

---

#### peel_front_at

`const` `inline`

```cpp
template<class NewE, class NewL, enable_if_t< _is_extents< NewE >::value, int > = 0> inline auto peel_front_at(offset_t lin, NewE, NewL) const
```

Defined in include/teeny/dynamic.h:390

---

#### peel_front

`const` `inline`

```cpp
template<long N> inline anyrank_front< T, offset_t, Meta, Space, Tail, TailS, static_cast< size_t >(N< 0 ? -N :0)> peel_front() const
```

Defined in include/teeny/dynamic.h:398

Peel the leading batch axes -> an iterable of fixed-rank-`|N|` sub-views (range-for, `[size()](#size-1)`, `operator[]`).

The `(*batch, *spatial, C)` boundary with `|N| = spatial + channels`: one kernel instantiation for `|N|`, not one per total rank. `N` is negative (keep the last |N| dims), as on the tensor.

---

#### size_front

`const` `inline` `noexcept`

```cpp
template<long N> inline offset_t size_front() const noexcept
```

Defined in include/teeny/dynamic.h:411

The number of cells `peel_front<N>()` would yield — the product of the peeled leading (batch) extents — computed directly, without building the range.

`N` is NEGATIVE (keep the last |N| dims), the same sign as `peel_front`; `size_front<-2>()` is the flattened batch count of a `(*batch, C, C)` carrier.

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr size_t` | [`tail_rank`](#tail_rank) `static` `constexpr` |  |
| `constexpr size_t` | [`head_rank`](#head_rank) `static` `constexpr` |  |
| `constexpr size_t` | [`ends_rank`](#ends_rank) `static` `constexpr` |  |
| `constexpr storage` | [`space`](#space) `static` `constexpr` |  |
| `constexpr bool` | [`is_device`](#is_device) `static` `constexpr` |  |
| `constexpr storage` | [`view_space`](#view_space) `static` `constexpr` |  |
| `constexpr size_t` | [`max_rank`](#max_rank) `static` `constexpr` |  |
| `constexpr bool` | [`device_passable`](#device_passable) `static` `constexpr` |  |

---

#### tail_rank

`static` `constexpr`

```cpp
constexpr size_t tail_rank = Tail::rank()
```

Defined in include/teeny/dynamic.h:262

---

#### head_rank

`static` `constexpr`

```cpp
constexpr size_t head_rank = Head::rank()
```

Defined in include/teeny/dynamic.h:263

---

#### ends_rank

`static` `constexpr`

```cpp
constexpr size_t ends_rank =  + 
```

Defined in include/teeny/dynamic.h:264

---

#### space

`static` `constexpr`

```cpp
constexpr storage space = Space
```

Defined in include/teeny/dynamic.h:273

---

#### is_device

`static` `constexpr`

```cpp
constexpr bool is_device = (Space)
```

Defined in include/teeny/dynamic.h:274

---

#### view_space

`static` `constexpr`

```cpp
constexpr storage view_space = (Space)
```

Defined in include/teeny/dynamic.h:276

---

#### max_rank

`static` `constexpr`

```cpp
constexpr size_t max_rank =
        Meta::extents_type::static_extent(0) != dynamic_extent
            ? Meta::extents_type::static_extent(0) : size_t()
```

Defined in include/teeny/dynamic.h:280

---

#### device_passable

`static` `constexpr`

```cpp
constexpr bool device_passable =
        (Meta::extents_type::static_extent(0) != dynamic_extent)
```

Defined in include/teeny/dynamic.h:294

### Public Types

| Name | Description |
|------|-------------|
| [`tail_type`](#tail_type)  |  |
| [`tail_stride_type`](#tail_stride_type)  |  |
| [`head_type`](#head_type)  |  |
| [`head_stride_type`](#head_stride_type)  |  |

---

#### tail_type

```cpp
using tail_type = Tail
```

Defined in include/teeny/dynamic.h:258

---

#### tail_stride_type

```cpp
using tail_stride_type = TailS
```

Defined in include/teeny/dynamic.h:259

---

#### head_type

```cpp
using head_type = Head
```

Defined in include/teeny/dynamic.h:260

---

#### head_stride_type

```cpp
using head_stride_type = HeadS
```

Defined in include/teeny/dynamic.h:261



## anyrank_front

```cpp
#include <dynamic.h>
```

```cpp
template<class T, class offset_t, class Meta, storage Space, class Tail, class TailS, size_t Sr>
struct anyrank_front
```

Defined in include/teeny/dynamic.h:423

A range of fixed-rank-`Sr` sub-views over an `anyrank`'s batch axes.

Inherits the carrier's `Space`, so each cell is a host or `gpu_view` view accordingly.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`src`](#src) | `variable` | Declared here |
| [`size`](#size-2) | `function` | Declared here |
| [`operator[]`](#operator-6) | `function` | Declared here |
| [`begin`](#begin) | `function` | Declared here |
| [`end`](#end) | `function` | Declared here |
| [`subrange`](#subrange) | `function` | Declared here |
| [`enumerate`](#enumerate) | `function` | Declared here |
| [`MaxNb`](#maxnb) | `variable` | Declared here |
| [`Cell`](#cell) | `typedef` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `anyrank< T, offset_t, Meta, Space, Tail, TailS >` | [`src`](#src)  |  |

---

#### src

```cpp
anyrank< T, offset_t, Meta, Space, Tail, TailS > src
```

Defined in include/teeny/dynamic.h:424

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `offset_t` | [`size`](#size-2) `const` `inline` `noexcept` |  |
| `auto` | [`operator[]`](#operator-6) `const` `inline` |  |
| `iterator` | [`begin`](#begin) `const` `inline` |  |
| `iterator` | [`end`](#end) `const` `inline` |  |
| `subrange_t` | [`subrange`](#subrange) `const` `inline` |  |
| `enum_range` | [`enumerate`](#enumerate) `const` `inline` |  |

---

#### size

`const` `inline` `noexcept`

```cpp
inline offset_t size() const noexcept
```

Defined in include/teeny/dynamic.h:428

---

#### operator[]

`const` `inline`

```cpp
inline auto operator[](offset_t i) const
```

Defined in include/teeny/dynamic.h:430

---

#### begin

`const` `inline`

```cpp
inline iterator begin() const
```

Defined in include/teeny/dynamic.h:478

---

#### end

`const` `inline`

```cpp
inline iterator end() const
```

Defined in include/teeny/dynamic.h:479

---

#### subrange

`const` `inline`

```cpp
inline subrange_t subrange(offset_t lo, offset_t hi) const
```

Defined in include/teeny/dynamic.h:489

---

#### enumerate

`const` `inline`

```cpp
inline enum_range enumerate() const
```

Defined in include/teeny/dynamic.h:531

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr size_t` | [`MaxNb`](#maxnb) `static` `constexpr` |  |

---

#### MaxNb

`static` `constexpr`

```cpp
constexpr size_t MaxNb = <T, offset_t, Meta, Space, Tail, TailS>::max_rank
```

Defined in include/teeny/dynamic.h:426

### Public Types

| Name | Description |
|------|-------------|
| [`Cell`](#cell)  |  |

---

#### Cell

```cpp
using Cell = _tail_cell< T, offset_t, Sr, Tail, TailS, storage_view_of(Space)>
```

Defined in include/teeny/dynamic.h:425



## coord

```cpp
#include <dynamic.h>
```

```cpp
struct coord
```

Defined in include/teeny/dynamic.h:503

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`ctr`](#ctr) | `variable` | Declared here |
| [`nb`](#nb) | `variable` | Declared here |
| [`lin`](#lin) | `variable` | Declared here |
| [`operator[]`](#operator-7) | `function` | Declared here |
| [`rank`](#rank-1) | `function` | Declared here |
| [`linear`](#linear) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `const offset_t *` | [`ctr`](#ctr)  |  |
| `int` | [`nb`](#nb)  |  |
| `offset_t` | [`lin`](#lin)  |  |

---

#### ctr

```cpp
const offset_t * ctr
```

Defined in include/teeny/dynamic.h:504

---

#### nb

```cpp
int nb
```

Defined in include/teeny/dynamic.h:504

---

#### lin

```cpp
offset_t lin
```

Defined in include/teeny/dynamic.h:504

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `offset_t` | [`operator[]`](#operator-7) `const` `inline` `noexcept` |  |
| `int` | [`rank`](#rank-1) `const` `inline` `noexcept` |  |
| `offset_t` | [`linear`](#linear) `const` `inline` `noexcept` |  |

---

#### operator[]

`const` `inline` `noexcept`

```cpp
inline offset_t operator[](int d) const noexcept
```

Defined in include/teeny/dynamic.h:505

---

#### rank

`const` `inline` `noexcept`

```cpp
inline int rank() const noexcept
```

Defined in include/teeny/dynamic.h:506

---

#### linear

`const` `inline` `noexcept`

```cpp
inline offset_t linear() const noexcept
```

Defined in include/teeny/dynamic.h:507



## enum_iterator

```cpp
#include <dynamic.h>
```

```cpp
struct enum_iterator
```

Defined in include/teeny/dynamic.h:510

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`it`](#it) | `variable` | Declared here |
| [`operator*`](#operator-8) | `function` | Declared here |
| [`operator++`](#operator-9) | `function` | Declared here |
| [`operator!=`](#operator-10) | `function` | Declared here |
| [`operator==`](#operator-11) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `iterator` | [`it`](#it)  |  |

---

#### it

```cpp
iterator it
```

Defined in include/teeny/dynamic.h:511

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `item` | [`operator*`](#operator-8) `const` `inline` |  |
| `enum_iterator &` | [`operator++`](#operator-9) `inline` |  |
| `bool` | [`operator!=`](#operator-10) `const` `inline` |  |
| `bool` | [`operator==`](#operator-11) `const` `inline` |  |

---

#### operator*

`const` `inline`

```cpp
inline item operator*() const
```

Defined in include/teeny/dynamic.h:512

---

#### operator++

`inline`

```cpp
inline enum_iterator & operator++()
```

Defined in include/teeny/dynamic.h:513

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const enum_iterator & o) const
```

Defined in include/teeny/dynamic.h:514

---

#### operator==

`const` `inline`

```cpp
inline bool operator==(const enum_iterator & o) const
```

Defined in include/teeny/dynamic.h:515



## enum_range

```cpp
#include <dynamic.h>
```

```cpp
struct enum_range
```

Defined in include/teeny/dynamic.h:517

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`r`](#r) | `variable` | Declared here |
| [`begin`](#begin-1) | `function` | Declared here |
| [`end`](#end-1) | `function` | Declared here |
| [`subrange`](#subrange-1) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `anyrank_front` | [`r`](#r)  |  |

---

#### r

```cpp
anyrank_front r
```

Defined in include/teeny/dynamic.h:518

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`begin`](#begin-1) `const` `inline` |  |
| `enum_iterator` | [`end`](#end-1) `const` `inline` |  |
| `enum_subrange` | [`subrange`](#subrange-1) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline enum_iterator begin() const
```

Defined in include/teeny/dynamic.h:519

---

#### end

`const` `inline`

```cpp
inline enum_iterator end() const
```

Defined in include/teeny/dynamic.h:520

---

#### subrange

`const` `inline`

```cpp
inline enum_subrange subrange(offset_t lo, offset_t hi) const
```

Defined in include/teeny/dynamic.h:526



## enum_subrange

```cpp
#include <dynamic.h>
```

```cpp
struct enum_subrange
```

Defined in include/teeny/dynamic.h:521

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`b`](#b) | `variable` | Declared here |
| [`e`](#e) | `variable` | Declared here |
| [`begin`](#begin-2) | `function` | Declared here |
| [`end`](#end-2) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`b`](#b)  |  |
| `enum_iterator` | [`e`](#e)  |  |

---

#### b

```cpp
enum_iterator b
```

Defined in include/teeny/dynamic.h:522

---

#### e

```cpp
enum_iterator e
```

Defined in include/teeny/dynamic.h:522

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`begin`](#begin-2) `const` `inline` |  |
| `enum_iterator` | [`end`](#end-2) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline enum_iterator begin() const
```

Defined in include/teeny/dynamic.h:523

---

#### end

`const` `inline`

```cpp
inline enum_iterator end() const
```

Defined in include/teeny/dynamic.h:524



## item

```cpp
#include <dynamic.h>
```

```cpp
struct item
```

Defined in include/teeny/dynamic.h:509

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`index`](#index) | `variable` | Declared here |
| [`cell`](#cell-1) | `variable` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `coord` | [`index`](#index)  |  |
| `Cell` | [`cell`](#cell-1)  |  |

---

#### index

```cpp
coord index
```

Defined in include/teeny/dynamic.h:509

---

#### cell

```cpp
Cell cell
```

Defined in include/teeny/dynamic.h:509



## iterator

```cpp
#include <dynamic.h>
```

```cpp
struct iterator
```

Defined in include/teeny/dynamic.h:437

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`tmpl`](#tmpl) | `variable` | Declared here |
| [`base`](#base) | `variable` | Declared here |
| [`ctr`](#ctr-1) | `variable` | Declared here |
| [`ext`](#ext) | `variable` | Declared here |
| [`str`](#str) | `variable` | Declared here |
| [`nb`](#nb-1) | `variable` | Declared here |
| [`off`](#off) | `variable` | Declared here |
| [`lin`](#lin-1) | `variable` | Declared here |
| [`operator*`](#operator-12) | `function` | Declared here |
| [`operator++`](#operator-13) | `function` | Declared here |
| [`operator!=`](#operator-14) | `function` | Declared here |
| [`operator==`](#operator-15) | `function` | Declared here |
| [`index`](#index-1) | `function` | Declared here |
| [`nbatch`](#nbatch) | `function` | Declared here |
| [`linear`](#linear-1) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `Cell` | [`tmpl`](#tmpl)  |  |
| `T *` | [`base`](#base)  |  |
| `offset_t` | [`ctr`](#ctr-1)  |  |
| `offset_t` | [`ext`](#ext)  |  |
| `offset_t` | [`str`](#str)  |  |
| `int` | [`nb`](#nb-1)  |  |
| `offset_t` | [`off`](#off)  |  |
| `offset_t` | [`lin`](#lin-1)  |  |

---

#### tmpl

```cpp
Cell tmpl
```

Defined in include/teeny/dynamic.h:438

---

#### base

```cpp
T * base
```

Defined in include/teeny/dynamic.h:439

---

#### ctr

```cpp
offset_t ctr
```

Defined in include/teeny/dynamic.h:440

---

#### ext

```cpp
offset_t ext
```

Defined in include/teeny/dynamic.h:441

---

#### str

```cpp
offset_t str
```

Defined in include/teeny/dynamic.h:442

---

#### nb

```cpp
int nb
```

Defined in include/teeny/dynamic.h:443

---

#### off

```cpp
offset_t off
```

Defined in include/teeny/dynamic.h:444

---

#### lin

```cpp
offset_t lin
```

Defined in include/teeny/dynamic.h:444

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `Cell` | [`operator*`](#operator-12) `const` `inline` |  |
| `iterator &` | [`operator++`](#operator-13) `inline` |  |
| `bool` | [`operator!=`](#operator-14) `const` `inline` |  |
| `bool` | [`operator==`](#operator-15) `const` `inline` |  |
| `offset_t` | [`index`](#index-1) `const` `inline` `noexcept` |  |
| `int` | [`nbatch`](#nbatch) `const` `inline` `noexcept` |  |
| `offset_t` | [`linear`](#linear-1) `const` `inline` `noexcept` |  |

---

#### operator*

`const` `inline`

```cpp
inline Cell operator*() const
```

Defined in include/teeny/dynamic.h:445

---

#### operator++

`inline`

```cpp
inline iterator & operator++()
```

Defined in include/teeny/dynamic.h:446

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const iterator & o) const
```

Defined in include/teeny/dynamic.h:454

---

#### operator==

`const` `inline`

```cpp
inline bool operator==(const iterator & o) const
```

Defined in include/teeny/dynamic.h:455

---

#### index

`const` `inline` `noexcept`

```cpp
inline offset_t index(int d) const noexcept
```

Defined in include/teeny/dynamic.h:461

---

#### nbatch

`const` `inline` `noexcept`

```cpp
inline int nbatch() const noexcept
```

Defined in include/teeny/dynamic.h:462

---

#### linear

`const` `inline` `noexcept`

```cpp
inline offset_t linear() const noexcept
```

Defined in include/teeny/dynamic.h:463



## subrange_t

```cpp
#include <dynamic.h>
```

```cpp
struct subrange_t
```

Defined in include/teeny/dynamic.h:484

A `[lo, hi)` slice of the batch cells for chunked/threaded sweeps: seed the incremental cursor once at `lo`, then O(1) per step.

Split `[0, [size()](#size-2))` across threads/blocks; each sweeps its own chunk.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`b`](#b-1) | `variable` | Declared here |
| [`e`](#e-1) | `variable` | Declared here |
| [`begin`](#begin-3) | `function` | Declared here |
| [`end`](#end-3) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `iterator` | [`b`](#b-1)  |  |
| `iterator` | [`e`](#e-1)  |  |

---

#### b

```cpp
iterator b
```

Defined in include/teeny/dynamic.h:485

---

#### e

```cpp
iterator e
```

Defined in include/teeny/dynamic.h:485

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `iterator` | [`begin`](#begin-3) `const` `inline` |  |
| `iterator` | [`end`](#end-3) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline iterator begin() const
```

Defined in include/teeny/dynamic.h:486

---

#### end

`const` `inline`

```cpp
inline iterator end() const
```

Defined in include/teeny/dynamic.h:487



## anyshape

```cpp
#include <alias.h>
```

```cpp
template<auto... Es>
struct anyshape
```

Defined in include/teeny/alias.h:193

The shape spelling for the rank-erased `anyrank` boundary: exactly one `etc` marks the dynamic-rank region, the dims AFTER it are the static **Tail** (anchored at `ndim`), the dims BEFORE it are the static **Head** (anchored at 0).

Each non-`etc` slot is a per-dim static extent or `-1` (dynamic), exactly like `shape<...>`. Hand it to `as_anyrank(..., anyshape<etc,-1,-1,3>{})` or `from_dlpack<T, anyshape<etc,-1,-1,3>>(m)` so the peeled cells fold those inner dims — `anyshape<etc,-1,-1,3>` == `(*batch, spatial, spatial, C=3)`.

A static leading **Head** (dims BEFORE `etc`) is allowed too: `anyshape<A, B, etc, C, D>` == `(A, B, *middle, C, D)` — e.g. `anyshape<3, etc, 5>` for `(C_in=3, *spatial, C_out=5)`. The Head folds in `fixed`/`dispatch_rank` (full-rank materialisation); `peel_front<-Sr>` stays trailing-oriented (a leading Head is normally peeled into the batch).

Unlike a plain `shape<...>` (a concrete fixed-rank `extents`), an `anyshape` is a SPEC, not a tensor type — a runtime-rank object needs the data + runtime arrays, not just a type.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`head`](#head) | `typedef` | Declared here |
| [`tail`](#tail) | `typedef` | Declared here |

### Public Types

| Name | Description |
|------|-------------|
| [`head`](#head)  |  |
| [`tail`](#tail)  |  |

---

#### head

```cpp
using head = typename _sp::head
```

Defined in include/teeny/alias.h:197

---

#### tail

```cpp
using tail = typename _sp::tail
```

Defined in include/teeny/alias.h:198



## axis

```cpp
#include <alias.h>
```

```cpp
template<long... Axes>
struct axis
```

Defined in include/teeny/alias.h:213

Compile-time **axis selector** — a value tag carrying a list of axes, the sibling of `shape<...>` for axis arguments.

It lets axis-taking ops be spelled by VALUE (deducing the axes from the argument type) instead of an explicit template list, so on a type-dependent receiver they need no `.template`: `peel(t, axis<0,1>{})` == `peel<0,1>(t)`, `t.slice_along(axis<0,2>{}, i, slice(1,4))` == `t.slice_along<0,2>(i, slice(1,4))`.

Like numpy's `axis: int | list[int]`, one variadic tag covers both a single axis (`axis<0>{}`) and a list (`axis<0,2>{}`); axes are **signed** (negatives count from the back, as everywhere in teeny). `rank` is the axis count.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`rank`](#rank-2) | `variable` | Declared here |

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr size_t` | [`rank`](#rank-2) `static` `constexpr` |  |

---

#### rank

`static` `constexpr`

```cpp
constexpr size_t rank = sizeof...(Axes)
```

Defined in include/teeny/alias.h:213



## compute_type

```cpp
#include <half.h>
```

```cpp
template<class T>
struct compute_type
```

Defined in include/teeny/half.h:137

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

Defined in include/teeny/half.h:137



## compute_type< bfloat16 >

```cpp
#include <half.h>
```

```cpp
struct compute_type< bfloat16 >
```

Defined in include/teeny/half.h:139

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

Defined in include/teeny/half.h:139



## compute_type< half >

```cpp
#include <half.h>
```

```cpp
struct compute_type< half >
```

Defined in include/teeny/half.h:138

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

Defined in include/teeny/half.h:138



## copy_meta_t

```cpp
#include <dynamic.h>
```

```cpp
struct copy_meta_t
```

Defined in include/teeny/dynamic.h:207

Tag for `as_anyrank(..., copy_meta)`: COPY shape/stride into an inline, device-passable store instead of wrapping the caller's arrays.

Named `copy_meta`, not `copy`: a bare `copy` variable in `tny` would, under `using namespace tny`, shadow an unqualified `std::copy(...)` call (finding a variable suppresses ADL) — a nasty surprise.



## cpp_alloc

```cpp
#include <storage.h>
```

```cpp
struct cpp_alloc
```

Defined in include/teeny/storage.h:112

Host allocator using C++ `new[]` / `delete[]`.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`allocate`](#allocate) | `function` | Declared here |
| [`allocate_uninit`](#allocate_uninit) | `function` | Declared here |
| [`deallocate`](#deallocate) | `function` | Declared here |

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `T *` | [`allocate`](#allocate) `static` `inline` |  |
| `T *` | [`allocate_uninit`](#allocate_uninit) `static` `inline` |  |
| `void` | [`deallocate`](#deallocate) `static` `inline` |  |

---

#### allocate

`static` `inline`

```cpp
template<class T> static inline T * allocate(size_t n)
```

Defined in include/teeny/storage.h:113

---

#### allocate_uninit

`static` `inline`

```cpp
template<class T> static inline T * allocate_uninit(size_t n)
```

Defined in include/teeny/storage.h:114

---

#### deallocate

`static` `inline`

```cpp
template<class T> static inline void deallocate(T * p)
```

Defined in include/teeny/storage.h:115



## cuda_gpu_alloc

```cpp
#include <cuda.h>
```

```cpp
struct cuda_gpu_alloc
```

Defined in include/teeny/cuda.h:30

Device (GPU) memory (`cudaMalloc`).

Not host-dereferenceable.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`allocate`](#allocate-1) | `function` | Declared here |
| [`allocate_uninit`](#allocate_uninit-1) | `function` | Declared here |
| [`deallocate`](#deallocate-1) | `function` | Declared here |

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `T *` | [`allocate`](#allocate-1) `static` `inline` |  |
| `T *` | [`allocate_uninit`](#allocate_uninit-1) `static` `inline` |  |
| `void` | [`deallocate`](#deallocate-1) `static` `inline` |  |

---

#### allocate

`static` `inline`

```cpp
template<class T> static inline T * allocate(size_t n)
```

Defined in include/teeny/cuda.h:31

---

#### allocate_uninit

`static` `inline`

```cpp
template<class T> static inline T * allocate_uninit(size_t n)
```

Defined in include/teeny/cuda.h:34

---

#### deallocate

`static` `inline`

```cpp
template<class T> static inline void deallocate(T * p)
```

Defined in include/teeny/cuda.h:35



## cuda_mapped_alloc

```cpp
#include <cuda.h>
```

```cpp
struct cuda_mapped_alloc
```

Defined in include/teeny/cuda.h:48

Page-locked + device-mapped (zero-copy) host memory (`cudaHostAlloc`).

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`allocate`](#allocate-2) | `function` | Declared here |
| [`allocate_uninit`](#allocate_uninit-2) | `function` | Declared here |
| [`deallocate`](#deallocate-2) | `function` | Declared here |

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `T *` | [`allocate`](#allocate-2) `static` `inline` |  |
| `T *` | [`allocate_uninit`](#allocate_uninit-2) `static` `inline` |  |
| `void` | [`deallocate`](#deallocate-2) `static` `inline` |  |

---

#### allocate

`static` `inline`

```cpp
template<class T> static inline T * allocate(size_t n)
```

Defined in include/teeny/cuda.h:49

---

#### allocate_uninit

`static` `inline`

```cpp
template<class T> static inline T * allocate_uninit(size_t n)
```

Defined in include/teeny/cuda.h:52

---

#### deallocate

`static` `inline`

```cpp
template<class T> static inline void deallocate(T * p)
```

Defined in include/teeny/cuda.h:53



## cuda_pinned_alloc

```cpp
#include <cuda.h>
```

```cpp
struct cuda_pinned_alloc
```

Defined in include/teeny/cuda.h:39

Page-locked ("pinned") host memory (`cudaMallocHost`).

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`allocate`](#allocate-3) | `function` | Declared here |
| [`allocate_uninit`](#allocate_uninit-3) | `function` | Declared here |
| [`deallocate`](#deallocate-3) | `function` | Declared here |

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `T *` | [`allocate`](#allocate-3) `static` `inline` |  |
| `T *` | [`allocate_uninit`](#allocate_uninit-3) `static` `inline` |  |
| `void` | [`deallocate`](#deallocate-3) `static` `inline` |  |

---

#### allocate

`static` `inline`

```cpp
template<class T> static inline T * allocate(size_t n)
```

Defined in include/teeny/cuda.h:40

---

#### allocate_uninit

`static` `inline`

```cpp
template<class T> static inline T * allocate_uninit(size_t n)
```

Defined in include/teeny/cuda.h:43

---

#### deallocate

`static` `inline`

```cpp
template<class T> static inline void deallocate(T * p)
```

Defined in include/teeny/cuda.h:44



## dtype

```cpp
#include <alias.h>
```

```cpp
template<class T>
struct dtype
```

Defined in include/teeny/alias.h:238

Compile-time **element-type tag** — a value carrier for `T`, the sibling of `axis<...>` for the dtype argument.

It lets a type-parameterised call be spelled by VALUE (deducing `T` from the argument) instead of an explicit `<T>` template argument, so on a type-dependent receiver it needs no `.template`: `empty(shape<3,3>{}, dtype<double>{})` == `empty<double>(shape<3,3>{})`, `a.to(dtype<float>{})` == `a.to<float>()`. Numpy's `dtype=` keyword is the namesake — including reuse as the reduction accumulator/result type: `sum(a, dtype<double>{})` == `sum<double>(a)`, matching `np.sum(a, dtype=...)`.



## into_t

```cpp
#include <tensor.h>
```

```cpp
template<class D>
struct into_t
```

Defined in include/teeny/tensor.h:32

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`dest`](#dest) | `variable` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `D &` | [`dest`](#dest)  |  |

---

#### dest

```cpp
D & dest
```

Defined in include/teeny/tensor.h:32



## keep_strides

```cpp
#include <layout.h>
```

```cpp
struct keep_strides
```

Defined in include/teeny/layout.h:231

Sentinel `Layout` selector for `recast<NewShape, [keep_strides](#keep_strides)>()` (the default): PRESERVE the source strides (fold where the source layout makes them derivable, keep runtime otherwise).

Contrast an explicit layout — `recast<NewShape, ccontiguous>()` reinterprets AS that layout, deriving the strides from the extents (the "I promise this is C-contiguous" form). Not a real layout (it has no mapping) — only a recast tag.



## keepdims_t

```cpp
#include <alias.h>
```

```cpp
struct keepdims_t
```

Defined in include/teeny/alias.h:267

numpy/pytorch `keepdims=True` tag for axis reductions — pass as any trailing keyword (composes with `dtype<...>`/`axis<...>`/`into(dest)` in any order) to keep the reduced axes as size-1 instead of removing them, so the result broadcasts back against the input: `sum<0>(a, keepdims)`, `sum(a, axis<0,2>{}, keepdims)`.

A distinct empty-tag type, like `all`/`none`, so it never collides with another argument.



## none_t

```cpp
#include <indexing.h>
```

```cpp
struct none_t
```

Defined in include/teeny/indexing.h:117

Open-ended slice sentinel — teeny's `None` (python `a[:n]` / `a[m:]`).

`slice(none, n)` starts at 0, `slice(m, none)` runs to the end, and `slice(none, none)`**folds** to `full_extent` — so `all == slice(none, none)`, keeping the axis and its static extent (`all` is built from it). Combined with runtime bounds it resolves at run time, so the one sentinel covers both.

A BARE `none`**argument** to `operator()`/`uget` is a different thing: numpy `newaxis` (`a[None]`), which inserts a size-1 axis — see `_is_newaxis` below. `newaxis` is a named alias of `none` for that bare-argument spelling (numpy calls the same value `None` when it's a slice bound and `np.newaxis` when it's inserting an axis; teeny's `none`/`newaxis` mirror that with one type).



## owning_storage

```cpp
#include <storage.h>
```

```cpp
template<class T, class Alloc>
struct owning_storage
```

Defined in include/teeny/storage.h:129

Generic owning storage (move-only, no ref-counting), parameterised by an allocator policy.

Shared by all owning `storage` modes.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p) | `variable` | Declared here |
| [`owning_storage`](#owning_storage-1) | `function` | Declared here |
| [`owning_storage`](#owning_storage-2) | `function` | Declared here |
| [`owning_storage`](#owning_storage-3) | `function` | Declared here |
| [`owning_storage`](#owning_storage-4) | `function` | Declared here |
| [`owning_storage`](#owning_storage-5) | `function` | Declared here |
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

Defined in include/teeny/storage.h:130

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
|  | [`owning_storage`](#owning_storage-1)  | Defaulted constructor. |
|  | [`owning_storage`](#owning_storage-2) `inline` `explicit` |  |
|  | [`owning_storage`](#owning_storage-3) `inline` |  |
|  | [`owning_storage`](#owning_storage-4)  | Deleted constructor. |
|  | [`owning_storage`](#owning_storage-5) `inline` `noexcept` |  |
| `T *` | [`data`](#data-1) `inline` `noexcept` |  |
| `const T *` | [`data`](#data-2) `const` `inline` `noexcept` |  |

---

#### owning_storage

```cpp
owning_storage() = default
```

Defined in include/teeny/storage.h:131

Defaulted constructor.

---

#### owning_storage

`inline` `explicit`

```cpp
inline explicit owning_storage(size_t n)
```

Defined in include/teeny/storage.h:132

---

#### owning_storage

`inline`

```cpp
inline owning_storage(size_t n, _uninit_t)
```

Defined in include/teeny/storage.h:133

---

#### owning_storage

```cpp
owning_storage(const owning_storage &) = delete
```

Defined in include/teeny/storage.h:134

Deleted constructor.

---

#### owning_storage

`inline` `noexcept`

```cpp
inline owning_storage(owning_storage && o) noexcept
```

Defined in include/teeny/storage.h:136

---

#### data

`inline` `noexcept`

```cpp
inline T * data() noexcept
```

Defined in include/teeny/storage.h:142

---

#### data

`const` `inline` `noexcept`

```cpp
inline const T * data() const noexcept
```

Defined in include/teeny/storage.h:143



## peel_range

```cpp
#include <iterate.h>
```

```cpp
template<class MD, storage OW, size_t... Axes>
struct peel_range
```

Defined in include/teeny/iterate.h:154

A range of sub-views obtained by peeling `Axes...`.

Supports `[size()](#size-3)`, random-access `operator[]` (grid-stride loops), range-for (an INCREMENTAL cursor — #110 — that advances the pointer instead of re-decoding each step, and builds the loop-invariant sub-view mapping once), and `subrange(lo,hi)` for chunked/threaded sweeps.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`src`](#src-1) | `variable` | Declared here |
| [`size`](#size-3) | `function` | Declared here |
| [`operator[]`](#operator-16) | `function` | Declared here |
| [`begin`](#begin-4) | `function` | Declared here |
| [`end`](#end-4) | `function` | Declared here |
| [`subrange`](#subrange-2) | `function` | Declared here |
| [`enumerate`](#enumerate-1) | `function` | Declared here |
| [`Nd`](#nd) | `variable` | Declared here |
| [`index_type`](#index_type) | `typedef` | Declared here |
| [`Cell`](#cell-2) | `typedef` | Declared here |
| [`El`](#el) | `typedef` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `MD` | [`src`](#src-1)  |  |

---

#### src

```cpp
MD src
```

Defined in include/teeny/iterate.h:157

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `index_type` | [`size`](#size-3) `const` `inline` `noexcept` |  |
| `auto` | [`operator[]`](#operator-16) `const` `inline` |  |
| `iterator` | [`begin`](#begin-4) `const` `inline` |  |
| `iterator` | [`end`](#end-4) `const` `inline` |  |
| `subrange_t` | [`subrange`](#subrange-2) `const` `inline` |  |
| `enum_range` | [`enumerate`](#enumerate-1) `const` `inline` |  |

---

#### size

`const` `inline` `noexcept`

```cpp
inline index_type size() const noexcept
```

Defined in include/teeny/iterate.h:159

---

#### operator[]

`const` `inline`

```cpp
inline auto operator[](index_type i) const
```

Defined in include/teeny/iterate.h:167

---

#### begin

`const` `inline`

```cpp
inline iterator begin() const
```

Defined in include/teeny/iterate.h:202

---

#### end

`const` `inline`

```cpp
inline iterator end() const
```

Defined in include/teeny/iterate.h:203

---

#### subrange

`const` `inline`

```cpp
inline subrange_t subrange(index_type lo, index_type hi) const
```

Defined in include/teeny/iterate.h:213

---

#### enumerate

`const` `inline`

```cpp
inline enum_range enumerate() const
```

Defined in include/teeny/iterate.h:248

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr size_t` | [`Nd`](#nd) `static` `constexpr` |  |

---

#### Nd

`static` `constexpr`

```cpp
constexpr size_t Nd = sizeof...(Axes)
```

Defined in include/teeny/iterate.h:156

### Public Types

| Name | Description |
|------|-------------|
| [`index_type`](#index_type)  |  |
| [`Cell`](#cell-2)  |  |
| [`El`](#el)  |  |

---

#### index_type

```cpp
using index_type = typename MD::index_type
```

Defined in include/teeny/iterate.h:155

---

#### Cell

```cpp
using Cell = decltype(_md::peel_at_ow< OW, Axes... >(declval< const MD & >(), index_type(0)))
```

Defined in include/teeny/iterate.h:172

---

#### El

```cpp
using El = typename Cell::element_type
```

Defined in include/teeny/iterate.h:173



## enum_iterator

```cpp
#include <iterate.h>
```

```cpp
struct enum_iterator
```

Defined in include/teeny/iterate.h:227

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`it`](#it-1) | `variable` | Declared here |
| [`operator*`](#operator-17) | `function` | Declared here |
| [`operator++`](#operator-18) | `function` | Declared here |
| [`operator!=`](#operator-19) | `function` | Declared here |
| [`operator==`](#operator-20) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `iterator` | [`it`](#it-1)  |  |

---

#### it

```cpp
iterator it
```

Defined in include/teeny/iterate.h:228

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `item` | [`operator*`](#operator-17) `const` `inline` |  |
| `enum_iterator &` | [`operator++`](#operator-18) `inline` |  |
| `bool` | [`operator!=`](#operator-19) `const` `inline` |  |
| `bool` | [`operator==`](#operator-20) `const` `inline` |  |

---

#### operator*

`const` `inline`

```cpp
inline item operator*() const
```

Defined in include/teeny/iterate.h:229

---

#### operator++

`inline`

```cpp
inline enum_iterator & operator++()
```

Defined in include/teeny/iterate.h:230

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const enum_iterator & o) const
```

Defined in include/teeny/iterate.h:231

---

#### operator==

`const` `inline`

```cpp
inline bool operator==(const enum_iterator & o) const
```

Defined in include/teeny/iterate.h:232



## enum_range

```cpp
#include <iterate.h>
```

```cpp
struct enum_range
```

Defined in include/teeny/iterate.h:234

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`r`](#r-1) | `variable` | Declared here |
| [`begin`](#begin-5) | `function` | Declared here |
| [`end`](#end-5) | `function` | Declared here |
| [`subrange`](#subrange-3) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `peel_range` | [`r`](#r-1)  |  |

---

#### r

```cpp
peel_range r
```

Defined in include/teeny/iterate.h:235

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`begin`](#begin-5) `const` `inline` |  |
| `enum_iterator` | [`end`](#end-5) `const` `inline` |  |
| `enum_subrange` | [`subrange`](#subrange-3) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline enum_iterator begin() const
```

Defined in include/teeny/iterate.h:236

---

#### end

`const` `inline`

```cpp
inline enum_iterator end() const
```

Defined in include/teeny/iterate.h:237

---

#### subrange

`const` `inline`

```cpp
inline enum_subrange subrange(index_type lo, index_type hi) const
```

Defined in include/teeny/iterate.h:243



## enum_subrange

```cpp
#include <iterate.h>
```

```cpp
struct enum_subrange
```

Defined in include/teeny/iterate.h:238

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`b`](#b-2) | `variable` | Declared here |
| [`e`](#e-2) | `variable` | Declared here |
| [`begin`](#begin-6) | `function` | Declared here |
| [`end`](#end-6) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`b`](#b-2)  |  |
| `enum_iterator` | [`e`](#e-2)  |  |

---

#### b

```cpp
enum_iterator b
```

Defined in include/teeny/iterate.h:239

---

#### e

```cpp
enum_iterator e
```

Defined in include/teeny/iterate.h:239

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`begin`](#begin-6) `const` `inline` |  |
| `enum_iterator` | [`end`](#end-6) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline enum_iterator begin() const
```

Defined in include/teeny/iterate.h:240

---

#### end

`const` `inline`

```cpp
inline enum_iterator end() const
```

Defined in include/teeny/iterate.h:241



## item

```cpp
#include <iterate.h>
```

```cpp
struct item
```

Defined in include/teeny/iterate.h:226

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`index`](#index-2) | `variable` | Declared here |
| [`cell`](#cell-3) | `variable` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `array< index_type, Nd ? Nd :1 >` | [`index`](#index-2)  |  |
| `Cell` | [`cell`](#cell-3)  |  |

---

#### index

```cpp
array< index_type, Nd ? Nd :1 > index
```

Defined in include/teeny/iterate.h:226

---

#### cell

```cpp
Cell cell
```

Defined in include/teeny/iterate.h:226



## iterator

```cpp
#include <iterate.h>
```

```cpp
struct iterator
```

Defined in include/teeny/iterate.h:174

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`tmpl`](#tmpl-1) | `variable` | Declared here |
| [`base`](#base-1) | `variable` | Declared here |
| [`cur`](#cur) | `variable` | Declared here |
| [`operator*`](#operator-21) | `function` | Declared here |
| [`operator++`](#operator-22) | `function` | Declared here |
| [`operator!=`](#operator-23) | `function` | Declared here |
| [`operator==`](#operator-24) | `function` | Declared here |
| [`index`](#index-3) | `function` | Declared here |
| [`index`](#index-4) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `Cell` | [`tmpl`](#tmpl-1)  |  |
| `El *` | [`base`](#base-1)  |  |
| `_md::peel_cursor< index_type, Nd >` | [`cur`](#cur)  |  |

---

#### tmpl

```cpp
Cell tmpl
```

Defined in include/teeny/iterate.h:175

---

#### base

```cpp
El * base
```

Defined in include/teeny/iterate.h:176

---

#### cur

```cpp
_md::peel_cursor< index_type, Nd > cur
```

Defined in include/teeny/iterate.h:177

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `Cell` | [`operator*`](#operator-21) `const` `inline` |  |
| `iterator &` | [`operator++`](#operator-22) `inline` |  |
| `bool` | [`operator!=`](#operator-23) `const` `inline` |  |
| `bool` | [`operator==`](#operator-24) `const` `inline` |  |
| `index_type` | [`index`](#index-3) `const` `inline` `noexcept` |  |
| `array< index_type, Nd ? Nd :1 >` | [`index`](#index-4) `const` `inline` `noexcept` |  |

---

#### operator*

`const` `inline`

```cpp
inline Cell operator*() const
```

Defined in include/teeny/iterate.h:178

---

#### operator++

`inline`

```cpp
inline iterator & operator++()
```

Defined in include/teeny/iterate.h:179

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const iterator & o) const
```

Defined in include/teeny/iterate.h:180

---

#### operator==

`const` `inline`

```cpp
inline bool operator==(const iterator & o) const
```

Defined in include/teeny/iterate.h:181

---

#### index

`const` `inline` `noexcept`

```cpp
inline index_type index(size_t d) const noexcept
```

Defined in include/teeny/iterate.h:186

---

#### index

`const` `inline` `noexcept`

```cpp
inline array< index_type, Nd ? Nd :1 > index() const noexcept
```

Defined in include/teeny/iterate.h:187



## subrange_t

```cpp
#include <iterate.h>
```

```cpp
struct subrange_t
```

Defined in include/teeny/iterate.h:208

A `[lo, hi)` slice of the cells for chunked/threaded sweeps: seed the incremental cursor once at `lo`, then O(1) per step within the chunk.

(Split `[0,[size()](#size-3))` across threads/blocks; each sweeps its chunk.)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`b`](#b-3) | `variable` | Declared here |
| [`e`](#e-3) | `variable` | Declared here |
| [`begin`](#begin-7) | `function` | Declared here |
| [`end`](#end-7) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `iterator` | [`b`](#b-3)  |  |
| `iterator` | [`e`](#e-3)  |  |

---

#### b

```cpp
iterator b
```

Defined in include/teeny/iterate.h:209

---

#### e

```cpp
iterator e
```

Defined in include/teeny/iterate.h:209

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `iterator` | [`begin`](#begin-7) `const` `inline` |  |
| `iterator` | [`end`](#end-7) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline iterator begin() const
```

Defined in include/teeny/iterate.h:210

---

#### end

`const` `inline`

```cpp
inline iterator end() const
```

Defined in include/teeny/iterate.h:211



## ptr_storage

```cpp
#include <storage.h>
```

```cpp
template<class T>
struct ptr_storage
```

Defined in include/teeny/storage.h:159

> **Subclassed by:** [`gpu_view, N >`](#gpu_viewn), [`mapped_view, N >`](#mapped_viewn), [`pinned_view, N >`](#pinned_viewn), [`view, N >`](#viewn)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p-1) | `variable` | Declared here |
| [`ptr_storage`](#ptr_storage-1) | `function` | Declared here |
| [`ptr_storage`](#ptr_storage-2) | `function` | Declared here |
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

Defined in include/teeny/storage.h:160

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
|  | [`ptr_storage`](#ptr_storage-1)  | Defaulted constructor. |
| `constexpr` | [`ptr_storage`](#ptr_storage-2) `inline` `constexpr` `noexcept` |  |
| `constexpr T *` | [`data`](#data-3) `const` `inline` `constexpr` `noexcept` |  |

---

#### ptr_storage

```cpp
ptr_storage() = default
```

Defined in include/teeny/storage.h:161

Defaulted constructor.

---

#### ptr_storage

`inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr ptr_storage(T * q) noexcept
```

Defined in include/teeny/storage.h:162

---

#### data

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr T * data() const noexcept
```

Defined in include/teeny/storage.h:163



## storage_policy

```cpp
template<class T, storage O, size_t N>
struct storage_policy
```

Defined in include/teeny/storage.h:151



## gpu, N >

```cpp
#include <cuda.h>
```

```cpp
template<class T, size_t N>
struct gpu, N >
```

Defined in include/teeny/cuda.h:61

> **Inherits:** [`owning_storage< T, cuda_gpu_alloc >`](#owning_storage)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p) | `variable` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-3) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-4) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-5) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |

### Inherited from [`owning_storage`](#owning_storage)

| Kind | Name | Description |
|------|------|-------------|
| `variable` | [`p`](#p)  |  |
| `function` | [`owning_storage`](#owning_storage-1)  | Defaulted constructor. |
| `function` | [`owning_storage`](#owning_storage-2) `inline` `explicit` |  |
| `function` | [`owning_storage`](#owning_storage-3) `inline` |  |
| `function` | [`owning_storage`](#owning_storage-4)  | Deleted constructor. |
| `function` | [`owning_storage`](#owning_storage-5) `inline` `noexcept` |  |
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

Defined in include/teeny/storage.h:166

> **Inherits:** [`ptr_storage< T >`](#ptr_storage)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p-1) | `variable` | Inherited from [`ptr_storage`](#ptr_storage) |
| [`ptr_storage`](#ptr_storage-1) | `function` | Inherited from [`ptr_storage`](#ptr_storage) |
| [`ptr_storage`](#ptr_storage-2) | `function` | Inherited from [`ptr_storage`](#ptr_storage) |
| [`data`](#data-3) | `function` | Inherited from [`ptr_storage`](#ptr_storage) |

### Inherited from [`ptr_storage`](#ptr_storage)

| Kind | Name | Description |
|------|------|-------------|
| `variable` | [`p`](#p-1)  |  |
| `function` | [`ptr_storage`](#ptr_storage-1)  | Defaulted constructor. |
| `function` | [`ptr_storage`](#ptr_storage-2) `inline` `constexpr` `noexcept` |  |
| `function` | [`data`](#data-3) `const` `inline` `constexpr` `noexcept` |  |



## heap, N >

```cpp
#include <storage.h>
```

```cpp
template<class T, size_t N>
struct heap, N >
```

Defined in include/teeny/storage.h:186

> **Inherits:** [`owning_storage< T, cpp_alloc >`](#owning_storage)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p) | `variable` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-3) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-4) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-5) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |

### Inherited from [`owning_storage`](#owning_storage)

| Kind | Name | Description |
|------|------|-------------|
| `variable` | [`p`](#p)  |  |
| `function` | [`owning_storage`](#owning_storage-1)  | Defaulted constructor. |
| `function` | [`owning_storage`](#owning_storage-2) `inline` `explicit` |  |
| `function` | [`owning_storage`](#owning_storage-3) `inline` |  |
| `function` | [`owning_storage`](#owning_storage-4)  | Deleted constructor. |
| `function` | [`owning_storage`](#owning_storage-5) `inline` `noexcept` |  |
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

Defined in include/teeny/cuda.h:69

> **Inherits:** [`owning_storage< T, cuda_mapped_alloc >`](#owning_storage)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p) | `variable` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-3) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-4) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-5) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |

### Inherited from [`owning_storage`](#owning_storage)

| Kind | Name | Description |
|------|------|-------------|
| `variable` | [`p`](#p)  |  |
| `function` | [`owning_storage`](#owning_storage-1)  | Defaulted constructor. |
| `function` | [`owning_storage`](#owning_storage-2) `inline` `explicit` |  |
| `function` | [`owning_storage`](#owning_storage-3) `inline` |  |
| `function` | [`owning_storage`](#owning_storage-4)  | Deleted constructor. |
| `function` | [`owning_storage`](#owning_storage-5) `inline` `noexcept` |  |
| `function` | [`data`](#data-1) `inline` `noexcept` |  |
| `function` | [`data`](#data-2) `const` `inline` `noexcept` |  |



## mapped_view, N >

```cpp
#include <storage.h>
```

```cpp
template<class T, size_t N>
struct mapped_view, N >
```

Defined in include/teeny/storage.h:168

> **Inherits:** [`ptr_storage< T >`](#ptr_storage)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p-1) | `variable` | Inherited from [`ptr_storage`](#ptr_storage) |
| [`ptr_storage`](#ptr_storage-1) | `function` | Inherited from [`ptr_storage`](#ptr_storage) |
| [`ptr_storage`](#ptr_storage-2) | `function` | Inherited from [`ptr_storage`](#ptr_storage) |
| [`data`](#data-3) | `function` | Inherited from [`ptr_storage`](#ptr_storage) |

### Inherited from [`ptr_storage`](#ptr_storage)

| Kind | Name | Description |
|------|------|-------------|
| `variable` | [`p`](#p-1)  |  |
| `function` | [`ptr_storage`](#ptr_storage-1)  | Defaulted constructor. |
| `function` | [`ptr_storage`](#ptr_storage-2) `inline` `constexpr` `noexcept` |  |
| `function` | [`data`](#data-3) `const` `inline` `constexpr` `noexcept` |  |



## pinned, N >

```cpp
#include <cuda.h>
```

```cpp
template<class T, size_t N>
struct pinned, N >
```

Defined in include/teeny/cuda.h:65

> **Inherits:** [`owning_storage< T, cuda_pinned_alloc >`](#owning_storage)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p) | `variable` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-3) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-4) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`owning_storage`](#owning_storage-5) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-1) | `function` | Inherited from [`owning_storage`](#owning_storage) |
| [`data`](#data-2) | `function` | Inherited from [`owning_storage`](#owning_storage) |

### Inherited from [`owning_storage`](#owning_storage)

| Kind | Name | Description |
|------|------|-------------|
| `variable` | [`p`](#p)  |  |
| `function` | [`owning_storage`](#owning_storage-1)  | Defaulted constructor. |
| `function` | [`owning_storage`](#owning_storage-2) `inline` `explicit` |  |
| `function` | [`owning_storage`](#owning_storage-3) `inline` |  |
| `function` | [`owning_storage`](#owning_storage-4)  | Deleted constructor. |
| `function` | [`owning_storage`](#owning_storage-5) `inline` `noexcept` |  |
| `function` | [`data`](#data-1) `inline` `noexcept` |  |
| `function` | [`data`](#data-2) `const` `inline` `noexcept` |  |



## pinned_view, N >

```cpp
#include <storage.h>
```

```cpp
template<class T, size_t N>
struct pinned_view, N >
```

Defined in include/teeny/storage.h:167

> **Inherits:** [`ptr_storage< T >`](#ptr_storage)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p-1) | `variable` | Inherited from [`ptr_storage`](#ptr_storage) |
| [`ptr_storage`](#ptr_storage-1) | `function` | Inherited from [`ptr_storage`](#ptr_storage) |
| [`ptr_storage`](#ptr_storage-2) | `function` | Inherited from [`ptr_storage`](#ptr_storage) |
| [`data`](#data-3) | `function` | Inherited from [`ptr_storage`](#ptr_storage) |

### Inherited from [`ptr_storage`](#ptr_storage)

| Kind | Name | Description |
|------|------|-------------|
| `variable` | [`p`](#p-1)  |  |
| `function` | [`ptr_storage`](#ptr_storage-1)  | Defaulted constructor. |
| `function` | [`ptr_storage`](#ptr_storage-2) `inline` `constexpr` `noexcept` |  |
| `function` | [`data`](#data-3) `const` `inline` `constexpr` `noexcept` |  |



## stack, N >

```cpp
#include <storage.h>
```

```cpp
template<class T, size_t N>
struct stack, N >
```

Defined in include/teeny/storage.h:172

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`a`](#a) | `variable` | Declared here |
| [`storage_policy`](#storage_policy-1) | `function` | Declared here |
| [`storage_policy`](#storage_policy-2) | `function` | Declared here |
| [`data`](#data-4) | `function` | Declared here |
| [`data`](#data-5) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `array< T, N >` | [`a`](#a)  |  |

---

#### a

```cpp
array< T, N > a
```

Defined in include/teeny/storage.h:173

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `constexpr` | [`storage_policy`](#storage_policy-1) `inline` `constexpr` `noexcept` |  |
|  | [`storage_policy`](#storage_policy-2) `inline` `noexcept` |  |
| `constexpr T *` | [`data`](#data-4) `inline` `constexpr` `noexcept` |  |
| `constexpr const T *` | [`data`](#data-5) `const` `inline` `constexpr` `noexcept` |  |

---

#### storage_policy

`inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr storage_policy() noexcept
```

Defined in include/teeny/storage.h:178

---

#### storage_policy

`inline` `noexcept`

```cpp
inline storage_policy(_uninit_t) noexcept
```

Defined in include/teeny/storage.h:179

---

#### data

`inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr T * data() noexcept
```

Defined in include/teeny/storage.h:180

---

#### data

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const T * data() const noexcept
```

Defined in include/teeny/storage.h:181



## view, N >

```cpp
#include <storage.h>
```

```cpp
template<class T, size_t N>
struct view, N >
```

Defined in include/teeny/storage.h:165

> **Inherits:** [`ptr_storage< T >`](#ptr_storage)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`p`](#p-1) | `variable` | Inherited from [`ptr_storage`](#ptr_storage) |
| [`ptr_storage`](#ptr_storage-1) | `function` | Inherited from [`ptr_storage`](#ptr_storage) |
| [`ptr_storage`](#ptr_storage-2) | `function` | Inherited from [`ptr_storage`](#ptr_storage) |
| [`data`](#data-3) | `function` | Inherited from [`ptr_storage`](#ptr_storage) |

### Inherited from [`ptr_storage`](#ptr_storage)

| Kind | Name | Description |
|------|------|-------------|
| `variable` | [`p`](#p-1)  |  |
| `function` | [`ptr_storage`](#ptr_storage-1)  | Defaulted constructor. |
| `function` | [`ptr_storage`](#ptr_storage-2) `inline` `constexpr` `noexcept` |  |
| `function` | [`data`](#data-3) `const` `inline` `constexpr` `noexcept` |  |



## storage_size

```cpp
#include <storage.h>
```

```cpp
template<class Mapping, bool Stack>
struct storage_size
```

Defined in include/teeny/storage.h:192

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

Defined in include/teeny/storage.h:192



## storage_size< Mapping, true >

```cpp
#include <storage.h>
```

```cpp
template<class Mapping>
struct storage_size< Mapping, true >
```

Defined in include/teeny/storage.h:194

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

Defined in include/teeny/storage.h:195



## strides

```cpp
#include <layout.h>
```

```cpp
template<int64_t... S>
struct strides
```

Defined in include/teeny/layout.h:82

An mdspan layout policy with **per-dimension static or dynamic strides** — the stride analogue of `extents`/`shape`.

`ccontiguous`/`fcontiguous` (mdspan `layout_right`/`layout_left`) give contiguous (extent-derived) strides; `layout_stride` stores every stride at run time. `strides<S...>` bakes the KNOWN strides into the type (folding to immediates) — **including negative strides** — while any dimension marked `dynamic_stride` is supplied at run time: 
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
| [`static_stride`](#static_stride) | `function` | Declared here |
| [`ndyn`](#ndyn) | `function` | Declared here |
| [`all_static`](#all_static) | `function` | Declared here |
| [`slot`](#slot) | `function` | Declared here |

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr size_t` | [`N`](#n) `static` `constexpr` |  |

---

#### N

`static` `constexpr`

```cpp
constexpr size_t N = sizeof...(S)
```

Defined in include/teeny/layout.h:83

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `constexpr int64_t` | [`static_stride`](#static_stride) `static` `inline` `constexpr` `noexcept` | The compile-time stride of dimension `r`&ndash;`dynamic_stride` when that dimension's stride is only known at run time. |
| `constexpr size_t` | [`ndyn`](#ndyn) `static` `inline` `constexpr` `noexcept` | How many dimensions carry a runtime stride (== the mapping's stored size). |
| `constexpr bool` | [`all_static`](#all_static) `static` `inline` `constexpr` `noexcept` |  |
| `constexpr size_t` | [`slot`](#slot) `static` `inline` `constexpr` `noexcept` |  |

---

#### static_stride

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr int64_t static_stride(size_t r) noexcept
```

Defined in include/teeny/layout.h:125

The compile-time stride of dimension `r`&ndash;`dynamic_stride` when that dimension's stride is only known at run time.

(`0` for a rank-0 `strides<>`, whose `r` is never a valid dimension.)

---

#### ndyn

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr size_t ndyn() noexcept
```

Defined in include/teeny/layout.h:130

How many dimensions carry a runtime stride (== the mapping's stored size).

---

#### all_static

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr bool all_static() noexcept
```

Defined in include/teeny/layout.h:133

---

#### slot

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr size_t slot(size_t r) noexcept
```

Defined in include/teeny/layout.h:135



## mapping

```cpp
#include <layout.h>
```

```cpp
template<class Shape>
struct mapping
```

Defined in include/teeny/layout.h:145

> **Inherits:** `ndyn()>`, `Shape`

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`mapping`](#mapping-1) | `function` | Declared here |
| [`mapping`](#mapping-2) | `function` | Declared here |
| [`mapping`](#mapping-3) | `function` | Declared here |
| [`extents`](#extents) | `function` | Declared here |
| [`stride`](#stride-1) | `function` | Declared here |
| [`operator()`](#operator-25) | `function` | Declared here |
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
| `constexpr index_type` | [`operator()`](#operator-25) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr index_type` | [`required_span_size`](#required_span_size) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_unique`](#is_unique) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_exhaustive`](#is_exhaustive) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_strided`](#is_strided) `const` `inline` `constexpr` `noexcept` |  |

---

#### mapping

```cpp
mapping() = default
```

Defined in include/teeny/layout.h:153

Defaulted constructor.

---

#### mapping

`inline` `constexpr`

```cpp
template<size_t M = strides::ndyn(), enable_if_t< M==0, int > = 0> constexpr inline constexpr mapping(const Shape & e)
```

Defined in include/teeny/layout.h:157

Fully-static strides: construct from extents only.

---

#### mapping

`inline` `constexpr`

```cpp
template<class OtherIdx> constexpr inline constexpr mapping(const Shape & e, const array< OtherIdx, strides::ndyn()> & dyn)
```

Defined in include/teeny/layout.h:174

Mixed strides: extents + the runtime strides (dim order, dynamic ones only).

Templated on the array's element type so a `reindex` (narrowing the offset index width) can pass its wider source strides — each is cast to `index_type`; symmetric with mdspan's `layout_stride`.

---

#### extents

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const Shape & extents() const noexcept
```

Defined in include/teeny/layout.h:177

---

#### stride

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr index_type stride(rank_type r) const noexcept
```

Defined in include/teeny/layout.h:178

---

#### operator()

`const` `inline` `constexpr` `noexcept`

```cpp
template<class... I> constexpr inline constexpr index_type operator()(I... i) const noexcept
```

Defined in include/teeny/layout.h:183

---

#### required_span_size

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr index_type required_span_size() const noexcept
```

Defined in include/teeny/layout.h:194

---

#### is_unique

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_unique() const noexcept
```

Defined in include/teeny/layout.h:205

---

#### is_exhaustive

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_exhaustive() const noexcept
```

Defined in include/teeny/layout.h:206

---

#### is_strided

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_strided() const noexcept
```

Defined in include/teeny/layout.h:207

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

Defined in include/teeny/layout.h:202

---

#### is_always_exhaustive

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr bool is_always_exhaustive() noexcept
```

Defined in include/teeny/layout.h:203

---

#### is_always_strided

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr bool is_always_strided() noexcept
```

Defined in include/teeny/layout.h:204

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

Defined in include/teeny/layout.h:146

---

#### index_type

```cpp
using index_type = typename Shape::index_type
```

Defined in include/teeny/layout.h:147

---

#### rank_type

```cpp
using rank_type = typename Shape::rank_type
```

Defined in include/teeny/layout.h:148

---

#### layout_type

```cpp
using layout_type = strides
```

Defined in include/teeny/layout.h:149



## tensor

```cpp
#include <tensor.h>
```

```cpp
template<class T, class Shape, class Layout, storage O>
struct tensor
```

Defined in include/teeny/tensor.h:351

> **Inherits:** `template mapping< Shape >`

One N-dimensional tensor, parameterised by ownership.

The layout / extents / offset mapping is delegated to `cuda::std::mdspan` (the mapping lives in an empty base, so a fully-static tensor is exactly the size of its data). Ownership is a policy: `[storage::view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a1bda80f2be4d3658e0baa43fbe7ae8c1)` (non-owning, trivially copyable, kernel-passable), `[storage::stack](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508afac2a47adace059aff113283a03f6760)` (inline storage, static shape), `[storage::heap](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a4d4a9aa362b6ffe089fd2e992ccf4f5f)` (host-only, move-only), the CUDA owners `[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947)`/`pinned`/`mapped` (from `[cuda.h](#cudah)`), and the space-carrying views `[storage::gpu_view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a716b431c57855c3a30f4c286ad4f0299)`/`pinned_view`/ `mapped_view` (a view of device / page-locked memory keeps its space). The tensor's copy/move semantics are induced by the storage member, not hand-written.

#### Template Parameters
* `T` Element type. 

* `Shape` The shape: any `cuda::std::extents<Idx, E...>` (static or dynamic per dim). Spell it with the `shape<...>` alias. 

* `Layout` mdspan layout policy (default `ccontiguous`). 

* `O` Ownership kind (default `[storage::view](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a1bda80f2be4d3658e0baa43fbe7ae8c1)`).

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
| [`tensor`](#tensor-7) | `function` | Declared here |
| [`tensor`](#tensor-8) | `function` | Declared here |
| [`tensor`](#tensor-9) | `function` | Declared here |
| [`tensor`](#tensor-10) | `function` | Declared here |
| [`mapping`](#mapping-4) | `function` | Declared here |
| [`extents`](#extents-1) | `function` | Declared here |
| [`extent`](#extent) | `function` | Declared here |
| [`extent`](#extent-1) | `function` | Declared here |
| [`shape`](#shape-2) | `function` | Declared here |
| [`shape`](#shape-3) | `function` | Declared here |
| [`strides`](#strides-1) | `function` | Declared here |
| [`stride`](#stride-2) | `function` | Declared here |
| [`stride`](#stride-3) | `function` | Declared here |
| [`numel`](#numel) | `function` | Declared here |
| [`is_dense`](#is_dense) | `function` | Declared here |
| [`is_dense`](#is_dense-1) | `function` | Declared here |
| [`is_dense`](#is_dense-2) | `function` | Declared here |
| [`is_contiguous`](#is_contiguous) | `function` | Declared here |
| [`is_contiguous`](#is_contiguous-1) | `function` | Declared here |
| [`data`](#data-6) | `function` | Declared here |
| [`data`](#data-7) | `function` | Declared here |
| [`mdspan`](#mdspan) | `function` | Declared here |
| [`mdspan`](#mdspan-1) | `function` | Declared here |
| [`view`](#view-1) | `function` | Declared here |
| [`view`](#view-2) | `function` | Declared here |
| [`operator()`](#operator-26) | `function` | Declared here |
| [`operator()`](#operator-27) | `function` | Declared here |
| [`at`](#at) | `function` | Declared here |
| [`at`](#at-1) | `function` | Declared here |
| [`operator()`](#operator-28) | `function` | Declared here |
| [`operator()`](#operator-29) | `function` | Declared here |
| [`uget`](#uget) | `function` | Declared here |
| [`uget`](#uget-1) | `function` | Declared here |
| [`uget`](#uget-2) | `function` | Declared here |
| [`uget`](#uget-3) | `function` | Declared here |
| [`uget`](#uget-4) | `function` | Declared here |
| [`uget`](#uget-5) | `function` | Declared here |
| [`uat`](#uat) | `function` | Declared here |
| [`uat`](#uat-1) | `function` | Declared here |
| [`operator()`](#operator-30) | `function` | Declared here |
| [`operator()`](#operator-31) | `function` | Declared here |
| [`operator()`](#operator-32) | `function` | Declared here |
| [`operator()`](#operator-33) | `function` | Declared here |
| [`at`](#at-2) | `function` | Declared here |
| [`at`](#at-3) | `function` | Declared here |
| [`uget`](#uget-6) | `function` | Declared here |
| [`uget`](#uget-7) | `function` | Declared here |
| [`uat`](#uat-2) | `function` | Declared here |
| [`uat`](#uat-3) | `function` | Declared here |
| [`operator T`](#operatort) | `function` | Declared here |
| [`item`](#item-2) | `function` | Declared here |
| [`slice_along`](#slice_along) | `function` | Declared here |
| [`slice_along`](#slice_along-1) | `function` | Declared here |
| [`slice_along`](#slice_along-2) | `function` | Declared here |
| [`slice_along`](#slice_along-3) | `function` | Declared here |
| [`subsample`](#subsample) | `function` | Declared here |
| [`subsample`](#subsample-1) | `function` | Declared here |
| [`subsample`](#subsample-2) | `function` | Declared here |
| [`subsample`](#subsample-3) | `function` | Declared here |
| [`unfold`](#unfold) | `function` | Declared here |
| [`unfold`](#unfold-1) | `function` | Declared here |
| [`unfold`](#unfold-2) | `function` | Declared here |
| [`unfold`](#unfold-3) | `function` | Declared here |
| [`index_select`](#index_select) | `function` | Declared here |
| [`index_select`](#index_select-1) | `function` | Declared here |
| [`index_select`](#index_select-2) | `function` | Declared here |
| [`index_select`](#index_select-3) | `function` | Declared here |
| [`permute`](#permute) | `function` | Declared here |
| [`permute`](#permute-1) | `function` | Declared here |
| [`flip`](#flip) | `function` | Declared here |
| [`flip`](#flip-1) | `function` | Declared here |
| [`flip`](#flip-2) | `function` | Declared here |
| [`flip`](#flip-3) | `function` | Declared here |
| [`clone`](#clone) | `function` | Declared here |
| [`to`](#to-2) | `function` | Declared here |
| [`to`](#to-3) | `function` | Declared here |
| [`to`](#to-4) | `function` | Declared here |
| [`to`](#to-5) | `function` | Declared here |
| [`to`](#to-6) | `function` | Declared here |
| [`to`](#to-7) | `function` | Declared here |
| [`reshape`](#reshape) | `function` | Declared here |
| [`reshape`](#reshape-1) | `function` | Declared here |
| [`can_reshape_without_copy`](#can_reshape_without_copy) | `function` | Declared here |
| [`recast`](#recast) | `function` | Declared here |
| [`recast`](#recast-1) | `function` | Declared here |
| [`index_fits`](#index_fits-1) | `function` | Declared here |
| [`reindex`](#reindex-2) | `function` | Declared here |
| [`reindex`](#reindex-3) | `function` | Declared here |
| [`flatten`](#flatten) | `function` | Declared here |
| [`flatten`](#flatten-1) | `function` | Declared here |
| [`unsqueeze`](#unsqueeze) | `function` | Declared here |
| [`unsqueeze`](#unsqueeze-1) | `function` | Declared here |
| [`unsqueeze`](#unsqueeze-2) | `function` | Declared here |
| [`unsqueeze`](#unsqueeze-3) | `function` | Declared here |
| [`squeeze`](#squeeze) | `function` | Declared here |
| [`squeeze`](#squeeze-1) | `function` | Declared here |
| [`squeeze`](#squeeze-2) | `function` | Declared here |
| [`squeeze`](#squeeze-3) | `function` | Declared here |
| [`flip`](#flip-4) | `function` | Declared here |
| [`flip`](#flip-5) | `function` | Declared here |
| [`squeeze`](#squeeze-4) | `function` | Declared here |
| [`squeeze`](#squeeze-5) | `function` | Declared here |
| [`unsqueeze`](#unsqueeze-4) | `function` | Declared here |
| [`unsqueeze`](#unsqueeze-5) | `function` | Declared here |
| [`permute`](#permute-2) | `function` | Declared here |
| [`permute`](#permute-3) | `function` | Declared here |
| [`squeeze`](#squeeze-6) | `function` | Declared here |
| [`squeeze`](#squeeze-7) | `function` | Declared here |
| [`unsqueeze`](#unsqueeze-6) | `function` | Declared here |
| [`unsqueeze`](#unsqueeze-7) | `function` | Declared here |
| [`flip`](#flip-6) | `function` | Declared here |
| [`flip`](#flip-7) | `function` | Declared here |
| [`permute`](#permute-4) | `function` | Declared here |
| [`permute`](#permute-5) | `function` | Declared here |
| [`reshape`](#reshape-2) | `function` | Declared here |
| [`reshape`](#reshape-3) | `function` | Declared here |
| [`recast`](#recast-2) | `function` | Declared here |
| [`recast`](#recast-3) | `function` | Declared here |
| [`recast`](#recast-4) | `function` | Declared here |
| [`recast`](#recast-5) | `function` | Declared here |
| [`add_`](#add_) | `function` | Declared here |
| [`sub_`](#sub_) | `function` | Declared here |
| [`mul_`](#mul_) | `function` | Declared here |
| [`div_`](#div_) | `function` | Declared here |
| [`add_`](#add_-1) | `function` | Declared here |
| [`sub_`](#sub_-1) | `function` | Declared here |
| [`mul_`](#mul_-1) | `function` | Declared here |
| [`div_`](#div_-1) | `function` | Declared here |
| [`minimum_`](#minimum_) | `function` | Declared here |
| [`maximum_`](#maximum_) | `function` | Declared here |
| [`minimum_`](#minimum_-1) | `function` | Declared here |
| [`maximum_`](#maximum_-1) | `function` | Declared here |
| [`add_`](#add_-2) | `function` | Declared here |
| [`sub_`](#sub_-2) | `function` | Declared here |
| [`atomic_add_`](#atomic_add_) | `function` | Declared here |
| [`atomic_sub_`](#atomic_sub_) | `function` | Declared here |
| [`atomic_add_`](#atomic_add_-1) | `function` | Declared here |
| [`atomic_sub_`](#atomic_sub_-1) | `function` | Declared here |
| [`operator+=`](#operator-34) | `function` | Declared here |
| [`operator-=`](#operator-35) | `function` | Declared here |
| [`operator*=`](#operator-36) | `function` | Declared here |
| [`operator/=`](#operator-37) | `function` | Declared here |
| [`copy_`](#copy_) | `function` | Declared here |
| [`fill_`](#fill_) | `function` | Declared here |
| [`zero_`](#zero_) | `function` | Declared here |
| [`iota_`](#iota_) | `function` | Declared here |
| [`add`](#add) | `function` | Declared here |
| [`sub`](#sub) | `function` | Declared here |
| [`mul`](#mul) | `function` | Declared here |
| [`div`](#div) | `function` | Declared here |
| [`pow`](#pow) | `function` | Declared here |
| [`add`](#add-1) | `function` | Declared here |
| [`sub`](#sub-1) | `function` | Declared here |
| [`mul`](#mul-1) | `function` | Declared here |
| [`div`](#div-1) | `function` | Declared here |
| [`pow`](#pow-1) | `function` | Declared here |
| [`add`](#add-2) | `function` | Declared here |
| [`sub`](#sub-2) | `function` | Declared here |
| [`add`](#add-3) | `function` | Declared here |
| [`sub`](#sub-3) | `function` | Declared here |
| [`maximum`](#maximum-4) | `function` | Declared here |
| [`minimum`](#minimum-4) | `function` | Declared here |
| [`maximum`](#maximum-5) | `function` | Declared here |
| [`clamp`](#clamp-2) | `function` | Declared here |
| [`clamp`](#clamp-3) | `function` | Declared here |
| [`normalize`](#normalize-6) | `function` | Declared here |
| [`normalize`](#normalize-7) | `function` | Declared here |
| [`normalize`](#normalize-8) | `function` | Declared here |
| [`normalize`](#normalize-9) | `function` | Declared here |
| [`normalize`](#normalize-10) | `function` | Declared here |
| [`normalize`](#normalize-11) | `function` | Declared here |
| [`normalize`](#normalize-12) | `function` | Declared here |
| [`normalize`](#normalize-13) | `function` | Declared here |
| [`normalize`](#normalize-14) | `function` | Declared here |
| [`normalize`](#normalize-15) | `function` | Declared here |
| [`cross`](#cross-2) | `function` | Declared here |
| [`cross`](#cross-3) | `function` | Declared here |
| [`map_`](#map_) | `function` | Declared here |
| [`zip_with_`](#zip_with_) | `function` | Declared here |
| [`map`](#map) | `function` | Declared here |
| [`map`](#map-1) | `function` | Declared here |
| [`all`](#all-1) | `function` | Declared here |
| [`any`](#any) | `function` | Declared here |
| [`dot`](#dot-1) | `function` | Declared here |
| [`dot`](#dot-2) | `function` | Declared here |
| [`sqdist`](#sqdist-1) | `function` | Declared here |
| [`sqdist`](#sqdist-2) | `function` | Declared here |
| [`dist`](#dist-1) | `function` | Declared here |
| [`dist`](#dist-2) | `function` | Declared here |
| [`allclose`](#allclose-4) | `function` | Declared here |
| [`allclose`](#allclose-5) | `function` | Declared here |
| [`allclose`](#allclose-6) | `function` | Declared here |
| [`allclose`](#allclose-7) | `function` | Declared here |
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
| [`normalize_`](#normalize_) | `function` | Declared here |
| [`normalize_`](#normalize_-1) | `function` | Declared here |
| [`cross_`](#cross_) | `function` | Declared here |
| [`operator++`](#operator-38) | `function` | Declared here |
| [`operator--`](#operator-39) | `function` | Declared here |
| [`operator++`](#operator-40) | `function` | Declared here |
| [`operator--`](#operator-41) | `function` | Declared here |
| [`add_`](#add_-3) | `function` | Declared here |
| [`sub_`](#sub_-3) | `function` | Declared here |
| [`mul_`](#mul_-2) | `function` | Declared here |
| [`div_`](#div_-2) | `function` | Declared here |
| [`add_`](#add_-4) | `function` | Declared here |
| [`sub_`](#sub_-4) | `function` | Declared here |
| [`minimum_`](#minimum_-2) | `function` | Declared here |
| [`maximum_`](#maximum_-2) | `function` | Declared here |
| [`add_`](#add_-5) | `function` | Declared here |
| [`sub_`](#sub_-5) | `function` | Declared here |
| [`atomic_add_`](#atomic_add_-2) | `function` | Declared here |
| [`atomic_sub_`](#atomic_sub_-2) | `function` | Declared here |
| [`copy_`](#copy_-1) | `function` | Declared here |
| [`map_`](#map_-1) | `function` | Declared here |
| [`zip_with_`](#zip_with_-1) | `function` | Declared here |
| [`cross_`](#cross_-1) | `function` | Declared here |
| [`normalize_`](#normalize_-2) | `function` | Declared here |
| [`minimum`](#minimum-5) | `function` | Declared here |
| [`ownership`](#ownership) | `variable` | Declared here |
| [`is_static`](#is_static) | `variable` | Declared here |
| [`is_view`](#is_view) | `variable` | Declared here |
| [`is_owning`](#is_owning) | `variable` | Declared here |
| [`is_device`](#is_device-1) | `variable` | Declared here |
| [`is_host_accessible`](#is_host_accessible) | `variable` | Declared here |
| [`buffer_size`](#buffer_size) | `variable` | Declared here |
| [`is_strides_layout`](#is_strides_layout) | `variable` | Declared here |
| [`is_contiguous_layout`](#is_contiguous_layout) | `variable` | Declared here |
| [`rank`](#rank-3) | `function` | Declared here |
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
| `storage_policy< T, O, buffer_size >` | [`store_`](#store_)  |  |

---

#### store_

```cpp
storage_policy< T, O, buffer_size > store_ {}
```

Defined in include/teeny/tensor.h:371

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
|  | [`tensor`](#tensor-1)  | Defaulted constructor. |
|  | [`tensor`](#tensor-2)  | Defaulted constructor. |
|  | [`tensor`](#tensor-3)  | Defaulted constructor. |
|  | [`tensor`](#tensor-4) `inline` | View constructor: wrap `p` with the given mapping. |
|  | [`tensor`](#tensor-5) `inline` `explicit` | View constructor from a pointer alone — for a fully-static geometry (static extents AND a fully determined layout: contiguous, or an all-static `strides<...>`). |
|  | [`tensor`](#tensor-6) `inline` | View constructor from a pointer + extents (contiguous / static-stride layouts). |
|  | [`tensor`](#tensor-7) `inline` `explicit` | Owning constructor: allocate storage for `m` (heap/device/host/pinned). |
|  | [`tensor`](#tensor-8) `inline` `explicit` | Owning constructor from extents (contiguous / static-stride layouts). |
|  | [`tensor`](#tensor-9) `inline` `explicit` | UNINITIALISED constructors (numpy `np.empty`) used by `[empty()](#empty)`: the buffer is left indeterminate — fill before reading. |
|  | [`tensor`](#tensor-10) `inline` |  |
| `constexpr const mapping_type &` | [`mapping`](#mapping-4) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr const Shape &` | [`extents`](#extents-1) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr auto` | [`extent`](#extent) `const` `inline` `constexpr` `noexcept` | Extent of an axis given by a STATIC index (`extent(Int<0>())`): a compile-time `integral_constant` when that extent is static, else a runtime `index_type`. |
| `constexpr index_type` | [`extent`](#extent-1) `const` `inline` `constexpr` `noexcept` | Extent of an axis given by a RUNTIME index (`extent(0)`). |
| `constexpr auto` | [`shape`](#shape-2) `const` `inline` `constexpr` `noexcept` | `[shape()](#shape)` — the extents as an array-like accessor: `[shape()](#shape)[Int<k>()]` folds to a compile-time value where static, `[shape()](#shape)[i]` (runtime) is a value, and it converts to the raw `extents()` for interop. |
| `constexpr auto` | [`shape`](#shape-3) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr auto` | [`strides`](#strides-1) `const` `inline` `constexpr` `noexcept` | `strides()` — the strides as an array-like accessor (twin of `[shape()](#shape)`): `strides()[Int<k>()]` folds where the layout makes the stride derivable, `strides()[i]` (runtime) is a value. |
| `constexpr auto` | [`stride`](#stride-2) `const` `inline` `constexpr` `noexcept` | Stride of an axis given by a STATIC index (`stride(Int<0>())`): a compile-time `integral_constant` when known statically (static- stride layout; a contiguous layout over static extents; or the always-unit stride of a contiguous layout even for dynamic shapes). |
| `constexpr index_type` | [`stride`](#stride-3) `const` `inline` `constexpr` `noexcept` | Stride of an axis given by a RUNTIME index (`stride(0)`). |
| `constexpr auto` | [`numel`](#numel) `const` `inline` `constexpr` `noexcept` | Number of elements. |
| `constexpr bool` | [`is_dense`](#is_dense) `const` `inline` `constexpr` `noexcept` | Whether the elements occupy a **dense block of memory**, in *some* axis order — true for a C- or F-contiguous tensor, and also for a permuted one (a permuted C-contiguous view still packs the same memory densely). |
| `bool` | [`is_dense`](#is_dense-1) `const` `inline` `noexcept` | Exact denseness in layout `L` (e.g. |
| `bool` | [`is_dense`](#is_dense-2) `const` `inline` `noexcept` |  |
| `bool` | [`is_contiguous`](#is_contiguous) `const` `inline` `noexcept` | Whether the elements are **contiguous in a specific order** — **C-order by default** (numpy/pytorch's `is_contiguous`), or F-order via `is_contiguous<fcontiguous>()`. |
| `bool` | [`is_contiguous`](#is_contiguous-1) `const` `inline` `noexcept` |  |
| `T *` | [`data`](#data-6) `inline` `noexcept` |  |
| `const T *` | [`data`](#data-7) `const` `inline` `noexcept` |  |
| `view_type` | [`mdspan`](#mdspan) `inline` `noexcept` | The raw `cuda::std::mdspan` over this tensor's storage. |
| `const_view_type` | [`mdspan`](#mdspan-1) `const` `inline` `noexcept` |  |
| `auto` | [`view`](#view-1) `inline` `noexcept` | A non-owning teeny **view** of this tensor's storage — a `view` (or `gpu_view`, for a device tensor) that aliases the same memory (no copy), keeping the source layout. |
| `auto` | [`view`](#view-2) `const` `inline` `noexcept` |  |
| `T &` | [`operator()`](#operator-26) `inline` `noexcept` | Element access when every argument is an integer (negatives wrap). |
| `const T &` | [`operator()`](#operator-27) `const` `inline` `noexcept` |  |
| `auto` | [`at`](#at) `inline` `noexcept` | `at(i...)` — a single element as a **rank-0 VIEW** (all-integer args; negatives wrap). |
| `auto` | [`at`](#at-1) `const` `inline` `noexcept` |  |
| `auto` | [`operator()`](#operator-28) `inline` `noexcept` | Sub-view when any argument is a slice (`all`, `slice(a,b[,step])`) or a bare `none` (numpy `newaxis`). |
| `auto` | [`operator()`](#operator-29) `const` `inline` `noexcept` |  |
| `T &` | [`uget`](#uget) `inline` `noexcept` |  |
| `const T &` | [`uget`](#uget-1) `const` `inline` `noexcept` |  |
| `auto` | [`uget`](#uget-2) `inline` `noexcept` |  |
| `auto` | [`uget`](#uget-3) `const` `inline` `noexcept` |  |
| `decltype(auto)` | [`uget`](#uget-4) `inline` `noexcept` |  |
| `decltype(auto)` | [`uget`](#uget-5) `const` `inline` `noexcept` |  |
| `auto` | [`uat`](#uat) `inline` `noexcept` | Unchecked `at`: a single element as a rank-0 VIEW, no negative wrap. |
| `auto` | [`uat`](#uat-1) `const` `inline` `noexcept` |  |
| `decltype(auto)` | [`operator()`](#operator-30) `inline` `noexcept` | Ellipsis form: exactly one `ellipsis` in the args expands to `rank - (#other args)` copies of `all`, then the call re-runs — so `t(1, ellipsis, 2)` on rank 5 is `t(1, all, all, all, 2)`. |
| `decltype(auto)` | [`operator()`](#operator-31) `const` `inline` `noexcept` |  |
| `decltype(auto)` | [`operator()`](#operator-32) `inline` `noexcept` | **Tuple-unpack form** — `t(m)` where `m` is a single tuple-like index pack (a `cuda::std::array` or `cuda::std::tuple`) holding the WHOLE index list: exactly numpy's `x[(a, b, c)] == x[a, b, c]`. |
| `decltype(auto)` | [`operator()`](#operator-33) `const` `inline` `noexcept` |  |
| `decltype(auto)` | [`at`](#at-2) `inline` `noexcept` | Tuple-unpack `at`: `t.at(m)` == `t.at(m[0], m[1], ...)` — the element as a rank-0 VIEW. |
| `decltype(auto)` | [`at`](#at-3) `const` `inline` `noexcept` |  |
| `decltype(auto)` | [`uget`](#uget-6) `inline` `noexcept` | Tuple-unpack `uget` / `uat`: the unchecked twins (no negative-index wrap), same unpack, same result types as the checked forms. |
| `decltype(auto)` | [`uget`](#uget-7) `const` `inline` `noexcept` |  |
| `decltype(auto)` | [`uat`](#uat-2) `inline` `noexcept` |  |
| `decltype(auto)` | [`uat`](#uat-3) `const` `inline` `noexcept` |  |
|  | [`operator T`](#operatort) `const` `inline` `noexcept` |  |
| `T` | [`item`](#item-2) `const` `inline` `noexcept` | The single element of a rank-0 tensor (explicit reader). |
| `auto` | [`slice_along`](#slice_along) `inline` `noexcept` | Index/slice one or more named axes; other axes are kept. |
| `auto` | [`slice_along`](#slice_along-1) `const` `inline` `noexcept` |  |
| `auto` | [`slice_along`](#slice_along-2) `inline` `noexcept` | Value form: `t.slice_along(axis<0,2>{}, i, slice(1,4))` == `t.slice_along<0,2>(i, slice(1,4))`. |
| `auto` | [`slice_along`](#slice_along-3) `const` `inline` `noexcept` |  |
| `auto` | [`subsample`](#subsample) `inline` `noexcept` | Subsample a coloured/strided sub-lattice: bind named axes to a `slice(start,none,k)` each, sharing one STEP `k` across all of them but taking a separate START per axis — sugar for `slice_along` (#258), for the "every `k`-th voxel, offset per axis" pattern coloured Gauss-Seidel relaxation needs (`loc[d] % k == digit_d(n)`). |
| `auto` | [`subsample`](#subsample-1) `const` `inline` `noexcept` |  |
| `auto` | [`subsample`](#subsample-2) `inline` `noexcept` | Value form: `t.subsample(axis<0,1>{}, k, s0, s1)` == `t.subsample<0,1>(k, s0, s1)` — leading `axis<...>` selector, same placement as `slice_along`'s own value form (a second variadic pack, the starts, needs the disambiguating tag up front rather than trailing). |
| `auto` | [`subsample`](#subsample-3) `const` `inline` `noexcept` |  |
| `auto` | [`unfold`](#unfold) `inline` `noexcept` | Sliding/strided window along axis `Axis` (pytorch `Tensor.unfold`): appends a NEW trailing axis of width `size`, stepped by `step` along `Axis` -> a rank-(N+1) view. |
| `auto` | [`unfold`](#unfold-1) `const` `inline` `noexcept` |  |
| `auto` | [`unfold`](#unfold-2) `inline` `noexcept` | Value form: `t.unfold(Int<0>(), K, s)` == `t.unfold<0>(K, s)` — a single-axis selector (like `flip`/`squeeze`/`unsqueeze`'s own `Int<k>()` twin), so no `.template` is needed on a dependent receiver. |
| `auto` | [`unfold`](#unfold-3) `const` `inline` `noexcept` |  |
| `auto` | [`index_select`](#index_select) `const` `inline` |  |
| `auto` | [`index_select`](#index_select-1) `const` `inline` |  |
| `auto &` | [`index_select`](#index_select-2) `const` `inline` |  |
| `auto &` | [`index_select`](#index_select-3) `const` `inline` | `into(dest)` form: writes the gather straight into `dest` — one pass, no allocation, `_TNY_API` (device-safe). |
| `auto` | [`permute`](#permute) `inline` `noexcept` | Reorder the axes (a permutation of 0..N-1; negatives wrap) -> a rank-N view. |
| `auto` | [`permute`](#permute-1) `const` `inline` `noexcept` |  |
| `auto` | [`flip`](#flip) `inline` `noexcept` | Reverse axis `Ax` (negatives wrap) -> a view (numpy `flip`). |
| `auto` | [`flip`](#flip-1) `const` `inline` `noexcept` |  |
| `auto` | [`flip`](#flip-2) `inline` `noexcept` | Reverse SEVERAL axes at once (numpy `flip(a, axis=(...))`) -> a rank-N view. |
| `auto` | [`flip`](#flip-3) `const` `inline` `noexcept` |  |
| `auto` | [`clone`](#clone) `const` `inline` |  |
| `auto` | [`to`](#to-2) `const` `inline` `&` | pytorch-like `.to<T2>()`: convert the element type to `T2`. |
| `auto` | [`to`](#to-3) `const` `inline` `&` |  |
| `auto` | [`to`](#to-4) `const` `inline` `&&` |  |
| `auto` | [`to`](#to-5) `const` `inline` `&&` |  |
| `auto` | [`to`](#to-6) `const` `inline` `&` |  |
| `auto` | [`to`](#to-7) `const` `inline` `&&` |  |
| `auto` | [`reshape`](#reshape) `inline` `noexcept` | View this tensor as a new shape — numpy semantics: a **VIEW** whenever the layout can be regrouped without a copy (not only when C-contiguous; a strided/permuted source often still views — split a contiguous axis, merge a contiguous run). |
| `auto` | [`reshape`](#reshape-1) `const` `inline` `noexcept` |  |
| `bool` | [`can_reshape_without_copy`](#can_reshape_without_copy) `const` `inline` `noexcept` | Whether `reshape<NewExt...>()` can produce a VIEW (no copy) of this tensor's actual layout — numpy's rule: not just C-contiguity, but any stride-compatible regrouping (splitting an axis, merging a contiguous run). |
| `auto` | [`recast`](#recast) `inline` | Reinterpret with a MORE-STATIC extents type of the same rank — recover statically-known inner dims at the dynamic (ndarray) boundary: a runtime `(n,3,3)` view -> `.recast<shape<-1,3,3>>()` so the `3`s (extents) fold. |
| `auto` | [`recast`](#recast-1) `const` `inline` |  |
| `bool` | [`index_fits`](#index_fits-1) `const` `inline` `noexcept` | Does every element offset of this view fit the index type `Idx2`? Computes the SIGNED reach directly (teeny has negative-stride views, so `required_span_size`'s non-negative assumption doesn't apply): `max = Σ_{s>0}(e−1)·s`, `min = Σ_{s<0}(e−1)·s`; fits ⟺ `min..max` ⊆ `Idx2`. |
| `auto` | [`reindex`](#reindex-2) `inline` | No-copy, **layout-preserving** retype of the offset index width to `Idx2`: same pointer, same layout KIND, the extents' `index_type` and any dynamic strides narrowed to `Idx2` (a `strides<...>` literal pack is unchanged). |
| `auto` | [`reindex`](#reindex-3) `const` `inline` |  |
| `auto` | [`flatten`](#flatten) `inline` `noexcept` | View as 1-D (`ravel`) — a VIEW whenever the layout is mergeable into a single contiguous run without a copy (numpy semantics; `[clone()](#clone)` first otherwise). |
| `auto` | [`flatten`](#flatten-1) `const` `inline` `noexcept` |  |
| `auto` | [`unsqueeze`](#unsqueeze) `inline` `noexcept` | Insert a size-1 axis at position `Ax` (numpy `newaxis`/`unsqueeze`) -> a rank-(N+1) view. |
| `auto` | [`unsqueeze`](#unsqueeze-1) `const` `inline` `noexcept` |  |
| `auto` | [`unsqueeze`](#unsqueeze-2) `inline` `noexcept` | Insert size-1 axes at SEVERAL positions at once (numpy `expand_dims(a, axis=(...))`) -> a rank-(N+k) view. |
| `auto` | [`unsqueeze`](#unsqueeze-3) `const` `inline` `noexcept` |  |
| `auto` | [`squeeze`](#squeeze) `inline` `noexcept` | Drop a size-1 axis `Ax` (negatives wrap) -> a rank-(N-1) view. |
| `auto` | [`squeeze`](#squeeze-1) `const` `inline` `noexcept` |  |
| `auto` | [`squeeze`](#squeeze-2) `inline` `noexcept` | Drop SEVERAL size-1 axes at once (numpy `squeeze(axis=(...))`) -> a rank-(N-k) view. |
| `auto` | [`squeeze`](#squeeze-3) `const` `inline` `noexcept` |  |
| `auto` | [`flip`](#flip-4) `inline` `noexcept` |  |
| `auto` | [`flip`](#flip-5) `const` `inline` `noexcept` |  |
| `auto` | [`squeeze`](#squeeze-4) `inline` `noexcept` |  |
| `auto` | [`squeeze`](#squeeze-5) `const` `inline` `noexcept` |  |
| `auto` | [`unsqueeze`](#unsqueeze-4) `inline` `noexcept` |  |
| `auto` | [`unsqueeze`](#unsqueeze-5) `const` `inline` `noexcept` |  |
| `auto` | [`permute`](#permute-2) `inline` `noexcept` |  |
| `auto` | [`permute`](#permute-3) `const` `inline` `noexcept` |  |
| `auto` | [`squeeze`](#squeeze-6) `inline` `noexcept` | Value form: `t.squeeze(axis<0,2>{})` == `t.squeeze<0,2>()`, likewise `unsqueeze`/`flip`/`permute`. |
| `auto` | [`squeeze`](#squeeze-7) `const` `inline` `noexcept` |  |
| `auto` | [`unsqueeze`](#unsqueeze-6) `inline` `noexcept` |  |
| `auto` | [`unsqueeze`](#unsqueeze-7) `const` `inline` `noexcept` |  |
| `auto` | [`flip`](#flip-6) `inline` `noexcept` |  |
| `auto` | [`flip`](#flip-7) `const` `inline` `noexcept` |  |
| `auto` | [`permute`](#permute-4) `inline` `noexcept` |  |
| `auto` | [`permute`](#permute-5) `const` `inline` `noexcept` |  |
| `auto` | [`reshape`](#reshape-2) `inline` `noexcept` |  |
| `auto` | [`reshape`](#reshape-3) `const` `inline` `noexcept` |  |
| `auto` | [`recast`](#recast-2) `inline` |  |
| `auto` | [`recast`](#recast-3) `const` `inline` |  |
| `auto` | [`recast`](#recast-4) `inline` |  |
| `auto` | [`recast`](#recast-5) `const` `inline` |  |
| `tensor &` | [`add_`](#add_)  |  |
| `tensor &` | [`sub_`](#sub_)  |  |
| `tensor &` | [`mul_`](#mul_)  |  |
| `tensor &` | [`div_`](#div_)  |  |
| `tensor &` | [`add_`](#add_-1)  |  |
| `tensor &` | [`sub_`](#sub_-1)  |  |
| `tensor &` | [`mul_`](#mul_-1)  |  |
| `tensor &` | [`div_`](#div_-1)  |  |
| `tensor &` | [`minimum_`](#minimum_)  |  |
| `tensor &` | [`maximum_`](#maximum_)  |  |
| `tensor &` | [`minimum_`](#minimum_-1)  |  |
| `tensor &` | [`maximum_`](#maximum_-1)  |  |
| `tensor &` | [`add_`](#add_-2)  |  |
| `tensor &` | [`sub_`](#sub_-2)  |  |
| `tensor &` | [`atomic_add_`](#atomic_add_)  |  |
| `tensor &` | [`atomic_sub_`](#atomic_sub_)  |  |
| `tensor &` | [`atomic_add_`](#atomic_add_-1)  |  |
| `tensor &` | [`atomic_sub_`](#atomic_sub_-1)  |  |
| `tensor &` | [`operator+=`](#operator-34) `inline` |  |
| `tensor &` | [`operator-=`](#operator-35) `inline` |  |
| `tensor &` | [`operator*=`](#operator-36) `inline` |  |
| `tensor &` | [`operator/=`](#operator-37) `inline` |  |
| `tensor &` | [`copy_`](#copy_)  |  |
| `tensor &` | [`fill_`](#fill_)  |  |
| `tensor &` | [`zero_`](#zero_)  |  |
| `tensor &` | [`iota_`](#iota_)  |  |
| `auto` | [`add`](#add) `const` |  |
| `auto` | [`sub`](#sub) `const` |  |
| `auto` | [`mul`](#mul) `const` |  |
| `auto` | [`div`](#div) `const` |  |
| `auto` | [`pow`](#pow) `const` |  |
| `auto &` | [`add`](#add-1) `const` |  |
| `auto &` | [`sub`](#sub-1) `const` |  |
| `auto &` | [`mul`](#mul-1) `const` |  |
| `auto &` | [`div`](#div-1) `const` |  |
| `auto &` | [`pow`](#pow-1) `const` |  |
| `auto` | [`add`](#add-2) `const` |  |
| `auto` | [`sub`](#sub-2) `const` |  |
| `auto &` | [`add`](#add-3) `const` |  |
| `auto &` | [`sub`](#sub-3) `const` |  |
| `auto` | [`maximum`](#maximum-4) `const` |  |
| `auto &` | [`minimum`](#minimum-4) `const` |  |
| `auto &` | [`maximum`](#maximum-5) `const` |  |
| `auto` | [`clamp`](#clamp-2) `const` |  |
| `auto &` | [`clamp`](#clamp-3) `const` |  |
| `auto` | [`normalize`](#normalize-6) `const` |  |
| `auto &` | [`normalize`](#normalize-7) `const` |  |
| `auto` | [`normalize`](#normalize-8) `const` |  |
| `auto` | [`normalize`](#normalize-9) `const` |  |
| `auto` | [`normalize`](#normalize-10) `const` |  |
| `auto` | [`normalize`](#normalize-11) `const` |  |
| `auto &` | [`normalize`](#normalize-12) `const` |  |
| `auto &` | [`normalize`](#normalize-13) `const` |  |
| `auto &` | [`normalize`](#normalize-14) `const` |  |
| `auto &` | [`normalize`](#normalize-15) `const` |  |
| `auto` | [`cross`](#cross-2) `const` |  |
| `auto &` | [`cross`](#cross-3) `const` |  |
| `tensor &` | [`map_`](#map_)  |  |
| `tensor &` | [`zip_with_`](#zip_with_)  |  |
| `auto` | [`map`](#map) `const` |  |
| `auto &` | [`map`](#map-1) `const` |  |
| `bool` | [`all`](#all-1) `const` |  |
| `bool` | [`any`](#any) `const` |  |
| `class Eb class Lb storage Ob auto` | [`dot`](#dot-1) `const` |  |
| `decltype(auto)` | [`dot`](#dot-2) `const` |  |
| `auto` | [`sqdist`](#sqdist-1) `const` |  |
| `decltype(auto)` | [`sqdist`](#sqdist-2) `const` |  |
| `auto` | [`dist`](#dist-1) `const` |  |
| `decltype(auto)` | [`dist`](#dist-2) `const` |  |
| `bool` | [`allclose`](#allclose-4) `const` |  |
| `decltype(auto)` | [`allclose`](#allclose-5) `const` |  |
| `decltype(auto)` | [`allclose`](#allclose-6) `const` |  |
| `decltype(auto)` | [`allclose`](#allclose-7) `const` |  |
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
| `tensor &` | [`normalize_`](#normalize_)  |  |
| `tensor &` | [`normalize_`](#normalize_-1)  |  |
| `tensor &` | [`cross_`](#cross_)  |  |
| `tensor &` | [`operator++`](#operator-38) `inline` |  |
| `tensor &` | [`operator--`](#operator-39) `inline` |  |
| `tensor< T, Shape, ccontiguous, storage::stack >` | [`operator++`](#operator-40) `inline` |  |
| `tensor< T, Shape, ccontiguous, storage::stack >` | [`operator--`](#operator-41) `inline` |  |
| `tensor< T, E, L, O > &` | [`add_`](#add_-3)  |  |
| `tensor< T, E, L, O > &` | [`sub_`](#sub_-3)  |  |
| `tensor< T, E, L, O > &` | [`mul_`](#mul_-2)  |  |
| `tensor< T, E, L, O > &` | [`div_`](#div_-2)  |  |
| `tensor< T, E, L, O > &` | [`add_`](#add_-4)  |  |
| `tensor< T, E, L, O > &` | [`sub_`](#sub_-4)  |  |
| `tensor< T, E, L, O > &` | [`minimum_`](#minimum_-2)  |  |
| `tensor< T, E, L, O > &` | [`maximum_`](#maximum_-2)  |  |
| `tensor< T, E, L, O > &` | [`add_`](#add_-5)  |  |
| `tensor< T, E, L, O > &` | [`sub_`](#sub_-5)  |  |
| `tensor< T, E, L, O > &` | [`atomic_add_`](#atomic_add_-2)  |  |
| `tensor< T, E, L, O > &` | [`atomic_sub_`](#atomic_sub_-2)  |  |
| `tensor< T, E, L, O > &` | [`copy_`](#copy_-1)  |  |
| `tensor< T, E, L, O > &` | [`map_`](#map_-1)  |  |
| `tensor< T, E, L, O > &` | [`zip_with_`](#zip_with_-1)  |  |
| `tensor< T, E, L, O > &` | [`cross_`](#cross_-1)  |  |
| `tensor< T, E, L, O > &` | [`normalize_`](#normalize_-2)  |  |
| `u_abs u_log u_cos u_tanh u_ceil u_trunc auto` | [`minimum`](#minimum-5) `const` |  |

---

#### tensor

```cpp
tensor() = default
```

Defined in include/teeny/tensor.h:374

Defaulted constructor.

---

#### tensor

```cpp
tensor(const tensor &) = default
```

Defined in include/teeny/tensor.h:380

Defaulted constructor.

---

#### tensor

```cpp
tensor(tensor &&) = default
```

Defined in include/teeny/tensor.h:381

Defaulted constructor.

---

#### tensor

`inline`

```cpp
template<storage OO = O, enable_if_t< storage_is_view(OO), int > = 0> inline tensor(T * p, mapping_type m)
```

Defined in include/teeny/tensor.h:385

View constructor: wrap `p` with the given mapping.

---

#### tensor

`inline` `explicit`

```cpp
template<storage OO = O, enable_if_t< storage_is_view(OO) &&is_static &&(_contiguous_layout< Layout >::value||_strides_all_static< Layout >::value), int > = 0> inline explicit tensor(T * p)
```

Defined in include/teeny/tensor.h:392

View constructor from a pointer alone — for a fully-static geometry (static extents AND a fully determined layout: contiguous, or an all-static `strides<...>`).

e.g. `tensor<float, shape<3,4>, strides<4,1>>(ptr)`.

---

#### tensor

`inline`

```cpp
template<storage OO = O, enable_if_t< storage_is_view(OO) &&is_constructible< mapping_type, Shape >::value, int > = 0> inline tensor(T * p, Shape e)
```

Defined in include/teeny/tensor.h:396

View constructor from a pointer + extents (contiguous / static-stride layouts).

---

#### tensor

`inline` `explicit`

```cpp
template<storage OO = O, enable_if_t< storage_is_owning(OO), int > = 0> inline explicit tensor(mapping_type m)
```

Defined in include/teeny/tensor.h:408

Owning constructor: allocate storage for `m` (heap/device/host/pinned).

---

#### tensor

`inline` `explicit`

```cpp
template<storage OO = O, enable_if_t< storage_is_owning(OO) &&is_constructible< mapping_type, Shape >::value, int > = 0> inline explicit tensor(Shape e)
```

Defined in include/teeny/tensor.h:413

Owning constructor from extents (contiguous / static-stride layouts).

---

#### tensor

`inline` `explicit`

```cpp
template<storage OO = O, enable_if_t< OO==storage::stack, int > = 0> inline explicit tensor(_uninit_t)
```

Defined in include/teeny/tensor.h:420

UNINITIALISED constructors (numpy `np.empty`) used by `[empty()](#empty)`: the buffer is left indeterminate — fill before reading.

`local<...>{}` and `zeros(...)` keep their zero-fill; this is the opt-out.

---

#### tensor

`inline`

```cpp
template<storage OO = O, enable_if_t< storage_is_owning(OO) &&is_constructible< mapping_type, Shape >::value, int > = 0> inline tensor(Shape e, _uninit_t)
```

Defined in include/teeny/tensor.h:422

---

#### mapping

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const mapping_type & mapping() const noexcept
```

Defined in include/teeny/tensor.h:426

---

#### extents

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const Shape & extents() const noexcept
```

Defined in include/teeny/tensor.h:427

---

#### extent

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx, enable_if_t< _is_ic< Idx >::value, int > = 0> constexpr inline constexpr auto extent(Idx) const noexcept
```

Defined in include/teeny/tensor.h:435

Extent of an axis given by a STATIC index (`extent(Int<0>())`): a compile-time `integral_constant` when that extent is static, else a runtime `index_type`.

---

#### extent

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx, enable_if_t<!_is_ic< Idx >::value, int > = 0> constexpr inline constexpr index_type extent(Idx d) const noexcept
```

Defined in include/teeny/tensor.h:439

Extent of an axis given by a RUNTIME index (`extent(0)`).

---

#### shape

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr auto shape() const noexcept
```

Defined in include/teeny/tensor.h:446

`[shape()](#shape)` — the extents as an array-like accessor: `[shape()](#shape)[Int<k>()]` folds to a compile-time value where static, `[shape()](#shape)[i]` (runtime) is a value, and it converts to the raw `extents()` for interop.

`[shape(d)](#shape)` is the per-axis shorthand (== `extent(d)`).

---

#### shape

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx> constexpr inline constexpr auto shape(Idx d) const noexcept
```

Defined in include/teeny/tensor.h:447

---

#### strides

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr auto strides() const noexcept
```

Defined in include/teeny/tensor.h:451

`strides()` — the strides as an array-like accessor (twin of `[shape()](#shape)`): `strides()[Int<k>()]` folds where the layout makes the stride derivable, `strides()[i]` (runtime) is a value.

`strides(d)` == `stride(d)`.

---

#### stride

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx, enable_if_t< _is_ic< Idx >::value, int > = 0> constexpr inline constexpr auto stride(Idx) const noexcept
```

Defined in include/teeny/tensor.h:458

Stride of an axis given by a STATIC index (`stride(Int<0>())`): a compile-time `integral_constant` when known statically (static- stride layout; a contiguous layout over static extents; or the always-unit stride of a contiguous layout even for dynamic shapes).

---

#### stride

`const` `inline` `constexpr` `noexcept`

```cpp
template<class Idx, enable_if_t<!_is_ic< Idx >::value, int > = 0> constexpr inline constexpr index_type stride(Idx d) const noexcept
```

Defined in include/teeny/tensor.h:462

Stride of an axis given by a RUNTIME index (`stride(0)`).

---

#### numel

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr auto numel() const noexcept
```

Defined in include/teeny/tensor.h:480

Number of elements.

A **fully static** shape folds to an `integral_constant` (so it propagates into later compile-time arithmetic, like `extent(Int<k>())`); any dynamic dim -> a runtime `index_type`.

---

#### is_dense

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_dense() const noexcept
```

Defined in include/teeny/tensor.h:501

Whether the elements occupy a **dense block of memory**, in *some* axis order — true for a C- or F-contiguous tensor, and also for a permuted one (a permuted C-contiguous view still packs the same memory densely).

Formally: the strides are a permutation of a dense nested packing (`1, e0, e0·e1, ...`). Size-1 axes are ignored (their stride is unconstrained); an empty tensor is trivially dense. Negative strides (flips) are *not* dense in this sense -> false.

Pass a layout for an **exact** check: `is_dense<ccontiguous>()` / `is_dense<fcontiguous>()` test C-/F-contiguity specifically (or any layout whose mapping is derivable from the extents). For the C-order question specifically, `[is_contiguous()](#is_contiguous)` (below) reads clearer.

---

#### is_dense

`const` `inline` `noexcept`

```cpp
template<class L> inline bool is_dense() const noexcept
```

Defined in include/teeny/tensor.h:528

Exact denseness in layout `L` (e.g.

`ccontiguous`/`fcontiguous`): the actual strides equal what `L` produces for these extents. Two spellings — `t.is_dense<ccontiguous>()` (type form) and `t.is_dense(ccontiguous())` (value form, layout deduced from the argument).

---

#### is_dense

`const` `inline` `noexcept`

```cpp
template<class L> inline bool is_dense(L) const noexcept
```

Defined in include/teeny/tensor.h:539

---

#### is_contiguous

`const` `inline` `noexcept`

```cpp
template<class L = ccontiguous> inline bool is_contiguous() const noexcept
```

Defined in include/teeny/tensor.h:547

Whether the elements are **contiguous in a specific order** — **C-order by default** (numpy/pytorch's `is_contiguous`), or F-order via `is_contiguous<fcontiguous>()`.

A thin alias of `is_dense<Layout>()`; this (not `[is_dense()](#is_dense)`) is what `reshape`/`flatten` need. Value form: `is_contiguous(ccontiguous{})`.

---

#### is_contiguous

`const` `inline` `noexcept`

```cpp
template<class L> inline bool is_contiguous(L) const noexcept
```

Defined in include/teeny/tensor.h:549

---

#### data

`inline` `noexcept`

```cpp
inline T * data() noexcept
```

Defined in include/teeny/tensor.h:552

---

#### data

`const` `inline` `noexcept`

```cpp
inline const T * data() const noexcept
```

Defined in include/teeny/tensor.h:553

---

#### mdspan

`inline` `noexcept`

```cpp
inline view_type mdspan() noexcept
```

Defined in include/teeny/tensor.h:555

The raw `cuda::std::mdspan` over this tensor's storage.

---

#### mdspan

`const` `inline` `noexcept`

```cpp
inline const_view_type mdspan() const noexcept
```

Defined in include/teeny/tensor.h:556

---

#### view

`inline` `noexcept`

```cpp
inline auto view() noexcept
```

Defined in include/teeny/tensor.h:563

A non-owning teeny **view** of this tensor's storage — a `view` (or `gpu_view`, for a device tensor) that aliases the same memory (no copy), keeping the source layout.

On an already-non-owning tensor it re-wraps the same pointer (an equivalent view). For the raw mdspan, use `[mdspan()](#mdspan)`.

---

#### view

`const` `inline` `noexcept`

```cpp
inline auto view() const noexcept
```

Defined in include/teeny/tensor.h:564

---

#### operator()

`inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _all_index< Args... >::value, int > = 0> inline T & operator()(Args... a) noexcept
```

Defined in include/teeny/tensor.h:774

Element access when every argument is an integer (negatives wrap).

---

#### operator()

`const` `inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _all_index< Args... >::value, int > = 0> inline const T & operator()(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:777

---

#### at

`inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _all_index< Args... >::value, int > = 0> inline auto at(Args... a) noexcept
```

Defined in include/teeny/tensor.h:787

`at(i...)` — a single element as a **rank-0 VIEW** (all-integer args; negatives wrap).

Unlike `operator()`, which returns a plain `T&`, this is a view, so the whole tensor API applies to one element: `x.at(i,j) = 3` writes it, `float v = x.at(i,j)` reads it (rank-0 tensors convert to/from `T`), and `x.at(i,j).atomic_add_(v)` is an atomic scatter.

---

#### at

`const` `inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _all_index< Args... >::value, int > = 0> inline auto at(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:792

---

#### operator()

`inline` `noexcept`

```cpp
template<class... Args, enable_if_t<!_all_index< Args... >::value &&!_has_ellipsis< Args... >::value &&!_is_pack_call< Args... >::value, int > = 0> inline auto operator()(Args... a) noexcept
```

Defined in include/teeny/tensor.h:805

Sub-view when any argument is a slice (`all`, `slice(a,b[,step])`) or a bare `none` (numpy `newaxis`).

Integer args drop their axis, `all` keeps it, a range keeps a strided window, and a bare `none` inserts a size-1 axis (static extent 1, stride 0) at its position — all via the one gather (folds static strides into `strides<...>`; works on any source layout). `t(none,all,all)` == `unsqueeze<0>()`.

---

#### operator()

`const` `inline` `noexcept`

```cpp
template<class... Args, enable_if_t<!_all_index< Args... >::value &&!_has_ellipsis< Args... >::value &&!_is_pack_call< Args... >::value, int > = 0> inline auto operator()(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:809

---

#### uget

`inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _all_index< Args... >::value, int > = 0> inline T & uget(Args... a) noexcept
```

Defined in include/teeny/tensor.h:834

---

#### uget

`const` `inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _all_index< Args... >::value, int > = 0> inline const T & uget(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:837

---

#### uget

`inline` `noexcept`

```cpp
template<class... Args, enable_if_t<!_all_index< Args... >::value &&!_has_ellipsis< Args... >::value &&!_is_pack_call< Args... >::value, int > = 0> inline auto uget(Args... a) noexcept
```

Defined in include/teeny/tensor.h:843

---

#### uget

`const` `inline` `noexcept`

```cpp
template<class... Args, enable_if_t<!_all_index< Args... >::value &&!_has_ellipsis< Args... >::value &&!_is_pack_call< Args... >::value, int > = 0> inline auto uget(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:847

---

#### uget

`inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _has_ellipsis< Args... >::value, int > = 0> inline decltype(auto) uget(Args... a) noexcept
```

Defined in include/teeny/tensor.h:852

---

#### uget

`const` `inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _has_ellipsis< Args... >::value, int > = 0> inline decltype(auto) uget(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:855

---

#### uat

`inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _all_index< Args... >::value, int > = 0> inline auto uat(Args... a) noexcept
```

Defined in include/teeny/tensor.h:860

Unchecked `at`: a single element as a rank-0 VIEW, no negative wrap.

---

#### uat

`const` `inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _all_index< Args... >::value, int > = 0> inline auto uat(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:865

---

#### operator()

`inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _has_ellipsis< Args... >::value, int > = 0> inline decltype(auto) operator()(Args... a) noexcept
```

Defined in include/teeny/tensor.h:875

Ellipsis form: exactly one `ellipsis` in the args expands to `rank - (#other args)` copies of `all`, then the call re-runs — so `t(1, ellipsis, 2)` on rank 5 is `t(1, all, all, all, 2)`.

What remains decides the result (all integers -> element, else view).

---

#### operator()

`const` `inline` `noexcept`

```cpp
template<class... Args, enable_if_t< _has_ellipsis< Args... >::value, int > = 0> inline decltype(auto) operator()(Args... a) const noexcept
```

Defined in include/teeny/tensor.h:878

---

#### operator()

`inline` `noexcept`

```cpp
template<class P, enable_if_t< _is_index_pack< P >::value, int > = 0> inline decltype(auto) operator()(const P & p) noexcept
```

Defined in include/teeny/tensor.h:898

**Tuple-unpack form** — `t(m)` where `m` is a single tuple-like index pack (a `cuda::std::array` or `cuda::std::tuple`) holding the WHOLE index list: exactly numpy's `x[(a, b, c)] == x[a, b, c]`.

The pack is unpacked and re-dispatched through the ordinary variadic call, so everything that call does still applies: its elements may be integers/`Int<>`, `all`, `slice(...)`, a bare `none` (newaxis) or one `ellipsis`, and the result is an element (`T&`) or a view exactly as if they had been written out. Arity and validity are diagnosed by that call's own `static_assert`s.

This is pure packing sugar and is **single-argument only** — the pack IS the index list, and is never mixed with other positional arguments. It closes the loop with a peel range's `enumerate()` / `it.index()`, whose multi-index is a `cuda::std::array`: `for (auto [m, cell] : peel(t, axis<0,1>{}).enumerate()) out(m) = f(cell);`. Also available on `at`/`uget`/`uat` (and, on C++23, `t[m]`).

---

#### operator()

`const` `inline` `noexcept`

```cpp
template<class P, enable_if_t< _is_index_pack< P >::value, int > = 0> inline decltype(auto) operator()(const P & p) const noexcept
```

Defined in include/teeny/tensor.h:900

---

#### at

`inline` `noexcept`

```cpp
template<class P, enable_if_t< _is_index_pack< P >::value, int > = 0> inline decltype(auto) at(const P & p) noexcept
```

Defined in include/teeny/tensor.h:905

Tuple-unpack `at`: `t.at(m)` == `t.at(m[0], m[1], ...)` — the element as a rank-0 VIEW.

All-integer packs only (as with variadic `at`).

---

#### at

`const` `inline` `noexcept`

```cpp
template<class P, enable_if_t< _is_index_pack< P >::value, int > = 0> inline decltype(auto) at(const P & p) const noexcept
```

Defined in include/teeny/tensor.h:907

---

#### uget

`inline` `noexcept`

```cpp
template<class P, enable_if_t< _is_index_pack< P >::value, int > = 0> inline decltype(auto) uget(const P & p) noexcept
```

Defined in include/teeny/tensor.h:912

Tuple-unpack `uget` / `uat`: the unchecked twins (no negative-index wrap), same unpack, same result types as the checked forms.

---

#### uget

`const` `inline` `noexcept`

```cpp
template<class P, enable_if_t< _is_index_pack< P >::value, int > = 0> inline decltype(auto) uget(const P & p) const noexcept
```

Defined in include/teeny/tensor.h:914

---

#### uat

`inline` `noexcept`

```cpp
template<class P, enable_if_t< _is_index_pack< P >::value, int > = 0> inline decltype(auto) uat(const P & p) noexcept
```

Defined in include/teeny/tensor.h:916

---

#### uat

`const` `inline` `noexcept`

```cpp
template<class P, enable_if_t< _is_index_pack< P >::value, int > = 0> inline decltype(auto) uat(const P & p) const noexcept
```

Defined in include/teeny/tensor.h:918

---

#### operator T

`const` `inline` `noexcept`

```cpp
template<size_t R = rank(), enable_if_t< R==0, int > = 0> inline operator T() const noexcept
```

Defined in include/teeny/tensor.h:937

---

#### item

`const` `inline` `noexcept`

```cpp
template<size_t R = rank(), enable_if_t< R==0, int > = 0> inline T item() const noexcept
```

Defined in include/teeny/tensor.h:942

The single element of a rank-0 tensor (explicit reader).

---

#### slice_along

`inline` `noexcept`

```cpp
template<long... Axes, class... Args> inline auto slice_along(Args... args) noexcept
```

Defined in include/teeny/tensor.h:996

Index/slice one or more named axes; other axes are kept.

`slice_along<Axes...>(args...)` applies `args[k]` to axis `Axes[k]` (each an integer &ndash; negatives wrap &ndash; or a slice `all`/`rng`) and keeps every other axis, returning a view. e.g. `t.slice_along<1>(2)` drops axis 1 at index 2; `t.slice_along<0,2>(i, rng(1,4))` binds axes 0 and 2 at once.

NB this is NOT numpy's `take_along_axis` / pytorch's `take_along_dim` (a data-dependent gather driven by an index TENSOR &ndash; that is teeny's `index_select`). `slice_along` binds compile-time-named axes to a scalar index or a slice, so it is always an affine view: pytorch's `select`/`narrow` generalised to several axes at once (#423).

---

#### slice_along

`const` `inline` `noexcept`

```cpp
template<long... Axes, class... Args> inline auto slice_along(Args... args) const noexcept
```

Defined in include/teeny/tensor.h:1003

---

#### slice_along

`inline` `noexcept`

```cpp
template<long... Axes, class... Args> inline auto slice_along(axis< Axes... >, Args... args) noexcept
```

Defined in include/teeny/tensor.h:1014

Value form: `t.slice_along(axis<0,2>{}, i, slice(1,4))` == `t.slice_along<0,2>(i, slice(1,4))`.

The leading `axis<...>` selector is a single distinct-typed argument, so it needs no `.template` on a dependent receiver AND disambiguates cleanly from the template form.

---

#### slice_along

`const` `inline` `noexcept`

```cpp
template<long... Axes, class... Args> inline auto slice_along(axis< Axes... >, Args... args) const noexcept
```

Defined in include/teeny/tensor.h:1016

---

#### subsample

`inline` `noexcept`

```cpp
template<long... Axes, class K, class... Starts> inline auto subsample(K k, Starts... starts) noexcept
```

Defined in include/teeny/tensor.h:1033

Subsample a coloured/strided sub-lattice: bind named axes to a `slice(start,none,k)` each, sharing one STEP `k` across all of them but taking a separate START per axis — sugar for `slice_along` (#258), for the "every `k`-th voxel, offset per
       axis" pattern coloured Gauss-Seidel relaxation needs (`loc[d] % k == digit_d(n)`).

Pure sugar, no new addressing power: `t.subsample<0,1>(k, s0, s1)` == `t.slice_along<0,1>(slice(s0,none,k), slice(s1,none,k))`. `k` and each `start` accept a runtime value OR a compile-time one (`Int<k>()`) — folds through `[slice()](#slice-2)`'s own static-range machinery, so a fully-static `(start,k)` pair keeps a folded static output extent/stride, same as a hand-written `[slice()](#slice-2)`.

---

#### subsample

`const` `inline` `noexcept`

```cpp
template<long... Axes, class K, class... Starts> inline auto subsample(K k, Starts... starts) const noexcept
```

Defined in include/teeny/tensor.h:1038

---

#### subsample

`inline` `noexcept`

```cpp
template<long... Axes, class K, class... Starts> inline auto subsample(axis< Axes... >, K k, Starts... starts) noexcept
```

Defined in include/teeny/tensor.h:1048

Value form: `t.subsample(axis<0,1>{}, k, s0, s1)` == `t.subsample<0,1>(k, s0, s1)` — leading `axis<...>` selector, same placement as `slice_along`'s own value form (a second variadic pack, the starts, needs the disambiguating tag up front rather than trailing).

---

#### subsample

`const` `inline` `noexcept`

```cpp
template<long... Axes, class K, class... Starts> inline auto subsample(axis< Axes... >, K k, Starts... starts) const noexcept
```

Defined in include/teeny/tensor.h:1050

---

#### unfold

`inline` `noexcept`

```cpp
template<long Axis, class Sz, class St = integral_constant<long,1>> inline auto unfold(Sz size, St step = St{}) noexcept
```

Defined in include/teeny/tensor.h:1078

Sliding/strided window along axis `Axis` (pytorch `Tensor.unfold`): appends a NEW trailing axis of width `size`, stepped by `step` along `Axis` -> a rank-(N+1) view.

`Axis`'s own extent shrinks to the window COUNT `([shape(Axis)](#shape) - size) / step + 1`, e.g. `t.unfold<0>(K, s)` == pytorch's `t.unfold(0, K, s)`. `size`/`step` accept a runtime value OR a compile-time one (`Int<k>()`), folding the output extent/stride to static where derivable (like `[slice()](#slice-2)`). `size` must be in `[1, [shape(Axis)](#shape)]` and `step >= 1` — a `static_assert` when both are known at compile time, a debug-time check otherwise. ND windows compose by chaining: `t.unfold<0>(K0,s0).unfold<1>(K1,s1)` appends TWO window axes at the end (nitorch's nd-unfold pattern) — no separate nd-unfold primitive is needed.

---

#### unfold

`const` `inline` `noexcept`

```cpp
template<long Axis, class Sz, class St = integral_constant<long,1>> inline auto unfold(Sz size, St step = St{}) const noexcept
```

Defined in include/teeny/tensor.h:1090

---

#### unfold

`inline` `noexcept`

```cpp
template<class I, class Sz, class St = integral_constant<long,1>, enable_if_t< _is_ic< I >::value, int > = 0> inline auto unfold(I, Sz size, St step = St{}) noexcept
```

Defined in include/teeny/tensor.h:1106

Value form: `t.unfold(Int<0>(), K, s)` == `t.unfold<0>(K, s)` — a single-axis selector (like `flip`/`squeeze`/`unsqueeze`'s own `Int<k>()` twin), so no `.template` is needed on a dependent receiver.

---

#### unfold

`const` `inline` `noexcept`

```cpp
template<class I, class Sz, class St = integral_constant<long,1>, enable_if_t< _is_ic< I >::value, int > = 0> inline auto unfold(I, Sz size, St step = St{}) const noexcept
```

Defined in include/teeny/tensor.h:1108

---

#### index_select

`const` `inline`

```cpp
template<long Axis, class Ti, class Ei, class Li, storage Oi, enable_if_t< _md::index_select_extents< Shape, _norm_axis(Axis, rank()), _shape_static_extent< Ei >(0)>::rank_dynamic() !=0, int > = 0> inline auto index_select(const tensor< Ti, Ei, Li, Oi > & idx, const tensor< Ti, Ei, Li, Oi > & idx) const
```

Defined in include/teeny/tensor.h:1161

---

#### index_select

`const` `inline`

```cpp
template<class Ti, class Ei, class Li, storage Oi, long Axis, enable_if_t< _md::index_select_extents< Shape, _norm_axis(Axis, rank()), _shape_static_extent< Ei >(0)>::rank_dynamic() !=0, int > = 0> inline auto index_select(const tensor< Ti, Ei, Li, Oi > & idx, axis< Axis >, const tensor< Ti, Ei, Li, Oi > & idx, axis< Axis >) const
```

Defined in include/teeny/tensor.h:1190

---

#### index_select

`const` `inline`

```cpp
template<class Ti, class Ei, class Li, storage Oi, long Axis, class D> inline auto & index_select(const tensor< Ti, Ei, Li, Oi > & idx, axis< Axis >, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1192

---

#### index_select

`const` `inline`

```cpp
template<long Axis, class Ti, class Ei, class Li, storage Oi, class D> inline auto & index_select(const tensor< Ti, Ei, Li, Oi > & idx, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1203

`into(dest)` form: writes the gather straight into `dest` — one pass, no allocation, `_TNY_API` (device-safe).

Returns `dest&`. `dest`'s extents must match (axis `Axis` == `idx.numel()`, checked; every other axis == this tensor's own, checked by the underlying `copy_`). `dest` must not ALIAS this tensor's storage — an aliased in-place gather is unsupported (each `j` overwrites a slot of `dest` that a LATER `j` may still need to read from `*this`) and silently reorders instead of erroring.

---

#### permute

`inline` `noexcept`

```cpp
template<long... Perm> inline auto permute() noexcept
```

Defined in include/teeny/tensor.h:1227

Reorder the axes (a permutation of 0..N-1; negatives wrap) -> a rank-N view.

---

#### permute

`const` `inline` `noexcept`

```cpp
template<long... Perm> inline auto permute() const noexcept
```

Defined in include/teeny/tensor.h:1230

---

#### flip

`inline` `noexcept`

```cpp
template<long Ax = 0> inline auto flip() noexcept
```

Defined in include/teeny/tensor.h:1240

Reverse axis `Ax` (negatives wrap) -> a view (numpy `flip`).

Uses a negative stride, so the index type must be signed (`shape<...>` is).

An EMPTY tensor (`[numel()](#numel) == 0`) flips to a view over the *same* base pointer: there is no last element to move the origin to, so `[data()](#data-6)` is left exactly where it was.

---

#### flip

`const` `inline` `noexcept`

```cpp
template<long Ax = 0> inline auto flip() const noexcept
```

Defined in include/teeny/tensor.h:1243

---

#### flip

`inline` `noexcept`

```cpp
template<long Ax0, long Ax1, long... Rest> inline auto flip() noexcept
```

Defined in include/teeny/tensor.h:1260

Reverse SEVERAL axes at once (numpy `flip(a, axis=(...))`) -> a rank-N view.

The axes are relative to the source rank (negatives count from the back) and must be distinct, in ANY order — flipping axes commutes, so `t.flip<0,2>()`, `t.flip<2,0>()` and `t.flip<0>().flip<2>()` are the same view (same type, same elements). Each named axis gets its stride negated and the base pointer moved to its last element, all in ONE view — no chain of intermediates. Arity picks this overload; one axis (or none) still means `flip<Ax>()` above.

As for the single-axis form, an EMPTY tensor keeps its base pointer (`[data()](#data-6)` unchanged) — even when only *one* axis is empty and the others are flipped.

---

#### flip

`const` `inline` `noexcept`

```cpp
template<long Ax0, long Ax1, long... Rest> inline auto flip() const noexcept
```

Defined in include/teeny/tensor.h:1270

---

#### clone

`const` `inline`

```cpp
template<bool S = is_static, enable_if_t<!S, int > = 0> inline auto clone() const
```

Defined in include/teeny/tensor.h:1291

---

#### to

`const` `inline` `&`

```cpp
template<class T2 = element_type, bool Force = false, enable_if_t<!Force &&is_same< T2, element_type >::value, int > = 0> inline auto to() const &
```

Defined in include/teeny/tensor.h:1328

pytorch-like `.to<T2>()`: convert the element type to `T2`.

**No copy when it already matches** — if `T2` is the current element type and `Force` is false, this returns a (read-only) *view* of `*this`, no allocation, keeping the source layout. So `x.to<>()` is a zero-cost borrow, not a clone. Because it borrows, the result must not outlive the storage it points at — the same lifetime rule as `[view()](#view-1)`/`[permute()](#permute)`/slicing. Pass `Force = true` to always materialise a fresh owning copy even when the dtype already matches (`x.to<float, true>()` force-clones a `float` tensor); `x.clone()` is the unconditional-copy spelling.

**On a temporary** the borrow is only taken when it *cannot* dangle: a non-owning **view** rvalue (`view`/`gpu_view` — e.g. a slice or `.to<>()` result) points at storage owned elsewhere, so borrowing from it is safe and stays zero-cost. An **owning** rvalue (`stack`/`heap`/`gpu`/`pinned`/ `mapped`) would carry its storage off, so its matching-dtype case forces a fresh owning copy instead of a dangling borrow (mirroring the free `to<Space>(tensor&&)`).

When a conversion IS needed (`T2` differs, or `Force`), the result is a dense, row-major OWNING copy cast elementwise (via `copy_`): static shape -> stack (host+device), dynamic -> heap (host only). The copy runs on the HOST, so it cannot dereference DEVICE memory: the dynamic-shape (`_TNY_HOST`) copying overload `static_assert`s that the source is host-accessible. To also move across memory spaces (host <-> CUDA) — or to convert a `gpu`/`gpu_view` tensor at all — use the `to<[storage::gpu](#namespacetny_1a269cef88b5ff5cb896f1766df7f21508a0aa0be2a866411d9ff03515227454947), T2, Force>(x)` free functions from `<[teeny/cuda.h](#cudah)>`, which copy device-aware.

---

#### to

`const` `inline` `&`

```cpp
template<class T2 = element_type, bool Force = false, bool S = is_static, enable_if_t<(Force||!is_same< T2, element_type >::value) &&!S, int > = 0> inline auto to() const &
```

Defined in include/teeny/tensor.h:1336

---

#### to

`const` `inline` `&&`

```cpp
template<class T2 = element_type, bool Force = false, enable_if_t< storage_is_view(O) &&!Force &&is_same< T2, element_type >::value, int > = 0> inline auto to() const &&
```

Defined in include/teeny/tensor.h:1355

---

#### to

`const` `inline` `&&`

```cpp
template<class T2 = element_type, bool Force = false, bool S = is_static, enable_if_t<!(storage_is_view(O) &&!Force &&is_same< T2, element_type >::value) &&!S, int > = 0> inline auto to() const &&
```

Defined in include/teeny/tensor.h:1363

---

#### to

`const` `inline` `&`

```cpp
template<bool Force = false, class T2, bool S = is_static, enable_if_t<!(S||(!Force &&is_same< T2, element_type >::value)), int > = 0> inline auto to(dtype< T2 >, dtype< T2 >) const &
```

Defined in include/teeny/tensor.h:1384

---

#### to

`const` `inline` `&&`

```cpp
template<bool Force = false, class T2, bool S = is_static, enable_if_t<!(S||(storage_is_view(O) &&!Force &&is_same< T2, element_type >::value)), int > = 0> inline auto to(dtype< T2 >, dtype< T2 >) const &&
```

Defined in include/teeny/tensor.h:1390

---

#### reshape

`inline` `noexcept`

```cpp
template<long... NewExt> inline auto reshape() noexcept
```

Defined in include/teeny/tensor.h:1490

View this tensor as a new shape — numpy semantics: a **VIEW** whenever the layout can be regrouped without a copy (not only when C-contiguous; a strided/permuted source often still views — split a contiguous axis, merge a contiguous run).

The output is a folded `strides<...>` view (compile-time strides when the source is fully static). One extent may be **`-1`** (numpy-style), inferred from the total size: `t.reshape<6,-1>()`. A non-viewable reshape is a compile error (static source) or a debug check (dynamic) — `[clone()](#clone)` first, or query `can_reshape_without_copy`.

---

#### reshape

`const` `inline` `noexcept`

```cpp
template<long... NewExt> inline auto reshape() const noexcept
```

Defined in include/teeny/tensor.h:1491

---

#### can_reshape_without_copy

`const` `inline` `noexcept`

```cpp
template<long... NewExt> inline bool can_reshape_without_copy() const noexcept
```

Defined in include/teeny/tensor.h:1500

Whether `reshape<NewExt...>()` can produce a VIEW (no copy) of this tensor's actual layout — numpy's rule: not just C-contiguity, but any stride-compatible regrouping (splitting an axis, merging a contiguous run).

One `-1` may be inferred. `false` -> the reshape needs a `[clone()](#clone)`. (The result type of a viewable `reshape` is a folded `strides<...>` view.)

---

#### recast

`inline`

```cpp
template<class NewShape, class NewLayout = keep_strides> inline auto recast()
```

Defined in include/teeny/tensor.h:1572

Reinterpret with a MORE-STATIC extents type of the same rank — recover statically-known inner dims at the dynamic (ndarray) boundary: a runtime `(n,3,3)` view -> `.recast<shape<-1,3,3>>()` so the `3`s (extents) fold.

`NewLayout` chooses the STRIDES (default `[keep_strides](#keep_strides)`):

* **`[keep_strides](#keep_strides)`** (default) — PRESERVE the source strides; works on ANY layout (no copy, no contiguity requirement), a strided/transposed source keeps its strides, a `dynamic_strides` source keeps them at run time. Never mis-addresses.

* **`ccontiguous`/`fcontiguous`** — reinterpret AS that order, deriving the strides from the extents (folds the inner unit stride). The "I promise this is contiguous" form — UB if it isn't.

* **`strides<S...>`** — impose those (static) strides; a `dynamic_stride` slot comes from the source.

Each static dim of `NewShape` is validated against the actual extent. Functional form: `[recast(shape_value, layout_value)](#recast-4)` (both may mix static/dynamic; the runtime values only deduce the types).

---

#### recast

`const` `inline`

```cpp
template<class NewShape, class NewLayout = keep_strides> inline auto recast() const
```

Defined in include/teeny/tensor.h:1574

---

#### index_fits

`const` `inline` `noexcept`

```cpp
template<class Idx2> inline bool index_fits() const noexcept
```

Defined in include/teeny/tensor.h:1583

Does every element offset of this view fit the index type `Idx2`? Computes the SIGNED reach directly (teeny has negative-stride views, so `required_span_size`'s non-negative assumption doesn't apply): `max = Σ_{s>0}(e−1)·s`, `min = Σ_{s<0}(e−1)·s`; fits ⟺ `min..max` ⊆ `Idx2`.

Accumulates in a wide type; a broadcast (stride-0) axis adds 0. The precondition `reindex<Idx2>()` debug-checks.

---

#### reindex

`inline`

```cpp
template<class Idx2> inline auto reindex()
```

Defined in include/teeny/tensor.h:1604

No-copy, **layout-preserving** retype of the offset index width to `Idx2`: same pointer, same layout KIND, the extents' `index_type` and any dynamic strides narrowed to `Idx2` (a `strides<...>` literal pack is unchanged).

Narrowing the boundary view to `shape32` halves the by-value footprint and runs offset math in 32-bit (big device win). Orthogonal to `recast` (which staticizes the extent VALUES) — they compose. Debug-checks `index_fits<Idx2>()`; UB if the caller lies (same contract as `u*`).

---

#### reindex

`const` `inline`

```cpp
template<class Idx2> inline auto reindex() const
```

Defined in include/teeny/tensor.h:1609

---

#### flatten

`inline` `noexcept`

```cpp
inline auto flatten() noexcept
```

Defined in include/teeny/tensor.h:1618

View as 1-D (`ravel`) — a VIEW whenever the layout is mergeable into a single contiguous run without a copy (numpy semantics; `[clone()](#clone)` first otherwise).

Just `reshape<-1>()` (one inferred dim), spelled out for discoverability.

---

#### flatten

`const` `inline` `noexcept`

```cpp
inline auto flatten() const noexcept
```

Defined in include/teeny/tensor.h:1619

---

#### unsqueeze

`inline` `noexcept`

```cpp
template<long Ax = 0> inline auto unsqueeze() noexcept
```

Defined in include/teeny/tensor.h:1625

Insert a size-1 axis at position `Ax` (numpy `newaxis`/`unsqueeze`) -> a rank-(N+1) view.

Negative `Ax` counts from the back, so `.unsqueeze<-1>()` appends a trailing axis: `(H,W)` -> `(H,W,1)`.

---

#### unsqueeze

`const` `inline` `noexcept`

```cpp
template<long Ax = 0> inline auto unsqueeze() const noexcept
```

Defined in include/teeny/tensor.h:1628

---

#### unsqueeze

`inline` `noexcept`

```cpp
template<long Ax0, long Ax1, long... Rest> inline auto unsqueeze() noexcept
```

Defined in include/teeny/tensor.h:1639

Insert size-1 axes at SEVERAL positions at once (numpy `expand_dims(a, axis=(...))`) -> a rank-(N+k) view.

The positions are relative to the **final** rank `N + k` (negatives count from the back of it), and must be distinct (in ANY order — sorted internally, #275) — e.g. `(H,W).unsqueeze<1,3>()` -> `(H,1,W,1)`, `(H,W).unsqueeze<0,-1>()` -> `(1,H,W,1)`. Arity picks this overload; one axis still means `unsqueeze<Ax>()` above.

---

#### unsqueeze

`const` `inline` `noexcept`

```cpp
template<long Ax0, long Ax1, long... Rest> inline auto unsqueeze() const noexcept
```

Defined in include/teeny/tensor.h:1649

---

#### squeeze

`inline` `noexcept`

```cpp
template<long Ax = _ax_all> inline auto squeeze() noexcept
```

Defined in include/teeny/tensor.h:1681

Drop a size-1 axis `Ax` (negatives wrap) -> a rank-(N-1) view.

`[squeeze()](#squeeze)` (no axis) drops EVERY statically-size-1 axis.

---

#### squeeze

`const` `inline` `noexcept`

```cpp
template<long Ax = _ax_all> inline auto squeeze() const noexcept
```

Defined in include/teeny/tensor.h:1690

---

#### squeeze

`inline` `noexcept`

```cpp
template<long Ax0, long Ax1, long... Rest> inline auto squeeze() noexcept
```

Defined in include/teeny/tensor.h:1708

Drop SEVERAL size-1 axes at once (numpy `squeeze(axis=(...))`) -> a rank-(N-k) view.

The positions are relative to the **source** rank (negatives count from the back) and must be distinct (in ANY order — sorted internally, #275); every named axis must have extent 1 (`static_assert` where the extent is static, `_TNY_CHECK` where it is dynamic). e.g. a `(1,H,1,W)` view `.squeeze<0,2>()` -> `(H,W)`. Arity picks this overload; one axis (or none) still means `squeeze<Ax>()` above.

---

#### squeeze

`const` `inline` `noexcept`

```cpp
template<long Ax0, long Ax1, long... Rest> inline auto squeeze() const noexcept
```

Defined in include/teeny/tensor.h:1719

---

#### flip

`inline` `noexcept`

```cpp
template<class I, enable_if_t< _is_ic< I >::value, int > = 0> inline auto flip(I) noexcept
```

Defined in include/teeny/tensor.h:1734

---

#### flip

`const` `inline` `noexcept`

```cpp
template<class I, enable_if_t< _is_ic< I >::value, int > = 0> inline auto flip(I) const noexcept
```

Defined in include/teeny/tensor.h:1735

---

#### squeeze

`inline` `noexcept`

```cpp
template<class I, enable_if_t< _is_ic< I >::value, int > = 0> inline auto squeeze(I) noexcept
```

Defined in include/teeny/tensor.h:1736

---

#### squeeze

`const` `inline` `noexcept`

```cpp
template<class I, enable_if_t< _is_ic< I >::value, int > = 0> inline auto squeeze(I) const noexcept
```

Defined in include/teeny/tensor.h:1737

---

#### unsqueeze

`inline` `noexcept`

```cpp
template<class I, enable_if_t< _is_ic< I >::value, int > = 0> inline auto unsqueeze(I) noexcept
```

Defined in include/teeny/tensor.h:1738

---

#### unsqueeze

`const` `inline` `noexcept`

```cpp
template<class I, enable_if_t< _is_ic< I >::value, int > = 0> inline auto unsqueeze(I) const noexcept
```

Defined in include/teeny/tensor.h:1739

---

#### permute

`inline` `noexcept`

```cpp
template<class... I, enable_if_t<(sizeof...(I) > 0) &&_all_ic< I... >::value, int > = 0> inline auto permute(I...) noexcept
```

Defined in include/teeny/tensor.h:1740

---

#### permute

`const` `inline` `noexcept`

```cpp
template<class... I, enable_if_t<(sizeof...(I) > 0) &&_all_ic< I... >::value, int > = 0> inline auto permute(I...) const noexcept
```

Defined in include/teeny/tensor.h:1741

---

#### squeeze

`inline` `noexcept`

```cpp
template<long... Axes> inline auto squeeze(axis< Axes... >) noexcept
```

Defined in include/teeny/tensor.h:1765

Value form: `t.squeeze(axis<0,2>{})` == `t.squeeze<0,2>()`, likewise `unsqueeze`/`flip`/`permute`.

`squeeze`/`unsqueeze`/`flip`/`permute` are axis-LIST ops (like `peel`/`slice_along`/the reductions), so — unlike the single-axis `Int<k>()` form above — they take the `axis<...>` tag: a single distinct-typed argument, so no `.template` is needed on a dependent receiver.

An EMPTY list — `axis<>{}` — names no axis, so it is a **no-op**: the same shape and strides back, as a view (numpy's own rule for an empty axis tuple, `np.squeeze(a, axis=())` / `np.expand_dims(a, axis=())`; same identity `_keepdims<>`/`slice_along(axis<>{})`/`peel(t, axis<>{})` already have). It is NOT the same as the no-argument `[squeeze()](#squeeze)` (drop EVERY statically-size-1 axis) or `[unsqueeze()](#unsqueeze)` (insert at axis 0) — those keep their meanings; only the axis-LIST spelling reads an empty list as "no axes named" (#369). Generic code that computes an axis list therefore stays correct when the list comes out empty.

`permute` is the exception, and needs nothing added: it takes a FULL permutation, so its own `sizeof...(Perm) == [rank()](#rank-3)` check already accepts `axis<>{}` for a rank-0 tensor only (a no-op there — the one permutation of no axes) and rejects it at compile time for any other rank, rather than silently doing something else.

---

#### squeeze

`const` `inline` `noexcept`

```cpp
template<long... Axes> inline auto squeeze(axis< Axes... >) const noexcept
```

Defined in include/teeny/tensor.h:1767

---

#### unsqueeze

`inline` `noexcept`

```cpp
template<long... Axes> inline auto unsqueeze(axis< Axes... >) noexcept
```

Defined in include/teeny/tensor.h:1769

---

#### unsqueeze

`const` `inline` `noexcept`

```cpp
template<long... Axes> inline auto unsqueeze(axis< Axes... >) const noexcept
```

Defined in include/teeny/tensor.h:1771

---

#### flip

`inline` `noexcept`

```cpp
template<long... Axes> inline auto flip(axis< Axes... >) noexcept
```

Defined in include/teeny/tensor.h:1773

---

#### flip

`const` `inline` `noexcept`

```cpp
template<long... Axes> inline auto flip(axis< Axes... >) const noexcept
```

Defined in include/teeny/tensor.h:1775

---

#### permute

`inline` `noexcept`

```cpp
template<long... Axes> inline auto permute(axis< Axes... >) noexcept
```

Defined in include/teeny/tensor.h:1777

---

#### permute

`const` `inline` `noexcept`

```cpp
template<long... Axes> inline auto permute(axis< Axes... >) const noexcept
```

Defined in include/teeny/tensor.h:1778

---

#### reshape

`inline` `noexcept`

```cpp
template<class... I, enable_if_t<(sizeof...(I) > 0) &&_all_ic< I... >::value, int > = 0> inline auto reshape(I...) noexcept
```

Defined in include/teeny/tensor.h:1779

---

#### reshape

`const` `inline` `noexcept`

```cpp
template<class... I, enable_if_t<(sizeof...(I) > 0) &&_all_ic< I... >::value, int > = 0> inline auto reshape(I...) const noexcept
```

Defined in include/teeny/tensor.h:1780

---

#### recast

`inline`

```cpp
template<class NewE> inline auto recast(NewE)
```

Defined in include/teeny/tensor.h:1781

---

#### recast

`const` `inline`

```cpp
template<class NewE> inline auto recast(NewE) const
```

Defined in include/teeny/tensor.h:1782

---

#### recast

`inline`

```cpp
template<class NewE, class NewL> inline auto recast(NewE, NewL)
```

Defined in include/teeny/tensor.h:1787

---

#### recast

`const` `inline`

```cpp
template<class NewE, class NewL> inline auto recast(NewE, NewL) const
```

Defined in include/teeny/tensor.h:1788

---

#### add_

```cpp
template<bool Atomic = false, class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & add_(const B & b)
```

Defined in include/teeny/tensor.h:1795

---

#### sub_

```cpp
template<bool Atomic = false, class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & sub_(const B & b)
```

Defined in include/teeny/tensor.h:1796

---

#### mul_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & mul_(const B & b)
```

Defined in include/teeny/tensor.h:1797

---

#### div_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & div_(const B & b)
```

Defined in include/teeny/tensor.h:1798

---

#### add_

```cpp
template<bool Atomic = false> tensor & add_(T s)
```

Defined in include/teeny/tensor.h:1799

---

#### sub_

```cpp
template<bool Atomic = false> tensor & sub_(T s)
```

Defined in include/teeny/tensor.h:1800

---

#### mul_

```cpp
tensor & mul_(T s)
```

Defined in include/teeny/tensor.h:1801

---

#### div_

```cpp
tensor & div_(T s)
```

Defined in include/teeny/tensor.h:1802

---

#### minimum_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & minimum_(const B & b)
```

Defined in include/teeny/tensor.h:1808

---

#### maximum_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & maximum_(const B & b)
```

Defined in include/teeny/tensor.h:1809

---

#### minimum_

```cpp
tensor & minimum_(T s)
```

Defined in include/teeny/tensor.h:1810

---

#### maximum_

```cpp
tensor & maximum_(T s)
```

Defined in include/teeny/tensor.h:1811

---

#### add_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & add_(const B & b, T alpha)
```

Defined in include/teeny/tensor.h:1816

---

#### sub_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & sub_(const B & b, T alpha)
```

Defined in include/teeny/tensor.h:1817

---

#### atomic_add_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & atomic_add_(const B & b)
```

Defined in include/teeny/tensor.h:1826

---

#### atomic_sub_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> tensor & atomic_sub_(const B & b)
```

Defined in include/teeny/tensor.h:1827

---

#### atomic_add_

```cpp
tensor & atomic_add_(T s)
```

Defined in include/teeny/tensor.h:1828

---

#### atomic_sub_

```cpp
tensor & atomic_sub_(T s)
```

Defined in include/teeny/tensor.h:1829

---

#### operator+=

`inline`

```cpp
template<class B> inline tensor & operator+=(const B & b)
```

Defined in include/teeny/tensor.h:1833

---

#### operator-=

`inline`

```cpp
template<class B> inline tensor & operator-=(const B & b)
```

Defined in include/teeny/tensor.h:1834

---

#### operator*=

`inline`

```cpp
template<class B> inline tensor & operator*=(const B & b)
```

Defined in include/teeny/tensor.h:1835

---

#### operator/=

`inline`

```cpp
template<class B> inline tensor & operator/=(const B & b)
```

Defined in include/teeny/tensor.h:1836

---

#### copy_

```cpp
template<class B> tensor & copy_(const B & b)
```

Defined in include/teeny/tensor.h:1839

---

#### fill_

```cpp
tensor & fill_(T s)
```

Defined in include/teeny/tensor.h:1840

---

#### zero_

```cpp
tensor & zero_()
```

Defined in include/teeny/tensor.h:1841

---

#### iota_

```cpp
tensor & iota_(T start = T(0), T step = T(1))
```

Defined in include/teeny/tensor.h:1842

---

#### add

`const`

```cpp
template<class B> auto add(const B & b) const
```

Defined in include/teeny/tensor.h:1845

---

#### sub

`const`

```cpp
template<class B> auto sub(const B & b) const
```

Defined in include/teeny/tensor.h:1846

---

#### mul

`const`

```cpp
template<class B> auto mul(const B & b) const
```

Defined in include/teeny/tensor.h:1847

---

#### div

`const`

```cpp
template<class B> auto div(const B & b) const
```

Defined in include/teeny/tensor.h:1848

---

#### pow

`const`

```cpp
template<class B> auto pow(const B & b) const
```

Defined in include/teeny/tensor.h:1849

---

#### add

`const`

```cpp
template<class B, class D> auto & add(const B & b, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1851

---

#### sub

`const`

```cpp
template<class B, class D> auto & sub(const B & b, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1852

---

#### mul

`const`

```cpp
template<class B, class D> auto & mul(const B & b, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1853

---

#### div

`const`

```cpp
template<class B, class D> auto & div(const B & b, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1854

---

#### pow

`const`

```cpp
template<class B, class D> auto & pow(const B & b, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1855

---

#### add

`const`

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> auto add(const B & b, T alpha) const
```

Defined in include/teeny/tensor.h:1859

---

#### sub

`const`

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int > = 0> auto sub(const B & b, T alpha) const
```

Defined in include/teeny/tensor.h:1860

---

#### add

`const`

```cpp
template<class B, class D, enable_if_t<!is_arithmetic< B >::value, int > = 0> auto & add(const B & b, T alpha, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1861

---

#### sub

`const`

```cpp
template<class B, class D, enable_if_t<!is_arithmetic< B >::value, int > = 0> auto & sub(const B & b, T alpha, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1862

---

#### maximum

`const`

```cpp
template<class B> auto maximum(const B & b) const
```

Defined in include/teeny/tensor.h:1879

---

#### minimum

`const`

```cpp
template<class B, class D> auto & minimum(const B & b, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1880

---

#### maximum

`const`

```cpp
template<class B, class D> auto & maximum(const B & b, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1881

---

#### clamp

`const`

```cpp
auto clamp(T lo, T hi) const
```

Defined in include/teeny/tensor.h:1882

---

#### clamp

`const`

```cpp
template<class D> auto & clamp(T lo, T hi, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1883

---

#### normalize

`const`

```cpp
auto normalize() const
```

Defined in include/teeny/tensor.h:1884

---

#### normalize

`const`

```cpp
template<class D> auto & normalize(into_t< D > out) const
```

Defined in include/teeny/tensor.h:1885

---

#### normalize

`const`

```cpp
template<long... Axes, enable_if_t<(sizeof...(Axes) > 0) &&_md::_nrm_out_api< Shape, Axes... >::value, int > = 0> auto normalize() const
```

Defined in include/teeny/tensor.h:1893

---

#### normalize

`const`

```cpp
template<long... Axes, enable_if_t<(sizeof...(Axes) > 0) &&_md::_nrm_out_host< Shape, Axes... >::value, int > = 0> auto normalize() const
```

Defined in include/teeny/tensor.h:1895

---

#### normalize

`const`

```cpp
template<long... Axes, enable_if_t<(sizeof...(Axes) > 0) &&_md::_nrm_out_api< Shape, Axes... >::value, int > = 0> auto normalize(axis< Axes... >) const
```

Defined in include/teeny/tensor.h:1897

---

#### normalize

`const`

```cpp
template<long... Axes, enable_if_t<(sizeof...(Axes) > 0) &&_md::_nrm_out_host< Shape, Axes... >::value, int > = 0> auto normalize(axis< Axes... >) const
```

Defined in include/teeny/tensor.h:1899

---

#### normalize

`const`

```cpp
template<long... Axes, class D, enable_if_t<(sizeof...(Axes) > 0) &&_md::_nrm_kept_api< Shape, Axes... >::value, int > = 0> auto & normalize(into_t< D > out) const
```

Defined in include/teeny/tensor.h:1901

---

#### normalize

`const`

```cpp
template<long... Axes, class D, enable_if_t<(sizeof...(Axes) > 0) &&_md::_nrm_kept_host< Shape, Axes... >::value, int > = 0> auto & normalize(into_t< D > out) const
```

Defined in include/teeny/tensor.h:1903

---

#### normalize

`const`

```cpp
template<long... Axes, class D, enable_if_t<(sizeof...(Axes) > 0) &&_md::_nrm_kept_api< Shape, Axes... >::value, int > = 0> auto & normalize(axis< Axes... >, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1905

---

#### normalize

`const`

```cpp
template<long... Axes, class D, enable_if_t<(sizeof...(Axes) > 0) &&_md::_nrm_kept_host< Shape, Axes... >::value, int > = 0> auto & normalize(axis< Axes... >, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1907

---

#### cross

`const`

```cpp
template<class Tb, class Eb, class Lb, storage Ob> auto cross(const tensor< Tb, Eb, Lb, Ob > & b) const
```

Defined in include/teeny/tensor.h:1908

---

#### cross

`const`

```cpp
template<class Tb, class Eb, class Lb, storage Ob, class D> auto & cross(const tensor< Tb, Eb, Lb, Ob > & b, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1909

---

#### map_

```cpp
template<class F> tensor & map_(F f)
```

Defined in include/teeny/tensor.h:1915

---

#### zip_with_

```cpp
template<class G, class B> tensor & zip_with_(G g, const B & b)
```

Defined in include/teeny/tensor.h:1916

---

#### map

`const`

```cpp
template<class F> auto map(F f) const
```

Defined in include/teeny/tensor.h:1917

---

#### map

`const`

```cpp
template<class F, class D> auto & map(F f, into_t< D > out) const
```

Defined in include/teeny/tensor.h:1918

---

#### all

`const`

```cpp
bool all() const
```

Defined in include/teeny/tensor.h:1922

---

#### any

`const`

```cpp
bool any() const
```

Defined in include/teeny/tensor.h:1923

---

#### dot

`const`

```cpp
class Eb class Lb storage Ob auto dot(const tensor< Tb, Eb, Lb, Ob > & b) const
```

Defined in include/teeny/tensor.h:1970

---

#### dot

`const`

```cpp
template<class Acc = void, class Tb, class Eb, class Lb, storage Ob, class Tag0, class... Tags> decltype(auto) dot(const tensor< Tb, Eb, Lb, Ob > & b, Tag0 tag0, Tags... tags) const
```

Defined in include/teeny/tensor.h:1972

---

#### sqdist

`const`

```cpp
template<class Acc = void, class Tb, class Eb, class Lb, storage Ob> auto sqdist(const tensor< Tb, Eb, Lb, Ob > & b) const
```

Defined in include/teeny/tensor.h:1975

---

#### sqdist

`const`

```cpp
template<class Acc = void, class Tb, class Eb, class Lb, storage Ob, class Tag0, class... Tags> decltype(auto) sqdist(const tensor< Tb, Eb, Lb, Ob > & b, Tag0 tag0, Tags... tags) const
```

Defined in include/teeny/tensor.h:1977

---

#### dist

`const`

```cpp
template<class Acc = void, class Tb, class Eb, class Lb, storage Ob> auto dist(const tensor< Tb, Eb, Lb, Ob > & b) const
```

Defined in include/teeny/tensor.h:1979

---

#### dist

`const`

```cpp
template<class Acc = void, class Tb, class Eb, class Lb, storage Ob, class Tag0, class... Tags> decltype(auto) dist(const tensor< Tb, Eb, Lb, Ob > & b, Tag0 tag0, Tags... tags) const
```

Defined in include/teeny/tensor.h:1981

---

#### allclose

`const`

```cpp
template<class Acc = void, class Tb, class Eb, class Lb, storage Ob> bool allclose(const tensor< Tb, Eb, Lb, Ob > & b, double rtol = _allclose_rtol(), double atol = _allclose_atol()) const
```

Defined in include/teeny/tensor.h:1989

---

#### allclose

`const`

```cpp
template<class Acc = void, class Tb, class Eb, class Lb, storage Ob, class Tag0, class... Tags, enable_if_t< _kw::is_keyword< Tag0 >::value, int > = 0> decltype(auto) allclose(const tensor< Tb, Eb, Lb, Ob > & b, Tag0 tag0, Tags... tags) const
```

Defined in include/teeny/tensor.h:1993

---

#### allclose

`const`

```cpp
template<class Acc = void, class Tb, class Eb, class Lb, storage Ob, class Tag0, class... Tags, enable_if_t< _kw::is_keyword< Tag0 >::value, int > = 0> decltype(auto) allclose(const tensor< Tb, Eb, Lb, Ob > & b, double rtol, Tag0 tag0, Tags... tags) const
```

Defined in include/teeny/tensor.h:1996

---

#### allclose

`const`

```cpp
template<class Acc = void, class Tb, class Eb, class Lb, storage Ob, class Tag0, class... Tags, enable_if_t< _kw::is_keyword< Tag0 >::value, int > = 0> decltype(auto) allclose(const tensor< Tb, Eb, Lb, Ob > & b, double rtol, double atol, Tag0 tag0, Tags... tags) const
```

Defined in include/teeny/tensor.h:2000

---

#### neg_

```cpp
tensor & neg_()
```

Defined in include/teeny/tensor.h:2004

---

#### abs_

```cpp
tensor & abs_()
```

Defined in include/teeny/tensor.h:2005

---

#### exp_

```cpp
tensor & exp_()
```

Defined in include/teeny/tensor.h:2006

---

#### log_

```cpp
tensor & log_()
```

Defined in include/teeny/tensor.h:2007

---

#### sin_

```cpp
tensor & sin_()
```

Defined in include/teeny/tensor.h:2008

---

#### cos_

```cpp
tensor & cos_()
```

Defined in include/teeny/tensor.h:2009

---

#### sqrt_

```cpp
tensor & sqrt_()
```

Defined in include/teeny/tensor.h:2010

---

#### tanh_

```cpp
tensor & tanh_()
```

Defined in include/teeny/tensor.h:2011

---

#### floor_

```cpp
tensor & floor_()
```

Defined in include/teeny/tensor.h:2012

---

#### ceil_

```cpp
tensor & ceil_()
```

Defined in include/teeny/tensor.h:2013

---

#### round_

```cpp
tensor & round_()
```

Defined in include/teeny/tensor.h:2014

---

#### trunc_

```cpp
tensor & trunc_()
```

Defined in include/teeny/tensor.h:2015

---

#### sign_

```cpp
tensor & sign_()
```

Defined in include/teeny/tensor.h:2016

---

#### pow_

```cpp
tensor & pow_(T e)
```

Defined in include/teeny/tensor.h:2017

---

#### clamp_

```cpp
tensor & clamp_(T lo, T hi)
```

Defined in include/teeny/tensor.h:2018

---

#### normalize_

```cpp
tensor & normalize_()
```

Defined in include/teeny/tensor.h:2019

---

#### normalize_

```cpp
template<long... Axes, enable_if_t< _md::_nrm_kept_host< Shape, Axes... >::value, int > = 0> tensor & normalize_()
```

Defined in include/teeny/tensor.h:2025

---

#### cross_

```cpp
template<class Tb, class Eb, class Lb, storage Ob> tensor & cross_(const tensor< Tb, Eb, Lb, Ob > & b)
```

Defined in include/teeny/tensor.h:2027

---

#### operator++

`inline`

```cpp
inline tensor & operator++()
```

Defined in include/teeny/tensor.h:2033

---

#### operator--

`inline`

```cpp
inline tensor & operator--()
```

Defined in include/teeny/tensor.h:2034

---

#### operator++

`inline`

```cpp
template<bool S = is_static, enable_if_t< S, int > = 0> inline tensor< T, Shape, ccontiguous, storage::stack > operator++(int)
```

Defined in include/teeny/tensor.h:2036

---

#### operator--

`inline`

```cpp
template<bool S = is_static, enable_if_t< S, int > = 0> inline tensor< T, Shape, ccontiguous, storage::stack > operator--(int)
```

Defined in include/teeny/tensor.h:2038

---

#### add_

```cpp
template<bool Atomic, class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & add_(const B & b)
```

Defined in include/teeny/math.h:1331

---

#### sub_

```cpp
template<bool Atomic, class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & sub_(const B & b)
```

Defined in include/teeny/math.h:1337

---

#### mul_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & mul_(const B & b)
```

Defined in include/teeny/math.h:1343

---

#### div_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & div_(const B & b)
```

Defined in include/teeny/math.h:1345

---

#### add_

```cpp
template<bool Atomic> tensor< T, E, L, O > & add_(T s)
```

Defined in include/teeny/math.h:1347

---

#### sub_

```cpp
template<bool Atomic> tensor< T, E, L, O > & sub_(T s)
```

Defined in include/teeny/math.h:1353

---

#### minimum_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & minimum_(const B & b)
```

Defined in include/teeny/math.h:1362

---

#### maximum_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & maximum_(const B & b)
```

Defined in include/teeny/math.h:1364

---

#### add_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & add_(const B & b, T alpha)
```

Defined in include/teeny/math.h:1369

---

#### sub_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & sub_(const B & b, T alpha)
```

Defined in include/teeny/math.h:1371

---

#### atomic_add_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & atomic_add_(const B & b)
```

Defined in include/teeny/math.h:1375

---

#### atomic_sub_

```cpp
template<class B, enable_if_t<!is_arithmetic< B >::value, int >> tensor< T, E, L, O > & atomic_sub_(const B & b)
```

Defined in include/teeny/math.h:1377

---

#### copy_

```cpp
template<class B> tensor< T, E, L, O > & copy_(const B & b)
```

Defined in include/teeny/math.h:1381

---

#### map_

```cpp
template<class F> tensor< T, E, L, O > & map_(F f)
```

Defined in include/teeny/math.h:1531

---

#### zip_with_

```cpp
template<class G, class B> tensor< T, E, L, O > & zip_with_(G g, const B & b)
```

Defined in include/teeny/math.h:1533

---

#### cross_

```cpp
template<class Tb, class Eb, class Lb, storage Ob> tensor< T, E, L, O > & cross_(const tensor< Tb, Eb, Lb, Ob > & b)
```

Defined in include/teeny/math.h:2332

---

#### normalize_

```cpp
template<long... Axes, enable_if_t< _md::_nrm_kept_api< E, Axes... >::value, int >> tensor< T, E, L, O > & normalize_()
```

Defined in include/teeny/math.h:2355

---

#### minimum

`const`

```cpp
template<class B> u_abs u_log u_cos u_tanh u_ceil u_trunc auto minimum(const B & b) const
```

Defined in include/teeny/math.h:2590

### Public Static Attributes

| Return | Name | Description |
|--------|------|-------------|
| `constexpr storage` | [`ownership`](#ownership) `static` `constexpr` |  |
| `constexpr bool` | [`is_static`](#is_static) `static` `constexpr` |  |
| `constexpr bool` | [`is_view`](#is_view) `static` `constexpr` |  |
| `constexpr bool` | [`is_owning`](#is_owning) `static` `constexpr` |  |
| `constexpr bool` | [`is_device`](#is_device-1) `static` `constexpr` |  |
| `constexpr bool` | [`is_host_accessible`](#is_host_accessible) `static` `constexpr` |  |
| `constexpr size_t` | [`buffer_size`](#buffer_size) `static` `constexpr` |  |
| `constexpr bool` | [`is_strides_layout`](#is_strides_layout) `static` `constexpr` |  |
| `constexpr bool` | [`is_contiguous_layout`](#is_contiguous_layout) `static` `constexpr` |  |

---

#### ownership

`static` `constexpr`

```cpp
constexpr storage ownership = O
```

Defined in include/teeny/tensor.h:361

---

#### is_static

`static` `constexpr`

```cpp
constexpr bool is_static = _is_static_shape<Shape>()
```

Defined in include/teeny/tensor.h:362

---

#### is_view

`static` `constexpr`

```cpp
constexpr bool is_view = (O)
```

Defined in include/teeny/tensor.h:364

---

#### is_owning

`static` `constexpr`

```cpp
constexpr bool is_owning = (O)
```

Defined in include/teeny/tensor.h:365

---

#### is_device

`static` `constexpr`

```cpp
constexpr bool is_device = (O)
```

Defined in include/teeny/tensor.h:366

---

#### is_host_accessible

`static` `constexpr`

```cpp
constexpr bool is_host_accessible = (O)
```

Defined in include/teeny/tensor.h:367

---

#### buffer_size

`static` `constexpr`

```cpp
constexpr size_t buffer_size = <, O == >::value
```

Defined in include/teeny/tensor.h:368

---

#### is_strides_layout

`static` `constexpr`

```cpp
constexpr bool is_strides_layout = _is_strides<Layout>::value
```

Defined in include/teeny/tensor.h:428

---

#### is_contiguous_layout

`static` `constexpr`

```cpp
constexpr bool is_contiguous_layout = _contiguous_layout<Layout>::value
```

Defined in include/teeny/tensor.h:429

### Public Static Methods

| Return | Name | Description |
|--------|------|-------------|
| `constexpr size_t` | [`rank`](#rank-3) `static` `inline` `constexpr` `noexcept` |  |

---

#### rank

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr size_t rank() noexcept
```

Defined in include/teeny/tensor.h:425

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

Defined in include/teeny/tensor.h:352

---

#### extents_type

```cpp
using extents_type = Shape
```

Defined in include/teeny/tensor.h:353

---

#### shape_type

```cpp
using shape_type = Shape
```

Defined in include/teeny/tensor.h:354

---

#### layout_type

```cpp
using layout_type = Layout
```

Defined in include/teeny/tensor.h:355

---

#### index_type

```cpp
using index_type = typename Shape::index_type
```

Defined in include/teeny/tensor.h:356

---

#### mapping_type

```cpp
using mapping_type = typename Layout::template mapping< Shape >
```

Defined in include/teeny/tensor.h:357

---

#### view_type

```cpp
using view_type = mdspan< T, Shape, Layout >
```

Defined in include/teeny/tensor.h:358

---

#### const_view_type

```cpp
using const_view_type = mdspan< const T, Shape, Layout >
```

Defined in include/teeny/tensor.h:359



## mapping

```cpp
#include <layout.h>
```

```cpp
template<class Shape>
struct mapping
```

Defined in include/teeny/layout.h:145

> **Inherits:** `ndyn()>`, `Shape`

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`mapping`](#mapping-1) | `function` | Declared here |
| [`mapping`](#mapping-2) | `function` | Declared here |
| [`mapping`](#mapping-3) | `function` | Declared here |
| [`extents`](#extents) | `function` | Declared here |
| [`stride`](#stride-1) | `function` | Declared here |
| [`operator()`](#operator-25) | `function` | Declared here |
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
| `constexpr index_type` | [`operator()`](#operator-25) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr index_type` | [`required_span_size`](#required_span_size) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_unique`](#is_unique) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_exhaustive`](#is_exhaustive) `const` `inline` `constexpr` `noexcept` |  |
| `constexpr bool` | [`is_strided`](#is_strided) `const` `inline` `constexpr` `noexcept` |  |

---

#### mapping

```cpp
mapping() = default
```

Defined in include/teeny/layout.h:153

Defaulted constructor.

---

#### mapping

`inline` `constexpr`

```cpp
template<size_t M = strides::ndyn(), enable_if_t< M==0, int > = 0> constexpr inline constexpr mapping(const Shape & e)
```

Defined in include/teeny/layout.h:157

Fully-static strides: construct from extents only.

---

#### mapping

`inline` `constexpr`

```cpp
template<class OtherIdx> constexpr inline constexpr mapping(const Shape & e, const array< OtherIdx, strides::ndyn()> & dyn)
```

Defined in include/teeny/layout.h:174

Mixed strides: extents + the runtime strides (dim order, dynamic ones only).

Templated on the array's element type so a `reindex` (narrowing the offset index width) can pass its wider source strides — each is cast to `index_type`; symmetric with mdspan's `layout_stride`.

---

#### extents

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr const Shape & extents() const noexcept
```

Defined in include/teeny/layout.h:177

---

#### stride

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr index_type stride(rank_type r) const noexcept
```

Defined in include/teeny/layout.h:178

---

#### operator()

`const` `inline` `constexpr` `noexcept`

```cpp
template<class... I> constexpr inline constexpr index_type operator()(I... i) const noexcept
```

Defined in include/teeny/layout.h:183

---

#### required_span_size

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr index_type required_span_size() const noexcept
```

Defined in include/teeny/layout.h:194

---

#### is_unique

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_unique() const noexcept
```

Defined in include/teeny/layout.h:205

---

#### is_exhaustive

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_exhaustive() const noexcept
```

Defined in include/teeny/layout.h:206

---

#### is_strided

`const` `inline` `constexpr` `noexcept`

```cpp
constexpr inline constexpr bool is_strided() const noexcept
```

Defined in include/teeny/layout.h:207

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

Defined in include/teeny/layout.h:202

---

#### is_always_exhaustive

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr bool is_always_exhaustive() noexcept
```

Defined in include/teeny/layout.h:203

---

#### is_always_strided

`static` `inline` `constexpr` `noexcept`

```cpp
constexpr static inline constexpr bool is_always_strided() noexcept
```

Defined in include/teeny/layout.h:204

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

Defined in include/teeny/layout.h:146

---

#### index_type

```cpp
using index_type = typename Shape::index_type
```

Defined in include/teeny/layout.h:147

---

#### rank_type

```cpp
using rank_type = typename Shape::rank_type
```

Defined in include/teeny/layout.h:148

---

#### layout_type

```cpp
using layout_type = strides
```

Defined in include/teeny/layout.h:149



## item

```cpp
#include <iterate.h>
```

```cpp
struct item
```

Defined in include/teeny/iterate.h:226

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`index`](#index-2) | `variable` | Declared here |
| [`cell`](#cell-3) | `variable` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `array< index_type, Nd ? Nd :1 >` | [`index`](#index-2)  |  |
| `Cell` | [`cell`](#cell-3)  |  |

---

#### index

```cpp
array< index_type, Nd ? Nd :1 > index
```

Defined in include/teeny/iterate.h:226

---

#### cell

```cpp
Cell cell
```

Defined in include/teeny/iterate.h:226



## item

```cpp
#include <dynamic.h>
```

```cpp
struct item
```

Defined in include/teeny/dynamic.h:509

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`index`](#index) | `variable` | Declared here |
| [`cell`](#cell-1) | `variable` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `coord` | [`index`](#index)  |  |
| `Cell` | [`cell`](#cell-1)  |  |

---

#### index

```cpp
coord index
```

Defined in include/teeny/dynamic.h:509

---

#### cell

```cpp
Cell cell
```

Defined in include/teeny/dynamic.h:509



## coord

```cpp
#include <dynamic.h>
```

```cpp
struct coord
```

Defined in include/teeny/dynamic.h:503

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`ctr`](#ctr) | `variable` | Declared here |
| [`nb`](#nb) | `variable` | Declared here |
| [`lin`](#lin) | `variable` | Declared here |
| [`operator[]`](#operator-7) | `function` | Declared here |
| [`rank`](#rank-1) | `function` | Declared here |
| [`linear`](#linear) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `const offset_t *` | [`ctr`](#ctr)  |  |
| `int` | [`nb`](#nb)  |  |
| `offset_t` | [`lin`](#lin)  |  |

---

#### ctr

```cpp
const offset_t * ctr
```

Defined in include/teeny/dynamic.h:504

---

#### nb

```cpp
int nb
```

Defined in include/teeny/dynamic.h:504

---

#### lin

```cpp
offset_t lin
```

Defined in include/teeny/dynamic.h:504

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `offset_t` | [`operator[]`](#operator-7) `const` `inline` `noexcept` |  |
| `int` | [`rank`](#rank-1) `const` `inline` `noexcept` |  |
| `offset_t` | [`linear`](#linear) `const` `inline` `noexcept` |  |

---

#### operator[]

`const` `inline` `noexcept`

```cpp
inline offset_t operator[](int d) const noexcept
```

Defined in include/teeny/dynamic.h:505

---

#### rank

`const` `inline` `noexcept`

```cpp
inline int rank() const noexcept
```

Defined in include/teeny/dynamic.h:506

---

#### linear

`const` `inline` `noexcept`

```cpp
inline offset_t linear() const noexcept
```

Defined in include/teeny/dynamic.h:507



## iterator

```cpp
#include <iterate.h>
```

```cpp
struct iterator
```

Defined in include/teeny/iterate.h:174

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`tmpl`](#tmpl-1) | `variable` | Declared here |
| [`base`](#base-1) | `variable` | Declared here |
| [`cur`](#cur) | `variable` | Declared here |
| [`operator*`](#operator-21) | `function` | Declared here |
| [`operator++`](#operator-22) | `function` | Declared here |
| [`operator!=`](#operator-23) | `function` | Declared here |
| [`operator==`](#operator-24) | `function` | Declared here |
| [`index`](#index-3) | `function` | Declared here |
| [`index`](#index-4) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `Cell` | [`tmpl`](#tmpl-1)  |  |
| `El *` | [`base`](#base-1)  |  |
| `_md::peel_cursor< index_type, Nd >` | [`cur`](#cur)  |  |

---

#### tmpl

```cpp
Cell tmpl
```

Defined in include/teeny/iterate.h:175

---

#### base

```cpp
El * base
```

Defined in include/teeny/iterate.h:176

---

#### cur

```cpp
_md::peel_cursor< index_type, Nd > cur
```

Defined in include/teeny/iterate.h:177

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `Cell` | [`operator*`](#operator-21) `const` `inline` |  |
| `iterator &` | [`operator++`](#operator-22) `inline` |  |
| `bool` | [`operator!=`](#operator-23) `const` `inline` |  |
| `bool` | [`operator==`](#operator-24) `const` `inline` |  |
| `index_type` | [`index`](#index-3) `const` `inline` `noexcept` |  |
| `array< index_type, Nd ? Nd :1 >` | [`index`](#index-4) `const` `inline` `noexcept` |  |

---

#### operator*

`const` `inline`

```cpp
inline Cell operator*() const
```

Defined in include/teeny/iterate.h:178

---

#### operator++

`inline`

```cpp
inline iterator & operator++()
```

Defined in include/teeny/iterate.h:179

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const iterator & o) const
```

Defined in include/teeny/iterate.h:180

---

#### operator==

`const` `inline`

```cpp
inline bool operator==(const iterator & o) const
```

Defined in include/teeny/iterate.h:181

---

#### index

`const` `inline` `noexcept`

```cpp
inline index_type index(size_t d) const noexcept
```

Defined in include/teeny/iterate.h:186

---

#### index

`const` `inline` `noexcept`

```cpp
inline array< index_type, Nd ? Nd :1 > index() const noexcept
```

Defined in include/teeny/iterate.h:187



## enum_range

```cpp
#include <iterate.h>
```

```cpp
struct enum_range
```

Defined in include/teeny/iterate.h:234

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`r`](#r-1) | `variable` | Declared here |
| [`begin`](#begin-5) | `function` | Declared here |
| [`end`](#end-5) | `function` | Declared here |
| [`subrange`](#subrange-3) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `peel_range` | [`r`](#r-1)  |  |

---

#### r

```cpp
peel_range r
```

Defined in include/teeny/iterate.h:235

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`begin`](#begin-5) `const` `inline` |  |
| `enum_iterator` | [`end`](#end-5) `const` `inline` |  |
| `enum_subrange` | [`subrange`](#subrange-3) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline enum_iterator begin() const
```

Defined in include/teeny/iterate.h:236

---

#### end

`const` `inline`

```cpp
inline enum_iterator end() const
```

Defined in include/teeny/iterate.h:237

---

#### subrange

`const` `inline`

```cpp
inline enum_subrange subrange(index_type lo, index_type hi) const
```

Defined in include/teeny/iterate.h:243



## enum_subrange

```cpp
#include <iterate.h>
```

```cpp
struct enum_subrange
```

Defined in include/teeny/iterate.h:238

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`b`](#b-2) | `variable` | Declared here |
| [`e`](#e-2) | `variable` | Declared here |
| [`begin`](#begin-6) | `function` | Declared here |
| [`end`](#end-6) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`b`](#b-2)  |  |
| `enum_iterator` | [`e`](#e-2)  |  |

---

#### b

```cpp
enum_iterator b
```

Defined in include/teeny/iterate.h:239

---

#### e

```cpp
enum_iterator e
```

Defined in include/teeny/iterate.h:239

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`begin`](#begin-6) `const` `inline` |  |
| `enum_iterator` | [`end`](#end-6) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline enum_iterator begin() const
```

Defined in include/teeny/iterate.h:240

---

#### end

`const` `inline`

```cpp
inline enum_iterator end() const
```

Defined in include/teeny/iterate.h:241



## iterator

```cpp
#include <dynamic.h>
```

```cpp
struct iterator
```

Defined in include/teeny/dynamic.h:437

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`tmpl`](#tmpl) | `variable` | Declared here |
| [`base`](#base) | `variable` | Declared here |
| [`ctr`](#ctr-1) | `variable` | Declared here |
| [`ext`](#ext) | `variable` | Declared here |
| [`str`](#str) | `variable` | Declared here |
| [`nb`](#nb-1) | `variable` | Declared here |
| [`off`](#off) | `variable` | Declared here |
| [`lin`](#lin-1) | `variable` | Declared here |
| [`operator*`](#operator-12) | `function` | Declared here |
| [`operator++`](#operator-13) | `function` | Declared here |
| [`operator!=`](#operator-14) | `function` | Declared here |
| [`operator==`](#operator-15) | `function` | Declared here |
| [`index`](#index-1) | `function` | Declared here |
| [`nbatch`](#nbatch) | `function` | Declared here |
| [`linear`](#linear-1) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `Cell` | [`tmpl`](#tmpl)  |  |
| `T *` | [`base`](#base)  |  |
| `offset_t` | [`ctr`](#ctr-1)  |  |
| `offset_t` | [`ext`](#ext)  |  |
| `offset_t` | [`str`](#str)  |  |
| `int` | [`nb`](#nb-1)  |  |
| `offset_t` | [`off`](#off)  |  |
| `offset_t` | [`lin`](#lin-1)  |  |

---

#### tmpl

```cpp
Cell tmpl
```

Defined in include/teeny/dynamic.h:438

---

#### base

```cpp
T * base
```

Defined in include/teeny/dynamic.h:439

---

#### ctr

```cpp
offset_t ctr
```

Defined in include/teeny/dynamic.h:440

---

#### ext

```cpp
offset_t ext
```

Defined in include/teeny/dynamic.h:441

---

#### str

```cpp
offset_t str
```

Defined in include/teeny/dynamic.h:442

---

#### nb

```cpp
int nb
```

Defined in include/teeny/dynamic.h:443

---

#### off

```cpp
offset_t off
```

Defined in include/teeny/dynamic.h:444

---

#### lin

```cpp
offset_t lin
```

Defined in include/teeny/dynamic.h:444

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `Cell` | [`operator*`](#operator-12) `const` `inline` |  |
| `iterator &` | [`operator++`](#operator-13) `inline` |  |
| `bool` | [`operator!=`](#operator-14) `const` `inline` |  |
| `bool` | [`operator==`](#operator-15) `const` `inline` |  |
| `offset_t` | [`index`](#index-1) `const` `inline` `noexcept` |  |
| `int` | [`nbatch`](#nbatch) `const` `inline` `noexcept` |  |
| `offset_t` | [`linear`](#linear-1) `const` `inline` `noexcept` |  |

---

#### operator*

`const` `inline`

```cpp
inline Cell operator*() const
```

Defined in include/teeny/dynamic.h:445

---

#### operator++

`inline`

```cpp
inline iterator & operator++()
```

Defined in include/teeny/dynamic.h:446

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const iterator & o) const
```

Defined in include/teeny/dynamic.h:454

---

#### operator==

`const` `inline`

```cpp
inline bool operator==(const iterator & o) const
```

Defined in include/teeny/dynamic.h:455

---

#### index

`const` `inline` `noexcept`

```cpp
inline offset_t index(int d) const noexcept
```

Defined in include/teeny/dynamic.h:461

---

#### nbatch

`const` `inline` `noexcept`

```cpp
inline int nbatch() const noexcept
```

Defined in include/teeny/dynamic.h:462

---

#### linear

`const` `inline` `noexcept`

```cpp
inline offset_t linear() const noexcept
```

Defined in include/teeny/dynamic.h:463



## subrange_t

```cpp
#include <iterate.h>
```

```cpp
struct subrange_t
```

Defined in include/teeny/iterate.h:208

A `[lo, hi)` slice of the cells for chunked/threaded sweeps: seed the incremental cursor once at `lo`, then O(1) per step within the chunk.

(Split `[0,[size()](#size-3))` across threads/blocks; each sweeps its chunk.)

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`b`](#b-3) | `variable` | Declared here |
| [`e`](#e-3) | `variable` | Declared here |
| [`begin`](#begin-7) | `function` | Declared here |
| [`end`](#end-7) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `iterator` | [`b`](#b-3)  |  |
| `iterator` | [`e`](#e-3)  |  |

---

#### b

```cpp
iterator b
```

Defined in include/teeny/iterate.h:209

---

#### e

```cpp
iterator e
```

Defined in include/teeny/iterate.h:209

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `iterator` | [`begin`](#begin-7) `const` `inline` |  |
| `iterator` | [`end`](#end-7) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline iterator begin() const
```

Defined in include/teeny/iterate.h:210

---

#### end

`const` `inline`

```cpp
inline iterator end() const
```

Defined in include/teeny/iterate.h:211



## enum_iterator

```cpp
#include <iterate.h>
```

```cpp
struct enum_iterator
```

Defined in include/teeny/iterate.h:227

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`it`](#it-1) | `variable` | Declared here |
| [`operator*`](#operator-17) | `function` | Declared here |
| [`operator++`](#operator-18) | `function` | Declared here |
| [`operator!=`](#operator-19) | `function` | Declared here |
| [`operator==`](#operator-20) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `iterator` | [`it`](#it-1)  |  |

---

#### it

```cpp
iterator it
```

Defined in include/teeny/iterate.h:228

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `item` | [`operator*`](#operator-17) `const` `inline` |  |
| `enum_iterator &` | [`operator++`](#operator-18) `inline` |  |
| `bool` | [`operator!=`](#operator-19) `const` `inline` |  |
| `bool` | [`operator==`](#operator-20) `const` `inline` |  |

---

#### operator*

`const` `inline`

```cpp
inline item operator*() const
```

Defined in include/teeny/iterate.h:229

---

#### operator++

`inline`

```cpp
inline enum_iterator & operator++()
```

Defined in include/teeny/iterate.h:230

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const enum_iterator & o) const
```

Defined in include/teeny/iterate.h:231

---

#### operator==

`const` `inline`

```cpp
inline bool operator==(const enum_iterator & o) const
```

Defined in include/teeny/iterate.h:232



## enum_range

```cpp
#include <dynamic.h>
```

```cpp
struct enum_range
```

Defined in include/teeny/dynamic.h:517

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`r`](#r) | `variable` | Declared here |
| [`begin`](#begin-1) | `function` | Declared here |
| [`end`](#end-1) | `function` | Declared here |
| [`subrange`](#subrange-1) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `anyrank_front` | [`r`](#r)  |  |

---

#### r

```cpp
anyrank_front r
```

Defined in include/teeny/dynamic.h:518

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`begin`](#begin-1) `const` `inline` |  |
| `enum_iterator` | [`end`](#end-1) `const` `inline` |  |
| `enum_subrange` | [`subrange`](#subrange-1) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline enum_iterator begin() const
```

Defined in include/teeny/dynamic.h:519

---

#### end

`const` `inline`

```cpp
inline enum_iterator end() const
```

Defined in include/teeny/dynamic.h:520

---

#### subrange

`const` `inline`

```cpp
inline enum_subrange subrange(offset_t lo, offset_t hi) const
```

Defined in include/teeny/dynamic.h:526



## enum_subrange

```cpp
#include <dynamic.h>
```

```cpp
struct enum_subrange
```

Defined in include/teeny/dynamic.h:521

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`b`](#b) | `variable` | Declared here |
| [`e`](#e) | `variable` | Declared here |
| [`begin`](#begin-2) | `function` | Declared here |
| [`end`](#end-2) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`b`](#b)  |  |
| `enum_iterator` | [`e`](#e)  |  |

---

#### b

```cpp
enum_iterator b
```

Defined in include/teeny/dynamic.h:522

---

#### e

```cpp
enum_iterator e
```

Defined in include/teeny/dynamic.h:522

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`begin`](#begin-2) `const` `inline` |  |
| `enum_iterator` | [`end`](#end-2) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline enum_iterator begin() const
```

Defined in include/teeny/dynamic.h:523

---

#### end

`const` `inline`

```cpp
inline enum_iterator end() const
```

Defined in include/teeny/dynamic.h:524



## subrange_t

```cpp
#include <dynamic.h>
```

```cpp
struct subrange_t
```

Defined in include/teeny/dynamic.h:484

A `[lo, hi)` slice of the batch cells for chunked/threaded sweeps: seed the incremental cursor once at `lo`, then O(1) per step.

Split `[0, [size()](#size-2))` across threads/blocks; each sweeps its own chunk.

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`b`](#b-1) | `variable` | Declared here |
| [`e`](#e-1) | `variable` | Declared here |
| [`begin`](#begin-3) | `function` | Declared here |
| [`end`](#end-3) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `iterator` | [`b`](#b-1)  |  |
| `iterator` | [`e`](#e-1)  |  |

---

#### b

```cpp
iterator b
```

Defined in include/teeny/dynamic.h:485

---

#### e

```cpp
iterator e
```

Defined in include/teeny/dynamic.h:485

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `iterator` | [`begin`](#begin-3) `const` `inline` |  |
| `iterator` | [`end`](#end-3) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline iterator begin() const
```

Defined in include/teeny/dynamic.h:486

---

#### end

`const` `inline`

```cpp
inline iterator end() const
```

Defined in include/teeny/dynamic.h:487



## enum_iterator

```cpp
#include <dynamic.h>
```

```cpp
struct enum_iterator
```

Defined in include/teeny/dynamic.h:510

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`it`](#it) | `variable` | Declared here |
| [`operator*`](#operator-8) | `function` | Declared here |
| [`operator++`](#operator-9) | `function` | Declared here |
| [`operator!=`](#operator-10) | `function` | Declared here |
| [`operator==`](#operator-11) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `iterator` | [`it`](#it)  |  |

---

#### it

```cpp
iterator it
```

Defined in include/teeny/dynamic.h:511

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `item` | [`operator*`](#operator-8) `const` `inline` |  |
| `enum_iterator &` | [`operator++`](#operator-9) `inline` |  |
| `bool` | [`operator!=`](#operator-10) `const` `inline` |  |
| `bool` | [`operator==`](#operator-11) `const` `inline` |  |

---

#### operator*

`const` `inline`

```cpp
inline item operator*() const
```

Defined in include/teeny/dynamic.h:512

---

#### operator++

`inline`

```cpp
inline enum_iterator & operator++()
```

Defined in include/teeny/dynamic.h:513

---

#### operator!=

`const` `inline`

```cpp
inline bool operator!=(const enum_iterator & o) const
```

Defined in include/teeny/dynamic.h:514

---

#### operator==

`const` `inline`

```cpp
inline bool operator==(const enum_iterator & o) const
```

Defined in include/teeny/dynamic.h:515



## enum_subrange

```cpp
#include <iterate.h>
```

```cpp
struct enum_subrange
```

Defined in include/teeny/iterate.h:238

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`b`](#b-2) | `variable` | Declared here |
| [`e`](#e-2) | `variable` | Declared here |
| [`begin`](#begin-6) | `function` | Declared here |
| [`end`](#end-6) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`b`](#b-2)  |  |
| `enum_iterator` | [`e`](#e-2)  |  |

---

#### b

```cpp
enum_iterator b
```

Defined in include/teeny/iterate.h:239

---

#### e

```cpp
enum_iterator e
```

Defined in include/teeny/iterate.h:239

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`begin`](#begin-6) `const` `inline` |  |
| `enum_iterator` | [`end`](#end-6) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline enum_iterator begin() const
```

Defined in include/teeny/iterate.h:240

---

#### end

`const` `inline`

```cpp
inline enum_iterator end() const
```

Defined in include/teeny/iterate.h:241



## enum_subrange

```cpp
#include <dynamic.h>
```

```cpp
struct enum_subrange
```

Defined in include/teeny/dynamic.h:521

### List of all members

| Name | Kind | Owner |
|------|------|-------|
| [`b`](#b) | `variable` | Declared here |
| [`e`](#e) | `variable` | Declared here |
| [`begin`](#begin-2) | `function` | Declared here |
| [`end`](#end-2) | `function` | Declared here |

### Public Attributes

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`b`](#b)  |  |
| `enum_iterator` | [`e`](#e)  |  |

---

#### b

```cpp
enum_iterator b
```

Defined in include/teeny/dynamic.h:522

---

#### e

```cpp
enum_iterator e
```

Defined in include/teeny/dynamic.h:522

### Public Methods

| Return | Name | Description |
|--------|------|-------------|
| `enum_iterator` | [`begin`](#begin-2) `const` `inline` |  |
| `enum_iterator` | [`end`](#end-2) `const` `inline` |  |

---

#### begin

`const` `inline`

```cpp
inline enum_iterator begin() const
```

Defined in include/teeny/dynamic.h:523

---

#### end

`const` `inline`

```cpp
inline enum_iterator end() const
```

Defined in include/teeny/dynamic.h:524

Generated by [Moxygen](https://0state.com/moxygen)