# Dispatch & the ndarray boundary

Kernels want static shapes (for folding), but data from Python arrives with a
runtime rank and runtime sizes. teeny turns a runtime **value**, a runtime
**rank**, or a rank-erased pointer into a statically-typed tensor once, at the
boundary, then stays fast inside.

## `dispatch_value` — runtime value → compile-time

Pick a compile-time value from a candidate list. `f` is instantiated once per
candidate; the matching one runs with an `integral_constant` usable as a template
argument. Returns whether any matched.

```cpp
dispatch_value<1,2,3>(spatial_ndim, [&](auto D) {
    kernel<D.value>(view);  // D.value is a compile-time constant here
});
```

Use it for any small runtime value you want static: a spatial rank, an
interpolation order, a matrix size `C`. Replaces a hand-written `switch`.

## `anyrank` — the rank-erased carrier

A rank-erased carrier for the host boundary: a pointer, 1-D shape/stride tensors,
and a runtime `ndim`. By default (`as_anyrank(...)`) it **wraps** the caller's
arrays with no copy and is **host-only**; built with the `copy_meta` tag it
copies into an inline `TNY_MAX_RANK` store (default 32) and is then trivially
copyable, so it passes into a CUDA kernel by value (`anyrank::device_passable`).

`anyrank` has **no arithmetic** — it is a doorway, not a room. Turn it into a
static view at the boundary and compute on that.

```cpp
auto at = as_anyrank(data, shape, stride, ndim);        // -> anyrank, WRAPS the arrays (no copy)
auto ad = as_anyrank(data, shape, stride, ndim, copy_meta);  // -> anyrank, COPIES into an inline store
```

By default `as_anyrank` **wraps** the caller's shape/stride arrays with no copy —
e.g. straight off a DLPack tensor — which is host-only (those pointers aren't
valid in a device kernel; peel/dispatch on the host and pass the fixed-rank views
to the device). Pass the `copy_meta` tag to instead **copy** shape/stride into an
inline `TNY_MAX_RANK` store (default 32; `-DTNY_MAX_RANK=N`, or per-call
`as_anyrank<N>(..., copy_meta)`), making the carrier trivially copyable so it can be
passed into a CUDA kernel by value. DLPack strides are in **elements**; numpy
`__array_interface__` strides are in **bytes** (divide by the itemsize first).

### Memory space of the data

The carrier also carries a compile-time **memory space** for the `data` pointer —
`storage::view` (host) by default. Every view it hands out (`fixed`, `peel_front`,
`peel_front_at`) inherits it, so a device pointer yields `gpu_view`-tagged views
rather than host views over device memory:

```cpp
auto hd = as_anyrank(data, shape, stride, ndim);              // host  -> view cells
auto gd = as_anyrank<storage::gpu_view>(dptr, shape, stride, n);  // device -> gpu_view cells
```

`from_dlpack` sets this from the capsule and **checks it**: importing a `kDLCUDA`
capsule with the default host space trips a `_TNY_CHECK` — spell it
`from_dlpack<T, storage::gpu_view>(m)` (or `dispatch_dlpack<storage::gpu_view>(m, f)`) so
the views are correctly device-tagged. (The shape/stride *metadata* is host either
way; the space labels the data.)

### Importing from DLPack: two dispatch flavours

`from_dlpack<T[,Space]>(m)` returns the typed `anyrank`; two helpers add the dtype
dispatch on top of it:

- **`dispatch_dlpack<Space>(m, f)`** reads dtype **and** rank and hands `f` a
  **fixed-rank** view — one instantiation per (dtype × total rank).
- **`dispatch_dlpack_dtype<Space>(m, f)`** reads only the dtype and hands `f` the
  **typed `anyrank`** (rank still dynamic), so `f` drives the batch idiom below —
  the kernel instantiates **once per `Sr`**, not once per total rank. Prefer this for
  `(*batch, *spatial, C)` data.

Both instantiate `f` for every supported dtype (only the matching one runs), so `f`
must be generic over its element type.

`m` may be any DLPack carrier: a classic `DLManagedTensor*`, a **bare `DLTensor*`**
(unmanaged — nothing to free, the caller owns all lifetime), or a
**`DLManagedTensorVersioned*`** (DLPack 1.0+, what a modern `__dlpack__(max_version=…)`
emits). `from_dlpack` and both dispatchers accept all three; only who frees the carrier
differs. To hand teeny data *out* to a consumer that wants a bare `DLTensor`, use
`to_dltensor(t, shape_out, strides_out)` (borrowed — you own the two `int64_t` buffers);
`to_dlpack(t)` is the owning managed-capsule export.

### `peel_front<-Sr>` — the batch pattern (preferred)

For `(*batch, *spatial, C)` data, peel the runtime number of leading batch dims
and keep the trailing `Sr` "interesting" dims static. The kernel instantiates
**once per `Sr`**, not once per total rank.

The template argument is **negative** — you pass `-Sr` (`peel_front<-2>()` keeps
the last two dims), the same "negative = keep the last `|N|`" sign rule as the
tensor's [`peel_front`](structure.md#nd-peel--iterate-a-subset-of-axes). On
`anyrank` it must be negative: a
positive front-count would leave a *runtime* rank, which can't be a static view
(it's a `static_assert`).

```cpp
for (auto cell : at.peel_front<-Sr>()) kernel<Sr>(cell);  // Sr=2 -> peel_front<-2>; cell is rank-Sr
auto cell = at.peel_front_at<-Sr>(i);                     // i-th (grid-stride)
```

Each `cell` is a `dextents<_,Sr>` view (inner extents dynamic). Follow with
[`recast(shape<-1,c,c>{})`](structure.md#recover-static-inner-dims) to fold known
inner dims.

The **range-for is incremental**: it advances the base pointer by the batch
strides and reuses the loop-invariant cell mapping, so each step is O(1) rather
than an O(#batch) index decode. Two ways to parallelize:

```cpp
// device grid-stride: random access, each thread strides by nthreads
for (offset_t i = tid; i < at.size_front<-Sr>(); i += nthreads)
    kernel<Sr>(at.peel_front_at<-Sr>(i));

// CPU thread / device BLOCK owning a contiguous chunk: incremental sweep of [lo,hi)
for (auto cell : at.peel_front<-Sr>().subrange(lo, hi)) kernel<Sr>(cell);
```

Use `peel_front_at` (random access) for grid-stride — the odometer can't express a
`+= nthreads` stride. Use `subrange(lo, hi)` when a worker owns a contiguous block
of cells: it seeds the cursor once at `lo`, then advances incrementally.

### `dispatch_rank` / `fixed<R>` — general (per total rank)

When the whole rank must be static, dispatch on the runtime `ndim`. `f` is
instantiated once per possible total rank. Returns false if `ndim` exceeds
`MaxRank`. Prefer `peel_front<-Sr>` when only the trailing dims need to be static.

```cpp
dispatch_rank(at, [&](auto v) { kernel(v); });  // once per total rank
auto v3 = at.fixed<3>();                        // or force a known rank
```

### `dispatch_index` / `dispatch_rank<narrow_index>` — the int32 fast path

At the kernel boundary you can narrow the **offset index width** to 32-bit when the
element span provably fits (`index_fits`) — halving a dynamic view's by-value
footprint and running address math in 32-bit (a device register/occupancy win). It's
the [`reindex`](shapes-strides.md#the-index-type-shape32-reindex) transition made a
runtime dispatch: `f` is instantiated for **both** widths and the right one is picked
at run time.

```cpp
dispatch_index(v, [&](auto w) { kernel(w); });        // narrow a fixed-rank view (or a peel cell)
dispatch_rank<narrow_index>(at, [&](auto v) { kernel(v); });  // fuse it into the rank dispatch
```

`dispatch_rank<narrow_index>` nests **rank outer, width inner**, so only the leaf
instantiation doubles; plain `dispatch_rank(at, f)` (the default) is unchanged and
adds nothing. For the batch idiom, narrow each cell:

```cpp
for (auto cell : at.peel_front<-Sr>()) dispatch_index(cell, [&](auto c) { kernel<Sr>(c); });
```

Opt in per launch site — narrowing everything would silently double instantiation
counts. `dispatch_index<Idx2>` targets a width other than the `int32_t` default.

## The full boundary pattern

For a `(*batch, *spatial, C)` array from numpy / torch / cupy / DLPack:

```
DLPack / ndarray  ──as_anyrank(data, shape, stride, ndim)──►  anyrank
   │  (DLPack strides in ELEMENTS; numpy's __array_interface__ in BYTES)
   ▼  dispatch_value<1,2,3>(spatial_ndim)  -> static spatial rank D
   ▼  Sr = D + 1  (spatial + channel)
   for (auto cell : at.peel_front<-Sr>()) {   // negative: keep the last Sr dims
       kernel<D>(cell.recast(shape<-1,…static inner…>{}), …);  // parallelise this
   }
```

The worked-through version, with CPU-thread and CUDA drivers and a nanobind
wrapper (native DLPack), is the [DLPack → Python tutorial](tutorials/dlpack-python.md).
