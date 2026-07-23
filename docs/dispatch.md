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
`own::view` (host) by default. Every view it hands out (`fixed`, `peel_front`,
`peel_front_at`) inherits it, so a device pointer yields `gpu_view`-tagged views
rather than host views over device memory:

```cpp
auto hd = as_anyrank(data, shape, stride, ndim);              // host  -> view cells
auto gd = as_anyrank<own::gpu_view>(dptr, shape, stride, n);  // device -> gpu_view cells
```

`from_dlpack` sets this from the capsule and **checks it**: importing a `kDLCUDA`
capsule with the default host space trips a `_TNY_CHECK` — spell it
`from_dlpack<T, own::gpu_view>(m)` (or `dispatch_dlpack<own::gpu_view>(m, f)`) so
the views are correctly device-tagged. (The shape/stride *metadata* is host either
way; the space labels the data.)

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
[`recast<shape<-1,c,c>>()`](structure.md#recover-static-inner-dims) to fold known
inner dims.

### `dispatch_rank` / `fixed<R>` — general (per total rank)

When the whole rank must be static, dispatch on the runtime `ndim`. `f` is
instantiated once per possible total rank. Returns false if `ndim` exceeds
`MaxRank`. Prefer `peel_front<-Sr>` when only the trailing dims need to be static.

```cpp
dispatch_rank(at, [&](auto v) { kernel(v); });  // once per total rank
auto v3 = at.fixed<3>();                        // or force a known rank
```

## The full boundary pattern

For a `(*batch, *spatial, C)` array from numpy / torch / cupy / DLPack:

```
DLPack / ndarray  ──as_anyrank(data, shape, stride, ndim)──►  anyrank
   │  (DLPack strides in ELEMENTS; numpy's __array_interface__ in BYTES)
   ▼  dispatch_value<1,2,3>(spatial_ndim)  -> static spatial rank D
   ▼  Sr = D + 1  (spatial + channel)
   for (auto cell : at.peel_front<-Sr>()) {   // negative: keep the last Sr dims
       kernel<D>(cell.recast<shape<-1,…static inner…>>(), …);  // parallelise this
   }
```

The worked-through version, with CPU-thread and CUDA drivers and a nanobind
wrapper (native DLPack), is the [DLPack → Python tutorial](tutorials/dlpack-python.md).
