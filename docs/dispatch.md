# Dispatch & the anyrank boundary

Kernels want static shapes (so they fold to compile-time constants), but data from
Python arrives with a runtime rank and runtime sizes. teeny turns a runtime
**value**, a runtime **rank**, or a rank-erased pointer into a statically-typed
tensor once, at the boundary, then stays fast inside.

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

## `dispatch_values` — several values at once

Kernel libraries usually have more than one runtime knob to make static — a spatial
rank *and* an interpolation order *and* a boundary condition. Nesting `dispatch_value`
works, but buries the kernel three lambdas deep. `dispatch_values` takes one
**candidate list** per parameter and hands `f` one `integral_constant` per list:

```cpp
dispatch_values([&](auto D, auto O, auto B) {
                    kernel<D.value, O.value, B.value>(view);
                },
                candidates<1,2,3>(spatial_ndim),      // spatial rank
                candidates<0,1,2,3>(order),           // interpolation order
                candidates<0,1,2,3,4,5,6,7>(bnd));    // boundary condition
```

instead of

```cpp
dispatch_value<1,2,3>(spatial_ndim, [&](auto D) {
  dispatch_value<0,1,2,3>(order, [&](auto O) {
    dispatch_value<0,1,2,3,4,5,6,7>(bnd, [&](auto B) {
      kernel<D.value, O.value, B.value>(view); });});});
```

It is exactly that nesting, written once: the same one comparison per parameter at run
time, and `f` instantiated once per **combination** of candidates. You still own the
instantiation budget — it is the product of the list lengths (`3 × 4 × 8 = 96` above) —
and putting the lists next to each other is the point: the budget is readable in one
place instead of spread down a pyramid.

`f` comes **first** here (the one place teeny puts the callable ahead of its arguments)
because the candidate lists are variadic.

**A value outside its list simply doesn't fire**, per parameter, exactly as with
`dispatch_value`: no assert and no abort — `f` is not called and the call returns
`false`. It returns `true` only when every value matched and `f` ran.

```cpp
bool ran = dispatch_values(f, candidates<1,2,3>(7), candidates<0,1>(0));   // 7 ∉ {1,2,3}
// ran == false, f never called
```

**Enums dispatch directly.** The runtime value may be any integer *or enum* type, so a
`bound`/`order` enum needs no `static_cast` at the call site (the candidates stay plain
`int`s, and so does the `integral_constant` `f` receives):

```cpp
enum class bound { zero, replicate, dct1, dct2, dst1, dst2, dft, nocheck };
dispatch_values(f, candidates<0,1,2,3,4,5,6,7>(bnd));   // bnd is a `bound`
```

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

The keep-count is **negative** — you pass `-Sr` (`-2` keeps the last two dims), the
same "negative = keep the last `|N|`" sign rule as the tensor's
[`peel_front`](structure.md#nd-peel--iterate-a-subset-of-axes). On `anyrank` it must
be negative: a positive front-count would leave a *runtime* rank, which can't be a
static view (it's a `static_assert`).

Write it as a static integer value, `Int<-Sr>()`:

=== "value form"

    ```cpp
    for (auto cell : at.peel_front(Int<-Sr>())) kernel<Sr>(cell);  // Sr=2 -> Int<-2>(); cell is rank-Sr
    auto cell = at.peel_front_at(i, Int<-Sr>());                   // i-th (grid-stride)
    auto nb   = at.size_front(Int<-Sr>());                         // flattened batch count
    ```

=== "template form"

    ```cpp
    for (auto cell : at.peel_front<-Sr>()) kernel<Sr>(cell);
    auto cell = at.peel_front_at<-Sr>(i);
    auto nb   = at.size_front<-Sr>();
    ```

Both spellings are the same call. Reach for the value form first: because the
keep-count is *deduced* from the argument, it works unchanged when the carrier's
type is a template parameter — `at.peel_front<-Sr>()` there needs the
`at.template peel_front<-Sr>()` disambiguator, and the value form needs nothing.

Each `cell` is a `dextents<_,Sr>` view (inner extents dynamic). Recover the static
inner dims by peeling **directly** to the target shape — one call, no separate
`recast`:

```cpp
auto cell = at.peel_front_at(i, shape<-1,c,c>{});     // rank-3 cell, inner (c,c) static
```

`peel_front_at<NewE[, NewL]>(i)` (and its `peel_front_at(i, shape<…>{}[, layout{}])`
value form) fuses `peel_front_at<-NewE::rank()>(i).recast<NewE, NewL>()`. `NewE`'s rank
is the number of KEPT trailing dims; a static extent folds, a `-1` stays dynamic
(`shape<-1,-1,3>` = 2-D spatial + static C). It keeps the carrier's **runtime strides**
by default (an `anyrank` has no compile-time stride info); pass `,ccontiguous{}` to fold
them (a debug-checked promise) or run [`dispatch_layout`](#dispatch_layout--recover-static-contiguity)
on the result for a runtime-proven fold. Fusing also drops the hand-kept
`Sr ≡ recast-shape rank` invariant. The two value forms never collide: a `shape<…>{}`
in that second position asks for the fused recast, an `Int<-Sr>()` just names the
keep-count.

### Static trailing shape — carry the static inner dims in the carrier's type

The per-call `peel_front_at<shape<-1,c,c>>(i)` (and `recast`) is a *promise re-made
at every call site*. When the inner geometry is a **property of the data** — known at
the import boundary, e.g. `(*batch, *spatial, C)` with a static channel count `C` — bake
it into the carrier once instead:

The tag is an **`anyshape<...>`** — a single `etc` marks the erased dynamic-rank region,
and the dims after it are the static tail (anchored at `ndim`). `anyshape<etc,-1,-1,3>`
reads as `(*batch, spatial, spatial, C=3)`; the leading `etc` makes "this is a rank-erased
tail spec" unmistakable (versus a concrete rank-3 `shape<-1,-1,3>`).

```cpp
auto at = as_anyrank(data, shape, stride, ndim, anyshape<etc,-1,-1,3>{});   // static trailing (…,…,3)
// from a DLPack capsule:  auto at = from_dlpack<float, anyshape<etc,-1,-1,3>>(m);
for (auto cell : at.peel_front<-3>()) kernel(cell);   // every cell's inner extent is 3 — folded
auto v = at.fixed<4>();                               // extent(3) == 3 at compile time
```

The runtime trailing dims are **debug-checked against the tag once, here at the boundary**
(next to the producer), then trusted — the same contract class as `recast`, but asserted
in one place rather than re-promised per kernel. Every view the carrier hands out —
`fixed()`, `peel_front_at`, and the incremental `peel_front<-Sr>()` iterator (its cell is
*born* folded, so its mapping shrinks) — carries the folded extents automatically, and it
composes with `dispatch_rank`/`dispatch_value` (the static `C` types once, every rank arm
inherits it). Bare `anyshape<etc>` (empty tail) is exactly today's fully-dynamic carrier,
byte-identical.

**Folding the inner strides too.** Pass a **layout tag** after the shape (like `recast`'s
2nd argument) to also bake the trailing strides into the type:

```cpp
auto at = as_anyrank(data, shape, stride, ndim, anyshape<etc,-1,-1,3>{}, ccontiguous{});
// DLPack:  auto at = from_dlpack<float, anyshape<etc,-1,-1,3>>(m, ccontiguous{});
```

- `keep_strides` (default) — strides stay runtime (`layout_stride` cell), exactly the
  extents-only behaviour above.
- `ccontiguous`/`fcontiguous` — the inner block's strides fold to compile-time constants
  (`shape<-1,-1,3>` C-order → `strides<9,3,1>` — the outer stride is static because it is
  the product of the *static* trailing extents, even though its own extent is dynamic). A
  fully-static contiguous tail makes the cell's mapping **empty (EBO)** — the cell loses its
  stride words, fewer registers per thread.
- `strides<S...>` — impose those strides.

The runtime strides are **checked against the tag once, here at the boundary** (with
`recast`'s "extent ≤ 1 ⇒ stride unobservable" exemption). This is where the tail
**subsumes `dispatch_layout`** for the common *"the input must be C-contiguous"* precondition:
the fold is backed by a check that actually ran, at the boundary, versus `dispatch_layout`'s
per-call 2–3× runtime branch. Reach for `dispatch_layout` only when the layout is genuinely
unknown per call (accept-anything); use the tag when contiguity is a precondition you can
assert at import.

### Static leading Head — `anyshape<A, B, etc, C, D>`

`etc` need not come first: dims *before* it are a static **Head** (anchored at 0), dims
*after* it the static **Tail** (anchored at `ndim`), with the erased middle between —
`anyshape<3, etc, 5>` is `(C_in=3, *spatial, C_out=5)`.

```cpp
auto at = as_anyrank(data, shape, stride, ndim, anyshape<3, etc, 5>{});
auto v  = at.fixed<R>();   // extent(0) == 3 AND extent(R-1) == 5 at compile time
```

The Head folds in **`fixed<R>` / `dispatch_rank`** — a full-rank window has a compile-time
left edge, so a leading dim anchors. It does **not** fold in `peel_front<-Sr>`: that keeps a
*trailing* window whose left edge is `ndim - Sr` (runtime), and the Head is normally peeled
into the batch (inert there — the peel cell folds the Tail only). Head **extents** fold; head
**strides** stay runtime (a leading dim's contiguous stride spans the dynamic middle, so it
isn't a compile-time constant). An empty Head (`etc` first) is exactly the trailing-only
carrier, byte-identical.

Keep the per-call `peel_front_at<shape<…>>(i)` as the escape hatch for a carrier imported
without a tag.

The **range-for is incremental**: it advances the base pointer by the batch
strides and reuses the loop-invariant cell mapping, so each step is O(1) rather
than an O(#batch) index decode. Two ways to parallelize:

```cpp
// device grid-stride: random access, each thread strides by nthreads
for (offset_t i = tid; i < at.size_front(Int<-Sr>()); i += nthreads)
    kernel<Sr>(at.peel_front_at(i, Int<-Sr>()));

// CPU thread / device BLOCK owning a contiguous chunk: incremental sweep of [lo,hi)
for (auto cell : at.peel_front(Int<-Sr>()).subrange(lo, hi)) kernel<Sr>(cell);
```

Use `peel_front_at` (random access) for grid-stride — the odometer can't express a
`+= nthreads` stride. Use `subrange(lo, hi)` when a worker owns a contiguous block
of cells: it seeds the cursor once at `lo`, then advances incrementally.

Need the **batch coordinates** (a per-batch-axis table `param[d][m[d]]`)? `enumerate()`
pairs each cell with the batch multi-index — opt-in, so the bare `peel_front<-Sr>()` cell
stays lean (the tensor-side `peel` has the same `enumerate`):

```cpp
for (auto [m, cell] : at.peel_front<-Sr>().enumerate())
    kernel<Sr>(m[0], cell);   // m[d] = coord of batch axis d; m.rank() (runtime); m.linear() = flat batch idx
```

`m` is a lightweight **view of the live odometer** — valid within the loop body, don't store
it past the iteration. The raw iterator also exposes `it.index(d)` / `it.nbatch()` /
`it.linear()`. Composes with `subrange` (`.enumerate().subrange(lo,hi)`). The batch rank is
runtime, so there is no fixed-size `index()` tuple — read coordinates per axis with `m[d]`.

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

### Narrowing the whole carrier (the GPU spelling)

The two calls above narrow **one view at a time**, on the host. For a CUDA launch that
keeps the batch idiom — where `ndim` stays runtime and only `Sr` is static — narrowing
has to happen **once, before the launch, on the carrier itself**. `anyrank` carries the
same pair:

```cpp
auto at = from_dlpack<float>(&dlt);
if (at.index_fits<int32_t>()) launch(at.reindex<int32_t>());   // int32 carrier
else                          launch(at);                      // int64 carrier
```

`at.index_fits<Idx2>()` is the signed-reach test over the **whole** carrier (every
reachable offset must be representable in `Idx2`). `at.reindex<Idx2>()` returns the same
carrier — same data pointer, same memory space, same static `anyshape` head/tail geometry
— with `ndim` and the runtime shape/strides copied into an inline `Idx2` store. Since the
carrier's offset width is part of its type, **every cell it later hands out is already
`Idx2`-indexed**: nothing downstream changes.

```cpp
for (auto cell : at32.peel_front<-Sr>()) launch<Sr>(cell);   // cells are int32-indexed
```

That halves the meta store the carrier passes by value into a `__global__`
(`MaxRank × 2 × 8 B` → `× 4 B`) and runs the cells' address math in 32-bit — the SM has
no native 64-bit integer multiply-add, so each axis costs one `IMAD` instead of a
sequence. `reindex` always produces an inline (`copy_meta`) store: narrowing has to copy,
and there is nothing to narrow a *wrapped* array into. Capacity follows the source
carrier; `at.reindex<int32_t, 8>()` picks another.

It debug-checks `index_fits` and is UB if you lie — the same contract as the view's
`reindex`. `dispatch_index` accepts a carrier too, so the two arms above are just:

```cpp
dispatch_index(at, [&](auto a) { launch(a); });      // f instantiated for both widths
```

Import width is never changed for you: `from_dlpack` keeps DLPack's own `int64_t`, and
teeny applies no heuristic — narrowing is a per-boundary decision you make. On the CPU it
measures neutral; the win is on the device.

### `dispatch_layout` — recover static contiguity

The rank-erased boundary drops the producer's contiguity into `layout_stride`
(`dynamic_strides`), so a cell's `recast<shape<-1,c,c>>()` can only keep *runtime*
strides. `dispatch_layout` is the layout sibling of `dispatch_index`: it cheaply
classifies the runtime strides (`is_dense<ccontiguous>()` / `<fcontiguous>()` — a
stride compare, no data touched) and hands `f` the view **retyped** to
`ccontiguous` / `fcontiguous` / (else) `dynamic_strides`. In the contiguous arms the
strides are extent-derived, so the inner `recast` folds them to compile-time constants **safely** —
no `recast<…, ccontiguous>()` "I promise it's contiguous" (which is UB under `-DNDEBUG`
if wrong).

```cpp
for (auto cell : at.peel_front<-Sr>())
    dispatch_layout(cell, [&](auto v) { kernel<Sr>(v.recast(shape<-1,c,c>{})); });
```

The **C-order arm is the payoff** — C-order inner strides depend only on the static
inner extents, so `shape<-1,c,c>` folds them; an F-order inner stride multiplies the
*dynamic* batch, so the F arm mainly gives a typed extent-derived view. **Opt-in**: `f`
runs on up to 3 view types, and it composes multiplicatively with the rank/width
dispatchers — reach for it only where the inner block's folded strides matter (a small
static-`C` kernel).

## The full boundary pattern

For a `(*batch, *spatial, C)` array from numpy / torch / cupy / DLPack:

```text
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
