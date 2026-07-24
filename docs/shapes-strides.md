# Shapes & strides

Each dimension's size and stride may be static (a compile-time constant) or
dynamic (a run-time value) — independently. Static values fold into machine
code; dynamic ones are carried at run time. One kernel source covers every
combination.

## `shape<...>` — the sizes

`shape<...>` is the extents type: `cuda::std::extents<int64_t, ...>`. The
`int64_t` index type matches DLPack's `shape`, so it drops onto ndarray
bindings.

```cpp
shape<2,3,4>             // fully static 2×3×4
shape<-1,3>              // dynamic rows, static 3 columns   (-1 == dynamic)
shape<dynamic_extent,3>  // the same type, spelled out
```

A dynamic dimension is `-1` (numpy-style) or `dynamic_extent`. Construct a tensor
by supplying only the dynamic sizes:

```cpp
auto a = wrap(ptr, shape<2,3,4>{});      // all static — nothing to supply
auto b = wrap(ptr, shape<-1,3>{n});      // supply the one dynamic size, n
auto c = wrap(ptr, shape<-1,-1>{r, k});  // supply both
```

Query sizes with `extent` / `shape` (a python-friendly alias):

```cpp
t.rank();            // number of dimensions (static)
t.numel();           // total element count
t.extent(0);         // RUNTIME lookup -> index_type
t.extent(Int<0>());  // STATIC lookup -> integral_constant when that extent is
                     //   static (folds into later arithmetic)
t.shape(1); t.shape();  // aliases of extent(1) / extents()
t.extent(Int<-1>());    // negative axis: the last dimension
```

## `strides<...>` — the strides

The default layout is `ccontiguous` (C-order); strides derive from the extents.
For specific strides — a padded row, a channel-last view, a reversed axis — use
`strides<...>`, the stride analogue of `shape<...>`:

```cpp
tensor<float, shape<3,4>, strides<4,1>>(ptr);               // row stride 4 (padding), col 1
tensor<float, shape<-1,4>, strides<dynamic_stride,1>>(ptr, {n});  // outer runtime, inner 1
```

- Known strides fold to immediates; only the dynamic ones are stored. A
  fully-static `strides<...>` mapping is empty (a `strides<>` view is exactly a
  pointer).
- Strides are **signed**: `strides<-4,1>` is a real stride of −4 (a reversed
  view), not dynamic. A runtime stride is the sentinel `dynamic_stride`.
- `layout_static_stride<S...>` is a back-compat alias of `strides<S...>`.

Query a stride, static when derivable:

```cpp
t.stride(0);         // runtime
t.stride(Int<1>());  // static integral_constant for a strides<> layout, a
                     //   contiguous static shape, or a contiguous layout's
                     //   always-unit stride (even under a dynamic shape)
```

A `strides<...>` tensor is **fully sliceable**. Every view op — `operator()`
slicing, `take_along`, `permute`, `flip`, `squeeze`/`unsqueeze`, `peel` — builds
its view by hand (teeny does not call `cs::submdspan`), so it works on any source
layout including `strides<...>`, and folds the output strides the same way.
Slicing a contiguous static tensor keeps folded compile-time strides.

## The static/runtime idiom

The API accepts runtime integers, static integers (`integral_constant`), and
slices of either, and returns the matching output type. `alias.h` provides short
names — `Int<V>`, `Long<V>`, `Size<V>`, `UInt<V>`, `Int32<V>`, `Int64<V>`,
`Diff<V>`, `Bool<V>` — each converting implicitly to a runtime integer
and carrying `::value`.

Rule of thumb: pass `Int<k>()` to make the compiler fold; pass a plain
`int`/`long` when the value is only known at run time.

```cpp
t.extent(Int<0>());   // -> integral_constant<…,2>  (a constant, folds)
t.extent(runtime_d);  // -> int64_t                 (a value)
```

## The index type — `shape32` / `reindex`

Every offset computation runs in the extents' `index_type` (`int64_t` for
`shape<...>`, matching DLPack). At a **kernel boundary** you can narrow it to 32-bit
without a copy:

```cpp
auto v32 = t.reindex<int32_t>();     // free form: reindex<int32_t>(t)
```

`reindex` preserves the layout — same pointer and layout kind, with the extents and
any *dynamic* strides narrowed to the new width (a `strides<...>` literal pack is
unchanged). It's the index-width twin of `recast` (which recovers static extent
*values*): orthogonal, and they compose. `shape32<...>` == `shape_as<int32_t, ...>`
is the int32-indexed shape; `shape_as<Idx, ...>` picks any index type.

The payoff is on the device: a dynamic-stride view carries its strides by value, so
halving their width cuts the register footprint (rank-2: 40 → 24 bytes) and runs the
address math in 32-bit. Guard it with `t.index_fits<int32_t>()` — a signed-reach
check that handles negative-stride (flipped) and broadcast views — and only narrow
when the element span provably fits. See [Performance](performance.md).

### Mixing widths in a broadcast

When two operands of **different** index widths meet in a broadcast (`a + b`,
`a < b`, …), the result takes the **wider** of the two — an `int32`-indexed view
plus an `int64`-indexed one yields an `int64`-indexed result, in either order. This
is lossless (the broadcast engine already runs its offset math in the result's index
type) and avoids silently truncating the wide operand's strides to the narrow width.
Two equal-width operands are unchanged; note that broadening cannot rescue two `int32`
operands whose broadcast *span* overflows `int32` (an outer-product stretch) — that
stays the caller's concern, guarded by `index_fits`/`dispatch_index` at the boundary.
