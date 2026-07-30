# Shapes & strides

Each dimension's size and stride may be static (a compile-time constant) or
dynamic (a run-time value) — independently. Static values fold into machine
code; dynamic ones are carried at run time. One kernel source covers every
combination.

## `shape<...>` — the sizes

`shape<...>` lists the per-dimension sizes of a tensor. Its `int64_t` index type
matches DLPack's, so it drops onto ndarray bindings directly. (Under the hood it is
teeny's spelling of `cuda::std::extents<int64_t, ...>` — see
[mdspan vs teeny](mdspan-vs-teeny.md).)

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
auto d = wrap(ptr, rank<2>{r, k});       // rank<N> is the fully-dynamic rank-N shape
```

`rank<N>` is the fully-dynamic shape of rank `N`, so `rank<2>` is exactly
`shape<-1,-1>`. (There is no `rank{r, k}` / `shape{r, k}` shortcut: class-template
argument deduction does not apply to alias templates in C++17, so you always give
the rank — `rank<2>{r, k}` or `shape<-1,-1>{r, k}`.)

Query sizes with `shape`:

```cpp
t.rank();          // number of dimensions (static)
t.numel();         // total element count
t.shape();         // the whole shape as an array-like accessor
t.shape(1);        // RUNTIME lookup -> the size of axis 1 (an index_type value)
t.shape(Int<0>()); // STATIC lookup -> an integral_constant when that axis is static
                   //   (so it folds into later arithmetic)
t.shape(Int<-1>()); // negative axis: the last dimension
```

(These are teeny's numpy/pytorch-flavoured spellings. The underlying `mdspan`
spellings — `t.extent(d)`, `t.extents()` — still exist as an interop escape hatch;
see [mdspan vs teeny](mdspan-vs-teeny.md).)

## `strides<...>` — the strides

The default layout is `ccontiguous` (C-order). In this case, strides are computed
automatically from the shape and do not need to be provided. For specific strides —
a padded row, a channel-last view, a reversed axis — use `strides<...>`, the stride
analogue of `shape<...>`:

```cpp
auto a = wrap(ptr, shape<3,4>{}, strides<4,1>{});               // row stride 4 (padding), col 1
auto b = wrap<dynamic_stride,1>(ptr, shape<-1,4>{n}, {4});      // outer runtime, inner 1 (folds)
```

- Known strides become compile-time constants baked into the code, and only the
  dynamic ones are stored. A fully-static `strides<...>` mapping is empty, so a
  `strides<>` view is exactly a pointer.
- Strides are **signed**: `strides<-4,1>` is a real stride of −4 (a reversed
  view), not dynamic. A runtime stride is the sentinel `dynamic_stride`.

Query a stride, static when derivable:

```cpp
t.stride(0);         // runtime
t.stride(Int<1>());  // a compile-time constant for a strides<> layout, a
                     //   contiguous static shape, or a contiguous layout's
                     //   always-unit stride (even under a dynamic shape)
```

A `strides<...>` tensor is **fully sliceable**. Every view op — `operator()`
slicing, `slice_along`, `permute`, `flip`, `squeeze`/`unsqueeze`, `peel` — works on
any source layout including `strides<...>`, and folds the output strides the same
way. Slicing a contiguous static tensor keeps folded compile-time strides. (If you
know mdspan, the [mdspan vs teeny](mdspan-vs-teeny.md) page explains how `strides<>`
relates to `layout_static_stride` and why teeny builds these views by hand.)

Those views are also **usable from CUDA device code**: a `strides<...>` view passes
into a kernel by value and indexes there exactly like a contiguous one, so you can
slice on the host and hand the slice to a kernel, or slice again inside it.

## The static/runtime idiom

The API accepts runtime integers, static integers (`integral_constant`), and
slices of either, and returns the matching output type. `alias.h` provides short
names for these static integers: `Int<V>`, `Long<V>`, `Size<V>`, `UInt<V>`,
`Int32<V>`, `Int64<V>`, `Diff<V>`, `Bool<V>`. Each static type converts implicitly
to a runtime value and carries a `::value`.

Rule of thumb: pass `Int<k>()` to make the compiler fold; pass a plain
`int`/`long` when the value is only known at run time.

```cpp
t.extent(Int<0>());   // -> integral_constant<…,2>  (a constant, folds)
t.extent(runtime_d);  // -> int64_t                 (a value)
```

## The index type — `shape32` / `reindex`

Every offset computation runs in the shape's `index_type` (`int64_t` for
`shape<...>`, matching DLPack). At a **kernel boundary** you can narrow it to 32-bit
without a copy:

```cpp
auto v32 = t.reindex<int32_t>();     // free form: reindex<int32_t>(t)
```

`reindex` preserves the layout — same pointer and layout kind, with the shape and
any *dynamic* strides narrowed to the new width (a `strides<...>` literal pack is
unchanged). It's the index-width twin of `recast` (which recovers static shape
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

The two-operand reductions `dot(a, b)` and `sqdist(a, b)`/`dist(a, b)` follow the
same rule: they produce a scalar rather than a tensor, but their offset math also
runs in a type that covers both operands, so mixing an `int32`-indexed operand with
an `int64`-indexed one is safe in either order — and, as below, so is mixing
signedness. So does `allclose(a, b)`, which likewise walks both operands and hands
back a plain `bool`.

The **in-place** ops (`a.add_(b)`, `a.copy_(b)`, `a *= b`, …) and a caller-supplied
`into(dest)` are the one place where the destination cannot simply take the wider
type — `a` is *your* tensor, and its index width is part of its own type. There the
offset math runs in the widest of the types in play (the destination's and every
operand's), while every tensor keeps its own: the widening is internal, so
`a.add_(b)` still hands you back `a` with `a`'s index type unchanged. Mixing widths
in place is safe in either direction — a narrow-indexed destination combined with a
wide-indexed right-hand side does not truncate that operand's strides, and a
wide-indexed destination with a narrow right-hand side was never at risk.

That covers **every** producer that takes an `into(dest)`, not only the two-operand
elementwise ones: a scalar right-hand side (`a.mul(2.0, into(y))`,
`minimum(a, 1.0, into(y))`) and a unary op (`exp(a, into(y))`, `clamp(a, lo, hi,
into(y))`) each walk a source *and* a destination, so they too run their offsets in a
type covering both. Called **without** `into(dest)`, those producers allocate their
result from the source's own shape type, so the widths already agree and there is
nothing to widen.

Mixing **signedness** is safe too. Width alone does not settle how those offsets are
computed: `shape_as<Idx, ...>` accepts an unsigned index type, while a flipped or
negative-step view has negative strides (which is exactly why such a view needs a
*signed* index type). Where an unsigned-indexed tensor meets a signed-indexed one, the
shared offset math runs in a signed type wide enough for both sides, so a stride of
`-1` stays `-1` instead of turning into a huge positive offset. As above this is
internal — every tensor keeps its own index type — and the ordinary case, where every
tensor in the expression shares one signedness, computes exactly as it did before.

This holds everywhere two or more tensors share one pass: the elementwise broadcast
ops above, and equally `dot`/`sqdist`/`dist`, where a flipped (reversed) signed-indexed
operand next to an unsigned-indexed one is safe in either operand order.
