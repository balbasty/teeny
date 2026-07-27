# mdspan vs teeny

teeny is built on `cuda::std::mdspan` (from NVIDIA's CCCL). The guide pages lead
with teeny's own numpy/pytorch-flavoured vocabulary, because most users never need
to know how it maps to mdspan. This page is for the readers who **do** know mdspan,
or who want the escape hatch to reach the raw `cuda::std` objects for interop.

## Vocabulary map

teeny renames the mdspan concepts to shorter, numpy-flavoured names. They are the
same types underneath — teeny's names are thin aliases.

| teeny | mdspan (`cuda::std`) | what it is |
|---|---|---|
| **shape** — `shape<...>` | `extents<int64_t, ...>` | the per-dimension sizes |
| `shape_as<Idx, ...>` / `shape32<...>` | `extents<Idx, ...>` | a shape with a chosen index type |
| `rank<N>` | `dextents<int64_t, N>` | a fully-dynamic rank-`N` shape |
| **layout** `ccontiguous` | `layout_right` | C-order (row-major); the default |
| **layout** `fcontiguous` | `layout_left` | F-order (column-major) |
| **layout** `dynamic_strides` | `layout_stride` | arbitrary runtime strides |
| **layout** `strides<S...>` | `layout_static_stride<S...>` (teeny's own) | per-dimension compile-time strides |

`ccontiguous`/`fcontiguous`/`dynamic_strides` are literal `using`-aliases of the
mdspan layouts, so a teeny tensor with one of those layouts *is* an mdspan-layout
tensor. The one thing mdspan lacks is strides baked into the type; teeny adds that
as `strides<S...>`, and exposes the same idea under the mdspan-shaped name
`layout_static_stride<S...>` (a back-compat alias of `strides<S...>`).

## Accessors: teeny spelling vs raw mdspan

teeny's geometry accessors are array-like: indexable with a static `Int<k>()` (which
folds to an `integral_constant`) or a runtime integer, iterable, and rank-aware. The
raw mdspan objects are still reachable when you need them.

| teeny (primary) | raw mdspan (escape hatch) | notes |
|---|---|---|
| `t.shape()` | `t.extents()` | the whole shape / the raw `cs::extents` |
| `t.shape(d)` | `t.extent(d)` | per-axis size; the two are equal (`t.extent(d) == t.shape(d)`) |
| `t.strides()` | — (via `t.mapping()`) | the whole strides as an array-like accessor |
| `t.stride(d)` | `t.mapping().stride(d)` | per-axis stride |
| — | `t.mapping()` | the raw layout mapping |
| — | `t.mdspan()` | the raw `cuda::std::mdspan<T, E, L>` |

Reach for `extent` / `extents` / `mapping` / `mdspan()` only when you actually need
the underlying mdspan types — for example to hand a tensor to code that already
speaks mdspan.

## Wrapping an mdspan as a teeny tensor

If you have a `cuda::std::mdspan` (or a `submdspan` result), wrap it as a teeny view
with `as_tensor`, so the whole teeny API applies to it:

```cpp
cs::mdspan<double, cs::dextents<int64_t,2>> md = /* ... */;
auto t = as_tensor(md);   // a non-owning teeny view over the same memory
```

## Why teeny builds its views by hand

Every view-making op — `operator()` slicing, `take_along`, `permute`, `flip`,
`squeeze`/`unsqueeze`, `peel` — builds its result view directly rather than calling
`cs::submdspan`. Two reasons:

- **It folds the output strides.** The result layout is teeny's `strides<...>`, with
  each kept stride folded to a compile-time constant wherever it is derivable. So
  slicing a static tensor keeps compile-time strides instead of falling back to
  runtime ones.
- **It works on any source layout.** `cs::submdspan` does not accept teeny's
  `strides<...>` (`layout_static_stride`) layout, but teeny's hand-built gather does,
  so a `strides<...>` tensor is fully sliceable — the same as any other layout.

The upshot is that teeny does not call `cs::submdspan` at all; it does the offset and
stride bookkeeping itself so that staticity is preserved through a chain of views.
