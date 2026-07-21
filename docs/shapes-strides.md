# Shapes & strides

teeny's defining feature: **each dimension's size and stride may be static
(a compile-time constant) or dynamic (a run-time value) — independently.** Static
values fold into the machine code; dynamic ones are carried at run time. One
kernel source covers every combination.

## `shape<...>` — the sizes

`shape<...>` is the extents type. It's `cuda::std::extents<int64_t, ...>` — the
`int64_t` index type matches DLPack's `shape` exactly, so it drops onto ndarray
bindings.

```cpp
shape<2,3,4>            // fully static 2×3×4
shape<-1,3>            // dynamic rows, static 3 columns   (-1 == dynamic)
shape<dynamic_extent,3>// the same thing, spelled out
```

A dynamic dimension is written **`-1`** (numpy-style) or `dynamic_extent`; they
mean the same type. Construct a tensor by supplying only the dynamic sizes:

```cpp
auto a = view(ptr, shape<2,3,4>{});      // nothing to supply — all static
auto b = view(ptr, shape<-1,3>{n});      // supply the one dynamic size, n
auto c = view(ptr, shape<-1,-1>{r, k});  // supply both
```

Query sizes with `extent` / `shape` (a python-friendly alias):

```cpp
t.rank();                 // number of dimensions (static)
t.numel();                // total element count
t.extent(0);              // a RUNTIME lookup -> index_type
t.extent(Int<0>());       // a STATIC lookup -> integral_constant when that
                          //   extent is static (folds into later arithmetic)
t.shape(1);  t.shape();   // aliases of extent(1) / extents()
t.extent(Int<-1>());      // negative axis: the last dimension
```

## `strides<...>` — the strides

By default the layout is `layout_right` (C-order); strides are derived from the
extents. When you need **specific** strides — a padded row, a channel-last view,
a reversed axis — use `strides<...>`, the stride analogue of `shape<...>`:

```cpp
tensor<float, shape<3,4>, strides<4,1>>(ptr);   // row stride 4 (padding), col 1
tensor<float, shape<-1,4>, strides<dynamic_stride,1>>(ptr, {n});  // outer runtime, inner 1
```

- Known strides **fold to immediates**; only the dynamic ones are stored, and a
  fully-static `strides<...>` mapping is **empty** (a `strides<>` view is exactly
  a pointer).
- Strides are **signed**: `strides<-4,1>` is a real stride of −4 (a reversed
  view), *not* dynamic. A runtime stride is the sentinel `dynamic_stride`.

Query a stride, static when it can be:

```cpp
t.stride(0);          // runtime
t.stride(Int<1>());   // static integral_constant for a strides<> layout,
                      //   a contiguous static shape, or a contiguous layout's
                      //   ALWAYS-unit stride even under a dynamic shape
```

!!! warning "`submdspan` and `strides<...>`"
    CCCL only defines `submdspan` for the standard layouts, so slicing / `peel` /
    `take_along` / `permute` do **not** apply to a `strides<...>` tensor. Use it
    for whole-tensor access with folded strides; use `layout_right`/`layout_left`
    /`layout_stride` when you need to slice.

## The static/runtime idiom

The API accepts **runtime integers, static integers (`integral_constant`), and
slices of either**, and returns the right output type. `alias.h` provides short
names — `Int<V>`, `Long<V>`, `Size<V>`, `Int32<V>`, `Int64<V>`, `Bool<V>`,
`ic<V>` — each of which converts implicitly to a runtime integer *and* carries
`::value`.

**Rule of thumb:** pass `Int<k>()` when you want the compiler to fold; pass a
plain `int`/`long` when the value is only known at run time.

```cpp
t.extent(Int<0>());   // -> integral_constant<…,2>  (a constant, folds)
t.extent(runtime_d);  // -> int64_t                 (a value)
```
