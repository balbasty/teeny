# API reference

Everything is in `namespace tny`. `namespace cs = cuda::std`. Include
`<teeny/teeny.h>` for all of it; `<teeny/cuda.h>` adds the CUDA memory spaces.

---

## The tensor type

```cpp
template <class T, class Extents, class Layout = layout_right, own O = own::view>
struct tensor;
```

One tensor type parameterised by element type, `cuda::std::extents`, an mdspan
layout, and ownership. You rarely name it directly — use the aliases and
factories below. See [Tensors & ownership](tensors.md).

### Ownership aliases

```cpp
view_t<T, E, L = layout_right>   // non-owning view (default; the bare `tensor` is this)
local<T, E, L = layout_right>    // stack-owned (requires a fully static shape)
owned<T, E, L = layout_right>    // heap-owned, host, move-only
device<T, E, L>  host<T, E, L>  pinned<T, E, L>   // CUDA memory (from <teeny/cuda.h>)
```

### Factories

```cpp
view(ptr, extents);  view<Layout>(ptr, extents);   // a view (C-order / chosen layout)
view_strided<S...>(ptr, extents);                  // view with compile-time strides
as_tensor(any_mdspan);                             // wrap an mdspan/submdspan result

make_view(ptr, extents);                           // deduce the extents type
make_local<T>(extents);  make_heap<T>(extents);
make_device<T>(extents); make_host<T>(extents); make_pinned<T>(extents);

zeros<T>(extents);  ones<T>(extents);  full<T>(extents, v);   // static->stack, dyn->heap
arange<T>(n);                                                 // 1-D [0..n-1] (heap)
```

---

## Shapes & strides

```cpp
template <auto... E>       using shape   = cs::extents<int64_t, E...>;  // -1 == dynamic
template <int64_t... S>    struct strides;                             // signed; dynamic_stride sentinel
template <int64_t... S>    using layout_static_stride = strides<S...>;  // back-compat alias
constexpr int64_t dynamic_stride;                                      // a runtime stride
```

Static-integer aliases (compile-time indices/extents; each converts to a runtime
integer and carries `::value`):

```cpp
Int<V> Long<V> Size<V> Uint<V> Int32<V> Int64<V> Diff<V> Bool<V> ic<V>
```

See [Shapes & strides](shapes-strides.md).

---

## Geometry

```cpp
t.rank();  t.numel();  t.is_contiguous();
t.extent(d);          t.extent(Int<d>());     // runtime value / static integral_constant
t.shape(d);           t.shape();              // aliases of extent(d) / extents()
t.stride(d);          t.stride(Int<d>());     // static when derivable
t.data();  t.view();  t.extents();  t.mapping();
```

---

## Indexing & slicing

```cpp
t(i, j, k);                     // element access (negatives wrap)
t(0, all, slice(1, 4));         // any slice arg -> a VIEW
slice(start, stop);  slice(start, stop, step);   // half-open range, optional (neg) step
none;  all;                     // open slice end (== python None); keep-axis marker
t.take_along<Axes...>(args...); // bind named axes (negatives wrap), keep the rest
```

See [Indexing & slicing](indexing.md).

---

## Structure (views)

```cpp
t.permute<Perm...>();           // reorder axes
t.flip<Ax>();                   // reverse an axis (negative-stride view)
t.unsqueeze<Ax>();  t.squeeze<Ax>();          // insert / drop a size-1 axis
t.reshape<NewExt...>();         // contiguous-view reshape (one -1 inferred)
t.flatten();                    // 1-D contiguous view
t.clone();                      // dense row-major OWNING copy
t.recast<NewExtents>();         // reinterpret with a more-static same-rank extents
```

Axis template arguments are signed (negatives count from the back). See
[Views & structure](structure.md).

---

## nd-peel (iteration)

```cpp
peel<Axes...>(t);       peel_at<Axes...>(t, i);        // peel named axes
peel_front<N>(t);       peel_front_at<N>(t, i);        // peel the first N axes
batch_offset(md, lin);  channel(md, c);                // low-level offset helpers
```

See [Views & structure](structure.md#nd-peel-iterate-a-subset-of-axes).

---

## Math

```cpp
// in-place (broadcasts tensor rhs; also scalar rhs)
a.add_(x); a.sub_(x); a.mul_(x); a.div_(x);
a.neg_(); a.abs_(); a.exp_(); a.log_(); a.sin_(); a.cos_(); a.sqrt_(); a.tanh_(); a.pow_(e);
a.fill_(v); a.zero_(); a.copy_(b); a.iota_(start, step);
a.map_(f); a.zip_with_(g, b);  auto c = a.map(f);       // user functor (device-safe)
a.add_at(v, i...);  fetch_add(ptr, v);                  // scatter (atomic on device)

// out-of-place -> new tensor (promotes types; static->stack, dyn->heap)
auto c = a + b;  auto c = a.add(b);  auto c = a * 2.0;  auto c = a.pow(b);
auto c = neg(a); abs(a); exp(a); log(a); sin(a); cos(a); sqrt(a); tanh(a);

// reductions -> scalar
sum(a); prod(a); max(a); min(a); dot(a, b);
```

Promotion: C++ rules but lower-width float wins (`-DTNY_STD_PROMOTION` opts out).
See [Math & broadcasting](math.md).

---

## Half precision

```cpp
half;  bfloat16;                          // native __half/__nv_bfloat16 under nvcc
compute_type<T>;  compute_type_t<T>;      // half -> float; else T
```

See [Half precision](half.md).

---

## Dispatch (the ndarray boundary)

```cpp
dispatch_value<Vs...>(v, f);              // runtime value -> integral_constant
any(data, shape, stride, ndim);           // -> any_tensor (rank-erased, bounded)
dispatch_rank(any_tensor, f);             // runtime rank -> fixed-rank view
any_tensor.fixed<R>();                    // force a known rank
```

See [Dispatch & the ndarray boundary](dispatch.md).

---

## Compile flags

| flag | effect |
|---|---|
| `-DTNY_STD_PROMOTION` | standard C++ float promotion (wider wins) instead of lower-wins |
| `-DTNY_NO_NEGATIVE_INDEX` | drop python-style negative-index wrap from `operator()` (tightest codegen) |
| `-DTNY_PORTABLE_HALF` | force the portable software `half`/`bfloat16` even under nvcc |
| `-DNDEBUG` | strip debug shape/precondition checks (host-only, already off on device) |

---

## Generating this reference

The headers carry Doxygen `@brief` comments, so `doxygen` produces a full XML/HTML
reference out of the box; for a Markdown/MkDocs workflow, feed that to
[`doxybook2`](https://github.com/matusnovak/doxybook2) or
[`moxygen`](https://github.com/sourcey/moxygen). This page is hand-curated
because the surface is small and a guided reference reads better than a dumped
one.
