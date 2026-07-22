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
layout, and ownership. Rarely named directly — use the aliases and factories
below. See [Tensors & ownership](tensors.md).

### Ownership aliases

```cpp
view_t<T, E, L = layout_right>       // non-owning view (default; the bare `tensor` is this)
local<T, E, L = layout_right>        // stack-owned (requires a fully static shape)
owned<T, E, L = layout_right>        // heap-owned, host, move-only
gpu<T, E, L>  pinned<T, E, L>  mapped<T, E, L>  // CUDA memory (from <teeny/cuda.h>)
```

### Factories

```cpp
wrap(ptr, shape);  wrap<Layout>(ptr, shape);  // a view (C-order / chosen layout)
wrap_strided<S...>(ptr, shape);               // view with compile-time strides
as_tensor(any_mdspan);                        // wrap an mdspan/submdspan result

make_view(ptr, shape);           // deduce the extents type
make_local<T>(shape);  make_heap<T>(shape);            // T defaults to float
make_gpu<T>(shape); make_pinned<T>(shape); make_mapped<T>(shape);   // T defaults to float

zeros<T>(shape);  ones<T>(shape);   // T defaults to float; static->stack, dyn->heap
full(shape, v);                     // element type = the VALUE's type (full<T>(...) to force)
arange<T>(n);                       // 1-D [0..n-1] (heap); T defaults to int64
arange<T, N>();  arange<T>(Int<N>());  // static 1-D [0..N-1] (stack, folds)
```

---

## Shapes & strides (`layout.h`, `alias.h`)

```cpp
template <auto... E>       using shape   = cs::extents<int64_t, E...>;  // -1 == dynamic
template <int64_t... S>    struct strides;                // signed; dynamic_stride sentinel
template <int64_t... S>    using layout_static_stride = strides<S...>;  // back-compat alias
constexpr int64_t dynamic_stride;                         // a runtime stride
```

Static-integer aliases (each converts to a runtime integer and carries `::value`):

```cpp
Int<V> Long<V> Size<V> Uint<V> Diff<V> Bool<V> ic<V>          // classic
Int8/16/32/64<V>  Uint8/16/32/64<V>                          // fixed-width
I1 I2 I4 I8  U1 U2 U4 U8   // numpy short forms (BYTES): I4<V> == Int32<V>
```

Element **dtype** aliases (numpy short codes; width in bytes, `local<f4, shape<3>>`):

```cpp
i1 i2 i4 i8   u1 u2 u4 u8   f4(float) f8(double)   f2(half) bf16(bfloat16)
```

See [Shapes & strides](shapes-strides.md).

---

## Geometry

```cpp
t.rank();  t.numel();  t.is_contiguous();
t.extent(d);          t.extent(Int<d>());  // runtime value / static integral_constant
t.shape(d);           t.shape();           // aliases of extent(d) / extents()
t.stride(d);          t.stride(Int<d>());  // static when derivable
t.data();  t.view();  t.extents();  t.mapping();
```

---

## Indexing & slicing (`indexing.h`)

```cpp
t(i, j, k);                     // element access -> T& (negatives wrap)
t.at(i, j, k);                  // one element as a rank-0 VIEW (rank-0 <-> scalar, .item())
t(0, all, slice(1, 4));         // any slice arg -> a VIEW
t(1, ellipsis, 2);              // ellipsis = (rank - #other args) copies of `all`
t(ellipsis) = b;  t(0, all) = v;  // assign INTO a slice copies/fills (a = b rebinds)
slice(start, stop);  slice(start, stop, step);  // half-open range, optional (neg) step
none;  all;                     // open slice end (== python None); keep-axis marker
t.take_along<Axes...>(args...);  // bind named axes (negatives wrap), keep the rest
```

See [Indexing & slicing](indexing.md).

---

## Structure (views) (`axis.h`, `tensor.h`)

```cpp
t.permute<Perm...>();                 // reorder axes
t.flip<Ax>();                         // reverse an axis (negative-stride view)
t.unsqueeze<Ax>();  t.squeeze<Ax>();  // insert / drop a size-1 axis
t.reshape<NewExt...>();               // contiguous-view reshape (one -1 inferred)
t.flatten();                          // 1-D contiguous view
t.clone();                            // dense row-major OWNING copy
t.recast<NewExtents>();               // reinterpret with a more-static same-rank extents
```

Axis template arguments are signed (negatives count from the back). Every view op
folds output strides and works on any source layout (incl. `strides<...>`). See
[Views & structure](structure.md).

---

## nd-peel (iteration) (`iterate.h`)

```cpp
peel<Axes...>(t);       peel_at<Axes...>(t, i);  // peel named axes
peel_front<N>(t);       peel_front_at<N>(t, i);  // peel the first N axes
```

See [Views & structure](structure.md#nd-peel-iterate-a-subset-of-axes).

---

## Math (`math.h`)

```cpp
// in-place (broadcasts tensor rhs; also scalar rhs). add_/sub_ take a bool
// Atomic flag: add_<true>(x) commits with fetch_add (atomic on device).
a.add_(x); a.sub_(x); a.mul_(x); a.div_(x);   a.add_<true>(x); a.sub_<true>(s);
a += x; a -= s; a *= x; a /= s;  // compound-assign
++a; --a; auto old = a++;        // prefix in place; postfix (static) -> stack copy
a.neg_(); a.abs_(); a.exp_(); a.log_(); a.sin_(); a.cos_(); a.sqrt_(); a.tanh_();
a.floor_(); a.ceil_(); a.round_(); a.trunc_(); a.sign_(); a.pow_(e); a.clamp_(lo, hi);
a & b; a | b; a ^ b; ~a; a &= b; a |= s;  // bitwise (INTEGER element types only)
a.fill_(v); a.zero_(); a.copy_(b); a.iota_(start, step);
a.map_(f); a.zip_with_(g, b);  auto c = a.map(f);  // user functor (device-safe)
a.add_at(v, i...);  fetch_add(ptr, v);             // scatter (atomic on device)

// out-of-place -> new tensor (promotes types; static->stack, dyn->heap)
auto c = a + b;  a.add(b);  a * 2.0;  2.0 - a;  1.0 / a;  -a;  a.pow(b);
auto c = neg(a); abs(a); exp(a); log(a); sin(a); cos(a); sqrt(a); tanh(a);
auto c = floor(a); ceil(a); round(a); trunc(a); sign(a);
auto c = minimum(a, b); maximum(a, s); clamp(a, lo, hi);

// reductions -> scalar (all axes). Accumulate in (and return) the "reduce type":
//   double for small floats (float/double/half), item type for ints. Override
//   the accumulator with a leading TYPE arg: sum<float>(a), dot<float>(a,b).
sum(a); prod(a); max(a); min(a); mean(a); dot(a, b);   // sum<Acc>(a), mean<Acc>(a), ...
allclose(a, b, rtol=1e-5, atol=1e-8);  // |a-b| <= atol+rtol*|b| everywhere (broadcasts)
// axis reductions -> lower-rank tensor (named axes removed; negatives wrap). The
//   result element type is the accumulator. sum<Axes...>(a) uses the default;
//   sum<Acc, Axes...>(a) forces it (a leading TYPE = accumulator, a leading int = axis).
sum<Axes...>(a); prod<...>(a); max<...>(a); min<...>(a); mean<...>(a);  // sum<Acc,Axes...>(a)
```

Promotion: C++ rules but lower-width float wins (`-DTNY_STD_PROMOTION` opts out).
See [Math & broadcasting](math.md).

---

## Half precision (`half.h`)

```cpp
half;  bfloat16;                      // native __half/__nv_bfloat16 under nvcc
compute_type<T>;  compute_type_t<T>;  // half -> float; else T
```

See [Half precision](half.md).

---

## Dispatch (the ndarray boundary) (`dynamic.h`)

```cpp
dispatch_value<Vs...>(v, f);            // runtime value -> integral_constant
as_anyrank(data, shape, stride, ndim);  // -> anyrank; COPIES into an inline
                                        //   TNY_MAX_RANK store (device-passable)
as_anyrank_view(data, shape, stride, ndim);  // -> anyrank wrapping the arrays, NO copy
                                             //   (host only; arrays must outlive it)
at.peel_front<Sr>();  at.peel_front_at<Sr>(i);  // batch idiom: one kernel per Sr
dispatch_rank(at, f);                    // runtime rank -> fixed-rank view (per total rank)
at.fixed<R>();                           // force a known rank
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

The headers carry Doxygen `@brief` comments, so `doxygen` produces a full
XML/HTML reference; for a Markdown/MkDocs workflow, feed that to
[`doxybook2`](https://github.com/matusnovak/doxybook2) or
[`moxygen`](https://github.com/sourcey/moxygen). This page is hand-curated
because the surface is small.
