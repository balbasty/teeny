# API cheat sheet

The whole surface on one screen. For type-annotated tables (inputs/outputs) see
the [Reference](reference.md); for every symbol's extracted signature see
[Autodoc](api/index.md).

Everything is in `namespace tny`. `namespace cs = cuda::std`. Include
`<teeny/teeny.h>` for all of it; `<teeny/cuda.h>` adds the CUDA memory spaces.

---

## The tensor type

```cpp
template <class T, class Shape, class Layout = ccontiguous, own O = own::view>
struct tensor;  // Shape = any cuda::std::extents (spell it shape<...>); Layout: ccontiguous/fcontiguous/...
```

One tensor type parameterised by element type, `cuda::std::extents`, an mdspan
layout, and ownership. Rarely named directly — use the aliases and factories
below. See [Tensors & ownership](tensors.md).

### Ownership aliases

```cpp
view<T, E, L = ccontiguous>       // non-owning view (default; the bare `tensor` is this)
local<T, E, L = ccontiguous>        // stack-owned (requires a fully static shape)
owned<T, E, L = ccontiguous>        // heap-owned, host, move-only
gpu<T, E, L>  pinned<T, E, L>  mapped<T, E, L>  // CUDA memory (from <teeny/cuda.h>)
```

### Factories

```cpp
wrap(ptr, shape);  wrap<Layout>(ptr, shape);  // a view (C-order / chosen layout)
wrap(ptr, shape, {s0, s1, ...});              // view with RUNTIME strides (dynamic_strides)
wrap(ptr, shape, strides<S...>{});           // view with COMPILE-TIME strides (fold into type)
as_tensor(any_mdspan);                        // wrap an mdspan/submdspan result

make_view(ptr, shape);           // alias of wrap that deduces the extents type
empty<T>(shape);                 // UNINITIALISED; deduces stack (static) / heap (dynamic)
empty<T, own::gpu>(shape);       // ...or name a backend: stack/heap/gpu/pinned/mapped
empty<T>(shape, own_c<own::gpu>{});   // value-tag backend form (same result)
make_local<T>(shape);  make_heap<T>(shape);            // thin spellings of empty<T,own::stack/heap>
make_gpu<T>(shape); make_pinned<T>(shape); make_mapped<T>(shape);   // empty<T,own::gpu/...>; T defaults to float

zeros<T>(shape);  ones<T>(shape);   // T defaults to float; static->stack, dyn->heap
full(shape, v);                     // element type = the VALUE's type (full<T>(...) to force)
arange<T>(n);                       // 1-D [0..n-1] (heap); T defaults to int64
arange<T, N>();  arange<T>(Int<N>());  // static 1-D [0..N-1] (stack, folds)
zeros<T, own::pinned>(shape);  arange<T>(n, own_c<own::pinned>{});   // ...or name a
                    // host-accessible backend (stack/heap/pinned/mapped); a gpu fill
                    // static_asserts -> use to<own::gpu>(zeros<T>(shape))
```

---

## Shapes & strides (`layout.h`, `alias.h`)

```cpp
template <auto... E>       using shape   = cs::extents<int64_t, E...>;  // -1 == dynamic
template <size_t N>        using rank    = shape<-1 ...N times>;        // fully-dynamic rank-N shape
template <int64_t... S>    struct strides;                // signed; dynamic_stride sentinel
template <int64_t... S>    using layout_static_stride = strides<S...>;  // back-compat alias
constexpr int64_t dynamic_stride;                         // a runtime stride
```

Static-integer aliases (each converts to a runtime integer and carries `::value`);
pass them where a static index/size is wanted, e.g. `x.extent(Int<0>())`:

```cpp
Int<0>{};  Long<3>{};  UInt<2>{};  Bool<true>{};  // classic:     Int Long Size UInt Diff Bool
Int32<4>{};  UInt64<8>{};                         // fixed-width: Int8/16/32/64  UInt8/16/32/64
I4<3>{};  U8<9>{};                                // numpy short (BYTES): I4==Int32, U8==UInt64
```

Element **dtype** aliases — numpy short codes, width in **bytes**:

```cpp
auto z = zeros<i4>(sh);        // i1 i2 i4 i8   u1 u2 u4 u8   (i4 == int32_t, u8 == uint64_t)
auto m = local<f4, shape<3>>{};// f4 == float, f8 == double, f2 == half, bf16 == bfloat16
```

See [Shapes & strides](shapes-strides.md).

---

## Geometry

```cpp
x.rank();  x.numel();  x.is_contiguous();
x.extent(d);          x.extent(Int<d>());  // runtime value / static integral_constant
x.shape(d);           x.shape();           // aliases of extent(d) / extents()
x.stride(d);          x.stride(Int<d>());  // static when derivable
x.data();  x.view();  x.mdspan();  x.extents();  x.mapping();
                      // view() = non-owning teeny view (gpu_view if device); mdspan() = raw cs::mdspan
```

---

## Indexing & slicing (`indexing.h`)

```cpp
x(i, j, k);                     // element access -> T& (negatives wrap)
x.at(i, j, k);                  // one element as a rank-0 VIEW (rank-0 <-> scalar, .item())
x(0, all, slice(1, 4));         // any slice arg -> a VIEW
x(1, ellipsis, 2);              // ellipsis = (rank - #other args) copies of `all`
x(ellipsis) = b;  x(0, all) = v;  // assign INTO a slice copies/fills (a = b rebinds)
slice(start, stop);  slice(start, stop, step);  // half-open range, optional (neg) step
none;  all;                     // open slice end (== python None); keep-axis marker
x.take_along<Axes...>(args...);  // bind named axes (negatives wrap), keep the rest
x.uget(i, j, k);  x.uat(i...);  x.uslice(0, slice(1,4));  x.uadd_at(v, i...);
                    // `u`-prefixed unchecked twins of ()/at/slice/add_at: skip the
                    // negative-index wrap for known-non-negative RUNTIME indices
                    // (per-call -DTNY_NO_NEGATIVE_INDEX). Same result type; static
                    // bounds still fold. A negative runtime index is then UB.
```

See [Indexing & slicing](indexing.md).

---

## Structure (views) (`axis.h`, `tensor.h`)

```cpp
x.permute<Perm...>();                 // reorder axes
x.flip<Ax>();                         // reverse an axis (negative-stride view)
x.unsqueeze<Ax>();  x.squeeze<Ax>();  // insert / drop a size-1 axis
x.reshape<NewExt...>();               // contiguous-view reshape (one -1 inferred)
x.flatten();                          // 1-D contiguous view
x.clone();                            // dense row-major OWNING copy
x.recast<NewExtents>();               // reinterpret with a more-static same-rank extents
x.to<T2>();                           // dtype convert (matching dtype -> no-copy borrow; else owning copy)
to<own::gpu>(x);                      // memory-space move: to<Space,ET,Force>(x) (from <teeny/cuda.h>)
```

Axis template arguments are signed (negatives count from the back). **Every** view
op — `operator()`/`take_along`/`peel` **and** `permute`/`flip`/`unsqueeze`/`squeeze`
— folds its output strides into a static `strides<...>` (compile-time where the
source strides are static), on any source layout. See [Views & structure](structure.md).

---

## nd-peel (iteration) (`iterate.h`)

```cpp
peel<Axes...>(x);       peel_at<Axes...>(x, i);  // peel named axes
peel_front<N>(x);       peel_front_at<N>(x, i);  // peel the first N axes
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

// reductions -> scalar (all axes). ACCUMULATE in the "reduce type" (double for
//   small floats float/double/half, item type for ints), then CAST the result to
//   the tensor's element type: sum(float)->float. A leading TYPE arg makes that
//   type BOTH accumulator and result: sum<double>(a), dot<double>(a,b).
sum(a); prod(a); max(a); min(a); mean(a); dot(a, b);   // sum<Acc>(a), mean<Acc>(a), ...
allclose(a, b, rtol=1e-5, atol=1e-8);  // |a-b| <= atol+rtol*|b| everywhere (broadcasts)
// axis reductions -> lower-rank tensor (named axes removed; negatives wrap). Same
//   rule: accumulate in reduce_type, result element type = the tensor's type;
//   sum<Acc, Axes...>(a) makes Acc accumulator AND result (leading TYPE = acc, int = axis).
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
as_anyrank(data, shape, stride, ndim);        // -> anyrank WRAPPING the arrays, NO copy
                                              //   (default; host only, arrays must outlive it)
as_anyrank(data, shape, stride, ndim, copy_meta);  // -> anyrank COPYING into an inline
                                              //   TNY_MAX_RANK store (device-passable)
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
| `-DTNY_HARDENED` | turn ON element-access bounds checks (OFF by default, like `mdspan`; the `u*` accessors always skip them; always off on device) |
| `-DNDEBUG` | strip the debug shape/precondition checks (`_TNY_CHECK`; on by default host-side, already off on device) |

---

This page is the quick scan. For type-annotated tables see the
[Reference](reference.md); for the Doxygen-extracted signatures of every symbol
see [Autodoc](api/index.md).
