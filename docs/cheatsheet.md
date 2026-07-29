# API cheat sheet

The whole surface on one screen. For type-annotated tables (inputs/outputs) see
the [Reference](reference.md); for every symbol's extracted signature see
[Autodoc](api/index.md).

Everything is in `namespace tny`. `namespace cs = cuda::std`. Include
`<teeny/teeny.h>` for all of it; `<teeny/cuda.h>` adds the CUDA memory spaces.

---

## The tensor type

```cpp
template <class T, class Shape, class Layout = ccontiguous, storage O = storage::view>
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
wrap(ptr, shape, fcontiguous{});               // value-tag layout (deduced, no .template)
wrap(ptr, shape, {s0, s1, ...});              // view with RUNTIME strides (dynamic_strides)
wrap(ptr, shape, strides<S...>{});           // view with COMPILE-TIME strides (fold into type)
as_tensor(any_mdspan);  wrap(any_mdspan);      // wrap an mdspan/submdspan result (same thing)

make_view(ptr, shape);           // alias of wrap that deduces the extents type
empty<T>(shape);                 // UNINITIALISED (np.empty); deduces stack (static) / heap (dynamic)
                                 //   (a value-initialised local<...>{} / owned(e) stays zeroed; empty opts out)
empty<T, storage::gpu>(shape);       // ...or name a backend: stack/heap/gpu/pinned/mapped
empty<T>(shape, storage_c<storage::gpu>{});   // value-tag backend form (same result)
empty(shape, dtype<T>{});            // value-tag ELEMENT-TYPE form: deduces T, no .template needed.
                                     //   O stays a leading explicit template arg: empty<storage::gpu>(shape, dtype<T>{})
empty(shape, fcontiguous{}, dtype<double>{});   // a layout tag composes with dtype/storage_c, any
                                                 //   order/subset (also zeros/ones/full; arange has no
                                                 //   layout keyword, #277-#281)
make_local<T>(shape);  make_heap<T>(shape);            // thin spellings of empty<T,storage::stack/heap>
make_gpu<T>(shape); make_pinned<T>(shape); make_mapped<T>(shape);   // empty<T,storage::gpu/...>; T defaults to float

zeros<T>(shape);  ones<T>(shape);   // T defaults to float; static->stack, dyn->heap
full(shape, v);                     // element type = the VALUE's type (full<T>(...) to force)
arange<T>(n);                       // 1-D [0..n-1] (heap); T defaults to int64
arange<T, N>();  arange<T>(Int<N>());  // static 1-D [0..N-1] (stack, folds)
zeros<T, storage::pinned>(shape);  arange<T>(n, storage_c<storage::pinned>{});   // ...or name a
                    // host-accessible backend (stack/heap/pinned/mapped); a gpu fill
                    // static_asserts -> use to<storage::gpu>(zeros<T>(shape))
zeros(shape, dtype<T>{});  full(shape, v, dtype<T>{});  arange(n, dtype<T>{});   // value-tag T (also ones)
zeros(shape, dtype<T>{}, storage_c<S>{});  // ...or compose BOTH tags, either order (also
                    //   storage_c<S>{}, dtype<T>{}) — no explicit template arg at all (also empty/ones/full)
```

---

## Shapes & strides (`layout.h`, `alias.h`)

```cpp
template <auto... E>       using shape   = cs::extents<int64_t, E...>;  // -1 == dynamic
template <class Idx, auto... E> using shape_as = cs::extents<Idx, E...>; // shape<> with a chosen index type
template <auto... E>       using shape32 = shape_as<int32_t, E...>;     // int32-indexed boundary view
template <size_t N>        using rank    = shape<-1 ...N times>;        // fully-dynamic rank-N shape
template <int64_t... S>    struct strides;                // signed; dynamic_stride sentinel
template <int64_t... S>    using layout_static_stride = strides<S...>;  // back-compat alias
constexpr int64_t dynamic_stride;                         // a runtime stride
```

`t.reindex<int32_t>()` (free: `reindex<int32_t>(t)`) retypes the offset index width
without a copy (layout preserved; extents/dynamic strides narrowed) — halves a
dynamic view's footprint and runs offset math in 32-bit at the kernel boundary.
`t.index_fits<int32_t>()` is the signed-reach guard. See [Performance](performance.md).

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
x.rank();  x.numel();  x.is_contiguous();  x.is_dense();  // is_contiguous = C-order; is_dense = any order
x.shape();            x.shape(d);          // the teeny SHAPE: array-like accessor / size of axis d
x.strides();          x.stride(d);         // the teeny STRIDES: array-like accessor / stride of axis d
x.shape()[Int<k>()];  x.strides()[Int<k>()];  // static index folds to integral_constant; runtime [i] is a value
x.data();  x.view();                       // base ptr / non-owning teeny view (gpu_view if device)
// raw-mdspan escape hatch (interop only):
x.mdspan();  x.extents();  x.mapping();     // raw cs::mdspan / raw cs::extents / layout mapping
x.extent(d);  x.extent(Int<d>());          // mdspan-side per-axis size (== x.shape(d)); static Int<d> folds
```

!!! note "mdspan equivalent"
    `x.shape()`/`x.strides()` are teeny's array-like accessors and the primary
    spelling; `x.extents()`/`x.mapping()` return the raw `cuda::std` mdspan
    objects. Per-axis, `x.extent(d) == x.shape(d)`. See
    [mdspan vs teeny](mdspan-vs-teeny.md) for the full vocabulary map.

---

## Indexing & slicing (`indexing.h`)

```cpp
x(i, j, k);                     // element access -> T& (negatives wrap)
x.at(i, j, k);                  // one element as a rank-0 VIEW (rank-0 <-> scalar, .item())
x(0, all, slice(1, 4));         // any slice arg -> a VIEW
x(1, ellipsis, 2);              // ellipsis = (rank - #other args) copies of `all`
x(1, etc, 2);                   // `etc` == `ellipsis` (one marker, two names)
x(ellipsis) = b;  x(0, all) = v;  // assign INTO a slice copies/fills (a = b rebinds)
slice(start, stop);  slice(start, stop, step);  // half-open range, optional (neg) step
none;  all;                     // open slice end (== python None); keep-axis marker
x(none, all, all);  x(0, newaxis, all);   // BARE none/newaxis arg -> insert a size-1
                    //   axis (== unsqueeze at that position); newaxis is an alias of none
x.take_along<Axes...>(args...);  // bind named axes (negatives wrap), keep the rest
x.subsample<Axes...>(k, starts...);  // coloured/strided sub-lattice: take_along + slice(start,
                    //   none,k) per named axis, one shared step k, per-axis start; k/starts
                    //   accept runtime or Int<> (folds static). Value form: x.subsample(
                    //   axis<Axes...>{}, k, starts...) -- same LEADING placement as take_along's
x.unfold<Axis>(size, step);  x.unfold<Axis>(size);  // pytorch Tensor.unfold: appends a NEW
                    //   trailing axis of width `size` stepped by `step` along Axis (step
                    //   defaults to 1); Axis's own extent shrinks to the window COUNT
                    //   (shape(Axis)-size)/step+1. size/step accept runtime or Int<> (folds
                    //   static); size in [1,shape(Axis)] and step>=1, checked (static_assert
                    //   or debug-time). ND windows compose by chaining: x.unfold<0>(k0,s0).unfold<1>(k1,s1).
                    //   Value form: x.unfold(Int<Axis>(), size, step) -- single-axis Int<k>()
                    //   selector (like flip/squeeze), not an axis<...> list (only ONE axis binds).
x.index_select<Axis>(idx);       // gather along Axis by a rank-1 integer index TENSOR
                    //   (runtime data, unlike take_along); idx values wrap negative;
                    //   static idx shape -> stack result, else heap (source must be
                    //   host-accessible). into(dest) form too: x.index_select<Axis>(idx,
                    //   into(dest)) (no alloc, device-safe; dest's axis-Axis extent must
                    //   match idx's, checked; dest must not alias x). Value form takes a
                    //   TRAILING axis<...>{} (no .template on a dependent receiver -- unlike
                    //   take_along/subsample's LEADING tag): x.index_select(idx, axis<Axis>{})
x.uget(i, j, k);  x.uget(0, slice(1,4));  x.uget(1, ellipsis);  x.uat(i...);
                    // uget = unchecked twin of operator() (element/slice/ellipsis,
                    // one entry point); uat = unchecked at. Skip the negative-index
                    // wrap for known-non-negative RUNTIME indices (per-call
                    // -DTNY_NO_NEGATIVE_INDEX). Same result type; static bounds
                    // still fold. A negative runtime index is then UB.
```

See [Indexing & slicing](indexing.md).

---

## Structure (views) (`axis.h`, `tensor.h`)

```cpp
x.permute<Perm...>();                 // reorder axes
x.flip<Ax>();                         // reverse an axis (negative-stride view)
x.unsqueeze<Ax>();  x.squeeze<Ax>();  // insert / drop a size-1 axis
x.unsqueeze<Ax0,Ax1,...>();  x.squeeze<Ax0,Ax1,...>();  // insert/drop SEVERAL at once (arity picks this
                                      //   overload); unsqueeze positions are relative to the FINAL rank
                                      //   (smallest-first fold), squeeze to the SOURCE rank (largest-first)
x.reshape<NewExt...>();               // contiguous-view reshape (one -1 inferred)
x.flatten();                          // 1-D contiguous view
x.clone();                            // dense row-major OWNING copy (copies on the HOST; a gpu/gpu_view
                                      //   tensor must use the free to<Space>(x) below — dynamic clone static_asserts it)
x.recast<NewExtents>();               // reinterpret w/ a more-static same-rank extents (keeps source strides)
x.recast<NewExtents, ccontiguous>();  // ...AS contiguous (fold the strides; "I promise it's contiguous")
x.recast(shape<...>{}, ccontiguous{}); // functional form (shape + layout, both may mix static/dynamic)
x.to<T2>();                           // dtype convert (matching dtype -> no-copy borrow; else owning copy).
                                      //   The copy runs on the HOST; convert a gpu/gpu_view tensor via to<Space>(x)
x.to(dtype<T2>{});                    // value-tag twin (deduces T2, no .template on a dependent receiver)
to<storage::gpu>(x);                      // memory-space move: to<Space,ET,Force>(x) (from <teeny/cuda.h>);
                                      //   device-aware copy — use this (not clone()/member .to<>()) for a gpu source
```

Axis template arguments are signed (negatives count from the back); each `<Ax>` op
also has a **value form** — `t.permute(Int<2>(),Int<0>(),Int<1>())` == `t.permute<2,0,1>()`,
`t.recast(shape<-1,3,3>{})` == `t.recast<shape<-1,3,3>>()`. The axis-**list** ops
(`peel`/`peel_at`/`take_along`) take an `axis<...>{}` selector (a compile-time axis
list, sibling of `shape<...>`, like numpy's `axis: int | list[int]`):
`peel(t, axis<0,1>{})` == `peel<0,1>(t)`, `t.take_along(axis<0,2>{}, i, slice(1,4))`.
Value forms are deduced, so a type-dependent receiver needs no `.template`.
**Every** view op —
`operator()`/`take_along`/`peel` **and** `permute`/`flip`/`unsqueeze`/`squeeze` —
folds its output strides into a static `strides<...>` (compile-time where the
source strides are static), on any source layout. See [Views & structure](structure.md).

---

## nd-peel (iteration) (`iterate.h`)

```cpp
peel<Axes...>(x);       peel_at<Axes...>(x, i);  // peel named axes. range-for is INCREMENTAL (O(1)/cell,
                                                 //   no per-cell decode); peel_at = random access (grid-stride)
peel(x, axis<Axes...>{}); peel_at(x, i, axis<Axes...>{});  // value form (numpy-like axis selector)
peel_front<N>(x);       peel_front_at<N>(x, i);  // peel the first N axes (COUNT -> template-only)
peel<Axes...>(x).subrange(lo, hi);               // a [lo,hi) chunk for a CPU thread / device block
                                                 //   (seed once, then O(1)/step); peel_front<N>(x) too
for (auto [m, cell] : peel<Axes...>(x).enumerate()) ...;  // ALSO yield the peeled multi-index m
                                                 //   (m[d] = coord of peeled axis d) for a per-axis table
                                                 //   axtab[d][m[d]]. OPT-IN (bare cell stays lean); composes
                                                 //   with .subrange(lo,hi). Or it.index(d) on the raw iterator.
size_front<N>(x);                                // # cells peel_front<N> yields (no range built)

for (auto [a,b,c] : peel_zip<Axes...>(x,y,z)) ...;  // zip-peel 2 or 3 tensors in LOCK-STEP: one
                                                 //   cs::tuple<ViewX,ViewY,ViewZ> per step (a
                                                 //   DISTINCT name from peel, not an overload).
                                                 //   Operands may differ in shape if BROADCAST-
                                                 //   compatible (numpy right-align); Axes... are
                                                 //   in the BROADCAST rank's numbering.
peel_zip(x, y, axis<Axes...>{});                 // value form: axis<...> TRAILING (after every
                                                 //   positional tensor -- unlike take_along/peel_at's
                                                 //   leading tag)
peel_zip<Axes...>(x,y).enumerate();  peel_zip<Axes...>(x,y).subrange(lo,hi);  // same shape as peel's

scan_<Axis>(x, init, f);            // sequential fold along Axis, batched (peeled) over the
                                    //   rest: carry=init, then carry=f(carry,elem); elem=carry
                                    //   for each element (increasing order). f is a device-safe
                                    //   functor (like map_'s convention). Value form: scan_(x,
                                    //   axis<Axis>{}, init, f) -- single-axis tag right after x.
                                    //   Reverse sweep: scan_<Axis>(x.flip<Axis>(), init, f) -- a
                                    //   temporary view binds fine (lvalue + rvalue overloads).
auto y = scan<Axis>(x, init, f);    // out-of-place: fresh dense copy, scanned (static->stack,
                                    //   dynamic->heap host-only, built on clone()); x untouched.
scan<Axis>(x, init, f, into(dest)); // no fresh allocation beyond the copy into dest; returns dest&.
                                    //   dest must match x's shape EXACTLY (checked -- unlike
                                    //   copy_'s own broadcast, since scan_ walks dest's own axes)
```

See [Views & structure](structure.md#nd-peel-iterate-a-subset-of-axes).

---

## Math (`math.h`)

```cpp
// in-place (broadcasts tensor rhs; also scalar rhs). atomic_add_/atomic_sub_
// accumulate ATOMICALLY, host and device (underlying form: add_<Atomic>/sub_<Atomic>).
a.add_(x); a.sub_(x); a.mul_(x); a.div_(x);   a.atomic_add_(x); a.atomic_sub_(s);
a.minimum_(x); a.maximum_(x);   // running min/max update: *this = min/max(*this, x)
y.add_(x, alpha); y.sub_(x, alpha);  // fused axpy: y += alpha*x / y -= alpha*x (x broadcasts)
a += x; a -= s; a *= x; a /= s;  // compound-assign
++a; --a; auto old = a++;        // prefix in place; postfix (static) -> stack copy
a.neg_(); a.abs_(); a.exp_(); a.log_(); a.sin_(); a.cos_(); a.sqrt_(); a.tanh_();
a.floor_(); a.ceil_(); a.round_(); a.trunc_(); a.sign_(); a.pow_(e); a.clamp_(lo, hi);
a & b; a | b; a ^ b; ~a; a &= b; a |= s;  // bitwise (INTEGER element types only)
a.fill_(v); a.zero_(); a.copy_(b); a.iota_(start, step);
a.map_(f); a.zip_with_(g, b);  auto c = a.map(f);  // user functor (device-safe)
a.at(i...).atomic_add_(v);                         // scatter-accumulate (atomic, host and device)

// out-of-place -> new tensor (promotes types; static->stack, dyn->heap)
auto c = a + b;  a.add(b);  a * 2.0;  2.0 - a;  1.0 / a;  -a;  a.pow(b);
auto c = neg(a); abs(a); exp(a); log(a); sin(a); cos(a); sqrt(a); tanh(a);
auto c = floor(a); ceil(a); round(a); trunc(a); sign(a);
auto c = minimum(a, b); maximum(a, s); clamp(a, lo, hi);
auto c = a.add(b, alpha);  a.sub(b, alpha);  // fused out-of-place axpy: a +/- alpha*b (b broadcasts)

// ...or write into a preallocated dest (one fused pass, no alloc) -> dest&: `into(y)` last
a.add(b, into(y));  a.mul(b, into(y));  a.add(2.0, into(y));  a.add(b, alpha, into(y));
exp(a, into(y)); sqrt(a, into(y)); minimum(a, b, into(y)); clamp(a, lo, hi, into(y));
normalize(a, into(y));  cross(a, b, into(N.at(i)));   // cross into a slot ("crossto")

// reductions -> scalar (all axes). ACCUMULATE in the "reduce type" (double for
//   small floats float/double/half, item type for ints), then CAST the result to
//   the tensor's element type: sum(float)->float. A leading TYPE arg makes that
//   type BOTH accumulator and result: sum<double>(a), dot<double>(a,b).
sum(a); prod(a); max(a); min(a); mean(a); dot(a, b);   // sum<Acc>(a), mean<Acc>(a), ...
a.sum(); a.mean(); a.dot(b); a.sum<0>(); a.mean(axis<1>{});  // ALSO methods (parity; same overloads)
sum(a, dtype<double>{});  a.sum(dtype<double>{});  // value-tag Acc == sum<double>(a) — a GENERIC
                      //   trailing keyword bag (tny::_kw, kwargs.h): dtype/axis/keepdims/into
                      //   compose in ANY subset, ANY order (see below), not just this bare form.
sum(a, into(cell)); dot(a, b, into(cell)); dot(a, b, dtype<float>{}, into(cell));  // into(dest):
                      //   FULL reduction -> a RANK-0 dest (local<T,shape<>>{} or wrap(&x,shape<>{}));
                      //   dtype casts, returns dest&. dot composes dtype+into too (no axis concept).
allclose(a, b, rtol=1e-5, atol=1e-8);  // |a-b| <= atol+rtol*|b| everywhere (broadcasts)
// axis reductions -> lower-rank tensor (named axes removed; negatives wrap). Same
//   rule: accumulate in reduce_type, result element type = the tensor's type;
//   sum<Acc, Axes...>(a) makes Acc accumulator AND result (leading TYPE = acc, int = axis).
//   That <Acc,Axes...> vs <Axes...> EXPLICIT TEMPLATE split stays (no universal template
//   param in C++17) — everything past it (axis<...>/dtype/keepdims/into) is a generic bag.
sum<Axes...>(a); prod<...>(a); max<...>(a); min<...>(a); mean<...>(a);  // sum<Acc,Axes...>(a)
sum(a, axis<0,2>{}); mean(a, axis<-1>{}); sum<double>(a, axis<0>{});    // numpy `axis=` value form
sum<0>(a, into(buf)); mean(a, axis<1>{}, into(buf));  // into(dest) -> copies lower-rank result
sum<0>(a, keepdims);  sum(a, axis<0>{}, keepdims);    // keepdims: reduced axis stays size-1 (numpy
                      //   keepdims=True) -> broadcasts back over a. Every axis reduction.
sum(a, dtype<double>{}, axis<0>{}, keepdims, into(buf));  // ...and it ALL composes, any subset/order:
                      //   dtype x axis x keepdims x into == sum<double,0>(a, keepdims, into(buf))

// vector algebra & geometry (contained exact math; on views, host+device)
sqnorm(a);            // Σaᵢ² over all axes (== dot(a,a)); sqnorm<Acc> forces acc+result
norm(a);              // √Σaᵢ² (L2/Frobenius); floating result (int -> double); norm<Acc> too
sqdist(a,b); dist(a,b);// Σ(aᵢ-bᵢ)² / √Σ(aᵢ-bᵢ)² (one fused pass, no a-b intermediate); binary
                      //   only (no axis form, like dot); sqdist<Acc>/dist<Acc>, dtype<Acc>{}/into
a.normalize_();       // in place a /= norm(a) (floating types); zero vector -> NaN
auto u = normalize(a);// out-of-place unit vector -> new tensor (static->stack, dyn->heap)
auto c = cross(a, b);  a.cross_(b);          // 3D cross (rank-1 length-3): new / in place (a = a×b)
                                             //   into a slot: slot.copy_(cross(a, b))
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

## Dispatch (the anyrank boundary) (`dynamic.h`)

```cpp
dispatch_value<Vs...>(v, f);            // runtime value -> integral_constant
as_anyrank(data, shape, stride, ndim);        // -> anyrank WRAPPING the arrays, NO copy
                                              //   (default; host only, arrays must outlive it)
as_anyrank(data, shape, stride, ndim, copy_meta);  // -> anyrank COPYING into an inline
                                              //   TNY_MAX_RANK store (device-passable)
as_anyrank(data, shape, stride, ndim, anyshape<etc,-1,-1,3>{});  // STATIC TRAILING shape in the type
                                              //   (anyshape<etc,...>: etc = erased batch, the rest = static tail):
                                              //   peeled cells fold the inner extents (no per-call recast). Checked
                                              //   vs the runtime shape once here, then trusted. `anyshape<etc>` (bare)
                                              //   == today's carrier. Also from_dlpack<T, anyshape<etc,-1,-1,3>>(m).
as_anyrank(data, shape, stride, ndim, anyshape<etc,-1,-1,3>{}, ccontiguous{});  // + LAYOUT tag folds the inner
                                              //   STRIDES too (checked vs runtime strides here). Fully-static tail ->
                                              //   EBO cell. Subsumes dispatch_layout for the "input contiguous"
                                              //   precondition. from_dlpack passes it by value: (m, ccontiguous{}).
at.peel_front<-Sr>();  at.peel_front_at<-Sr>(i);  // batch idiom (arg NEGATIVE: keep last Sr); 1 kernel per Sr
at.size_front<-Sr>();                             // flattened batch count (no range built)
dispatch_rank(at, f);                    // runtime rank -> fixed-rank view (per total rank)
dispatch_rank<narrow_index>(at, f);      // ...+ int32 offsets when the span fits (rank outer, width inner)
dispatch_index(v, f);                    // narrow one fixed view's offset width to int32 (else keep it)
at.fixed<R>();                           // force a known rank
```

See [Dispatch & the anyrank boundary](dispatch.md).

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
