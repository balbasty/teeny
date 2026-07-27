# Math & broadcasting

A valarray-like math layer that works on any tensor or view. Lambda-free, so it
builds under `nvcc` without `--extended-lambda`. All elementwise/reduction math
computes in `float` for half types (`compute_type<T>`).

## In-place ops (mutate `*this`)

Never allocate. A tensor right-hand side **broadcasts** numpy-style (a size-1
axis is stretched); a scalar applies elementwise.

```cpp
a.add_(b); a.sub_(b); a.mul_(b); a.div_(b);  // tensor rhs broadcasts
a.add_(2.0); a.mul_(0.5);                    // scalar rhs

y.add_(x, alpha);  y.sub_(x, alpha);    // fused scaled accumulate (BLAS axpy):
                                        //   y += alpha*x / y -= alpha*x (x broadcasts).
                                        //   scaled copy y = alpha*x is y.zero_().add_(x, alpha)

a += b;  a -= 2.0;  a *= b;  a /= 2.0;  // compound-assign (scalar or tensor rhs)
++a; --a;                               // prefix: add/sub 1 in place
auto old = a++;                         // postfix: pre-value as a stack copy
                                        //   (STATIC shape only)
```

!!! warning "Don't write in place through a self-overlapping view"
    `wrap` trusts the strides you pass, so a **stride-0** axis (or a stride smaller
    than an inner extent) makes a view where several indices alias the same element.
    *Reading* one is fine — that's how a broadcast RHS works — but an in-place
    **write** into such a destination applies the update to the same element
    repeatedly (`v.add_(b)` double-counts). A host-debug check rejects an in-place
    write whose destination has an `extent > 1` axis with stride 0; `clone()` to a
    dense tensor first if you need to write.

`atomic_add_`/`atomic_sub_` accumulate a delta **atomically on device** — the
scatter/"push" accumulate. Both a broadcasting tensor rhs and a scalar rhs:

```cpp
a.atomic_add_(b);    // accumulate a delta, atomic on device
a.atomic_sub_(2.0);
```

These are the readable spelling of the underlying `add_<Atomic>`/`sub_<Atomic>`
form (`a.atomic_add_(x)` == `a.add_<true>(x)`).

Unary math (in place):

```cpp
a.neg_(); a.abs_(); a.exp_(); a.log_();
a.sin_(); a.cos_(); a.sqrt_(); a.tanh_(); a.pow_(3.0);
a.floor_(); a.ceil_(); a.round_(); a.trunc_(); a.sign_();  // sign_ -> -1/0/+1
a.clamp_(lo, hi);                                          // clamp to [lo,hi]
```

An in-place op with a **scalar** rhs (`a *= 2`, `a.add_(1)`) or a **unary** op
(`a.exp_()`), plus `iota_`/`fill_`/`zero_`, is a single-array read-modify-write and
**auto-vectorizes** (one pointer, nothing to alias — see [Performance](performance.md#open-work)).
The scalar/unary ones apply over any **dense** view (even transposed); only an in-place
op with a *tensor* rhs (`a.add_(b)`) can't, since `b` may overlap `a`.

Bitwise ops are available for **integer element types only**:

```cpp
a & b;  a | b;  a ^ b;  ~a;  // out-of-place (tensor or scalar rhs), unary NOT
a &= b; a |= 1; a ^= b;      // in-place
```

Broadcasting is **numpy-style**: operands are aligned from the **right**, and a
lower-rank operand's missing leading axes are treated as size 1. So a `(H,W)`
tensor broadcasts against a `(W,)` vector directly, and the result rank is the
larger of the two. (In-place `a.op_(b)` still needs `b`'s rank ≤ `a`'s — it can't
grow the destination.) A size-1 axis stretches over its partner as usual;
`unsqueeze` is only needed to align size-1 axes that aren't at the front.

If the two operands have different offset [index widths](shapes-strides.md#mixing-widths-in-a-broadcast)
(e.g. an `int32`-indexed view and an `int64`-indexed one), the result takes the
**wider** of the two — lossless, and it never truncates the wide operand's strides.

Broadcasting example — per-channel scale/bias over a `(C,H,W)` image with
`(C,1,1)` parameters:

```cpp
img.mul_(scale);  // scale is (C,1,1) -> stretched over H and W
img.add_(bias);
```

## Assignment, scatter, generic

```cpp
a.fill_(0.0); a.zero_(); a.copy_(b);  // b broadcasts into a
a.iota_(start, step);                 // start, start+step, … (row-major)
a.map_(f);                            // *this = f(*this)      (user functor)
a.zip_with_(g, b);                    // *this = g(*this, b)   (broadcasts)
auto c = a.map(f);                    // out-of-place variant
a.at(i, j).atomic_add_(v);            // scatter: a(i,j) += v — ATOMIC on device
```

`map_`/`zip_with_` take a functor **struct** (a lambda would need
`--extended-lambda` under nvcc). `at(i...).atomic_add_(v)` is the write half of a
scatter/"push" kernel (`atomicAdd` on device).

## Out-of-place ops → a new tensor

`a + b`, `a.add(b)`, unary free functions, `minimum`/`maximum`, `clamp`, and
`dot` return a new tensor. A fully-static result is stack-owned (host and
device); a dynamic result is heap-owned (host only). When every operand is
C-contiguous and the same shape as the result (no broadcast), these ops take a
`__restrict__` linear fast path over the fresh result and **auto-vectorize**
(#161); a broadcast or strided operand falls back to the general decode.

```cpp
auto c = a + b;    auto d = a.add(b);  // tensor+tensor (broadcasts) or +scalar
auto e = a * 2.0;  auto f = 2.0 * a;   // + and * commute
auto g = 2.0 - a;  auto h = 1.0 / a;   // scalar on the left: reversed op
auto n = -a;                           // unary minus
auto p = a.pow(b);                     // elementwise power

auto e1 = exp(a); auto s1 = sqrt(a);  // unary free functions:
// neg abs exp log sin cos sqrt tanh floor ceil round trunc sign

auto mn = minimum(a, b);   auto mx = maximum(a, 2.0);  // elementwise binary min/max
auto cl = clamp(a, lo, hi);                            // elementwise clamp
```

Every out-of-place producer is **also a method**, for parity with `a.add(b)` and
for chaining — `a.exp()`, `a.sqrt()`, `a.minimum(b)`, `a.clamp(lo, hi)`,
`a.normalize()`, `a.cross(b)`, … — and each takes the same optional `into(y)`
(`a.exp(into(y))`). The free forms (`exp(a)`) and the in-place `_` forms
(`a.exp_()`) still exist; pick whichever reads best.

### Writing into a preallocated destination — `into(dest)`

Pass `into(y)` as the **last** argument to any of these producers to write the
result straight into `y` — one fused pass, **no allocation** — instead of
returning a fresh tensor. It returns `y&`. This is the kernel-friendly form (the
numpy/pytorch `out=`), and the way to write into a strided slot of a larger array.

```cpp
a.add(b, into(y));            // y = a + b   (one pass; y may be any compatible shape/dtype)
a.mul(b, into(y));  a.div(b, into(y));  a.pow(b, into(y));
a.add(2.0, into(y));          // scalar rhs works too
exp(a, into(y));  sqrt(a, into(y));  neg(a, into(y));   // every unary
minimum(a, b, into(y));  maximum(a, s, into(y));  clamp(a, lo, hi, into(y));
normalize(a, into(y));
cross(a, b, into(N.at(i)));   // 3D cross into a preallocated slot ("crossto")
```

`into(y)` is a **distinct type**, so it never collides with a scalar argument —
which is what lets `add`/`sub` also offer the **fused out-of-place axpy** (the
out-of-place twin of `add_(b, alpha)`):

```cpp
auto c = a.add(b, alpha);     // -> new,  a + alpha*b
a.add(b, alpha, into(y));     // -> y,    a + alpha*b   (sub likewise)
```

`into(y)` writes in a single pass; `y` may share memory with an operand, and its
extents are checked against the (broadcast) result. `y`'s dtype need not match —
the result is cast to it. Reductions do not take `into`.

### Type promotion

The result type follows the usual C++ arithmetic conversions, **except among
floating types where the lower precision wins** (`float16 > float32 > float64`,
pytorch-style), so the compact type survives a chain of ops.

```cpp
half   + float  -> half
float  + double -> float
int    + double -> double
int8   + int8   -> int8  // numpy-like, not C++'s int-promoted result
```

Opt back to standard (wider-float-wins) promotion with `-DTNY_STD_PROMOTION`.

## Reductions

Over all axes → a scalar:

```cpp
sum(a); prod(a); max(a); min(a); mean(a); dot(a, b);
```

Over named axes → a lower-rank **tensor** (the named axes are removed; negatives
wrap):

=== "value form"

    ```cpp
    sum(a, axis<0>{});      // remove axis 0 — numpy's `np.sum(a, axis=0)`
    mean(a, axis<0,2>{});   // remove axes 0 and 2
    max(a, axis<1>{});  min(a, axis<-1>{});  prod(a, axis<0>{});
    sum<double>(a, axis<0>{});   // leading TYPE = accumulator; axes deduced from the tag
    ```

=== "template form"

    ```cpp
    sum<0>(a);     // remove axis 0
    mean<0,2>(a);  // remove axes 0 and 2
    max<1>(a);  min<-1>(a);  prod<0>(a);
    ```

The `axis<...>{}` value form (a compile-time axis list, sibling of `shape<...>`,
like numpy's `axis: int | list[int]`) is deduced, so it needs no `.template` on a
type-dependent receiver; a leading TYPE arg still selects the accumulator
(`sum<double>(a, axis<0>{})` == `sum<double,0>(a)`).

A fully-static result is stack-owned (host and device); any dynamic extent makes
it heap-owned (host only — it allocates, so it is not callable on the device
path). Reducing over every axis is the scalar form above.

### Accumulator type vs result type

A reduction **accumulates** in a wide **reduce type** for precision, then **casts
the result back to the tensor's element type** (pytorch-like):

| element type | accumulator (default) |
|---|---|
| `float`, `double`, `half`, `bfloat16` (≤ 8-byte floats) | `double` |
| a wider float (`long double`) | itself |
| integers, everything else | the item type |

So `sum(float_tensor)` **returns `float`** — but the summation runs in `double`,
so many low-precision values still add up accurately before the final cast. The
accumulator trait is `reduce_type_t<T>`. A leading **type** argument makes that
type BOTH the accumulator and the result (like torch's `dtype=`):

```cpp
sum(a);               // float tensor -> float result (accumulated in double)
sum<double>(a);       // accumulate AND return double
mean<double>(a);      // force double throughout
dot<float>(a, b);     // float accumulator and result
sum<int>(int8_view);  // widen an int8 sum to avoid overflow (item type would overflow)
```

Axis reductions follow the same rule — default result element type = the tensor's
type (accumulated wide); a leading type is the accumulator+result, a leading
integer is an axis, so the two never collide:

```cpp
sum<0>(a);          // float tensor -> float result (accumulated in double)
sum<double, 0>(a);  // double accumulator -> double result
mean<double, 1>(a); // force the accumulator+result on an axis mean
```

## Vector algebra & geometry

Small **exact** linear-algebra / geometry helpers on views — the compact building
blocks for kernels like point-to-triangle mesh distance (teeny stops short of
*algorithms*: no solves, matrix inversion, or optimisation).

```cpp
sqnorm(a);            // Σ aᵢ²  over ALL axes — this is dot(a, a)
norm(a);              // √Σ aᵢ² (L2). floating result; an INTEGER input -> double (like mean)
sqnorm<double>(a);    // leading TYPE = accumulator AND result (as with sum/dot)

sqnorm<1>(a);  norm<0,2>(a);  norm(a, axis<-1>{});   // OVER NAMED AXES -> lower-rank tensor
norm<double,1>(a);                                   // ...with a leading accumulator type (like sum)

a.normalize_();       // in place: a /= norm(a)   (floating element types)
auto u = normalize(a);// out-of-place unit vector -> new tensor (static->stack, dynamic->heap)
a.normalize_<1>();  normalize<-1>(a);  normalize(a, axis<1>{});   // OVER NAMED AXES (keepdim broadcast)

auto c = cross(a, b);       // 3D cross product a × b -> new stack 3-vector (rank-1, length 3)
a.cross_(b);                // in place: a becomes a × b (mirrors add_/mul_; aliasing-safe)
slot.copy_(cross(a, b));    // write into a preallocated slot with no new vocabulary
```

`sqnorm`/`norm` are reductions: with no axes they reduce over everything (so `norm`
of a matrix is the Frobenius norm); with `<Axes...>` (or the `axis<...>` value form,
or a leading accumulator type) they reduce over just those axes into a lower-rank
tensor — the same API as `sum`/`mean`. `normalize`/`normalize_` mirror it: with
`<Axes...>` each sub-vector is divided by its norm over those axes (the reduced axes
are kept as size-1 so the norm broadcasts back). Axes must be distinct and ascending.
`normalize` of a zero vector yields NaNs — this is exact math with no epsilon; add
one at the call site if you need it. `cross` is defined only for rank-1, length-3
operands (a `static_assert` catches a wrong static length; a runtime length is
debug-checked). Its in-place form is the member `a.cross_(b)` (`a = a × b`); to
write into a *separate* buffer, `slot.copy_(cross(a, b))` — the result is a tiny
stack 3-vector, so there is no meaningful copy cost.

## Comparisons → a bool tensor

`==`, `!=`, `<`, `<=`, `>`, `>=` broadcast like the arithmetic and return a
`bool` tensor. A scalar may be on either side (`s < a` is `a > s`).

```cpp
auto m = a < b;                         // bool tensor, broadcast
auto p = a >= 2.0;   auto q = 3.0 < a;  // scalar either side
```

Reduce a mask with `.all()` / `.any()` — **members**, because `all` is the slice
keyword. They chain after a comparison:

```cpp
if ((a < b).all())  ...  // every element a < b
if ((a > 0).any())  ...  // some element positive
```

`sum()` preserves dtype, so `sum(mask)` is a saturating `bool`, not a count — use
`.all()`/`.any()`, or cast the mask, to reduce a comparison.
