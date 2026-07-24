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

a += b;  a -= 2.0;  a *= b;  a /= 2.0;  // compound-assign (scalar or tensor rhs)
++a; --a;                               // prefix: add/sub 1 in place
auto old = a++;                         // postfix: pre-value as a stack copy
                                        //   (STATIC shape only)
```

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
device); a dynamic result is heap-owned (host only).

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

```cpp
sum<0>(a);     // remove axis 0
mean<0,2>(a);  // remove axes 0 and 2
max<1>(a);  min<-1>(a);  prod<0>(a);
```

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
