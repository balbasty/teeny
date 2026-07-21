# Math & broadcasting

teeny has a small valarray-like math layer. It works on **any** tensor or view
and is lambda-free, so it builds under `nvcc` without `--extended-lambda`.

## In-place ops (mutate `*this`)

In-place ops work on any tensor/view and never allocate. A tensor right-hand
side **broadcasts** numpy-style (a size-1 axis is stretched); a scalar applies
elementwise:

```cpp
a.add_(b);  a.sub_(b);  a.mul_(b);  a.div_(b);   // tensor rhs broadcasts
a.add_(2.0); a.mul_(0.5);                        // scalar rhs
a.neg_(); a.abs_(); a.exp_(); a.log_();          // unary
a.sin_(); a.cos_(); a.sqrt_(); a.tanh_(); a.pow_(3.0);
```

Broadcasting example — a per-channel scale/bias over a `(C,H,W)` image with
`(C,1,1)` parameters:

```cpp
img.mul_(scale);   // scale is (C,1,1) -> stretched over H and W
img.add_(bias);
```

## Assignment, scatter, generic

```cpp
a.fill_(0.0); a.zero_(); a.copy_(b);   // b broadcasts into a
a.iota_(start, step);                  // 0,1,2,… in row-major order
a.map_(f);                             // *this = f(*this)      (user functor)
a.zip_with_(g, b);                     // *this = g(*this, b)   (broadcasts)
auto c = a.map(f);                     // out-of-place variant
a.add_at(v, i, j);                     // scatter: a(i,j) += v — ATOMIC on device
```

`map_`/`zip_with_` take a **functor struct** (device-safe — a lambda would need
`--extended-lambda` under nvcc). `add_at` / the free `fetch_add(ptr, v)` are the
write half of a scatter/"push" kernel and use `atomicAdd` on the device.

## Out-of-place ops → a new tensor

`a + b`, `a.add(b)`, unary free functions, and `dot` return a **new** tensor. A
fully-static result is stack-owned (host **and** device); a dynamic result is
heap-owned (host only).

```cpp
auto c = a + b;    auto d = a.add(b);     // tensor+tensor (broadcasts) or +scalar
auto e = a * 2.0;  auto f = 2.0 * a;      // scalar (+ and * commute)
auto g = a.pow(b);                        // elementwise power
auto h = exp(a);   auto k = sqrt(a);      // unary free functions
```

### Type promotion

The result type follows the usual C++ arithmetic conversions, **except among
floating types where the lower precision wins** (`float16 > float32 > float64`,
pytorch-style) — so the compact type survives a chain of ops.

```cpp
half   + float  -> half
float  + double -> float
int    + double -> double
int8   + int8   -> int8    // numpy-like, not C++'s int-promoted result
```

Opt back to standard (wider-float-wins) promotion with `-DTNY_STD_PROMOTION`.

## Reductions → a scalar

```cpp
sum(a);  prod(a);  max(a);  min(a);  dot(a, b);
```

!!! note "Half precision accumulates in float"
    For `half`/`bfloat16` tensors, reductions accumulate in `float` (via
    `compute_type<T>`) and cast back, so summing many 16-bit values doesn't
    stall. See [Half precision](half.md).
