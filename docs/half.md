# Half precision

teeny ships two 16-bit floating element types — the ones PyTorch uses:

- **`half`** — IEEE binary16 (1-5-10).
- **`bfloat16`** — brain float (1-8-7).

They are ordinary element types: use them anywhere a `float`/`double` goes.

```cpp
auto a = local<half, shape<64,64>>();
a.fill_(half(1.5));
a.mul_(2.0);
auto s = sum(a);  // accumulated in double (reduce type), then cast back to half
```

## Native under nvcc, portable elsewhere

- Under **`nvcc`**, `tny::half` / `tny::bfloat16` *are* the native CUDA
  `__half` / `__nv_bfloat16`, so device kernels get hardware half math.
- On a **host compiler** (g++/clang++, no CUDA) teeny provides portable software
  types with the identical 16-bit layout and correct round-to-nearest-even
  conversions.

Force the portable types even under nvcc with `-DTNY_PORTABLE_HALF`. The bit
layout matches the CUDA types, so a `reinterpret_cast` between them is valid.

## Precision: compute wide, store narrow

**Elementwise** math computes in **`float`** for half element types
(`compute_type<half> == float`), then rounds the result back — so the engines
never depend on native half *host* operators. **Reductions** go a step wider:
they accumulate in the **reduce type**, which is **`double`** for the small
floats (`reduce_type<half/float/double> == double`), then cast the result back to
the tensor's element type. Either way, summing many 16-bit values doesn't stall
at the 16-bit gap.

```cpp
// 2049 halves of 1.0: fp16 can't represent 2049 (gap > 1 above 2048), but the
// double accumulator sums correctly, then rounds to fp16 2048 — not ~1024.
auto big = local<half, shape<2049>>(); big.fill_(half(1.0));
float s = sum(big);  // ~2048
```

## Type promotion with half

Because teeny prefers the lower-width float in a mixed op, half stays half:

```cpp
half + float  -> half
half + double -> half
half + int    -> half
```

(Opt out to standard wider-wins promotion with `-DTNY_STD_PROMOTION`; see
[Math](math.md).)
