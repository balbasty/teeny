# Writing efficient kernels

teeny is designed so that clean, readable kernel code is also fast — *if* you let
the compiler see what's constant. This page is the short version of "how do I make
a teeny kernel fast?"; each idiom links to the page that explains it in depth.

The whole model reduces to one rule:

!!! tip "The one rule"
    **Keep the hot loop running over a view whose geometry is static (or at least
    contiguous). Push everything dynamic to the launch boundary.** Static shapes and
    strides fold to immediates; a fresh contiguous result auto-vectorizes. What's left
    dynamic — the batch count, an arbitrary DLPack stride — should live in the *outer*
    loop, where it's hoisted once and costs nothing per element.

## 1. Bake known geometry into the type

A dimension whose size or stride you know at compile time should be *in the type*, not
a runtime value. Then it vanishes from both the footprint and the offset math.

```cpp
local<double, shape<3,3>> m;             // 9 doubles on the stack — sizeof(m) == 72, no pointer
auto v = wrap(ptr, shape<-1,3,3>{n});    // n dynamic, the 3×3 folds: stride0 == 9 is a constant
```

- A **fully static** view is a bare pointer (EBO — zero geometry bytes).
- A **`ccontiguous`** view never stores strides; reach for `dynamic_strides` only when
  the strides are genuinely arbitrary (a transpose gap, an external layout).

Why it matters — footprint (GPU registers) and folded offsets — is measured in
[Performance](performance.md); the static/runtime spelling is in
[Shapes & strides](shapes-strides.md#the-staticruntime-idiom).

## 2. Views, not copies

Every structural op returns a **view** that aliases the same memory — no allocation,
no data movement. Slice, transpose, reverse, and iterate freely in the hot path.

```cpp
t(0, all, slice(1, 4));        // sub-view
t.permute(Int<2>(), Int<0>(), Int<1>());   // transpose (a view)
t.flip(Int<1>());              // reversed axis (negative-stride view)
for (auto line : peel(t, axis<0,1>{})) work(line);   // each `line` is a view
```

Views also **preserve folded strides**: slicing a static tensor yields a
`strides<...>` view whose strides are still compile-time constants, so a static source
stays static through a chain of views ([Views & structure](structure.md)). Only reach
for a copy (`clone()` / `to<...>()`) when you genuinely need dense, owned memory.

## 3. The `(*batch, …)` batch idiom

The workhorse pattern for field kernels: leading **batch** dims and a small, fixed
trailing block (a `C×C` matrix, or `*spatial, C`). Peel the batch dims into the pointer
so the inner kernel only sees the fixed block — `peel_front` bakes each batch offset into
the sub-view's *data handle*, so the batch strides never touch the inner loop. Which
spelling you use depends on **whether the batch rank is known at compile time**.

**Runtime batch rank — the general case** (a DLPack tensor of arbitrary rank). The input
arrives rank-erased as an [`anyrank`](dispatch.md), so its rank is a *runtime* value. Peel
with a **negative** front: `peel_front<-Sr>()` keeps the last `Sr` dims static and folds
*however many* batch dims there are into the pointer. Each `cell` is a `dextents<_,Sr>`
view (rank `Sr`, dynamic inner extents); `recast` folds the known inner dims:

```cpp
auto at = as_anyrank(data, shape, stride, ndim);   // rank-erased carrier (runtime rank, no copy)
for (auto cell : at.peel_front<-2>())              // keep the trailing 2 dims; peel the rest
    op(cell.recast(shape<C, C>{}));                // cell: dextents<_,2> view -> folded C×C
```

Combine with `dispatch_value<1,2,3>(spatial_ndim, …)` to also turn a runtime *spatial
rank* into a static one — the full `(*batch, *spatial, C)` walk-through is in
[Dispatch & the ndarray boundary](dispatch.md).

**Known batch rank** (a plain tensor). A `tensor` always has a *static rank* — the rank is
a compile-time property even when the extents are dynamic — so if you know the batch count
you can peel a **positive** front directly. `peel_front<Nbatch>(t)` drops the first
`Nbatch` dims; each `cell` is a rank-`(rank − Nbatch)` view with the trailing extents
(folded where the source is static):

```cpp
auto t = wrap(data, shape<-1,-1,C,C>{b0, b1});  // static rank 4, dynamic batch extents
for (auto cell : peel_front<2>(t))              // peel the 2 batch dims -> cell is a C×C view
    op(cell);
```

## 4. Narrow the offset width at the boundary (device)

Public tensors index in `int64` (matching DLPack). On the **device**, a dynamic view's
strides are carried by value, and 64-bit offset math costs registers. When the element
span provably fits 32 bits, narrow it — no copy, just a retype:

```cpp
dispatch_index(v, [&](auto w) { kernel(w); });   // int32 arm when index_fits, else int64
```

`dispatch_index` instantiates the kernel for both widths and picks the narrow one at
run time; it halves a dynamic view's footprint and runs address math in 32-bit. It's
opt-in per launch site — see [Shapes & strides](shapes-strides.md#the-index-type-shape32-reindex)
and [Dispatch](dispatch.md#dispatch_index-dispatch_ranknarrow_index-the-int32-fast-path).

## 5. Contiguous elementwise math auto-vectorizes

Contiguous elementwise ops take a linear fast path (in place of the per-element decode)
that **auto-vectorizes**. Two flavours, by whether a second array is involved:

```cpp
auto c = a + b;     // OUT-OF-PLACE: fresh result, can't alias -> SIMD write
auto d = exp(a);    // likewise (needs operands same-shape + contiguous, no broadcast)
a *= 2;  a.add_(1); // IN-PLACE SCALAR: one array, nothing to alias -> SIMD
a.exp_(); a.neg_(); // IN-PLACE UNARY: same — and over ANY dense view (transposed too)
a.add_(b);          // IN-PLACE TENSOR rhs: `b` may overlap `a`, so this stays scalar
```

- **Out-of-place** (`a + b`, `exp(a)`, `a < b`) writes a **fresh** result that can't alias
  its operands — vectorizes when every operand is C-contiguous and the same shape.
- **In-place scalar / unary** (`a *= 2`, `a.exp_()`), plus `iota_`/`fill_`/`zero_`, is a
  single-array read-modify-write — one pointer, nothing to alias. The scalar/unary ones
  are order-independent, so they vectorize over **any dense view** (C/F/permuted — a
  transposed in-place op still SIMDs); `iota_` needs exact C-order.
- The **only** case left scalar is an in-place op with a **tensor rhs** (`a.add_(b)`):
  `b` may overlap the destination, so the compiler must assume aliasing.

A broadcast or a strided operand also falls back to the decode (correct, just not
vectorized). Mechanism and codegen: [Performance](performance.md#open-work).

## 6. Compute wide, store narrow

Math on `half`/`bfloat16` computes in `float`, and reductions accumulate in a wide type
(`double` for small floats, 64-bit for narrow ints) before casting back — so you get
precision without paying storage for it. Keep your data in the compact type; teeny
widens only inside the op.

```cpp
local<half, shape<64,64>> h;   // stored as 16-bit
auto s = sum(h);               // accumulated in double, returned as half
```

See [Half precision](half.md) and [Math → reductions](math.md#accumulator-type-vs-result-type).

## 7. Small static-C kernels — two codegen rules

A per-voxel kernel over a static `C` (a `C×C` solve, a packed-symmetric gather) folds to
hand-written quality *only* if you help the compiler in two places (both measured via
`-O2 -S`):

- **Snapshot inputs through a `local` before writing outputs.** If a kernel reads `in`
  and writes `out` where the two *could* alias, the compiler reloads `in` after every
  store. Copy the inputs into a `local` workspace first, compute from that, then write
  `out` — the reloads vanish (a probe dropped one clang kernel 68 → 37 instructions).
  (This is the read-side analogue of the out-of-place `__restrict__` fast path in §5.)
- **Fully unroll the small static loops with `TNY_UNROLL`.** A packed index like
  `sub2pak(C,i,j)` folds to an immediate offset only when its loop unrolls. clang and
  nvcc honour a bare `#pragma unroll`; **gcc silently ignores it** and needs
  `#pragma GCC unroll N`. Use the portable `TNY_UNROLL` (`defines.h`) immediately before
  the `for`:

  ```cpp
  TNY_UNROLL
  for (int j = 0; j < C; ++j) acc += L(i,j) * x(j);   // static C -> folds to immediates
  ```

  `TNY_UNROLL` is a full unroll with a generous fixed count (gcc's pragma needs a
  *literal*, so a per-count macro can't take a template-parameter `C`); for a partial
  unroll write the compiler pragma directly.

## Checklist

Before a kernel is "fast", check:

- [ ] The inner loop runs over a view whose **strides are static or contiguous**
      (`recast` / `dispatch` at the boundary if not).
- [ ] Known dimensions are **in the type** (`shape<3,3>`, `strides<...>`), not runtime.
- [ ] You're **slicing/peeling into views**, not cloning, on the hot path.
- [ ] Batch dims are **peeled into the pointer** (`peel_front`), not carried in the inner loop.
- [ ] On the device, offsets are **narrowed** (`dispatch_index`) where the span fits.
- [ ] Bulk elementwise work uses the **out-of-place** form when shapes already match.
- [ ] Small static-C inner loops are marked **`TNY_UNROLL`**, and inputs are **snapshotted
      through a `local`** before writing outputs (§7).
