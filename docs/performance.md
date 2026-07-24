# Performance

teeny's performance model rests on one idea: **fold everything that is known at
compile time, carry only what is genuinely dynamic.** A `strides<...>` layout with
known strides is an *empty base* (EBO) — it costs zero bytes and its offsets fold
to immediates. You pay, in footprint and in lost optimization, only for the axes
whose extent or stride is truly a runtime value.

This page is about *where that cost lands* — especially for **dynamic strides**,
which a kernel carries by value — and how to keep it off the hot path.

## Footprint: what a by-value view actually carries

A view is passed to a kernel by value (it is trivially copyable — a pointer plus
the *dynamic* parts of its geometry). `sizeof` the view is what you carry in
registers / kernel argument space:

| View type | `sizeof` | Why |
|---|---|---|
| `double*` (raw) | 8 | baseline |
| `view<double, shape<8>, strides<1>>` | **8** | fully static geometry → EBO; a bare pointer |
| `view<double, shape<-1>, strides<2>>` | 16 | static stride is free; only the dynamic **extent** adds 8 |
| `view<double, shape<-1,-1,-1>, ccontiguous>` | 32 | strides are **derived** from extents, not stored (8 + 3×8) |
| `view<double, shape<-1,-1,-1>, dynamic_strides>` | **56** | 8 ptr + 3 extents + **3 stored strides** |

Two things fall out immediately:

- **A fully-static view is a bare pointer.** Bake known dims/strides into the type
  (`shape<3,3>`, `strides<...>`) and the geometry vanishes.
- **`ccontiguous` never stores strides** — it derives them from the extents. So even
  for a dynamic shape, a *contiguous* view is 24 bytes lighter than a
  `dynamic_strides` one of the same rank. Reach for `dynamic_strides` only when the
  strides are genuinely arbitrary (a transpose gap, an external DLPack layout).

On the **GPU** this footprint is the dominant cost: a rank-3 `dynamic_strides` view
is ~7 registers *per thread, per view*; a kernel juggling several fields can lose
occupancy to it. This is the cost your instinct should flag.

## The per-element hot path is *not* the problem

A natural worry: "runtime strides are extra weight carried on every element access."
For a well-structured loop, **the compiler removes it.** Loop-invariant strides are
hoisted once and the address is strength-reduced to a pointer increment. The same
`for i: d(i) += a(i)` loop, compiled three ways at `-O2`, has an **identical**
three-instruction body:

| Layout | Loop body | Stride handling |
|---|---|---|
| `ccontiguous` (stride 1) | `movsd; addsd; movsd` | folded into the ×8 address scale |
| `strides<2>` (static) | `movsd; addsd; movsd` | strength-reduced — index steps by 16 |
| `dynamic_strides` (runtime) | `movsd; addsd; movsd` | stride **hoisted** to a register before the loop; pointer strength-reduced |

No `imul`, no per-element stride reload in any of them. This works because
`operator()` is `_TNY_API inline` and the rank is `constexpr`, so the offset unrolls
and the strides are plainly loop-invariant to the optimizer. **In a simple march,
dynamic strides cost nothing per element.**

## Where dynamic strides *genuinely* cost

1. **Register pressure / GPU occupancy** — the footprint above. The real, measurable
   cost, and the reason to prefer static strides and (soon) 32-bit indexing.
2. **Vectorization** — an unknown stride can never be proven contiguous, so it can't
   SIMD. Worse, even a *contiguous* teeny op currently doesn't auto-vectorize,
   because the view's pointers aren't `restrict`-qualified and the compiler can't
   prove the destination and source don't overlap. (Measured: at `-O3`, a
   `__restrict__` pointer loop vectorizes; the may-alias version stays scalar.) This
   is a known gap — see *Open work* below.
3. **ND random-access gather** — the one place loop-invariant hoisting can't help.
   A spline pull/push inner neighborhood computes `base + Σ idxₖ·strideₖ` at scattered
   points, not a linear march. With dynamic spatial strides that is N runtime
   multiply-adds per gather; with **static** strides they fold to immediates. Keep the
   spatial strides static (below) and the gather folds.
4. **Index width** — dynamic offset math in `int64` is slower and uses more registers
   than `int32`, especially on the device. Addressed by #115.

## How to keep it fast — the fold in practice

teeny is built to route around the above; the idioms:

- **Bake known geometry into the type.** `local<double, shape<3,3>>`,
  `strides<9,3,1>` — zero footprint, immediate offsets.
- **Views preserve static strides.** Slicing, `peel`, `permute`, `flip`,
  `unsqueeze`/`squeeze`, and `reshape` all **fold their output strides** to
  compile-time constants where the source is static — so a static source stays static
  through a chain of views (see [Views & structure](structure.md)).
- **Recover static inner dims at the boundary.** `recast(shape<-1,3,3>{})` re-types a
  runtime `(n,3,3)` view so the `3`s fold, without a copy and preserving the source
  strides.
- **Peel the batch into the pointer.** `peel_front<Nbatch>` bakes the batch offset
  into each sub-view's *data handle*, so the inner kernel sees only the (usually
  static) spatial + channel strides — the batch strides never enter the inner loop
  (see [Dispatch & the ndarray boundary](dispatch.md)).
- **Dispatch runtime shape to static kernels.** `dispatch_value` / `dispatch_rank`
  instantiate a static-shape kernel from a runtime spatial rank, so the hot code is
  compiled against folded extents and strides.

Rule of thumb for a hot kernel: **the inner loop should run over a view whose
strides are static** (or contiguous). If it doesn't, `recast` / `dispatch` at the
launch boundary until it does; carry the dynamic strides only in the outer/batch
loop, where they hoist for free.

## Open work

- **32-bit offset dispatch (#115) — landed.** `t.reindex<int32_t>()` (→ `shape32`) is
  the no-copy, layout-preserving retype; narrowing a dynamic boundary view halves its
  footprint (rank-2: 40→24 B) and runs offset math in 32-bit — the biggest device win.
  `dispatch_index(v, f)` instantiates the kernel for both widths and picks `reindex`
  when `v.index_fits<int32_t>()`; `dispatch_rank<narrow_index>(at, f)` fuses it into the
  anyrank rank dispatch. Opt in per launch site. (Remaining: forbid cross-width
  broadcasting — assert equal `Idx`.)
- **`restrict`/no-alias fast path (#161).** teeny's elementwise engines carry
  non-`restrict` pointers, so even contiguous host loops don't auto-vectorize (the
  compiler must assume the destination may alias a source). A `__restrict__` path — or
  an explicit "these don't alias" opt-in — could unlock SIMD on the contiguous host
  path.

!!! note "Measurement over intuition"
    The numbers here (`sizeof`, loop bodies, SIMD) are from `g++ -O2/-O3` on the host.
    The *shape* of the argument — static folds to nothing, dynamic strides hoist per
    loop but cost footprint — holds on the device too, but exact GPU register/occupancy
    effects are best confirmed with `nvcc --ptxas-options=-v` on the real kernel.
