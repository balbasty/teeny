# Indexing & slicing

`operator()` does both element access and slicing, chosen by the argument types,
like NumPy's `[]`.

## Element access

All-integer arguments return a **reference to one element** (`T&`). Negative
indices count from the back:

```cpp
t(1, 2, 3);      // T& at (1,2,3)
t(0, -1);        // row 0, last column
t(Int<1>(), j);  // a static index folds; a runtime index is a value
t[1, 2, 3];      // C++23 only: multidimensional subscript, an exact alias of t(1,2,3)
```

`operator[]` is available when the compiler provides the C++23 multidimensional
subscript (`__cpp_multidimensional_subscript`) — it forwards to `operator()`, so
`t[i, j]`, `t[0, all, slice(1,4)]` and `t[ellipsis]` all behave identically
(`mdspan`'s own spelling). On C++17/20 use `operator()`.

The negative-index wrap is a signed compare that folds away for static
(`Int<k>()`) and unsigned arguments. For runtime signed indices in a hot loop
where non-negative is guaranteed, compile with `-DTNY_NO_NEGATIVE_INDEX` to drop
the check.

## `at` — one element as a rank-0 view

`at(i...)` takes the same all-integer indices but returns the element as a
**rank-0 view** instead of a `T&`, so the whole tensor API applies to a single
cell. Rank-0 tensors convert to and from `T` and have `.item()`:

```cpp
x.at(i,j) = 3;               // write
float v = x.at(i,j);         // read (implicit conversion to T)
float w = x.at(i,j).item();  // explicit read
x.at(i,j).atomic_add_(v);    // atomic scatter into one cell
```

## Unchecked accessors — `uget` / `uat`

The wrap that lets a negative index count from the back costs one compare-and-add
per axis. In a hot kernel where every index is already known non-negative that is
wasted work, so the indexing entry points have a `u`-prefixed twin that **skips the
negative-index wrap** for runtime signed args — the per-call equivalent of building
with `-DTNY_NO_NEGATIVE_INDEX`. There are just two:

- **`uget`** is the twin of **`operator()`** — one entry point covering *all three*
  forms, chosen by the argument types exactly as `operator()` does: all-integer →
  element `T&`, any slice arg → a view, one `ellipsis` → expand and re-dispatch.
- **`uat`** is the twin of **`at`** — one element as a rank-0 view.

```cpp
x.uget(i, j, k);              // like x(i,j,k)         -> T&        (element, no wrap)
x.uget(0, slice(1, 4));       // like x(0, slice(1,4)) -> a VIEW    (no wrap on runtime bounds)
x.uget(1, ellipsis, 2);       // like x(1, ellipsis, 2) — ellipsis expands, stays unchecked
x.uat(i, j);                  // like x.at(i,j)        -> rank-0 view
```

They return the **same type** as their checked counterparts, so they drop in
unchanged, and static (`Int<>`) bounds and `none` fold identically — only *runtime
signed* values skip the wrap. Slice ranges are still clamped, so a runtime negative
bound is now taken literally then clamped (the rule is just "no wrap": with a forward
step a negative *stop* yields an empty axis, a negative *start* clamps to `0`).

The one rule: **passing a negative runtime index to a `u`-accessor is undefined
behaviour** — that is the promise you make in exchange for the tighter codegen.

Bounds checking is off by default (like `mdspan`'s subscript), so normally `uget`
differs from `operator()` only by the dropped wrap. Under **`-DTNY_HARDENED`** the
*checked* accessors gain a per-index bounds check; the `u*` accessors always skip it,
so they are the deliberate opt-out. The bounds check is always off on the device.

## Slicing → a sub-view (no copy)

If any argument is a slice specifier, `operator()` returns a lower- or same-rank
view into the same memory:

| argument | meaning |
|---|---|
| an integer | drop that axis (fix it at this index) |
| `all` | keep the whole axis |
| `slice(a, b)` | half-open range `[a, b)` |
| `slice(a, b, step)` | strided range (`step` may be negative) |
| `none` (or `newaxis`) | inside a `slice`: an open end. As a **bare argument**: insert a new size-1 axis |
| `ellipsis` (or `etc`) | stand-in for as many `all` as it takes to fill the rank |

```cpp
t(1, all, all);                 // fix axis 0 -> lower-rank view
t(0, slice(1, 4));              // axis 1 range [1,4)
t(all, slice(none, 8, 2));      // every other element up to 8
t(all, slice(none, none, -1));  // reverse an axis (numpy a[:, ::-1])
```

`none` is python's `None`: `slice(none, k)` starts at 0, `slice(m, none)` runs to
the end, and `slice(none, none)` keeps the whole axis (it *is* `all`, and folds
to a static extent). Negative bounds wrap.

```cpp
t(slice(-2, none));  // last two rows
t(slice(3, 0, -1));  // rows 3,2,1  (stop excluded, like python)
```

**Ellipsis** (`...`, numpy's `Ellipsis`) fills the middle for you: it expands to
`rank - (#other args)` copies of `all` (possibly zero). At most one per call.

```cpp
t(1, ellipsis);      // == t(1, all, all)   drop the FIRST axis, keep the rest
t(ellipsis, 2);      // == t(all, all, 2)   drop the LAST axis
t(1, ellipsis, 2);   // == t(1, all, 2)     fill only the middle
t(ellipsis) = b;     // whole-tensor slice-assign (copies b's elements in)
t(1, etc, 2);        // `etc` is the same marker under another name (== ellipsis)
```

`ellipsis` and `etc` are two names for one marker — "the unspecified middle axes."
`etc` is the spelling used in an [`anyshape<etc, …>`](dispatch.md) boundary tag; both
names work in both places.

### Inserting a new axis — `none` / `newaxis`

A **bare** `none` argument (not inside a `slice`) is numpy's `newaxis`
(`a[None]` / `a[np.newaxis]`): it inserts a size-1 axis at that position and
consumes no source axis — the mirror of an integer, which consumes a source
axis and emits nothing. `newaxis` is just a named alias of `none`, for
readers who'd rather spell it that way (as far as teeny is concerned they are
the exact same value):

```cpp
t(none, all, all);      // (H,W) -> (1,H,W): insert a leading axis
t(all, all, none);      // (H,W) -> (H,W,1): insert a trailing axis
t(newaxis, all, all);   // identical to the first line
t(0, none, all);        // (H,W) -> (1,W): int drops an axis, none adds one back
t(none, ellipsis);      // ellipsis + none compose freely
```

`t(none, ...)` at position `k` is the same view as `t.unsqueeze<k>()`; the
inserted axis has static extent 1 and stride 0 (it's a broadcast axis, so
writing through it writes the single underlying row/column). Any number of
`none`/`newaxis` args can appear alongside integers, ranges, and (at most one)
ellipsis in the same call.

If what remains after expansion is all integers you get an element `T&`; anything
else yields a view. Assigning into a slice — `t(...) = b`, `t(0, all) = 5.0` —
copies (see [Tensors & ownership](tensors.md), which contrasts it with the
rebinding `a = b`). The copy runs front-to-back with no overlap check, so a slice
that **overlaps its own source** (`a(slice(1,5)) = a(slice(0,4))`) reads cells it
has already written — `clone()` the source first if the regions overlap.

Axes kept with `all` keep their static extent; a ranged axis becomes dynamic
(its size is a runtime value). Reach for `all` when you want the extent to keep
folding. The output layout is `strides<...>` with each kept stride folded to a
compile-time value where derivable.

**Runtime vs compile-time bounds.** `slice(a, b)` takes its bounds as ordinary
values; `slice<a,b>()` / `slice<a,b,step>()` bake them into the type, so a ranged
axis keeps a **static** extent and stride (folds like `all`) instead of a runtime
one. The runtime form is the value spelling; reach for the compile-time form when
you want the extent to keep folding:

=== "runtime bounds"

    ```cpp
    t(0, slice(1, 4));       // [1,4), extent is a runtime value
    t(all, slice(0, 8, 2));  // runtime stride 2
    ```

=== "compile-time bounds"

    ```cpp
    t(0, slice<1,4>());              // [1,4), static extent (folds like `all`)
    t(all, slice<0,8,2>());          // static stride 2
    t(0, slice<Int<1>, Int<4>>());   // type form (the only way to bake `none`)
    ```

## `take_along<Axes...>` — bind named axes

`operator()` is positional. To name only the axes you touch and keep the rest,
use `take_along`:

=== "value form"

    ```cpp
    t.take_along(axis<1>{}, 2);                // fix axis 1 at index 2, keep all others
    t.take_along(axis<0,2>{}, i, slice(1,4));  // bind axes 0 and 2 at once
    t.take_along(axis<-1>{}, k);               // negative axis: the last one
    ```

=== "template form"

    ```cpp
    t.take_along<1>(2);
    t.take_along<0,2>(i, slice(1,4));
    t.take_along<-1>(k);
    ```

Each bind argument is an integer (negatives wrap), `all`, or a `slice` — the same
specifiers `operator()` accepts. The **value form** leads with an `axis<...>{}`
selector (a compile-time axis list, sibling of `shape<...>`, like numpy's
`axis: int | list[int]`); being a single deduced argument it needs no `.template`
on a type-dependent receiver, and it's the one spelling that disambiguates
`take_along`'s two argument packs cleanly.

## `subsample<Axes...>` — a coloured/strided sub-lattice

Coloured Gauss-Seidel relaxation walks a sub-lattice selected by
`loc[d] % k == digit_d(n)` per axis — already expressible with `take_along` and
a `slice(start, none, k)` per named axis, just verbose to spell out when the
step `k` is shared across every axis and only the per-axis `start` differs.
`subsample` names that pattern:

```cpp
t.subsample<0,1>(k, s0, s1);   // == t.take_along<0,1>(slice(s0,none,k), slice(s1,none,k))
```

Pure sugar — no new addressing power, just a named shorthand for the multi-axis
strided slice teeny already does. `k` and each `start` accept either a runtime
value or a compile-time one (`Int<k>()`); a fully-static `(start, k)` pair folds
a static output extent, same as a hand-written `slice()`:

```cpp
t.subsample<0,1>(Int<2>(), Int<0>(), Int<0>());   // step/starts all compile-time -> static result
```

Has the same value form as `take_along` — a leading `axis<...>{}` selector, so
no `.template` is needed on a type-dependent receiver:

```cpp
t.subsample(axis<0,1>{}, k, s0, s1);   // == t.subsample<0,1>(k, s0, s1)
```

## `unfold<Axis>` — a sliding/strided window (pytorch `Tensor.unfold`)

Modelled directly on pytorch's `Tensor.unfold(dimension, size, step)`:
`unfold<Axis>` appends a **new trailing axis** of width `size`, stepped by
`step` along `Axis` — the "every stencil tap needs the `K`-wide window
starting at a per-call offset" pattern, spelled out today via
`t(..., slice(off, off+K), ...)`:

=== "value form"

    ```cpp
    t.unfold(Int<0>(), 3, 1);   // axis 0: width-3 windows, step 1
    t.unfold(Int<0>(), 3, 2);   // step 2 -> windows start 0, 2, 4, ...
    ```

=== "template form"

    ```cpp
    t.unfold<0>(3, 1);
    t.unfold<0>(3, 2);
    t.unfold<0>(3);     // step defaults to 1
    ```

`Axis`'s own extent shrinks to the window **count**,
`(extent(Axis) - size) / step + 1`; the new trailing axis holds one window's
`size` elements. `size`/`step` each accept a runtime value or a compile-time
one (`Int<k>()`), folding the output extent to static where derivable (same
convention as `slice()`):

```cpp
t.unfold<0>(Int<3>(), Int<2>());   // size/step both compile-time -> static result
```

`unfold` is pure sugar over the existing gather (no new addressing power) and
returns a **view** — write-through, and windows **alias** when `step < size`
(as in pytorch: writing one tap of an overlapping window mutates the element
every neighbouring window also sees). ND windows compose by chaining one
`unfold` per axis — each call appends its window axis after the previous
one's, matching how `nitorch.core.utils.unfold`'s nd-unfold is itself built
on the single-axis primitive:

```cpp
t.unfold<0>(2,1).unfold<1>(2,1);   // (H,W) -> (H-1,W-1,2,2): a 2x2 window per cell
```

## `index_select<Axis>` — gather by a runtime index tensor

`take_along`'s bind arguments are known at the call site (a literal, a variable
holding one index, a slice). `index_select` is for the opposite case: pulling
elements along one axis using an arbitrary integer index **tensor** — data you
don't know until runtime, e.g. an index buffer used to gather triangle vertices
out of a vertex buffer:

```cpp
auto verts = local<double, shape<5,3>>();   // 5 vertices, 3 coords each
auto idx   = local<long, shape<3>>();       // idx(0..2) = which vertices to pull
idx(0) = 2; idx(1) = 0; idx(2) = 4;
auto tri = verts.index_select<0>(idx);      // (3,3): rows 2, 0, 4 of verts
```

`idx` must be rank-1; axis `Axis`'s extent in the result becomes `idx`'s own
extent (static when `idx`'s shape is static, so a compile-time-sized index
buffer keeps the result on the stack). `idx`'s values wrap negative like any
other teeny index (`index_select` is built on `take_along`, which already
wraps), and repeated indices are fine — it's a gather, not a permutation.

Because the index values are runtime data, `index_select` can't return a view
(an arbitrary data-dependent gather isn't expressible as an affine mdspan
mapping) — it always materialises a copy: static result → stack, dynamic →
heap. Pass `into(dest)` to write straight into a preallocated buffer instead —
one pass, no allocation, the device-safe form to use inside a kernel:

```cpp
verts.index_select<0>(idx, into(dest));     // dest must already have the right shape
```

`dest`'s extent on axis `Axis` must equal `idx`'s (checked — a `static_assert`
when both are static shapes, a debug-time check otherwise) and `dest` must not
alias `verts`' own storage (an aliased in-place gather silently reorders rather
than erroring). The allocating form copies on the **host**, so the source must
be host-accessible — gather a `gpu`/`gpu_view` tensor into a preallocated
device `into(dest)` instead.

Like `take_along`, `index_select` has a value form leading with an `axis<...>{}`
selector — the one to reach for on a type-dependent receiver (inside a kernel
template), since it needs no `.template` disambiguator:

```cpp
verts.index_select(idx, axis<0>{});               // == verts.index_select<0>(idx)
verts.index_select(idx, axis<0>{}, into(dest));    // == verts.index_select<0>(idx, into(dest))
```
