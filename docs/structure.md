# Views & structure

These return **views** (no copy) that rearrange, reshape, or iterate a tensor.
Axis template arguments are signed — **negatives count from the back**.

The axis-**list** ops — `permute`, `squeeze`, `unsqueeze` — take an `axis<...>{}`
tag (a compile-time axis list, sibling of `shape<...>`, the same one
`peel`/`take_along`/the reductions use): `t.squeeze(axis<0,2>{})` == `t.squeeze<0,2>()`,
`t.permute(axis<2,0,1>{})` == `t.permute<2,0,1>()`. Being a single distinct-typed
argument, it needs no `.template` on a dependent receiver — reach for this spelling
first.

Every axis argument also has a plain **value form**: pass a static integer
(`Int<k>()`) instead of the `<k>` template argument — `t.squeeze(Int<1>())` ==
`t.squeeze<1>()`, and likewise for `permute`/`flip`/`unsqueeze`/`reshape`. All
three spellings compile to the same code; the tabs below lead with `axis<...>{}`
where an op supports it, then the `Int<k>()` value form, then the explicit
template form.

## Rearrange axes

=== "axis<...> form"

    ```cpp
    t.permute(axis<2,0,1>{});    // reorder axes (a permutation of 0..N-1)
    t.permute(axis<-1,0,1>{});   // negatives count from the back
    // flip is single-axis, not an axis-list op — use the Int<k>()/template form below
    ```

=== "value form"

    ```cpp
    t.permute(Int<2>(), Int<0>(), Int<1>());   // reorder axes (a permutation of 0..N-1)
    t.permute(Int<-1>(), Int<0>(), Int<1>());  // negatives count from the back
    t.flip(Int<1>());                          // reverse an axis (a negative-stride view; needs a
                                               //   signed index type, which shape<...> is)
    ```

=== "template form"

    ```cpp
    t.permute<2,0,1>();               // reorder axes (a permutation of 0..N-1)
    t.permute<-1,0,1>();              // negatives count from the back
    t.flip<1>();                      // reverse an axis (a negative-stride view)
    ```

## Add / drop / reshape

=== "axis<...> form"

    ```cpp
    t.unsqueeze(axis<2>{});    // insert a size-1 axis (numpy newaxis) -> rank+1
    t.unsqueeze(axis<-1>{});   // append a trailing axis, e.g. (H,W) -> (H,W,1)
    t.squeeze(axis<3>{});      // drop a specific size-1 axis -> rank-1
    t.squeeze();               // drop EVERY statically-size-1 axis
    // reshape/flatten aren't axis-list ops (they take dimension SIZES, not axis
    // positions) — use the Int<k>()/template form below
    ```

=== "value form"

    ```cpp
    t.unsqueeze(Int<2>());   // insert a size-1 axis (numpy newaxis) -> rank+1
    t.unsqueeze(Int<-1>());  // append a trailing axis, e.g. (H,W) -> (H,W,1)
    t.squeeze(Int<3>());     // drop a specific size-1 axis -> rank-1
    t.squeeze();             // drop EVERY statically-size-1 axis
    t.reshape(Int<6>(), Int<4>());   // view as a new shape (same numel) — no copy when viewable
    t.reshape(Int<6>(), Int<-1>());  // one -1 dimension is inferred from numel
    t.flatten();             // view as 1-D (ravel)
    ```

=== "template form"

    ```cpp
    t.unsqueeze<2>();   // insert a size-1 axis (numpy newaxis) -> rank+1
    t.unsqueeze<-1>();  // append a trailing axis, e.g. (H,W) -> (H,W,1)
    t.squeeze<3>();     // drop a specific size-1 axis -> rank-1
    t.squeeze();        // drop EVERY statically-size-1 axis
    t.reshape<6,4>();   // view as a new shape (same numel) — no copy when viewable
    t.reshape<6,-1>();  // one -1 dimension is inferred from numel
    t.flatten();        // view as 1-D (ravel)
    ```

## Multiple axes at once

`squeeze`/`unsqueeze` also take **several** axes in one call, inserting or
dropping them all at once instead of chaining single-axis calls:

```cpp
(h,w).unsqueeze<1,3>();          // (H,W) -> (H,1,W,1)
(1,h,1,w)_view.squeeze<0,2>();   // (1,H,1,W) -> (H,W)
```

The tricky part — axis positions shift as siblings get inserted or dropped — is
handled for you: `unsqueeze`'s axes are positions in the **final** (post-insert)
rank, and `squeeze`'s are positions in the **source** rank; both must be
**distinct**, but you can list them in **any order** — `t.squeeze<2,0>()` ==
`t.squeeze<0,2>()`. Internally each folds one axis at a time in the direction
that keeps the remaining positions valid — `unsqueeze` smallest-first (each
insert lands left of what's left to do), `squeeze` largest-first (each drop
never shifts an earlier position) — sorting the axes into that order at compile
time first, so you don't need to think about it or list them ascending yourself.

A single axis (or none, for `squeeze`) still means the one-axis form above —
arity alone picks the multi-axis overload.

`reshape`/`flatten` follow **numpy semantics**: they return a **view** whenever the
new shape is reachable without a copy — not only from a C-contiguous tensor, but any
layout that regroups in C-order (splitting a contiguous axis, merging a contiguous
run — so a strided or permuted source often still views). The output is a folded
`strides<...>` view (compile-time strides when the source is fully static). When the
requested regrouping genuinely needs a copy (e.g. it would cross a stride gap), it is
a **compile error** for a static source and a debug check for a dynamic one — query
first, or `clone()`:

```cpp
t.can_reshape_without_copy<6,4>();  // will reshape(Int<6>(),Int<4>()) be a view? (numpy's rule)
auto c = t.clone();                 // a dense, row-major OWNING copy (static -> stack, dyn -> heap)
c.reshape(Int<6>(), Int<4>());      // guaranteed viewable now
```

`is_dense()` with no argument asks only whether the elements occupy a dense block
of memory in *some* axis order — so a *permuted* C-contiguous view still counts.
`is_contiguous()` is the **C-order** question (numpy/pytorch's meaning); a
C-contiguous tensor is always reshapable to any matching shape. `is_contiguous<fcontiguous>()`
asks F-order. Both are thin
aliases of `is_dense<Layout>()` (the exact-layout check), so `is_contiguous()` ==
`is_dense<ccontiguous>()`. The exact-layout check takes its layout either way — as
an argument (value form) or as a template parameter:

=== "value form"

    ```cpp
    t.is_dense(ccontiguous{});        // dense in C-order?  (layout deduced from the argument)
    t.is_dense(fcontiguous{});        // ...F-order?
    t.is_contiguous(ccontiguous{});   // C-contiguous specifically
    ```

=== "template form"

    ```cpp
    t.is_dense<ccontiguous>();        // dense in C-order?
    t.is_dense<fcontiguous>();        // ...F-order?
    t.is_contiguous<fcontiguous>();   // F-contiguous
    ```

(`is_dense()` / `is_contiguous()` with no argument are the argument-free defaults —
any-order denseness, and C-order contiguity respectively.)

!!! note "mdspan equivalent"
    teeny leads with the numpy/pytorch vocabulary — `t.shape()`, `t.shape(d)`,
    `t.strides()`, `t.numel()`, `t.rank()`, `t.is_contiguous()`. The mdspan spellings
    are still there as an interop escape hatch: `t.extents()` / `t.mapping()` return
    the raw `cuda::std` objects and `t.extent(d) == t.shape(d)`. See
    [mdspan vs teeny](mdspan-vs-teeny.md) for the full map.

## Recover static inner dims

At the array boundary a view is often fully dynamic. `recast` reinterprets it
with a **more-static** `shape<...>` of the same rank, so known inner dims fold:

```cpp
auto dyn = wrap(ptr, shape<-1,-1,-1>{n,3,3});  // came in fully dynamic
```

=== "value form"

    ```cpp
    auto st = dyn.recast(shape<-1,3,3>{});  // the 3s are now compile-time
    ```

=== "template form"

    ```cpp
    auto st = dyn.recast<shape<-1,3,3>>();  // the 3s are now compile-time
    ```

Static dims of the target are validated against the actual shape. `recast`
**preserves the source's strides and works on any layout** (no copy, no contiguity
requirement): a strided or transposed source keeps its strides — they fold to
compile-time constants where the source layout makes them derivable (contiguous /
`strides<>`), and stay run-time for a `dynamic_strides` source. It only re-types
the *shape*; it never assumes row-major (which would silently mis-address a
non-contiguous view).

A second **layout** argument overrides that when you *want* to reinterpret the
data with a specific layout — e.g. to fold a `dynamic_strides` cell's inner strides
to compile-time constants when you know it's contiguous:

=== "value form"

    ```cpp
    auto sv = dyn.recast(shape<-1,3,3>{}, ccontiguous{});  // reinterpret AS row-major: strides fold to (9,3,1)
    ```

=== "template form"

    ```cpp
    auto st = dyn.recast<shape<-1,3,3>, ccontiguous>();    // reinterpret AS row-major: strides fold to (9,3,1)
    ```

`ccontiguous`/`fcontiguous` derive the strides from the shape (**you** promise
the data is contiguous in that order — UB if not); `strides<S...>` imposes explicit
strides. The default (`keep_strides`) preserves the source strides, as above.

## nd-peel — iterate a subset of axes

Peel some axes and get a lower-rank sub-view for each, without writing index
arithmetic.

```cpp
for (auto line : peel<0,1>(t)) work(line);  // peel axes 0,1; each is a view
auto s = peel_at<0,1>(t, i);                // the i-th sub-view (grid-stride style)

for (auto cell : peel_front<N>(t)) work(cell);  // peel the FIRST N axes
auto c = peel_front_at<N>(t, i);                // the i-th
```

The peeled-axis selector has a **value form** too — pass `axis<...>{}` (a
compile-time axis list, the sibling of `shape<...>`, like numpy's
`axis: int | list[int]`) instead of the `<...>` template list. It reads the same
and, being a deduced argument, needs no `.template` on a type-dependent receiver:

=== "value form"

    ```cpp
    for (auto line : peel(t, axis<0,1>{})) work(line);   // == peel<0,1>(t)
    auto s = peel_at(t, i, axis<0,1>{});                 // == peel_at<0,1>(t, i)
    ```

=== "template form"

    ```cpp
    for (auto line : peel<0,1>(t)) work(line);
    auto s = peel_at<0,1>(t, i);
    ```

(`peel_front<N>` / `size_front<N>` take a **count**, not an axis list, so they stay
template-only.) `peel_front<N>` handles **arbitrary batch rank**: for a `(*batch, *spatial, C)`
tensor, `peel_front<Nbatch>` yields `(*spatial, C)` sub-views to parallelise
over — one per CPU thread or CUDA thread. Each sub-view already has the batch
offset baked into its pointer, so the inner kernel sees only spatial strides.

### `peel_zip` — walk 2 or 3 tensors in lock-step

`peel<Axes...>(t)` iterates ONE tensor. For 2 or 3 tensors that should be walked
together — the classic case is a triangle's three per-vertex coordinate tensors —
`peel_zip` yields one `cs::tuple` of views per step instead of independently
peeling each and trusting the iteration stays in lock-step:

```cpp
for (auto [va, vb, vc] : peel_zip<0>(a, b, c)) work(va, vb, vc);  // one tuple per step
```

A **distinct name from `peel`**, not an overload: peeling 1 tensor yields a view
per step, but 2+ would yield a tuple — a silent return-type bifurcation on arity
that a typo'd extra argument could trigger unnoticed. (Python's own `zip()` is the
same call: its own name, not an overload of single-iterable iteration.)

The operands can differ in shape as long as they're **broadcast-compatible** —
numpy's right-aligned rule, the same one `a + b` uses — and `Axes...` name axes in
the **broadcast rank's** numbering (the larger of the operands' own ranks;
negatives wrap against it):

```cpp
auto verts = local<double, shape<4,3>>();   // 4 triangles' vertex 0, xyz each
auto w     = local<double, shape<4,1>>();   // one scalar weight per triangle
for (auto [v, wt] : peel_zip<0>(verts, w)) { ... }   // wt's extent-1 axis broadcasts against v's 3 coords
```

Composes with `axis<...>{}` (trailing, after every positional tensor — a
`peel_zip`-specific placement, unlike `take_along`/`peel_at`'s leading tag, since
`peel_zip`'s tensor arguments are each a single fixed-arity positional rather than
an open pack), `.enumerate()` (yields `(multi_index, tuple)`, same shape as the
single-tensor form), and `.subrange(lo,hi)` for chunked/threaded sweeps:

```cpp
for (auto [va,vb,vc] : peel_zip(a, b, c, axis<0>{})) { ... }       // == peel_zip<0>(a,b,c)
for (auto [m, cell] : peel_zip<0>(a, b).enumerate()) { ... }       // m[d] = coord of peeled axis d
for (auto cell : peel_zip<0>(a, b).subrange(lo, hi)) { ... }       // a [lo,hi) chunk
```

Every operand must be **all-mutable or all-const** — there's no mixed-mutability
overload. Passing one non-const and one const tensor silently resolves to the
all-const overload (so writing through the "mutable" one is a compile error at
the write site, not a runtime surprise, but it's easy to miss why); write into a
separate destination, or make every operand `const`, if you don't need to mutate.
