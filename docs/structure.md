# Views & structure

These return **views** (no copy) that rearrange, reshape, or iterate a tensor.
Axis template arguments are signed — **negatives count from the back**.

The axis-**list** ops — `permute`, `flip`, `squeeze`, `unsqueeze` — take an `axis<...>{}`
tag (a compile-time axis list, sibling of `shape<...>`, the same one
`peel`/`slice_along`/the reductions use): `t.squeeze(axis<0,2>{})` == `t.squeeze<0,2>()`,
`t.flip(axis<0,2>{})` == `t.flip<0,2>()`,
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
    t.flip(axis<1>{});           // reverse an axis (a negative-stride view; needs a
                                 //   signed index type, which shape<...> is)
    t.flip(axis<0,2>{});         // reverse SEVERAL axes at once (numpy flip(a, axis=(0,2)))
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
    t.flip<0,2>();                    // reverse SEVERAL axes at once
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

`squeeze`/`unsqueeze`/`flip` also take **several** axes in one call, inserting,
dropping or reversing them all at once instead of chaining single-axis calls:

```cpp
(h,w).unsqueeze<1,3>();          // (H,W) -> (H,1,W,1)
(1,h,1,w)_view.squeeze<0,2>();   // (1,H,1,W) -> (H,W)
t.flip<0,2>();                   // reverse axes 0 and 2 (numpy flip(a, axis=(0,2)))
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

`flip` has none of that trickiness: reversing one axis never disturbs another, so
its axis list is simply a **set** — distinct axes, any order, and `t.flip<0,2>()`,
`t.flip<2,0>()` and `t.flip<0>().flip<2>()` are the very same view (same type, same
elements). It builds that view in one pass rather than a chain: each named axis
gets its stride negated and the base pointer moved to its last element along that
axis, and every stride the source knows at compile time stays compile-time known
(negated).

One edge case worth naming: an **empty** tensor (`numel() == 0`) has no last
element for the origin to move to, so flipping it leaves `data()` exactly where
it was — whichever axes you name, and whether it is a named axis or some other
one that is empty. So `wrap(p, shape<-1,3>{0}).flip<0,1>().data() == p`, and the
view never forms a pointer past the (possibly zero-length) allocation.

A single axis (or none, for `squeeze`) still means the one-axis form above —
arity alone picks the multi-axis overload.

### An empty axis list is a no-op

`axis<>{}` names *no* axis, so `t.squeeze(axis<>{})`, `t.unsqueeze(axis<>{})` and
`t.flip(axis<>{})`
hand back the same shape and strides — the same rule numpy gives an empty axis
tuple (`np.squeeze(a, axis=())` and `np.expand_dims(a, axis=())` both leave the
shape alone). That matters for generic code that *computes* the axis list: when
the computed set comes out empty, the call does nothing instead of quietly
restructuring the tensor.

```cpp
auto t = wrap(buf, shape<1,2,1,3>{});
t.squeeze(axis<>{});      // (1,2,1,3) -- unchanged, singleton axes kept
t.unsqueeze(axis<>{});    // (1,2,1,3) -- unchanged, nothing inserted
t.flip(axis<>{});         // (1,2,1,3) -- unchanged, no axis reversed
t.squeeze();              // (2,3)     -- the NO-ARGUMENT form still drops every singleton
t.unsqueeze();            // (1,1,2,1,3) -- ...and still inserts at axis 0
t.flip();                 // reversed along axis 0 -- ...and still defaults to axis 0
```

The last three lines are the contrast worth remembering: **an empty axis *list* is
not the same as *no argument at all*.** `squeeze()` (drop every statically-size-1
axis), `unsqueeze()` (insert at axis 0) and `flip()` (reverse axis 0) keep their
own meanings; only the
`axis<...>{}` spelling reads an empty list as "no axes named".

`permute` needs a *full* permutation of `rank()` axes, so `axis<>{}` is accepted
there only for a rank-0 tensor (the one permutation of no axes, a no-op) and is a
compile error for any other rank — an under-specified permutation is a mistake,
not a no-op.

The **reductions** read an empty list the same way: `sum(a, axis<>{})` reduces over
no axis and gives `a`'s shape back, while the plain `sum(a)` — no axis argument at
all — is the full, every-axis reduction. Same contrast, same reason; see
[Math & reductions](math.md#an-empty-axis-list-reduces-over-no-axis).

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

(`peel_front<N>` / `size_front<N>` take a **count**, not an axis list, so there is no
`axis<...>` tag for them — and being free functions they need no `.template` either
way. Their `anyrank` counterparts are *members*, so those do take the count as a
value: `at.peel_front(Int<-Sr>())`, see
[dispatch](dispatch.md#peel_front-sr--the-batch-pattern-preferred).)
`peel_front<N>` handles **arbitrary batch rank**: for a `(*batch, *spatial, C)`
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

The operands need not share an **index type** either (the `Idx` in
`shape_as<Idx, ...>`). The cells `peel_zip` hands you carry one index type wide
enough — and, where the operands disagree on signedness, *signed* enough — to
address every operand exactly, so a reversed view (from `flip`, or a negative
step) zipped against an unsigned-indexed tensor still steps backwards rather
than wrapping to a huge positive offset.

Composes with `axis<...>{}` (trailing, after every positional tensor — like
`peel_at`'s own tag, and unlike `slice_along`'s leading one, since `peel_zip`'s
and `peel_at`'s only other arguments are each fixed-arity — `peel_at`'s a
single index, `peel_zip`'s a fixed run of tensors — rather than `slice_along`'s
open per-axis pack), `.enumerate()` (yields `(multi_index, tuple)`, same shape
as the single-tensor form), and `.subrange(lo,hi)` for chunked/threaded sweeps:

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

### `scan_` / `scan` — sequential fold along one axis, batched over the rest

A per-axis recurrence — `carry = f(carry, x)`, `x = carry` — is inherently
**sequential** along the axis it walks, but every OTHER axis is just batched,
exactly what `peel` already collapses for other ops. `scan_<Axis>` peels every
axis except `Axis`, walks `Axis` in increasing order within each batch cell,
and threads `carry` through a device-safe functor:

```cpp
scan_<0>(t, 0.0, sum_op{});   // t := running cumulative sum along axis 0, batched over the rest
```

`f` is a plain callable struct (lambda-free engines, like `map_`/`zip_with_`'s
own functor convention): `Carry operator()(Carry carry, T x) const`. The
returned value doubles as both the new carry AND the new element — a 1-D
Felzenszwalb-style min-plus sweep (`carry = min(carry + w, x)`) is exactly this
shape (see `examples/distance_transform.cpp`'s hand-written twin). A **reverse**
sweep needs no separate flag — it composes with the existing negative-stride
view:

```cpp
scan_<0>(t.flip<0>(), 0.0, sum_op{});   // a temporary flip() view binds fine (scan_
                                        // has both lvalue and rvalue overloads)
```

The value form puts `axis<Axis>{}` **last**, after every positional argument:
`scan_(t, 0.0, sum_op{}, axis<0>{})` == `scan_<0>(t, 0.0, sum_op{})`. That is the
library-wide keyword rule — keywords trail, and are order-free among themselves —
and it matches `index_select(idx, axis<A>{})` and the reduction family.
A *leading* tag — `scan_(t, axis<0>{}, 0.0, sum_op{})` — is a compile error naming
the right order; there is no alias for it. The selector is `axis<...>` rather than
`Int<k>()` because `init` is itself an arbitrary `Carry` value that the selector has
to stay visually and typewise distinct from (see [CLAUDE.md](../CLAUDE.md)'s
selector-vocabulary rule).

Out-of-place `scan<Axis>` is a fresh dense copy, scanned (static shape -> stack,
dynamic -> heap, host-only, like `clone()` — which it's built on); `into(dest)`
writes into a preallocated buffer instead (one copy, no fresh allocation).
Unlike `copy_`'s own numpy-style broadcast, `dest` must match `t`'s shape
EXACTLY (checked — a compile error when both are static, a debug-time check
otherwise), since `scan_` then walks `dest`'s own axis numbering:

```cpp
auto out = scan<0>(t, 0.0, sum_op{});             // fresh tensor; t itself untouched
scan<0>(t, 0.0, sum_op{}, into(dest));            // writes into dest, returns dest&
```

`scan` has two trailing keywords, `axis<Axis>{}` and `into(dest)`, and they ride
the same generic keyword mechanism the factories and the reductions use — so any
subset in any order is the same call:

```cpp
scan(t, 0.0, sum_op{}, axis<0>{});                // == scan<0>(t, 0.0, sum_op{})
scan(t, 0.0, sum_op{}, axis<0>{}, into(dest));    // == scan<0>(t, 0.0, sum_op{}, into(dest))
scan(t, 0.0, sum_op{}, into(dest), axis<0>{});    // ...and the keywords may be swapped
```

#### The forward + backward sweep — two-pass line recurrences

A whole family of separable kernels runs the *same* line recurrence twice, once
in each direction: an L1 (min-plus) distance transform, and the causal +
anticausal passes of an IIR spline-coefficient prefilter (`splinc`). Written
with `scan_`, that is two calls — the second one on a flipped view (`ax` below is
the `axis<...>{}` tag naming the swept axis):

```cpp
scan_(t,          init, f, ax);   // forward  pass (increasing along ax)
scan_(t.flip(ax), init, f, ax);   // backward pass (decreasing along ax)
```

**That pair is the idiom** — there is no `scan2_`, no direction flag, and no
reason to write the loop by hand. The second call sees the same storage through
a negative-stride view, so it mutates `t` in place just like the first, and both
calls batch over every other axis themselves: nothing in the pair mentions the
batch rank, so the same two lines work on a `(H,W)` image and on a
`(*batch,H,W,C)` volume.

Worked example — a separable **L1 distance transform** along the last axis,
`carry = min(carry + w, x)` in both directions (the `examples/distance_transform.cpp`
kernel, whose hand-written twin this reproduces element for element):

=== "value form"

    ```cpp
    #include <teeny/teeny.h>
    #include <limits>
    using namespace tny;

    struct minplus {                          // device-safe functor, like map_'s
        double w;
        double operator()(double carry, double x) const {
            const double c = carry + w;
            return c < x ? c : x;
        }
    };

    template <class Tensor>
    void distance_l1(Tensor & t, double w) {
        constexpr double inf = std::numeric_limits<double>::infinity();
        constexpr auto ax = axis<-1>{};                  // the swept axis, named once
        scan_(t,          inf, minplus{w}, ax);          // forward
        scan_(t.flip(ax), inf, minplus{w}, ax);          // backward
    }
    ```

=== "template form"

    ```cpp
    template <class Tensor>
    void distance_l1(Tensor & t, double w) {
        constexpr double inf = std::numeric_limits<double>::infinity();
        scan_<-1>(t,                     inf, minplus{w});   // forward
        scan_<-1>(t.template flip<-1>(), inf, minplus{w});   // backward
    }
    ```

    `Tensor` is a template parameter here, so `flip<-1>()` needs the
    `.template` disambiguator — which is the reason to prefer the value form
    above inside a kernel template.

`init = +inf` is what makes each pass leave its own first element alone
(`min(inf + w, x) == x`), which is exactly how a hand-written sweep seeds its
carry from element 0. Pick the identity of your own recurrence the same way: it
is the carry value for which `f(init, x) == x`.

A **causal + anticausal IIR pass** has the same shape — one pole `p` forward
(`carry = x + p*carry`), one backward — with the boundary initialiser supplying
`init` instead of an identity.

Running it over more than one axis is the same pair per axis; nothing else
changes, because each `scan_` re-derives its own batching:

```cpp
auto swapped = t.permute<0,2,1>();   // a named view whose last axis is t's axis 1
distance_l1(t, w);                   // sweep along the last axis
distance_l1(swapped, w);             // ...then along the one before it
```
