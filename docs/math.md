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

a.minimum_(b);  a.maximum_(2.0);        // running min/max update (#325): *this =
                                        //   min/max(*this, b); tensor rhs broadcasts,
                                        //   scalar rhs applies to all — the running-
                                        //   nearest-distance idiom: best.minimum_(candidate)
```

!!! warning "Don't write through a self-overlapping view"
    `wrap` trusts the strides you pass, so a **stride-0** axis (or a stride smaller
    than an inner extent) makes a view where several indices alias the same element.
    *Reading* one is fine — that's how a broadcast RHS works — but **writing** into
    one is wrong either way round:

    - an in-place write applies the update to that one element repeatedly
      (`v.add_(b)` double-counts);
    - an out-of-place [`into(dest)`](#writing-into-a-preallocated-destination-intodest)
      write stores many results into the one slot, so all but the last are silently
      discarded.

    A host-debug check rejects **any** write whose destination has an `extent > 1`
    axis with stride 0 — the in-place ops (`v.add_(b)`, `v.mul_(2.0)`, `v.exp_()`,
    `v.iota_()`, …) *and* every `into(dest)` producer (`a.add(b, into(v))`,
    `a.mul(2.0, into(v))`, `exp(a, into(v))`, `clamp(a, lo, hi, into(v))`, …).
    `clone()` to a dense tensor first if you need to write.

`atomic_add_`/`atomic_sub_` accumulate a delta **atomically**, on device
(`atomicAdd`) and on the host (`cuda::std::atomic_ref`) alike — the
scatter/"push" accumulate. Both a broadcasting tensor rhs and a scalar rhs:

```cpp
a.atomic_add_(b);    // accumulate a delta, atomic on host and device
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
(Should the two also disagree in *signedness*, the result steps up to a signed type
wide enough for both ranges, for the same reason the in-place offset math below does.)
In place (`a.add_(b)`) and into a caller-supplied `into(dest)` there is no new
result to widen, so the offset math itself runs in a type that covers every tensor in
play while each keeps its own: a narrow-indexed destination never truncates a
wide-indexed right-hand side, in either direction. That holds for every producer that
takes an `into(dest)` — a scalar right-hand side (`a.mul(2.0, into(y))`) and a unary
op (`exp(a, into(y))`) as much as a tensor one — and for `allclose(a, b)`, which walks
two operands of its own. The same holds across **signedness**: an unsigned-indexed
tensor next to a flipped (negative-stride) signed-indexed one computes in a signed
type, so the negative stride stays negative.

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
a.map(f, into(y));                    // ...into a preallocated buffer -> y& (no allocation)
a.at(i, j).atomic_add_(v);            // scatter: a(i,j) += v — ATOMIC on host and device
```

`map_`/`zip_with_` take a functor **struct** (a lambda would need
`--extended-lambda` under nvcc). `at(i...).atomic_add_(v)` is the write half of a
scatter/"push" kernel — `atomicAdd` on device, `cuda::std::atomic_ref` on the
host, so a push kernel parallelised with `std::thread`/OpenMP over overlapping
outputs is race-free on the host too.

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
`a.normalize()`, `a.normalize(axis<1>{})`, `a.cross(b)`, `a.map(f)`, … — and each
takes the same optional `into(y)` (`a.exp(into(y))`). The free forms (`exp(a)`)
and the in-place `_` forms (`a.exp_()`) still exist; pick whichever reads best.

### Writing into a preallocated destination — `into(dest)`

Pass `into(y)` as the **last** argument to any of these producers to write the
result straight into `y` — one fused pass, **no allocation** — instead of
returning a fresh tensor. It returns `y&`. This is the kernel-friendly form (the
numpy/pytorch `out=`), and the way to write into a strided slot of a larger array.

```cpp
a.add(b, into(y));            // y = a + b   (one pass; y holds the result's shape, any dtype)
a.mul(b, into(y));  a.div(b, into(y));  a.pow(b, into(y));
a.add(2.0, into(y));          // scalar rhs works too
exp(a, into(y));  sqrt(a, into(y));  neg(a, into(y));   // every unary
minimum(a, b, into(y));  maximum(a, s, into(y));  clamp(a, lo, hi, into(y));
normalize(a, into(y));
normalize(a, axis<1>{}, into(y));   // the axis form too (y matches `a`'s full shape exactly)
a.map(f, into(y));             // the user-functor producer
cross(a, b, into(N(i, all)));  // 3D cross straight into row i of a matrix ("crossto")
```

**The destination may be a slice, written in place.** Every view-producing op —
slicing, `at`, `permute`, `unsqueeze`, `slice_along`, … — hands back its view *by
value*, and `into()` takes one of those directly, so a slot of a bigger output
needs no named intermediate. The write goes through to the storage the slice
refers to:

```cpp
auto N = local<double, shape<4,3>>();        // four 3-vectors, one per row
cross(a, b, into(N(i, all)));                // row i        <- a × b
sum(m, axis<0>{}, into(rows(j, all)));       // row j        <- a column sum
auto cells = local<double, shape<2,2>>();    // a matrix of scalars
sum(a, into(cells.at(i, j)));                // one cell     <- a full reduction
```

Use such a call for its **effect** — don't keep the `dest&` it returns, since the
temporary view it refers to is gone at the end of the statement (`auto & r =
cross(a, b, into(N(i, all)))` dangles). The same rule applies to the `into(...)`
tag itself: don't name it and reuse it later (`auto tag = into(N(i, all));`
followed by a separate statement using `tag`) — the temporary view `into()`
captured is just as gone by then. A temporary *owning* tensor is not a
legal destination at all (`into(zeros<double>(shape<3>{}))` is a compile error):
its storage would die with the statement, so the result would be thrown away.

`into(y)` is a **distinct type**, so it never collides with a scalar argument —
which is what lets `add`/`sub` also offer the **fused out-of-place axpy** (the
out-of-place twin of `add_(b, alpha)`):

```cpp
auto c = a.add(b, alpha);     // -> new,  a + alpha*b
a.add(b, alpha, into(y));     // -> y,    a + alpha*b   (sub likewise)
```

`into(y)` writes in a single pass; `y` may share memory with an operand, and `y`'s
dtype need not match — the result is cast to it.

**Only the result is cast.** The arithmetic itself runs in the *operands'* own
precision — including a scalar right-hand side and the axpy coefficient — and `y`'s
dtype applies to the store at the end. So `a.op(b, into(y))` gives exactly the same
numbers as `y.copy_(a.op(b))`; the `into` form just skips the temporary:

```cpp
auto a = local<double, shape<3>>(); a(0)=1.5; a(1)=2.5; a(2)=3.5;
auto y = local<int, shape<3>>();

a.mul(0.5, into(y));          // y = {0, 1, 1}   (0.75, 1.25, 1.75 truncated on the store)
exp(a, into(y));              // y = {4, 12, 33} (exp of 1.5/2.5/3.5, then truncated)
```

This holds for a `half`/`bfloat16` operand too, not just for wider types: the
arithmetic runs in `float` either way, but `into(y)` rounds through the *allocating
twin's own element type* (`promote_t<Ta,Tb>` — a 16-bit float, when the source is)
before casting to `y`, exactly mirroring the extra rounding step
`y.copy_(a.op(b))` performs when it narrows its temporary result down to that
16-bit type on the way. Skipping that stop would make `into(y)` slightly more
precise than its twin (one rounding instead of two) — closer to the exact math, but
a different number, which breaks the equivalence this section promises.

The flip side is that the operands' own promotion rule still applies, whatever `y`
is: two integer tensors divide as integers even into a floating destination, exactly
as they would without `into` (`y`'s dtype never promotes the operation).

```cpp
auto n = local<int, shape<2>>(); n(0)=7; n(1)=9;
auto d = local<int, shape<2>>(); d.fill_(2);
auto f = local<double, shape<2>>();
n.div(d, into(f));            // f = {3, 4}  — integer division, == f.copy_(n.div(d))
```

**`y`'s shape is checked**: against the source's own shape for a unary or
scalar-rhs op, and axis by axis against `y` itself for a tensor-rhs one — each
operand must broadcast into the `y` you passed. Only the *operands* broadcast —
the destination never stretches, so a `y` that is smaller in any axis is an
error, not a repeated write. A mismatch is caught at **compile time** whenever
the extents in play are static, and by a debug-time check (an `assert`, compiled
out under `-DNDEBUG`) when any of them is dynamic — for every producer alike,
broadcasting or not:

```cpp
auto a = zeros(shape<8,8>{});
auto y = zeros(shape<2,2>{});
a.mul(2.0, into(y));          // compile error: dest's shape must match the source's
exp(a, into(y));              // same
a.add(a, into(y));            // same, for the broadcasting tensor-rhs form
```

The two rules differ only in what counts as a match. A unary or scalar-rhs op wants
**exact** equality — it has nothing to stretch. A tensor-rhs op broadcasts, so each
*operand* axis must either equal `y`'s or be 1 (an extent-1 operand axis stretches
over `y`). `y`'s own axes never stretch in either case:

```cpp
auto row = zeros(shape<1,3>{});
auto out = zeros(shape<2,3>{});
out.add(row, into(out));      // fine: the (1,3) operand stretches over (2,3)
out.add(out, into(row));      // compile error: a (1,3) DEST does not stretch
```

**A tensor-rhs `y` may be bigger than the natural result — deliberately.** Read
`a.add(b, into(y))` as "`y` = `a + b`, minus the allocation and the copy": `y` is
the result shape *you* chose, and the operands broadcast into it exactly as they
would into the left-hand side of `y.copy_(a + b)`. So a `y` that is larger — or
of higher rank — than the tensor `a + b` alone would have returned is legal
whenever every operand still stretches into it, and the smaller natural result
fills all of `y`:

```cpp
auto u = zeros(shape<1,3>{});  auto w = zeros(shape<1,3>{});
auto big = zeros(shape<5,3>{});
u.add(w, into(big));          // u + w alone is (1,3); here it fills all 5 rows
auto batch = zeros(shape<2,1,3>{});
u.add(w, into(batch));        // higher rank too: operands right-align and stretch
```

This is the same broadcasting the non-`into` spelling applies, pointed at the
destination you supplied — handy for filling a batch axis of a preallocated
buffer in one pass. What stays rejected is any axis where an operand can neither
match `y` nor stretch (its extent is neither `y`'s nor 1) — the smaller-`y` cases
above included. If you want the exact natural-result shape enforced, size `y`
that way: the check pins every operand against the `y` you pass.

Which rule a producer uses follows from the *result it promises*, not from how it
happens to be computed. `normalize(a, into(y))` is the case worth spelling out:
whether you normalize the whole tensor or only along named axes, the result is `a`
element for element, so **`y` must match `a` exactly** — for both spellings. The
axis form divides by a reduced-then-keepdim tensor internally, but that divisor is
not an operand you get to pick, so none of the broadcasting leeway above applies to
`y`:

```cpp
auto a = zeros(shape<1,3>{});
normalize<1>(a, into(zeros(shape<1,3>{})));   // fine: y has a's shape
normalize<1>(a, into(zeros(shape<5,3>{})));   // compile error: y is not a's shape
                                              //   (a's extent-1 axis does NOT stretch here)
normalize<1>(a, into(zeros(shape<2,1,3>{}))); // compile error: y's rank is not a's
```

The two contracts are not an inconsistency. A tensor-rhs op hands you the result
shape to choose because both of its operands legitimately broadcast; `normalize` —
like every unary and scalar-rhs producer — promises a result congruent with `a`,
so a bigger `y` could only ever hold replicas. If replicas are really what you
want, say so: `y.copy_(normalize(a))`.

`y` must also not **self-overlap** — no `extent > 1` axis with stride 0. Such a `y`
would take many results into one element and keep only the last; a debug-time check
rejects it, for every `into(dest)` producer alike (see the warning
[above](#in-place-ops-mutate-this)).

Reductions take `into(dest)` too. A **full** reduction (all axes) writes its
scalar into a **rank-0** destination; an **axis** reduction copies its lower-rank
result into `dest`:

```cpp
sum(a, into(cell));           // cell : local<double, shape<>>{} — a rank-0 scalar
dot(a, b, into(cell));        // full reductions: sum/prod/max/min/mean/sqnorm/norm/dot/sqdist/dist
allclose(a, b, into(flag));   // ...and allclose, whose answer lands in a rank-0 bool cell
sum<0>(m, into(colbuf));      // axis reduction -> a lower-rank dest
mean(m, axis<1>{}, into(rowbuf));  // value form takes into as well
```

A rank-0 destination can be a `local<T, shape<>>{}` or a view over a bare
address, `wrap(&x, shape<>{})` — so writing a full reduction "to an address" needs
no extra overload. A full reduction **requires** a rank-0 dest (a non-rank-0 dest is
a `static_assert`, so forgetting the `<axes>` fails to compile rather than silently
splatting the grand total); an axis reduction's dest is **broadcast-compatible**
with the reduced shape (it goes through `copy_`). As elsewhere, `dest`'s dtype need
not match — the accumulation runs in the reduction's own accumulator type and only
the final result is cast — and the destination is returned by reference. **Every**
pair of dtypes works here, the two 16-bit floats included: `sum(half_a,
into(bfloat16_cell))` accumulates as any other `half` reduction does and converts
the scalar to `bfloat16` on the way out, exactly like a wider destination.

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

## Keyword arguments

Reductions (and, elsewhere in the API, factories like `empty`/`zeros`/`wrap`)
accept a run of **keyword-like value tags** after their required positional
arguments — here that's `dtype<T>{}`, `axis<...>{}`, `keepdims`, `into(dest)`.
This is a design rule of the whole API, not something you need to know the
mechanism for to use: keywords are **trailing** (after every positional
argument), **order-free among themselves**, and there is **one of each kind
per call** — matched by distinct type, so a call site can tell a misplaced or
duplicated keyword apart from a real argument and reject it with a clear
compile error instead of silently doing the wrong thing. That's why every
combination below — `sum(a, dtype<double>{}, axis<0>{}, keepdims, into(buf))`,
in any order — just works, without a hand-written overload for every
arrangement. (The one exception: where a selector still needs an *explicit*
`<...>` template argument — the `<Acc, Axes...>` reduction split, since C++17
has no single template parameter that means "leading type = accumulator" *and*
"leading int = axis" — that split stays; it's the keyword bag *past* it that's
generic.) See [Tensors & ownership](tensors.md#factories) for the same
contract on `empty`/`zeros`/`wrap`'s `dtype`/`storage_c`/layout keywords.

## Reductions

Over all axes → a scalar:

```cpp
sum(a); prod(a); max(a); min(a); mean(a); dot(a, b);
a.sum(); a.mean(); a.dot(b);            // the same reductions are also methods
```

Every reduction (and `sqnorm`/`norm`/`sqdist`/`dist`) is available as a **method**
too, with the same overload shapes — `a.sum()`, `a.sum<0>()`, `a.mean(axis<1>{})`,
`a.norm()`, `a.dot(b)`, `a.sqdist(b)`, `a.sum(into(cell))`. The free `sum(a)` form
stays as well.

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

The axes must be **distinct**. You may list them in any order (`sum<2,0>(a)` ==
`sum<0,2>(a)`), but naming one twice is a **compile error** rather than a silently
dropped duplicate — `sum<0,0>(a)` and `sum(a, axis<0,0>{})` both fail with
*"sum: axes must be distinct — each axis may be reduced only once"*. Negative axes
are normalised first, so a duplicate spelled two different ways is caught too
(`mean<1,-2>(a)` on a rank-3 tensor). The rule is the same with and without
`keepdims`.

### An empty axis list reduces over *no* axis

`axis<>{}` names no axis, so `sum(a, axis<>{})` reduces over none of them: each
output cell aggregates the single element at its own index, and the result keeps
`a`'s shape. That is numpy's rule for an empty axis tuple (`np.sum(a, axis=())`
hands `a` back unchanged), and the same "an empty axis list asks for nothing" the
rest of teeny's axis-list ops follow (`t.squeeze(axis<>{})`, `peel(t, axis<>{})`).

```cpp
auto a = ...            // (2,3), values [[0,1,2],[3,4,5]]
sum(a, axis<>{});       // (2,3) [[0,1,2],[3,4,5]] — a copy; nothing was summed
sum(a);                 // 15 — NO axis argument still means EVERY axis
```

The second line is the contrast worth remembering: **an empty axis list and no
axis argument at all are different requests.** Leaving the keyword out is the
documented full reduction (every axis, giving a scalar); passing an empty list
asks to reduce over zero axes. So generic code that *computes* an axis list stays
correct when the computed list comes out empty, instead of silently collapsing the
whole tensor to a single number.

`sqnorm` and `norm` follow the same rule rather than coming out as plain copies,
because they are Σaᵢ² and √Σaᵢ² *over the named axes*: with no axis named, each
cell's sum runs over its own element alone, so `sqnorm(a, axis<>{})` is the
elementwise `a²` and `norm(a, axis<>{})` the elementwise `|a|`.
`sum`/`prod`/`max`/`min`/`mean` all come out as a copy of `a` in the result type.

The result is an owned tensor, exactly like any other axis reduction, and the
other keywords compose as usual — `dtype<Acc>{}` sets the result type, `into(dest)`
writes into a destination with `a`'s shape, and `keepdims` has no reduced axis to
keep, so it changes nothing.

### `keepdims` — keep the reduced axes as size-1

Pass `keepdims` (any subset, any order, alongside `dtype<Acc>{}`/`axis<...>{}`/
`into(dest)`) to keep the named axes as size-1 instead of removing them —
numpy/pytorch's `keepdims=True` — so the result **broadcasts back** against the
input without an extra `unsqueeze`:

```cpp
sum<0>(a, keepdims);              // (H,W) -> (1,W), instead of (W,)
sum(a, axis<0,2>{}, keepdims);    // value form takes it too
sum<double,0>(a, keepdims);       // composes with a leading Acc type
sum<0>(a, keepdims, into(dest));  // and with into(dest) — dest matches the kept-dims shape
```

Applies to every axis reduction (`sum`/`prod`/`max`/`min`/`mean`/`sqnorm`/`norm`).
`keepdims` is a distinct empty-tag value (like `all`/`none`), so it never collides
with another argument. The axes must be **distinct** (a repeat is a compile error),
but — as everywhere else in teeny — you may list them in **any order**:
`sum<2,0>(a, keepdims)` == `sum<0,2>(a, keepdims)`, kept axes and all. Both rules
hold identically with and without `keepdims`.

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

sqdist(a, b);         // Σ(aᵢ-bᵢ)² — mathematically sqnorm(a-b), one fused pass, no a-b
                      //   intermediate (not necessarily bit-identical; see prose below)
dist(a, b);           // √Σ(aᵢ-bᵢ)² — mathematically norm(a-b). Binary only (no axis form, like dot)
sqdist<double>(a, b); // leading TYPE = accumulator AND result (as with sqnorm/dot)
sqdist(a, b, dtype<double>{}, into(cell));   // dtype/into compose, same trailing bag as dot

a.normalize_();       // in place: a /= norm(a)   (floating element types)
auto u = normalize(a);// out-of-place unit vector -> new tensor (static->stack, dynamic->heap)
a.normalize_<1>();  normalize<-1>(a);  normalize(a, axis<1>{});   // OVER NAMED AXES (keepdim broadcast)
a.normalize(axis<1>{});  a.normalize<1>();                        // ...as a METHOD, either spelling
normalize(a, axis<1>{}, into(y));  a.normalize<1>(into(y));       // ...and each takes into(y)

auto c = cross(a, b);       // 3D cross product a × b -> new stack 3-vector (rank-1, length 3)
a.cross_(b);                // in place: a becomes a × b (mirrors add_/mul_; aliasing-safe)
cross(a, b, into(N(i, all)));   // straight into a preallocated slot — row i of a matrix
```

`sqnorm`/`norm` are reductions: with no axes they reduce over everything (so `norm`
of a matrix is the Frobenius norm); with `<Axes...>` (or the `axis<...>` value form,
or a leading accumulator type) they reduce over just those axes into a lower-rank
tensor — the same API as `sum`/`mean`. `sqdist`/`dist` are the two-operand siblings —
binary only, no axis-list form, mirroring `dot`'s own convenience-wrapper status
over a manual `sum(a*b)` (same `<Acc>`/`dtype<Acc>{}`/`into(dest)` composition).
Each difference is formed and squared directly in the accumulator type, so for a
narrow element type `sqdist(a,b)` can be *more* accurate than the un-fused
`sqnorm(a-b)` spelling (which rounds `a-b` to the operands' own type first) —
the two are only guaranteed bit-identical for `double` operands.
`normalize`/`normalize_` mirror `sqnorm`/`norm`: with
`<Axes...>` each sub-vector is divided by its norm over those axes (the reduced axes
are kept as size-1 so the norm broadcasts back). Axes must be distinct, but may be
listed in any order (`normalize<2,0>(a)` == `normalize<0,2>(a)`).
`normalize` of a zero vector yields NaNs — this is exact math with no epsilon; add
one at the call site if you need it.

The axis-scoped spellings follow the same host/device rule as every other
allocating op, with one useful wrinkle. `normalize<Axes...>` divides by
`norm<Axes...>(a)`, and that reduced norm is itself a *tensor* — stack-owned (so
usable on the device) when the reduced extents are static, heap-owned (host only)
when they are not. So `normalize<1>(a)` and `a.normalize<1>()`, which additionally
materialise a result the same shape as `a`, are device-callable when `a` is fully
static; while `a.normalize_<1>()` and `normalize<1>(a, into(y))`, which allocate
nothing but that norm, are device-callable whenever the *reduced* extents are
static. Reducing a `shape<-1,3>` tensor over axis 0 leaves a `shape<3>` stack norm,
so normalizing its columns in place works inside a kernel even though the source
has a dynamic axis.

`cross` is defined only for rank-1, length-3
operands (a `static_assert` catches a wrong static length; a runtime length is
debug-checked). Its in-place form is the member `a.cross_(b)` (`a = a × b`); to
write into a *separate* buffer, pass a destination — `cross(a, b, into(slot))`,
where `slot` may itself be a slice of a bigger array (`N(i, all)` for row `i` of
a matrix of 3-vectors), computed in one pass with nothing allocated.

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

### Approximate equality: `allclose`

`allclose(a, b)` answers numpy's question — is `|a-b| <= atol + rtol*|b|` for
*every* element? — in one pass, with no mask tensor in between. The operands
broadcast, and the two tolerances are optional positionals with numpy's defaults
(`rtol=1e-5`, `atol=1e-8`):

```cpp
allclose(a, b);                 // true if every element is within the default tolerances
allclose(a, b, 1e-3);           // looser relative tolerance
allclose(a, b, 0.0, 1e-6);      // absolute tolerance only
a.allclose(b);  a.allclose(b, 1e-3);   // also a method, like a.dot(b)
```

It is the binary sibling of `dot`/`sqdist`/`dist`, and takes the same trailing
keywords — in any subset, in any order, after the tolerances:

```cpp
allclose(a, b, dtype<float>{});          // carry the comparison out in float
allclose<float>(a, b);                   // == the same, as an explicit template argument
allclose(a, b, into(flag));              // flag : local<bool, shape<>>{} — a rank-0 cell
allclose(a, b, 1e-3, into(flag));        // a tolerance prefix, then the keywords
a.allclose(b, 1e-3, 1e-6, dtype<double>{}, into(flag));
```

`dtype<Acc>{}` picks the type the difference and the tolerance test are computed
in (the operands' compute type by default), which matters when a narrow type
rounds two values together — `1e10` and `1e10+1` are one `double` apart but the
same `float`. `into(dest)` writes the answer into a rank-0 destination and
returns it by reference, allocating nothing: a `bool` cell keeps the answer
exactly, any other element type takes the `0`/`1` cast.
