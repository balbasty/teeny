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
```

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
x.at(i,j).add_<true>(v);     // atomic scatter into one cell
x.add_at(v, i, j);           // shorthand for at(i,j).add_<true>(v)
```

## Slicing → a sub-view (no copy)

If any argument is a slice specifier, `operator()` returns a lower- or same-rank
view into the same memory:

| argument | meaning |
|---|---|
| an integer | drop that axis (fix it at this index) |
| `all` | keep the whole axis |
| `slice(a, b)` | half-open range `[a, b)` |
| `slice(a, b, step)` | strided range (`step` may be negative) |
| `none` | an open end inside a `slice` |

```cpp
t(1, all, all);                // fix axis 0 -> lower-rank view
t(0, slice(1, 4));             // axis 1 range [1,4)
t(all, slice(none, 8, 2));     // every other element up to 8
t(all, slice(none, none, -1)); // reverse an axis (numpy a[:, ::-1])
```

`none` is python's `None`: `slice(none, k)` starts at 0, `slice(m, none)` runs to
the end, and `slice(none, none)` keeps the whole axis (it *is* `all`, and folds
to a static extent). Negative bounds wrap.

```cpp
t(slice(-2, none));   // last two rows
t(slice(3, 0, -1));   // rows 3,2,1  (stop excluded, like python)
```

Axes kept with `all` keep their static extent; a ranged axis becomes dynamic
(its size is a runtime value). Reach for `all` when you want the extent to keep
folding. The output layout is `strides<...>` with each kept stride folded to a
compile-time value where derivable.

**Compile-time bounds.** `slice<a,b>()` / `slice<a,b,step>()` bake the bounds
into the type, so a ranged axis keeps a static extent and stride (folds like
`all`):

```cpp
t(0, slice<1,4>());        // [1,4), static
t(all, slice<0,8,2>());    // static stride 2
t(0, slice<Int<1>, Int<4>>());   // type form (the only way to bake `none`)
```

## `take_along<Axes...>` — bind named axes

`operator()` is positional. To name only the axes you touch and keep the rest,
use `take_along`:

```cpp
t.take_along<1>(2);                // fix axis 1 at index 2, keep all others
t.take_along<0,2>(i, slice(1,4));  // bind axes 0 and 2 at once
t.take_along<-1>(k);               // negative axis: the last one
```

Each argument is an integer (negatives wrap), `all`, or a `slice` — the same
specifiers `operator()` accepts.
