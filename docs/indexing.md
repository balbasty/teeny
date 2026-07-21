# Indexing & slicing

`operator()` does both **element access** and **slicing**, chosen by the argument
types — just like NumPy's `[]`.

## Element access

All-integer arguments return a reference to one element. Negative indices count
from the back (python-style):

```cpp
t(1, 2, 3);      // the element at (1,2,3)
t(0, -1);        // row 0, last column
t(Int<1>(), j);  // a static index folds; a runtime index is a value
```

!!! note "Negative indexing and codegen"
    The negative-index wrap is a signed compare that **folds away** for static
    (`Int<k>()`) and unsigned arguments. For runtime signed indices in a hot loop
    where you guarantee non-negative values, compile with
    `-DTNY_NO_NEGATIVE_INDEX` to drop the check entirely.

## Slicing → a sub-view (no copy)

If **any** argument is a slice specifier, `operator()` returns a lower- or
same-rank **view** into the same memory:

| argument | meaning |
|---|---|
| an integer | drop that axis (fix it at this index) |
| `all` | keep the whole axis |
| `slice(a, b)` | the half-open range `[a, b)` |
| `slice(a, b, step)` | strided range (`step` may be **negative**) |
| `none` | an open end inside a `slice` |

```cpp
t(1, all, all);              // fix axis 0 -> a lower-rank view
t(0, slice(1, 4));           // axis 1 range [1,4)
t(all, slice(none, 8, 2));   // every other element up to 8
t(all, slice(none, none, -1))// reverse an axis   (numpy a[:, ::-1])
```

`none` is python's `None`: `slice(none, k)` starts at 0, `slice(m, none)` runs to
the end, and `slice(none, none)` keeps the whole axis (it *is* `all`, and folds
to a static extent). Negative bounds wrap.

```cpp
t(slice(-2, none));          // the last two rows
t(slice(3, 0, -1));          // rows 3,2,1  (stop excluded, like python)
```

!!! info "What stays static"
    Axes kept with `all` keep their **static** extent. A *ranged* axis becomes
    dynamic (its size is a runtime value). This is the correct, predictable
    trade — reach for `all` when you want the extent to keep folding.

## `take_along<Axes...>` — bind named axes

`operator()` is positional (one argument per axis). When you'd rather name just
the axes you want to touch and keep the rest, use `take_along`:

```cpp
t.take_along<1>(2);            // fix axis 1 at index 2, keep all others
t.take_along<0,2>(i, slice(1,4)); // bind axes 0 and 2 at once
t.take_along<-1>(k);           // negative axis: the last one
```

Each argument is an integer (negatives wrap), `all`, or a `slice` — the same
specifiers `operator()` accepts.
