# Views & structure

These return **views** (no copy) that rearrange, reshape, or iterate a tensor.
Axis template arguments are signed — **negatives count from the back**.

Every axis argument has a **value form** too: pass a static integer
(`Int<k>()`) instead of the `<k>` template argument — `t.squeeze(Int<1>())` ==
`t.squeeze<1>()`, and likewise for `permute`/`flip`/`unsqueeze`/`reshape`. Both
spellings compile to the same code; the tabs below **lead with the value form**.

## Rearrange axes

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

=== "value form"

    ```cpp
    t.unsqueeze(Int<2>());   // insert a size-1 axis (numpy newaxis) -> rank+1
    t.unsqueeze(Int<-1>());  // append a trailing axis, e.g. (H,W) -> (H,W,1)
    t.squeeze(Int<3>());     // drop a specific size-1 axis -> rank-1
    t.squeeze();             // drop EVERY statically-size-1 axis
    t.reshape(Int<6>(), Int<4>());   // view as a new shape (needs C-contiguous, same numel)
    t.reshape(Int<6>(), Int<-1>());  // one -1 dimension is inferred from numel
    t.flatten();             // view as 1-D (ravel)
    ```

=== "template form"

    ```cpp
    t.unsqueeze<2>();   // insert a size-1 axis (numpy newaxis) -> rank+1
    t.unsqueeze<-1>();  // append a trailing axis, e.g. (H,W) -> (H,W,1)
    t.squeeze<3>();     // drop a specific size-1 axis -> rank-1
    t.squeeze();        // drop EVERY statically-size-1 axis
    t.reshape<6,4>();   // view as a new shape (needs C-contiguous, same numel)
    t.reshape<6,-1>();  // one -1 dimension is inferred from numel
    t.flatten();        // view as 1-D (ravel)
    ```

`reshape`/`flatten` require the tensor to be **C-contiguous** (they reinterpret
the same memory). If it isn't, materialise first:

```cpp
t.is_dense();                     // dense block in SOME order (C, F, or a permuted view)
t.is_contiguous();                // C-order specifically — what reshape/flatten need
auto c = t.clone();               // a dense, row-major OWNING copy (static -> stack, dyn -> heap)
c.flatten();                      // now contiguous
```

`is_dense()` with no argument asks only whether the elements occupy a dense block
of memory in *some* axis order — so a *permuted* C-contiguous view still counts.
`is_contiguous()` is the **C-order** question (numpy/pytorch's meaning) and is what
`reshape`/`flatten` need; `is_contiguous<fcontiguous>()` asks F-order. Both are thin
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
    the raw `cuda::std` objects and `t.extent(d) == t.shape(d)`.

## Recover static inner dims

At the ndarray boundary a view is often fully dynamic. `recast` reinterprets it
with a **more-static** shape (a `shape<...>`, i.e. the mdspan extents type) of the
same rank, so known inner dims fold:

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

Static dims of the target are validated against the actual extents. `recast`
**preserves the source's strides and works on any layout** (no copy, no contiguity
requirement): a strided or transposed source keeps its strides — they fold to
compile-time constants where the source layout makes them derivable (contiguous /
`strides<>`), and stay run-time for a `dynamic_strides` source. It only re-types
the *extents*; it never assumes row-major (which would silently mis-address a
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

`ccontiguous`/`fcontiguous` derive the strides from the extents (**you** promise
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

`peel_front<N>` handles **arbitrary batch rank**: for a `(*batch, *spatial, C)`
tensor, `peel_front<Nbatch>` yields `(*spatial, C)` sub-views to parallelise
over — one per CPU thread or CUDA thread. Each sub-view already has the batch
offset baked into its pointer, so the inner kernel sees only spatial strides.
