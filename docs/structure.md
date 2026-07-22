# Views & structure

These return **views** (no copy) that rearrange, reshape, or iterate a tensor.
Axis template arguments are signed — **negatives count from the back**.

## Rearrange axes

```cpp
t.permute<2,0,1>();   // reorder axes (a permutation of 0..N-1)
t.permute<-1,0,1>();  // negatives allowed
t.flip<1>();          // reverse an axis (a negative-stride view; needs a signed
                      //   index type, which shape<...> is)
```

## Add / drop / reshape

```cpp
t.unsqueeze<2>();     // insert a size-1 axis (numpy newaxis) -> rank+1
t.unsqueeze<-1>();    // append a trailing axis, e.g. (H,W) -> (H,W,1)
t.squeeze<3>();       // drop a size-1 axis -> rank-1
t.reshape<6,4>();     // view as a new shape (needs C-contiguous, same numel)
t.reshape<6,-1>();    // one -1 dimension is inferred from numel
t.flatten();          // view as 1-D (ravel)
```

`reshape`/`flatten` require the tensor to be **C-contiguous** (they reinterpret
the same memory). If it isn't, materialise first:

```cpp
t.is_contiguous();    // query
auto c = t.clone();   // a dense, row-major OWNING copy (static -> stack, dyn -> heap)
c.flatten();          // now contiguous
```

## Recover static inner dims

At the ndarray boundary a view is often fully dynamic. `recast` reinterprets it
with a **more-static** extents type of the same rank, so known inner dims fold:

```cpp
auto dyn = view(ptr, shape<-1,-1,-1>{n,3,3});  // came in fully dynamic
auto st  = dyn.recast<shape<-1,3,3>>();        // the 3s are now compile-time
```

Static dims of the target are validated against the actual extents.

## nd-peel — iterate a subset of axes

Peel some axes and get a lower-rank sub-view for each, without writing index
arithmetic.

```cpp
for (auto line : peel<0,1>(t)) work(line);   // peel axes 0,1; each is a view
auto s = peel_at<0,1>(t, i);                 // the i-th sub-view (grid-stride style)

for (auto cell : peel_front<N>(t)) work(cell);  // peel the FIRST N axes
auto c = peel_front_at<N>(t, i);                // the i-th
```

`peel_front<N>` handles **arbitrary batch rank**: for a `(*batch, *spatial, C)`
tensor, `peel_front<Nbatch>` yields `(*spatial, C)` sub-views to parallelise
over — one per CPU thread or CUDA thread. Each sub-view already has the batch
offset baked into its pointer, so the inner kernel sees only spatial strides.
