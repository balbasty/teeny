# Dispatch & the ndarray boundary

Kernels want static shapes (for folding), but data from Python arrives with a
runtime rank and runtime sizes. teeny gives you three tools to cross that seam:
turn a runtime **value**, a runtime **rank**, or a rank-erased pointer into a
statically-typed tensor once, at the boundary, and stay fast inside.

## `dispatch_value` — runtime value → compile-time

Pick a compile-time value from a candidate list. `f` is instantiated once per
candidate; the matching one runs with an `integral_constant` it can use as a
template argument.

```cpp
dispatch_value<1,2,3>(spatial_ndim, [&](auto D) {
    kernel<D.value>(view);          // D.value is a compile-time constant here
});
```

Use it for anything that's a small runtime value you want static: a spatial
rank, an interpolation order, a matrix size `C`. It replaces the giant `switch`
statements hand-written kernels use.

## `any_tensor` + `dispatch_rank` — runtime rank → static

A rank-erased, bounded tensor for the host boundary. It carries a pointer plus
bounded shape/stride arrays and a runtime `ndim`. You don't compute on it — you
dispatch it to a fixed-rank view:

```cpp
auto at = any(data, shape, stride, ndim);      // rank-erased (MaxRank default 8)
dispatch_rank(at, [&](auto v) { kernel(v); }); // instantiates kernel once per rank
auto v3 = at.fixed<3>();                        // or force a known rank
```

!!! tip "Recover static inner dims after rank dispatch"
    `fixed<R>()` gives a `dextents<_,R>` view — all inner extents are dynamic.
    Follow it with [`recast<shape<-1,c,c>>()`](structure.md#recover-static-inner-dims)
    to make the known inner dims fold again.

## The full boundary pattern

For a `(*batch, *spatial, C)` array from numpy / torch / cupy / DLPack:

```
DLPack / ndarray  ──any(data, shape, stride, ndim)──►  any_tensor
   │  (DLPack strides are in ELEMENTS; numpy's __array_interface__ is in BYTES)
   ▼  dispatch_rank / dispatch_value on total rank -> fixed<R>()
   ▼  dispatch_value<1,2,3>(spatial_ndim) -> static spatial rank D
   ▼  recast<shape<-1,…static inner…>>()  -> inner dims fold
   ▼  Nbatch = R - D - 1
   for (auto cell : peel_front<Nbatch>(t)) kernel<D>(cell, …);  // parallelise this
```

The worked-through version, with CPU-thread and CUDA drivers and a pybind11
wrapper, is the [DLPack → Python tutorial](tutorials/dlpack-python.md).
