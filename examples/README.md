# teeny examples

Standalone example algorithms that show the intended teeny idioms. Each is a
single `.cpp` with a `main()` that runs and self-checks (returns non-zero on
failure). Build them all with:

```sh
make examples          # from the repo root
make run-examples      # build + run, printing PASS/FAIL
```

| File | Shows |
|---|---|
| `pull_nd.cpp` | N-D separable spline interpolation ("pull"), rank-generic, with selectable boundary conditions; the tensor-product accumulation written **once** and folded per rank. The flagship fastfields idiom. |
| `distance_transform.cpp` | A separable 1-D transform applied along one axis, with **arbitrary batch rank** peeled by `peel<...>`. No hand-written `index2offset`. |
| `cholesky_solve.cpp` | Small SPD Cholesky factor + solve on a matrix with **per-dimension compile-time strides** (`wrap_strided`) and stack-owned work tensors that are exactly `sizeof` their data. |
| `broadcast_affine.cpp` | numpy-style broadcasting: per-channel affine (`x*scale + bias`) with `(C,1,1)` params over an `(C,H,W)` image, in-place and out-of-place. |
| `pushpull_adjoint.cpp` | the flagship fastfields kernel: rank-generic spline **pull** (gather) and **push** (scatter, atomic on device), validated by the adjoint identity `<Px,y>==<x,Pᵀy>` across interpolation orders 0–3 and 4 boundary conditions. |
| `batched_inverse.cpp` | **efficient kernel idiom**: a tensor with **static inner dims (C×C) and dynamic batch** (`shape<dynamic_extent,C,C>`), inverting one matrix per thread on **CPU threads and CUDA** (`peel_front_at` + `dispatch_value`). Includes proof the static access folds to 4 instructions. |

`fastfields/` holds the domain numerics the pushpull example builds on —
`bounds.hpp` (8 boundary conditions), `spline.hpp` (B-spline weights),
`pushpull.hpp` (separable gather/scatter). These are transcribed faithfully from
jitfields and are **not** part of teeny core (teeny is a pure tensor library);
they are the lift-ready reference for the fastfields port — see
`../docs/fastfields-port.md`.

These double as living documentation for `CLAUDE.md`. If you change the API,
update the example that exercises it.
