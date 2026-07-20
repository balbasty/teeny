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
| `distance_transform.cpp` | A separable 1-D transform applied along one axis, with **arbitrary batch rank** peeled by `slices<...>`. No hand-written `index2offset`. |
| `cholesky_solve.cpp` | Small SPD Cholesky factor + solve on a matrix with **per-dimension compile-time strides** (`view_strided`) and stack-owned work tensors that are exactly `sizeof` their data. |
| `broadcast_affine.cpp` | numpy-style broadcasting: per-channel affine (`x*scale + bias`) with `(C,1,1)` params over an `(C,H,W)` image, in-place and out-of-place. |

These double as living documentation for `CLAUDE.md`. If you change the API,
update the example that exercises it.
