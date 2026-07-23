# CLAUDE.md — working on teeny

teeny is a **header-only C++17 tensor library for host + CUDA device**, built on
NVIDIA CCCL's `cuda::std::mdspan`. One tensor type, with **per-dimension static
and/or dynamic shape and strides**, so a single kernel source folds to
immediates when shapes are static and stays generic when they are dynamic. It
exists to make numeric C++/CUDA kernels (spline interpolation, distance
transforms, small linear algebra) compact and readable.

## Golden rules

1. **mdspan does the heavy lifting.** Extents, layouts, offset mapping, and
   `submdspan` come from `cuda::std`. teeny adds *only* what mdspan lacks. Before
   writing new machinery, check whether `cuda::std::mdspan`/`submdspan`/`extents`
   already does it. Do not reinvent layouts or offset math.
2. **Everything must compile on both `clang++` and `g++` at `-std=c++17`.** Run
   `make CXX=clang++ run-test && make CXX=g++ run-test` before every commit. The
   floor is C++17 because CCCL refuses lower.
3. **Device-safe by construction.** No virtuals, no exceptions, no RTTI, no
   `Date.now`-style host-only calls in `_TNY_API` code. Engines are **lambda-free**
   (index-sequence folds + tiny functors) so they instantiate under `nvcc`
   *without* `--extended-lambda`. Keep them that way.
4. **`_TNY_API` = host+device, `_TNY_HOST` = host-only.** Anything that
   allocates (heap/CUDA storage, out-of-place ops on dynamic shapes) is
   `_TNY_HOST`. Element access, in-place math, views, static-shape out-of-place
   ops are `_TNY_API`. A `_TNY_API` function must not call a `_TNY_HOST` one on
   the device path.
5. **Keep it teeny.** Every header is small and single-purpose. Prefer deleting
   code to adding it. The whole point is compactness; resist feature creep that
   does not serve real kernels.

## Layout of the repo

```
include/teeny/
  defines.h        macros: _TNY_API / _TNY_HOST, namespace open/close
  alias.h          cs:: vocabulary into tny:: + Int<V>/... static ints + `all` + shape<...>
  half.h           `half` (IEEE binary16) + `bfloat16` element types + compute_type
  storage.h        `own` enum + storage policies (owning_storage<T,Alloc>, cpp_alloc)
  layout.h         strides<S...> — per-dim static/dynamic strides (extents for strides)
  indexing.h       free indexing/slicing vocabulary: slice()/none, _norm_axis,
                   _wrap_idx, slice_spec + traits, _compact output-extents
  tensor.h         the tensor class + view/local/owned aliases + as_tensor
                   + indexing/slicing, take_along, permute, unsqueeze/squeeze
  math.h           in-place & out-of-place elementwise (broadcasting) + unary math
                   + reductions (sum/prod/max/min/dot). Members declared in
                   tensor.h, DEFINED here.
  iterate.h        nd-peel: peel / peel_at / peel_front / peel_front_at
  dynamic.h        anyrank (rank-erased carrier) + peel_front<-Sr> + dispatch_rank
  cuda.h           gpu/pinned/mapped memory. Self-guarded (__has_include /
                   __CUDACC__): a no-op unless the CUDA runtime is reachable, so
                   teeny.h includes it unconditionally. TNY_NO_CUDA forces it off.
  dlpack.h         DLPack interchange (to_dlpack / from_dlpack / dispatch_dlpack).
                   Vendors the DLManagedTensor structs (guard DLPACK_DLPACK_H_).
                   OPT-IN: include <teeny/dlpack.h> explicitly (not in teeny.h).
  teeny.h          umbrella (includes everything, cuda.h included + self-guarded)
tests/             one file per feature; `make run-test` builds + runs all
examples/          standalone example algorithms (see examples/README.md)
external/cccl/     vendored CCCL 3.3.0 (libcudacxx). -I external/cccl/libcudacxx/include
```

`namespace tny` is the public namespace; internal code lives in `tny::_md`
(compute/iteration engines — the `math.h` elementwise/reduction folds and the
`iterate.h` peel machinery), `tny::_detail` (view/mapping builders and host
helpers — `axis.h`, `cuda.h`'s `dense_host`, `dynamic.h` dispatch, `half.h`
converters), `tny::_dl` (DLPack), and bare `_`-prefixed names directly in `tny`
for vocabulary that public templates must see unqualified (the `indexing.h` /
`layout.h` traits, `_norm_axis`/`_axis_in_range`, `math.h`'s `_promote` etc.). New
helpers should follow that split. `namespace cs = cuda::std;` throughout.

## The tensor type

```cpp
template <class T, class Shape, class Layout = ccontiguous, own O = own::view>
struct tensor;
```

- **`T`** element type. Any arithmetic type, plus `half` (IEEE binary16) and
  `bfloat16` (`half.h`). Under nvcc these ARE the native CUDA `__half` /
  `__nv_bfloat16`; on a host compiler they are portable software types with the
  same layout. All elementwise/reduction math computes in `float`
  (`compute_type<T>`), so precision holds and the engines never need native
  half *host* operators.
- **`Shape`** (the second template parameter, exposed as `shape_type`, and still
  `extents_type`) = any `cuda::std::extents<Idx, E0, E1, ...>`; each `Ei` is a
  compile-time size or `dynamic_extent`. Mix freely per dimension. Prefer the
  `shape<...>` alias (`= extents<int64_t, ...>`, DLPack's index type); a dynamic
  dim is `dynamic_extent` or numpy-style **`-1`** (`shape<-1,3>` ==
  `shape<dynamic_extent,3>`). `rank<N>` is the fully-dynamic rank-N shape
  (`rank<3>` == `shape<-1,-1,-1>`).
- **`Layout`** = `ccontiguous` (default, C-order; mdspan `layout_right`),
  `fcontiguous` (F-order; mdspan `layout_left`), `dynamic_strides` (runtime
  strides), or teeny's `strides<S...>` (compile-time strides).
- **`O`** ownership: `view` (non-owning host view, trivially copyable,
  kernel-passable), `stack` (inline array, requires fully static shape), `heap`
  (host `new[]`, move-only), CUDA `gpu`/`pinned`/`mapped` (from `cuda.h`), or
  `gpu_view` (a non-owning view of **device** memory). Slicing / permuting /
  peeling / `.at()` of a `gpu` tensor yields a `gpu_view`, not a `view`, so a
  device pointer is never mistaken for a host one in the type. `pinned`/`mapped`
  views likewise keep their space (`pinned_view`/`mapped_view`) so DLPack labels
  them `kDLCUDAHost`; they are still host-accessible and behave like `view`
  otherwise. Helpers: `own_is_device(O)` (gpu/gpu_view), `own_is_host_accessible(O)`,
  `own_is_view(O)` (view/gpu_view/pinned_view/mapped_view), `own_view_of(O)` (the
  view kind that preserves a source's space — used by every view-producing op).

The mapping lives in an **empty base** (private inheritance → EBO), so a
fully-static stack tensor is *exactly* `sizeof` its data.

### Aliases (prefer these over spelling out `tensor<...>`)

```cpp
view<T,E,L>   // non-owning view type
local<T,E,L>    // stack-owned (static shape)     e.g. local<double, shape<3,3>>{}
owned<T,E,L>    // heap-owned (host, move-only)    e.g. owned<double, DynE>(DynE{2,3})
gpu/pinned/mapped<T,E,L>   // from cuda.h
```

Factories: `wrap(ptr, extents)` / `wrap<Layout>(ptr, extents)`,
`wrap(ptr, extents, {s...})` (runtime strides -> dynamic_strides),
`wrap<S...>(ptr, extents, {dyn...})` (mixed static/runtime strides),
`wrap(ptr, extents, strides<S...>{})` (compile-time strides),
`as_tensor(any_mdspan)` (wrap a submdspan/mdspan result as a view). Functional
factories that deduce the extents type: `make_view(ptr,e)`, `make_local<T>(e)`,
`make_heap<T>(e)`, `make_gpu/pinned/mapped<T>(e)` (E deduced; **T defaults to
`float`**, override explicitly). The `make_*` owning factories are thin spellings
of one **`empty<T[, own::Space]>(e)`** factory: ownership is deduced from the
shape (static→stack, dynamic→heap) unless a backend is named as a template arg
(`empty<T, own::gpu>(e)`) or a value-tag (`empty<T>(e, own_c<own::gpu>{})`);
gpu/pinned/mapped need `<teeny/cuda.h>`.

Custom strided layout: `strides<Sx, Sy, ...>` is the stride analogue of `shape`,
folding known strides to immediates and storing only the dynamic ones (EBO when
all static): `tensor<float, shape<3,4>, strides<4,1>>(ptr)`. Strides are SIGNED,
so `-1` means a real stride of −1 (reversed view), NOT dynamic — a runtime
stride is spelled `dynamic_stride` (`strides<dynamic_stride,1>`, constructed with
the runtime strides). NB CCCL's `submdspan` does not apply to it, but teeny's own
slicing/take_along/permute/flip/peel DO (they build views by hand and fold the
output strides), so a strides<...> tensor is fully sliceable.

## API cheat-sheet

```cpp
// --- geometry (static index -> integral_constant, runtime index -> value) ---
t.rank();  t.numel();
t.extent(Int<0>());   // static lookup -> integral_constant when the extent is static
t.extent(0);          // runtime lookup -> index_type   (t.shape(...) is a python-y alias)
t.stride(Int<1>());   // static when derivable (static-stride layout; contiguous+static;
                      //   or a contiguous layout's UNIT stride even for a dynamic shape)
t.data();  t.view();  t.mdspan();  t.extents();  t.shape();  t.mapping();
                      //   view() = non-owning teeny view tensor (gpu_view if device);
                      //   mdspan() = the raw cuda::std::mdspan

// --- indexing / slicing (python-like) ---
t(1, 2, 3);           // element access -> T& ; negative indices wrap (count from the back)
t.at(1, 2, 3);        // same element as a rank-0 VIEW (has add_/etc.); rank-0 <-> scalar
                      //   (implicit to/from T, `.item()`). at(i...).add_<true>(v) = atomic scatter
t.uget(1,2,3); t.uat(i...); t.uslice(0,slice(1,4));  // UNCHECKED twins of
                      //   ()/at/slice: skip the negative-index wrap for known-non-negative
                      //   RUNTIME indices (per-call -DTNY_NO_NEGATIVE_INDEX). Same result type;
                      //   static bounds still fold; a negative runtime index is then UB.
t(0, all, slice(1,4));  // any slice arg -> a lower-/same-rank VIEW. all = keep axis,
                      //   slice(a,b) = half-open [a,b). Integer args drop that axis.
t(0, slice(none,4), slice(1,none,2));  // python-like: none = open end, 3rd arg = step.
t(1, ellipsis, 2);    // ellipsis (numpy ...) = (rank - #other args) copies of `all`; max one.
t(ellipsis) = b; t(0,all) = 3.0;  // assign INTO a slice copies/fills (b broadcasts);
                      //   `a = b` on a NAMED view REBINDS (shallow) — the contrast.
t(0, slice<1,4>()); t(0, slice<0,8,2>());  // compile-time slice (bounds fold like `all`);
                      //   slice<Int<1>,Int<4>>() is the type form (only way to bake `none`).
                      //   negative bounds wrap; all == slice(none,none) (folds).
                      //   NB a range outputs teeny's strides<...> layout (folding the
                      //   stride where derivable); a COMPILE-TIME range folds its extent
                      //   too (source static + static bounds), a runtime range's is dynamic.
t.take_along<0,2>(i, slice(1,4));  // bind named axes only; keep every other axis
t.permute<2,0,1>();   // reorder axes (a permutation of 0..N-1) -> view
t.flip<1>();          // reverse an axis (negative-stride view; needs signed index)
t.unsqueeze<2>();     // insert size-1 axis at pos 2 (numpy newaxis) -> rank+1 view
t.squeeze<3>();       // drop a size-1 axis -> rank-1 view
t.reshape<6,4>(); t.flatten();  // contiguous-view reshape / ravel (clone() first if not)
t.is_contiguous();              // dense in SOME order (C/F/permuted); <ccontiguous>() = exact C
t.clone();                      // materialise a dense row-major copy
t.to<double>();                 // pytorch-like dtype convert -> dense owning copy (static->stack, dyn->heap).
                                //   NO-COPY when it already matches: t.to<>() (same dtype) borrows a read-only
                                //   view; t.to<T,true>() forces a copy (clone() is the unconditional spelling).
to<own::gpu>(t);                // MEMORY-SPACE move (cuda.h free fn): to<Space,ET,Force>(x). Same no-copy/force
                                //   rule — to<own::gpu>(gpu_x) borrows; to<own::gpu,void,true>(x) force-clones.
t(all, slice(none,none,-1));    // reverse a range (negative step; a[::-1])
// AXIS template args are signed: negatives count from the back (numpy). e.g.
//   t.extent(Int<-1>()), t.unsqueeze<-1>() (append), t.permute<-1,0,1>(),
//   t.take_along<-2>(i), peel<0,-1>(t).

// --- math (in-place: any tensor/view; mutates *this) ---
a.add_(b); a.sub_(b); a.mul_(b); a.div_(b);   // tensor rhs BROADCASTS numpy-style
a.add_(2.0); a.mul_(0.5);                     // scalar rhs
a += b; a -= 2.0; a *= b; a /= 2.0;           // compound-assign sugar (scalar or tensor)
a.add_<true>(b); a.sub_<true>(2.0);           // ATOMIC accumulate (device scatter/push)
a.neg_(); a.abs_(); a.exp_(); a.log_();       // unary in-place
a.sin_(); a.cos_(); a.sqrt_(); a.tanh_(); a.pow_(3.0);
a.floor_(); a.ceil_(); a.round_(); a.trunc_(); a.sign_(); a.clamp_(lo,hi);
++a; --a; auto old = a++;                      // prefix in place; postfix (static shape) -> stack copy
a & b; a | b; a ^ b; ~a; a &= b; a |= 1;       // bitwise (INTEGER element types only)

// --- assignment / scatter / generic (kernel prologue/epilogue) ---
a.fill_(0.0); a.zero_(); a.copy_(b);          // b broadcasts into a
a.iota_(start, step);                         // 0,1,2,... (row-major)
a.map_(f); a.zip_with_(g, b); auto c = a.map(f);  // user functor (device-safe struct)
a.at(i, j).add_<true>(v);                     // scatter-accumulate: a(i,j) += v,
                                              //   ATOMIC on device (push/splat write)
auto z = zeros<T>(shape); ones<T>(sh); full(sh,v); arange<T>(n);  // creation. zeros/ones
                      //   default T=float; full's T = value type; arange defaults T=int64.
                      //   Static: arange<T,N>() / arange<T>(Int<N>()) -> stack [0..N-1].
                      //   Backend selector (like empty): zeros<T,own::pinned>(sh),
                      //   full<T>(sh,v,own_c<own::pinned>{}) — HOST-ACCESSIBLE only
                      //   (stack/heap/pinned/mapped); a gpu fill static_asserts ->
                      //   to<own::gpu>(zeros<T>(sh)).

// --- math (out-of-place -> NEW tensor; static shape -> stack, else heap/host) ---
// result type = promote(A,B): C++ rules, but among floats the LOWER width wins
//   (half>float>double, pytorch-style). Opt out with -DTNY_STD_PROMOTION.
auto c = a + b;  auto c = a.add(b);   // tensor+tensor (broadcasts) or tensor+scalar
auto c = a * 2.0;  auto c = 2.0 * a;  // scalar ops (+ and * commute; 2.0-a and 1.0/a reversed)
auto c = -a;                          // unary minus -> new tensor
auto c = a.pow(b);                    // element-wise power
auto e = exp(a); auto e = sqrt(a);    // unary free (neg/abs/exp/log/sin/cos/sqrt/tanh/floor/ceil/round/trunc/sign)
auto c = minimum(a,b); maximum(a,2.0); clamp(a,lo,hi);   // elementwise binary min/max, clamp

// --- comparisons -> a bool tensor (broadcast); reduce with .all()/.any() ---
auto m = a < b; a == 2.0; 3.0 < a;    // ==,!=,<,<=,>,>= ; scalar either side
(a > 0).all(); (a > 3).any();         // bool reductions (MEMBERS: `all` is the slice kw)

// --- reductions -> scalar (all axes). ACCUMULATE in the "reduce type" (double for
//   small floats float/double/half, item type for ints; reduce_type_t<T>), then
//   CAST the result to the tensor's element type: sum(float)->float. A leading TYPE
//   arg makes that type BOTH accumulator and result: sum<double>(a), dot<double>(a,b).
sum(a); prod(a); max(a); min(a); mean(a); dot(a,b);
allclose(a, b, rtol=1e-5, atol=1e-8);  // |a-b| <= atol+rtol*|b| everywhere (broadcasts)
// --- axis reductions -> a lower-rank TENSOR (named axes removed; negatives wrap).
//   Same rule: accumulate in reduce_type, result element type = the tensor's type.
//   sum<Acc,Axes...>(a) makes Acc accumulator AND result (leading TYPE = accumulator,
//   leading int = axis -> never collide).
sum<0>(a); mean<0,2>(a); max<1>(a); min<-1>(a); prod<0>(a); sum<double,0>(a);
//   static result -> stack (host+device); any dynamic -> heap (HOST ONLY: allocates)

// --- nd-peel: iterate a SUBSET of axes, each yielding a lower-rank view ---
for (auto line : peel<0,1>(t)) f(line);   // peel axes 0,1; each `line` is a view
auto s = peel_at<0,1>(t, i);               // the i-th peeled sub-view (grid-stride style)

// --- nd-peel: peel the FIRST N axes (arbitrary batch rank) ---
for (auto v : peel_front<N>(t)) f(v);      // v is (*spatial, C); N = #batch dims
auto v = peel_front_at<N>(t, i);            // the i-th (grid-stride style)

// --- dynamic-rank / dynamic-value host boundary (dynamic.h) ---
auto at = as_anyrank(data, shape, stride, ndim);    // -> anyrank: rank-erased carrier; WRAPS the
                                             //   shape/stride arrays (1-D tensor views), NO copy; HOST only (default)
auto ac = as_anyrank(data, shape, stride, ndim, copy_meta);  // COPIES into an inline TNY_MAX_RANK (default
                                             //   32) store -> trivially copyable, device-passable
dispatch_rank(at, [&](auto v){ kernel(v); });  // instantiates kernel once per TOTAL rank
auto v3 = at.fixed<3>();                      // or force a known rank
dispatch_value<1,2,3>(D, [&](auto d){ kern<d.value>(v); });  // runtime value -> static
// BATCH idiom (one kernel per Sr, not per total rank): peel the runtime batch
// dims, keep the trailing Sr "interesting" dims static. NB the arg is NEGATIVE
// (keep the last |N|), like the tensor's peel_front — anyrank asserts N<0.
for (auto cell : at.peel_front<-Sr>()) kernel<Sr>(cell);  // Sr=2 -> peel_front<-2>; cell is rank-Sr
auto cell = at.peel_front_at<-Sr>(i);         // i-th (grid-stride); .recast<shape<-1,c,c>>() folds inner dims
```

### Static vs runtime values (important idiom)

The API accepts **runtime integers, static integers (`integral_constant`), and
slices of either**, and dispatches to the right output type:

- `alias.h` provides short names for `cuda::std::integral_constant`:
  `Int<V>`, `Long<V>`, `Size<V>`, `UInt<V>`, `Int32<V>`, `Int64<V>`, `Diff<V>`,
  and `Bool<V>`. Each converts implicitly to a runtime integer *and*
  carries `::value`.
- `extent(Int<0>())` returns an `integral_constant` when that extent is static
  (folds into later arithmetic); `extent(0)` returns a runtime `index_type`.
- Same for `stride(...)`. A static-stride layout, or a contiguous layout over
  static extents, yields a static stride.

Rule of thumb: **pass a static index (`Int<k>()`) when you want the compiler to
fold**, a plain `int`/`long` when the value is only known at run time.

## How the hard parts work (so you don't re-derive them)

- **Broadcasting** (`math.h`, `_md::bzip`): numpy-style — operands are aligned
  from the **right** (`bc_ext`/`bc_str` right-align each operand into the result
  rank; a shorter operand's missing leading axes are extent 1 / stride 0), so the
  result rank is `bc_rank = max(rankA, rankB)`. A dimension of extent 1 gets
  stride 0 (it is stretched). The result extent per axis is computed at compile
  time by `bc1`/`bcast_extents` (`dynamic_extent` if either operand is dynamic).
  In-place `a.op_(b)` needs `rankB ≤ rankA` (can't grow the destination).
  Out-of-place: a fully static result → `own::stack` (host+device); any dynamic →
  `own::heap` (host only). The SFINAE keys on `bcast_extents<...>::rank_dynamic()`,
  **not** on instantiating a stack tensor (that would fire the "stack needs static
  shape" `static_assert`).
- **The gather** (`tensor.h` `_slice_range`, `iterate.h` `gather_peel`): ALL
  view-making ops — `operator()` slicing, `take_along`, `peel` — route through
  one hand-built gather (NO `cs::submdspan`). Per axis: an integer drops it (into
  the base offset), `all` keeps it, a range keeps a strided window. The output
  layout is teeny's `strides<Sfold...>`, folding each kept stride to a
  compile-time value where derivable (`_out_sstride`/`_str_compact` in
  `indexing.h`, `_src_sstride` in `layout.h`: source-static-stride × static-step).
  So slicing a static tensor keeps folded strides, and every op works on ANY
  source layout — including a `strides<...>` tensor. `permute`/`flip`/`unsqueeze`/
  `squeeze` (`axis.h`) likewise build views by hand.
- **layout_static_stride** (`layout.h`): the one thing mdspan lacks — strides
  baked into the type. It is now the OUTPUT layout of every slice/peel (folded),
  and a fully sliceable source. CCCL's `cs::submdspan` doesn't accept it, but
  teeny never calls `cs::submdspan` anymore.
- **EBO**: `tensor : private Layout::mapping<Extents>`. `mapping()` returns
  `*this`. Do not add non-static data members besides `store_`.

## Adding a feature — checklist

1. Is it really missing from mdspan? If not, alias it in `alias.h` instead.
2. Member function? **Declare** it in `tensor.h`, **define** it in `math.h`
   (math) or inline in `tensor.h` (structural). Watch declaration-before-use:
   `math.h` has a single `_md` engine block first, then all member/operator
   definitions — keep new engines in that block.
3. Keep it lambda-free and `_TNY_API` unless it allocates (then `_TNY_HOST`).
4. Add a `tests/test_<feature>.cpp` mixing `static_assert` (compile-time shape
   checks) and runtime asserts; wire it into `Makefile`'s `TESTS` list.
5. `make CXX=clang++ run-test && make CXX=g++ run-test` — all green.
6. Commit with a focused message.

## Testing

`make run-test` builds and runs every `tests/test_*.cpp`, printing `PASS`/`FAIL`.
Tests return non-zero on failure (the return code is the failing check number).
`test_cuda` compiles against a malloc-backed fake CUDA runtime in
`tests/fakecuda/`. `test_pull`, `test_distance_l1`, `test_posdef` port real
kernels and validate them numerically against hand-written references — when you
change math or layout code, these are the ones that catch regressions.

## Downstream: fastfields

teeny is the substrate for the `fastfields` kernels (spline pull/push, distance
transforms, small SPD solves). Those kernels care about: arbitrary **batch**
rank (shape `(*batch, *spatial, C)`, not a fixed number of batch dims), a small
set of **spatial** ranks (1/2/3D) worth dispatching to specialized code, and
configurable **boundary conditions** and **interpolation order**. Design teeny
changes so those stay expressible and fast.

- **Idioms:** `examples/` — `pushpull_adjoint.cpp` is the flagship (pull+push).
- **Reference numerics:** `examples/fastfields/{bounds,spline,pushpull}.hpp`
  (the 8 boundary conditions + spline weight tables + separable gather/scatter,
  transcribed from jitfields — domain code that lives outside teeny core).
- **Full porting plan:** `docs/fastfields-port.md` — the dispatch architecture
  (`(*batch,*spatial,C)` via `peel_front`/`dispatch_value`), the complete
  boundary/order spec, kernel-by-kernel mechanics, and repo mapping.
