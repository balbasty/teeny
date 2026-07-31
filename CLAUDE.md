# CLAUDE.md — working on teeny

teeny is a **header-only C++17 tensor library for host + CUDA device**, built on
NVIDIA CCCL's `cuda::std::mdspan`. One tensor type, with **per-dimension static
and/or dynamic shape and strides**, so a single kernel source folds to
immediates when shapes are static and stays generic when they are dynamic. It
exists to make numeric C++/CUDA kernels (spline interpolation, distance
transforms, small linear algebra) compact and readable.

## Follow the contribution guidelines

Read and follow [`CONTRIBUTING.md`](CONTRIBUTING.md) for the **process**: the
issue-based workflow (file an issue first; **every issue carries ≥1 label** —
`bug`/`enhancement`/`perf`/`maintainability`/`documentation`/`blocked`/`good first
issue`; break big work into sub-issues under an umbrella), **one branch and one PR
per task** (never bundle unrelated changes), conventional commit subjects that
reference the issue and `Closes #NN` from the PR, the build/test gate (both
compilers + sanitizers on touched host paths), and skeptical element-identity
review for core-path diffs. This file (CLAUDE.md) is the *design* companion to
that process guide.

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
   **No `static constexpr` array data member read at a RUNTIME index from
   `_TNY_API` code** (#389): nvcc never places a host `static constexpr` array
   into device memory, so a runtime index — which cannot constant-fold — has no
   device-side object to reach and is a hard device-compile error ("identifier
   `X` is undefined in device code"). Read a compile-time pack through a **fold**
   instead (see `strides<...>::static_stride`/`slot`, `layout.h`); a
   function-local array compiles but pessimises the hot path, since materialising
   it on the stack defeats the constant folding the immediates were for. And note
   nvcc is **eager**: it generates and checks the device version of every
   `_TNY_API` template at instantiation, *even from a host-only call site* — so a
   violation is a real compile error in any `.cu` TU, not a latent device-path
   hazard. (clang's CUDA front end defers until the function is genuinely
   device-reachable, so it will not show you this.)
4. **`_TNY_API` = host+device, `_TNY_HOST` = host-only.** Anything that
   allocates (heap/CUDA storage, out-of-place ops on dynamic shapes) is
   `_TNY_HOST`. Element access, in-place math, views, static-shape out-of-place
   ops are `_TNY_API`. A `_TNY_API` function must not call a `_TNY_HOST` one on
   the device path. **Corollary — split every forwarder that straddles the
   line.** Where a feature has a static/dynamic overload PAIR (static result →
   stack, `_TNY_API`; dynamic result → heap, `_TNY_HOST`), any *thin forwarder*
   over that pair — a value-tag twin (`axis<...>{}`, `dtype<T>{}`), a member
   wrapper around a free function — must itself be split in two, `enable_if`'d
   on the *same predicate* the pair uses, so each arm's annotation matches the
   overload it resolves to. A single unconditionally `_TNY_API` forwarder is a
   `__host__ __device__` function that calls a `__host__` allocator on the
   dynamic path. The host-only suite is **blind** to this (both macros expand to
   nothing without `__CUDACC__`), so it is caught only by the nvcc device pass —
   add any new such spelling to `tests/nvcc_smoke.cu`, in its **dynamic**-shaped
   form, since a static shape takes the safe arm. Existing instances of the
   pattern: `to(dtype)` (`tensor.h`), `_TNY_RED_AXIS_DEF`/`_TNY_RED_TAGGED_DEF`
   and `_red_finish_static`/`_red_finish_dynamic` (`math.h`), and — since #375 —
   `index_select`'s and `scan`'s `axis<...>{}` value forms.
5. **Keep it teeny.** Every header is small and single-purpose. Prefer deleting
   code to adding it. The whole point is compactness; resist feature creep that
   does not serve real kernels.

## Layout of the repo

```text
include/teeny/
  defines.h        macros: _TNY_API / _TNY_HOST (host+device / host-only), _TNY_EMPTY_BASES
                   (MSVC empty-base fold), _TNY_RESTRICT (portable __restrict spelling),
                   TNY_UNROLL (loop-unroll pragma), _TNY_CHECK (precondition assert,
                   compiled out under NDEBUG and always off on the device) / _TNY_BOUND
                   (opt-in TNY_HARDENED bounds check, itself assert-based), TNY_MAX_RANK
                   (anyrank's inline-store rank
                   cap, default 32), TNY_MAX_STATIC_UNROLL (largest STATIC element count
                   math.h's static-unroll fast paths still unroll, default 256 = clang's
                   hard fold-argument limit; bigger static shapes decode — #343),
                   namespace open/close; also the min/max/interface
                   Windows.h #undef block (outside the include guard, see MSVC traps below)
  alias.h          cs:: vocabulary into tny:: + Int<V>/... static ints + `all` + shape<...>
                   + axis<...>/dtype<...> value-carrier tags
  half.h           `half` (IEEE binary16) + `bfloat16` element types + compute_type
  kwargs.h         `tny::_kw` — generic keyword-argument primitive (find/get/has/count/
                   accepts/resolve/is_keyword) for the trailing value-carrier tags (dtype/
                   storage/layout/axis/into/keepdims). Backs every migrated call site (#277's
                   umbrella: empty/zeros/ones/full/arange/wrap/make_*, and the reduction
                   family sum/prod/max/min/mean/sqnorm/norm/dot). Two shared entry points,
                   so neither the guard nor the precedence rule is transcribed again (#376):
                   **`_TNY_KW_CHECK(SITE, EXPECTED, (PREDS...), Tags...)`** is the one-line
                   guard every call site opens with (it expands to the unrecognised-keyword
                   + duplicated-keyword `static_assert` pair; `accepts<Ps...>::known/unique/
                   check` are the predicates under it), and **`_kw::resolve`** is the one
                   copy of "explicit-template-arg > matching value tag > library default,
                   both given = `static_assert`". The per-keyword READERS stay next to each
                   tag's own definition, each a one-line alias over `resolve` supplying only
                   its own tag->answer step: `dtype_arg_t` (alias.h, unwraps `dtype<T>` ->
                   `T`), `storage_arg` (storage.h, travels in a `storage_c<O>` carrier since
                   its answer is a VALUE), `layout_arg_t` (layout.h, the tag IS the layout).
  storage.h        `storage` enum + storage policies (owning_storage<T,Alloc>, cpp_alloc)
  layout.h         strides<S...> — per-dim static/dynamic strides (extents for strides)
  indexing.h       free indexing/slicing vocabulary: slice()/none, _norm_axis,
                   _wrap_idx, slice_spec + traits, _compact output-extents
  tensor.h         the tensor class + view/local/owned aliases + as_tensor
                   + indexing/slicing, slice_along, index_select, permute, unsqueeze/squeeze
  math.h           in-place & out-of-place elementwise (broadcasting) + unary math
                   + reductions (sum/prod/max/min/dot). Members declared in
                   tensor.h, DEFINED here.
  iterate.h        nd-peel: peel / peel_at / peel_front / peel_front_at / peel_zip / scan_ / scan
  dynamic.h        anyrank (rank-erased carrier) + peel_front<-Sr> + dispatch_rank
  cuda.h           gpu/pinned/mapped memory. Self-guarded (__has_include /
                   __CUDACC__): a no-op unless the CUDA runtime is reachable, so
                   teeny.h includes it unconditionally. TNY_NO_CUDA forces it off.
  dlpack.h         DLPack interchange (to_dlpack -> managed capsule; to_dltensor ->
                   bare unmanaged DLTensor; from_dlpack / dispatch_dlpack /
                   dispatch_dlpack_dtype). from_dlpack + the dispatchers accept ALL
                   THREE carriers: DLManagedTensor*, bare DLTensor* (no deleter,
                   caller owns lifetime), DLManagedTensorVersioned* (DLPack 1.0+).
                   Uses the OFFICIAL DLPack header — the app's / a system
                   <dlpack/dlpack.h> if present, else the complete upstream copy
                   vendored under external/dlpack/ (never a hand-rolled subset).
                   OPT-IN: include <teeny/dlpack.h> explicitly (not in teeny.h).
  teeny.h          umbrella (includes everything, cuda.h included + self-guarded)
tests/             one file per feature; `make run-test` builds + runs all
examples/          standalone example algorithms (see examples/README.md)
external/cccl/     vendored CCCL v2.8.2 (libcudacxx) — last 2.x, spans CUDA 11.1–12.9
                   with mdspan; -I external/cccl/libcudacxx/include (see docs/cuda-compat.md)
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
template <class T, class Shape, class Layout = ccontiguous, storage O = storage::view>
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
  otherwise. Helpers: `storage_is_device(O)` (gpu/gpu_view), `storage_is_host_accessible(O)`,
  `storage_is_view(O)` (view/gpu_view/pinned_view/mapped_view), `storage_view_of(O)` (the
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

Factories: `wrap(ptr, extents)` / `wrap<Layout>(ptr, extents)` / `wrap(ptr, extents,
fcontiguous{})` (value-tag layout: same as `wrap<fcontiguous>`, deduced, no
`.template`),
`wrap(ptr, extents, {s...})` (runtime strides -> dynamic_strides),
`wrap<S...>(ptr, extents, {dyn...})` (mixed static/runtime strides),
`wrap(ptr, extents, strides<S...>{})` (compile-time strides),
`as_tensor(any_mdspan)` / `wrap(any_mdspan)` (wrap a submdspan/mdspan result as a
view. BOTH are public ON PURPOSE — layering, not an accidental duplicate (#351):
`wrap` is the one caller-facing factory name for "view existing memory, whatever
carrier you hold" (pointer+shape, pointer+strides, an mdspan — you never switch
names based on the carrier), with the family conventions (trailing keyword bag,
plain-backend space folding to its view kind); `as_tensor` is the mdspan-adaptation
PRIMITIVE under it — no keyword bag, its `<OW>` is the already-folded view kind —
which teeny's own view-producing ops call directly (pre-folded space, and it must
be a free function declared before the tensor class: its arg is a `cs::mdspan`, so
ADL can't find it) and which the mdspan-interop docs teach as the interop spelling.
The mdspan already carries element type/extents/layout, so the `wrap` form's
explicit template argument is the memory SPACE — `wrap<storage::gpu>(md)` — not a
layout type, and a 1-arg `wrap(x)` on a non-mdspan is a clean "no matching
function", #370). EVERY `wrap`
overload — the mdspan one included (#370) — takes an optional trailing memory-space tag `storage_c<Space>{}` (or the
no-braces `storage_v<Space>`), default `storage::view` (host). Pass the plain BACKEND the
memory lives in — `wrap(dptr, e, storage_v<storage::gpu>)` wraps a **device** pointer,
`storage::pinned`/`storage::mapped` page-locked host memory. Since `wrap` always yields a
VIEW, the space folds to its view kind (`gpu -> gpu_view`, …) via `storage_view_of`, so
you never spell the `_view` kinds (symmetric with `as_anyrank<Space>` /
`from_dlpack<T,Space>`). `storage::heap`/`storage::stack` name no distinct memory
space — they're ownership kinds, not backends — so passing one here just folds to a
plain `storage::view` like any other host-accessible tag (#395); for an OWNING
heap/stack tensor use `empty<T, storage::heap>(e)`/`make_heap<T>(e)` instead. Since #282,
that trailing tag is a generic keyword-tag
bag (kwargs mechanism, #277) rather than a fixed `storage_c<Space>` parameter —
same behavior today, but a future keyword (e.g. a `stream` tag) lands on all
`wrap` forms without touching any of them again. Functional
factories that deduce the extents type: `make_view(ptr,e)`, `make_local<T>(e)`,
`make_heap<T>(e)`, `make_gpu/pinned/mapped<T>(e)` (E deduced; **T defaults to
`float`**, override explicitly). `make_view` takes the layout the same two ways
`wrap` does — a positional value tag (`make_view(ptr, e, fcontiguous{})`) or the
explicit `make_view<fcontiguous>(ptr, e)` (#374) — plus the trailing `storage_c`
tag. The `make_*` owning factories are thin spellings
of one **`empty<T[, storage::Space]>(e)`** factory: ownership is deduced from the
shape (static→stack, dynamic→heap) unless a backend is named as a template arg
(`empty<T, storage::gpu>(e)`) or a value-tag (`empty<T>(e, storage_c<storage::gpu>{})`);
gpu/pinned/mapped need `<teeny/cuda.h>`. Since #280, `empty` also composes a
`dtype<T>{}`/`storage_c<O>{}`/layout tag in ANY order and ANY subset (the kwargs
mechanism, #277/#278/#279): `empty(e, fcontiguous{}, dtype<double>{})`,
`empty(e, storage_c<storage::heap>{}, fcontiguous{}, dtype<double>{})`, etc.
`empty` is **UNINITIALISED** (numpy
`np.empty` — fill it before reading), so a workspace a kernel fully overwrites pays
no zero-init; `zeros`/`ones`/`full`/`arange` and a value-initialised `local<...>{}`
(or `owned(e)`) stay zeroed/filled — zeroing is the opt-in, not the default.

Custom strided layout: `strides<Sx, Sy, ...>` is the stride analogue of `shape`,
folding known strides to immediates and storing only the dynamic ones (EBO when
all static): `tensor<float, shape<3,4>, strides<4,1>>(ptr)`. Strides are SIGNED,
so `-1` means a real stride of −1 (reversed view), NOT dynamic — a runtime
stride is spelled `dynamic_stride` (`strides<dynamic_stride,1>`, constructed with
the runtime strides). NB CCCL's `submdspan` does not apply to it, but teeny's own
slicing/slice_along/permute/flip/peel DO (they build views by hand and fold the
output strides), so a strides<...> tensor is fully sliceable.

## API cheat-sheet

```cpp
// --- geometry: LEAD with the python-y spelling (shape/strides/numel/rank) ---
t.rank();  t.numel();  t.is_contiguous();  // #axes / #elements / C-order? (numpy/pytorch default)
t.shape();  t.strides();  // the teeny shape/strides — ARRAY-LIKE accessors: [Int<k>()] folds to an
                      //   integral_constant where derivable, [i] (runtime) is a value; have rank(),
                      //   are iterable, and shape() converts to the raw extents
t.shape()[Int<1>()];  t.strides()[Int<1>()];  // static index folds; runtime index is a value
t.shape(1);  t.stride(Int<1>());  // per-axis size / stride (== extent/stride below). stride static when
                      //   derivable (static-stride layout; or a contiguous stride = product of the
                      //   STATIC extents it spans — folds even for shape<-1,3,3>: stride0=9)
t.data();  t.view();  // base ptr / non-owning teeny view tensor (gpu_view if device)
// --- raw-mdspan escape hatch (interop only; not the primary spelling) ---
t.mdspan();  t.extents();  t.mapping();  // the raw cuda::std::mdspan / raw cs::extents / layout mapping
t.extent(0);  t.extent(Int<0>());  // mdspan-side per-axis size (== t.shape(d)): runtime -> index_type,
                      //   static Int<k> -> integral_constant. Same folding as shape()/stride()

// --- indexing / slicing (python-like) ---
t(1, 2, 3);           // element access -> T& ; negative indices wrap (count from the back)
t.at(1, 2, 3);        // same element as a rank-0 VIEW (has add_/etc.); rank-0 <-> scalar
                      //   (implicit to/from T, `.item()`). at(i...).atomic_add_(v) = atomic scatter
t.uget(1,2,3); t.uget(0,slice(1,4)); t.uget(1,ellipsis); t.uat(i...);  // UNCHECKED:
                      //   uget = twin of operator() (element/slice/ellipsis, one entry point),
                      //   uat = twin of at. Skip the negative-index wrap for known-non-negative
                      //   RUNTIME indices (per-call -DTNY_NO_NEGATIVE_INDEX). Same result type;
                      //   static bounds still fold; a negative runtime index is then UB.
t(0, all, slice(1,4));  // any slice arg -> a lower-/same-rank VIEW. all = keep axis,
                      //   slice(a,b) = half-open [a,b). Integer args drop that axis.
t(0, slice(none,4), slice(1,none,2));  // python-like: none = open end, 3rd arg = step.
t(none, all, all); t(all, none); t(ellipsis, none);  // a BARE `none` arg = numpy newaxis
                      //   (`newaxis` is a named alias of `none` for this spelling):
                      //   inserts a size-1 axis (static extent 1, stride 0) -> rank+1 view,
                      //   == unsqueeze at that position (composes with int/range/ellipsis).
t(m); t.at(m); t.uget(m); t.uat(m);  // TUPLE-UNPACK: ONE tuple-like arg (cs::array / cs::tuple)
                      //   carrying the WHOLE index list — numpy's x[(a,b,c)] == x[a,b,c]. Its elements
                      //   may be anything the variadic call takes (int/Int<>/all/slice/none/one
                      //   ellipsis); it unpacks and re-dispatches, so the result TYPE is identical.
                      //   Single-arg ONLY (never mixed with other positional args); C++23 t[m] too.
                      //   Closes the loop with peel(...).enumerate(), whose m IS a cs::array:
                      //   for (auto [m, cell] : peel(a, axis<0,1,2>{}).enumerate()) b(m) = f(cell);
t(1, ellipsis, 2);    // ellipsis (numpy ...) = (rank - #other args, excl. none) copies of `all`; max one.
t(1, etc, 2);         // `etc` and `ellipsis` are ONE marker under two names (ellipsis_t == etc_t):
                      //   `etc` is the anyshape<etc,...> spelling; both names work in both contexts.
t(ellipsis) = b; t(0,all) = 3.0;  // assign INTO a slice copies/fills (b broadcasts);
                      //   `a = b` on a NAMED view REBINDS (shallow) — the contrast.
t(0, slice<1,4>()); t(0, slice<0,8,2>());  // compile-time slice (bounds fold like `all`);
                      //   slice<Int<1>,Int<4>>() is the type form (only way to bake `none`).
                      //   negative bounds wrap; all == slice(none,none) (folds).
                      //   NB a range outputs teeny's strides<...> layout (folding the
                      //   stride where derivable); a COMPILE-TIME range folds its extent
                      //   too (source static + static bounds), a runtime range's is dynamic.
t.slice_along<0,2>(i, slice(1,4));  // bind named axes only; keep every other axis (#423: NOT
                      //   numpy take_along_axis / torch take_along_dim — those are index-TENSOR
                      //   gathers, i.e. index_select. This is torch select/narrow over N axes.)
t.subsample<Axes...>(k, starts...);  // coloured/strided sub-lattice (#258): slice_along +
                      //   slice(start,none,k) per named axis, one shared step k, per-axis
                      //   start. Pure sugar, no new addressing power. k/starts accept a
                      //   runtime value or Int<k>() (folds through slice()'s own static-range
                      //   machinery -> a fully-static (start,k) pair keeps a folded static
                      //   result, same as a hand-written slice()). Value form: t.subsample(
                      //   axis<Axes...>{}, k, starts...) -- LEADING tag, same placement as
                      //   slice_along's own (a second variadic pack, the starts, needs the
                      //   disambiguating tag up front, not trailing).
t.unfold<Axis>(size, step);  t.unfold<Axis>(size);  // pytorch Tensor.unfold (#256): appends
                      //   a NEW trailing axis of width `size`, stepped by `step` along Axis
                      //   (step defaults to 1). Axis's own extent shrinks to the window COUNT
                      //   (shape(Axis)-size)/step+1; the new trailing axis holds one window's
                      //   `size` elements, at stride = Axis's ORIGINAL (un-stepped) stride.
                      //   size/step accept a runtime value or Int<k>() (folds static, like
                      //   slice()); size in [1,shape(Axis)] and step>=1, checked (static_assert
                      //   when both are static, debug-time _TNY_CHECK otherwise -- compared in
                      //   a SIGNED type so a bogus negative runtime value can't wrap through an
                      //   unsigned index_type and slip past the guard, #339 review).
                      //   Pure sugar over the existing gather -- a VIEW, so windows
                      //   ALIAS when step < size (write-through touches every neighbouring
                      //   window, as in pytorch). ND windows compose by chaining one unfold
                      //   per axis (matches nitorch's nd-unfold, itself built on the single-
                      //   axis primitive): t.unfold<0>(k0,s0).unfold<1>(k1,s1). Value form:
                      //   t.unfold(Int<Axis>(), size, step) -- single-axis Int<k>() selector
                      //   (like flip/squeeze), not an axis<...> list (only one axis binds).
t.index_select<Axis>(idx);         // gather along Axis by a rank-1 integer index
                      //   TENSOR (#326) — runtime DATA, unlike slice_along's compile-time
                      //   indices. idx values wrap negative (built on slice_along); static
                      //   idx shape -> stack result (host+device, works on a gpu/gpu_view
                      //   source, same rule as clone()), else heap (host only: source must
                      //   be host-accessible; a dynamic gpu gather: use a device
                      //   into(dest)). Always a COPY (an arbitrary data-dependent gather
                      //   isn't an affine mdspan view). into(dest) form too:
                      //   t.index_select<Axis>(idx, into(dest)) (no alloc, device-safe;
                      //   dest's axis-Axis extent must equal idx's, checked; dest must not
                      //   alias t). Value form: t.index_select(idx, axis<Axis>{}) (TRAILING
                      //   tag here -- unlike slice_along/subsample's leading one, since
                      //   index_select's only other arg, idx, is a single fixed positional,
                      //   not an open pack, so a trailing tag is unambiguous and deducible)
t.permute<2,0,1>();   // reorder axes (a permutation of 0..N-1) -> view
t.flip<1>();          // reverse an axis (negative-stride view; needs signed index)
t.flip<0,2>();        // reverse SEVERAL at once (numpy flip(a, axis=(0,2))): distinct axes in
                      //   ANY order -- flips COMMUTE, so flip<0,2> == flip<2,0> == flip<0>().flip<2>()
                      //   (same view TYPE + elements), built in ONE pass, folding each named
                      //   axis's static stride to its negation (#349)
t.unsqueeze<2>();     // insert size-1 axis at pos 2 (numpy newaxis) -> rank+1 view
t.squeeze<3>();       // drop a size-1 axis -> rank-1 view
t.unsqueeze<1,3>();   // insert SEVERAL at once: positions relative to the FINAL rank,
                      //   must be distinct, in ANY order (sorted internally, #275) -> rank+k view
t.squeeze<0,2>();     // drop SEVERAL at once: positions relative to the SOURCE rank,
                      //   must be distinct, in ANY order (sorted internally, #275) -> rank-k view
t.squeeze(axis<0,2>{});  t.unsqueeze(axis<1,3>{});  t.flip(axis<0,2>{});  t.permute(axis<2,0,1>{});
                      //   axis<...> value form (== the <...> template form) for these axis-LIST ops
t.squeeze(axis<>{});  t.unsqueeze(axis<>{});  t.flip(axis<>{});  // an EMPTY axis LIST names no axis -> a NO-OP:
                      //   same shape+strides back, as a view (numpy's axis=() rule, #369; the
                      //   same identity slice_along(axis<>{}) / peel(t,axis<>{}) already have).
                      //   NOT the no-ARGUMENT forms: t.squeeze() still drops EVERY static
                      //   singleton, t.unsqueeze() still inserts at axis 0 and t.flip() still
                      //   reverses axis 0 -- an empty
                      //   LIST and no argument at all are different things. permute needs a
                      //   FULL permutation, so permute(axis<>{}) is a no-op at rank 0 only
                      //   and a compile error otherwise (never silent).
t.reshape<6,4>(); t.flatten();  // reshape / ravel -> VIEW when regroupable without a copy
                                //   (numpy: not just C-contiguous; strided/permuted often view too).
                                //   Non-viewable = static_assert (static src) / debug-check (dyn); clone() first.
                                //   can_reshape_without_copy<...>() queries it. Output is a folded strides<...>.
t.recast<shape<-1,3,3>>();      // re-type extents (recover static dims), PRESERVE source strides (any
                                //   layout, no copy). 2nd arg = layout override: recast<E,ccontiguous>()
                                //   reinterprets AS contiguous (folds strides; "I promise"); strides<S> imposes.
                                //   Functional: recast(shape<...>{}, ccontiguous{}).
t.reindex<int32_t>();           // no-copy, layout-PRESERVING retype of the OFFSET INDEX WIDTH to Idx2
                                //   (recast staticizes EXTENTS; reindex swaps the index type — orthogonal,
                                //   compose). Narrows extents+dynamic strides to Idx2 (strides<S> pack kept);
                                //   halves a dynamic view's footprint + 32-bit device offset math. Free form:
                                //   reindex<int32_t>(t). Debug-guarded by t.index_fits<int32_t>() (signed reach;
                                //   UB if the caller lies). shape32<...> == shape_as<int32_t,...> is the int32 shape.
t.is_dense();                   // dense block in SOME order (C/F/permuted); is_dense<L>() = exact L
t.is_contiguous();              // C-order (numpy/pytorch default); is_contiguous<fcontiguous>() = F. alias of is_dense<L>
t.clone();                      // materialise a dense row-major copy. Copies on the HOST -> the dynamic-shape
                                //   overload static_asserts host-accessibility; a gpu/gpu_view tensor must use
                                //   the free to<storage::heap>(x)/to<storage::gpu>(x) (device-aware, <teeny/cuda.h>).
t.to<double>();                 // pytorch-like dtype convert -> dense owning copy (static->stack, dyn->heap).
                                //   NO-COPY when it already matches: t.to<>() (same dtype) borrows a read-only
                                //   view; t.to<T,true>() forces a copy (clone() is the unconditional spelling).
                                //   The copy runs on the HOST (dynamic overload static_asserts host-accessibility);
                                //   convert a gpu/gpu_view tensor via the free to<Space>(x) instead.
t.to(dtype<double>{});          // value-tag twin of to<double>() (deduces T, no .template on a dependent receiver)
to<storage::gpu>(t);                // MEMORY-SPACE move (cuda.h free fn): to<Space,ET,Force>(x). Same no-copy/force
                                //   rule — to<storage::gpu>(gpu_x) borrows; to<storage::gpu,void,true>(x) force-clones.
t(all, slice(none,none,-1));    // reverse a range (negative step; a[::-1])
// AXIS template args are signed: negatives count from the back (numpy). e.g.
//   t.extent(Int<-1>()), t.unsqueeze<-1>() (append), t.permute<-1,0,1>(),
//   t.slice_along<-2>(i), peel<0,-1>(t).

// --- math (in-place: any tensor/view; mutates *this) ---
a.add_(b); a.sub_(b); a.mul_(b); a.div_(b);   // tensor rhs BROADCASTS numpy-style
a.add_(2.0); a.mul_(0.5);                     // scalar rhs
y.add_(x, alpha); y.sub_(x, alpha);           // FUSED axpy: y += alpha*x / y -= alpha*x (x
                                              //   broadcasts). scaled copy y=alpha*x = y.zero_().add_(x,alpha)
a += b; a -= 2.0; a *= b; a /= 2.0;           // compound-assign sugar (scalar or tensor)
a.atomic_add_(b); a.atomic_sub_(2.0);         // ATOMIC accumulate, host and device (scatter/push);
                                              //   underlying form: add_<Atomic>/sub_<Atomic>
a.minimum_(b); a.maximum_(2.0);               // running min/max update (#325): *this =
                                              //   min/max(*this, b); tensor rhs BROADCASTS, scalar
                                              //   rhs applies to all; e.g. best.minimum_(candidate)
a.neg_(); a.abs_(); a.exp_(); a.log_();       // unary in-place
a.sin_(); a.cos_(); a.sqrt_(); a.tanh_(); a.pow_(3.0);
a.floor_(); a.ceil_(); a.round_(); a.trunc_(); a.sign_(); a.clamp_(lo,hi);
++a; --a; auto old = a++;                      // prefix in place; postfix (static shape) -> stack copy
a & b; a | b; a ^ b; ~a; a &= b; a |= 1;       // bitwise (INTEGER element types only)

// --- assignment / scatter / generic (kernel prologue/epilogue) ---
a.fill_(0.0); a.zero_(); a.copy_(b);          // b broadcasts into a
a.iota_(start, step);                         // 0,1,2,... (row-major)
a.map_(f); a.zip_with_(g, b); auto c = a.map(f); a.map(f, into(y));  // user functor (device-safe struct)
a.at(i, j).atomic_add_(v);                    // scatter-accumulate: a(i,j) += v,
                                              //   ATOMIC on host and device (push/splat write)
auto z = zeros<T>(shape); ones<T>(sh); full(sh,v); arange<T>(n);  // creation. zeros/ones
                      //   default T=float; full's T = value type; arange defaults T=int64.
                      //   Static: arange<T,N>() / arange<T>(Int<N>()) -> stack [0..N-1].
                      //   Backend selector (like empty): zeros<T,storage::pinned>(sh),
                      //   full<T>(sh,v,storage_c<storage::pinned>{}) — HOST-ACCESSIBLE only
                      //   (stack/heap/pinned/mapped); a gpu fill static_asserts ->
                      //   to<storage::gpu>(zeros<T>(sh)).
zeros(sh, dtype<T>{}); full(sh,v,dtype<T>{}); arange(n, dtype<T>{});  // value-tag T (deduced,
                      //   no .template); a LEADING explicit template arg still names the backend
                      //   — zeros<storage::pinned>(sh, dtype<T>{}). Same for empty().
zeros<storage::pinned>(sh, dtype<T>{}, fcontiguous{});  // ...and that leading backend arg takes the
                      //   SAME trailing keyword bag, ANY subset/ANY order (#373) — dtype and/or a
                      //   layout tag, or none at all (zeros<storage::pinned>(sh)). It used to accept
                      //   exactly ONE dtype{} tag and nothing else. A storage_c<...>{} tag on top of
                      //   it is the one rejection ("pick one" — that IS the backend keyword).
zeros(sh, dtype<T>{}, storage_c<storage::pinned>{});  // ...or compose BOTH value tags, either
                      //   order (zeros(sh, storage_c<...>{}, dtype<T>{}) too) — no explicit
                      //   template argument at all. Same for empty/ones/full/arange.
empty(sh, fcontiguous{}, dtype<T>{}, storage_c<storage::heap>{});  // the generic keyword
                      //   mechanism (#277-#281): dtype/storage_c/a layout tag compose in ANY
                      //   order/subset — empty/zeros/ones/full all take a layout tag this way;
                      //   arange has no layout keyword (1-D has no C/F distinction). `wrap`/
                      //   `make_*` (#282) are on the same mechanism for their `storage_c` tag
                      //   (no `dtype` keyword there — `T` is deduced from the pointer/shape).

// --- math (out-of-place -> NEW tensor; static shape -> stack, else heap/host) ---
// result type = promote(A,B): C++ rules, but among floats the LOWER width wins
//   (half>float>double, pytorch-style). Opt out with -DTNY_STD_PROMOTION.
auto c = a + b;  auto c = a.add(b);   // tensor+tensor (broadcasts) or tensor+scalar
auto c = a * 2.0;  auto c = 2.0 * a;  // scalar ops (+ and * commute; 2.0-a and 1.0/a reversed)
auto c = -a;                          // unary minus -> new tensor
auto c = a.pow(b);                    // element-wise power
auto e = exp(a); auto e = sqrt(a);    // unary free (neg/abs/exp/log/sin/cos/sqrt/tanh/floor/ceil/round/trunc/sign)
auto c = minimum(a,b); maximum(a,2.0); clamp(a,lo,hi);   // elementwise binary min/max, clamp
// Every out-of-place producer is ALSO a method (parity with a.add(b)), each taking into():
//   a.exp(); a.sqrt(); a.minimum(b); a.clamp(lo,hi); a.normalize(); a.normalize(axis<1>{});
//   a.cross(b); a.map(f);  a.exp(into(y)); ...
auto c = a.add(b,alpha); a.sub(b,alpha);  // FUSED out-of-place axpy: a +/- alpha*b (twin of add_(b,alpha))
// --- into(dest): write a producer's result into a preallocated buffer -> dest& ---
//   one fused pass, NO alloc; `into(y)` LAST arg. A distinct type (never conflated with
//   a scalar alpha). y may alias an operand / differ in dtype: the arithmetic runs in the
//   OPERANDS' compute type (scalar rhs and axpy alpha included) and only the RESULT is cast
//   to y, so a.op(b,into(y)) == y.copy_(a.op(b)) numerically (#379). y's SHAPE is
//   CHECKED against the result — the source's own shape for a scalar-rhs/unary op (only
//   operands broadcast, never the dest), the broadcast shape for a tensor rhs. Scalar-rhs/
//   unary: compile error when both shapes are static, debug-time _TNY_CHECK otherwise
//   (#357); tensor rhs: a compile error too when static (#361's bc_static_ok_dest gate),
//   but in the BROADCAST sense — each OPERAND axis == y's or 1 — else the _TNY_CHECK.
//   normalize(a,into(y)) wants EXACT equality on BOTH forms: the axis form's result is `a`
//   element-for-element, and its reduced keepdim divisor is not an operand the caller picks,
//   so it states that rule itself (check_into_same_shape) instead of inheriting the
//   broadcast engine's weaker one — a (1,3) source into a (5,3)/(2,1,3) y used to compile
//   AND pass the runtime check, silently replicating (#434).
//   y must also NOT SELF-OVERLAP (no extent>1 axis with stride 0): it would take many
//   results into one element and keep the last. Debug-checked for EVERY producer since
//   #364 (before that only the tensor-rhs one — check_dest_no_overlap was missing from
//   scalo_/unaryo_, so a.mul(2.0,into(y))/exp(a,into(y)) were silently wrong).
a.add(b, into(y)); a.mul(2.0, into(y)); a.add(b, alpha, into(y));  // elementwise / scalar / fused
exp(a, into(y)); sqrt(a, into(y)); minimum(a,b,into(y)); clamp(a,lo,hi,into(y));
normalize(a, into(y)); normalize(a, axis<1>{}, into(y)); a.map(f, into(y));
cross(a, b, into(N(i, all)));                          // cross into row i of a matrix (the "crossto")
// y may be a TEMPORARY VIEW (#380): every view-producing op (slicing/at/permute/slice_along/
//   peel_at) returns its view BY VALUE, and into() binds an RVALUE view, so "a slot of a
//   bigger output" needs no named intermediate -- cross(a,b,into(N(i,all))),
//   sum(a,into(cells.at(i,j))), sum(m,axis<0>{},into(rows(j,all))). Gated to the non-owning
//   VIEW storages (storage_is_view): a temporary view aliases storage the caller owns
//   elsewhere, so the write outlives the call, while a temporary OWNING dest
//   (into(zeros<T>(sh)), into(local<T,E>{})) is a static_assert -- its storage dies with the
//   full expression, so the result would be discarded. Use such a call for its EFFECT: the
//   dest& it returns dangles past the statement (same rule as for (auto v : peel<0>(t))).

// --- comparisons -> a bool tensor (broadcast); reduce with .all()/.any() ---
auto m = a < b; a == 2.0; 3.0 < a;    // ==,!=,<,<=,>,>= ; scalar either side
(a > 0).all(); (a > 3).any();         // bool reductions (MEMBERS: `all` is the slice kw)

// --- reductions -> scalar (all axes). ACCUMULATE in the "reduce type" (double for
//   small floats float/double/half; 64-bit int for narrow ints so sum/prod/dot
//   can't overflow mid-accumulation -> int64/uint64; reduce_type_t<T>), then
//   CAST the result to the tensor's element type: sum(float)->float, sum(int8)->int8
//   (defined truncation). A leading TYPE arg makes that type BOTH accumulator and
//   result: sum<double>(a), dot<double>(a,b), sum<int64_t>(int8_tensor) (untruncated).
//   mean(int_tensor) -> DOUBLE (numpy: integer mean is float64); mean(float)->T.
sum(a); prod(a); max(a); min(a); mean(a); dot(a,b);
a.sum(); a.mean(); a.dot(b); a.sum<0>(); a.mean(axis<1>{}); a.sum<double>();  // ALSO methods
                      //   (parity with the free forms): every reduction + sqnorm/norm/dot, all the
                      //   same overload shapes (full / axis / <Acc> / a generic trailing keyword bag).
sum(a, dtype<double>{});  a.sum(dtype<double>{});  // value-tag accumulator: == sum<double>(a)
                      //   (numpy's dtype=) — a GENERIC trailing keyword bag (kwargs.h's `_kw`):
                      //   dtype<...>/axis<...>/keepdims/into(dest) compose in ANY subset, ANY
                      //   order: sum(a, dtype<double>{}, axis<0>{}, keepdims, into(buf)) works,
                      //   same as sum<double,0>(a, keepdims, into(buf)). The explicit `<Acc,Axes...>`
                      //   template split still exists (C++17 has no universal template parameter to
                      //   unify "leading type = accumulator" vs "leading int = axis") — only the
                      //   TRAILING bag is generic. dot has the same dtype/into composition (no axis).
sum(a, into(cell)); dot(a,b,into(cell)); dot(a,b,dtype<float>{},into(cell));  // into(dest): a FULL
                      //   reduction writes its scalar into a RANK-0 dest (local<T,shape<>>{}, or
                      //   wrap(&x,shape<>{}) over an address; non-rank-0 dest = static_assert);
                      //   dtype casts, returns dest&.
allclose(a, b, rtol=1e-5, atol=1e-8);  // |a-b| <= atol+rtol*|b| everywhere (broadcasts) -> bool
a.allclose(b); a.allclose(b, rtol, atol);  // ALSO a method (parity with a.dot(b))
allclose(a,b,dtype<float>{}); allclose(a,b,rtol,atol,into(cell));  // dot/sqdist/dist's trailing bag:
                      //   dtype<Acc>{} == allclose<Acc>(a,b) picks the COMPARISON's compute type;
                      //   into(dest) writes the answer into a RANK-0 cell (a bool cell keeps it,
                      //   another dtype takes the 0/1 cast) and returns dest&. NOT a
                      //   _TNY_RED_BINARY_TAGGED invocation: the tolerances are POSITIONAL ahead of
                      //   the bag (C++17 can't default them before a trailing pack), so there is one
                      //   bag overload per tolerance arity — pass any prefix of (rtol, atol).
// --- vector algebra & geometry (contained exact math; NO solves/inversion/optimisation) ---
sqnorm(a);  norm(a);                   // Σaᵢ² / √Σaᵢ² over ALL axes. sqnorm==dot(a,a); norm floating
                      //   (int->double, mean rule). sqnorm<Acc>/norm<Acc> = leading TYPE = acc+result.
sqnorm<1>(a); norm<0,2>(a); norm(a,axis<-1>{});  // ...over NAMED AXES -> lower-rank tensor (reduction
                      //   API, like sum; sqnorm<Acc,Axes...>/norm<Acc,Axes...> too).
sqdist(a,b);  dist(a,b);               // Σ(aᵢ-bᵢ)² / √Σ(aᵢ-bᵢ)² -- mathematically sqnorm(a-b)/
                      //   norm(a-b), one fused pass, no a-b intermediate (more accurate than the
                      //   un-fused spelling for narrow types; bit-exact only for double). Binary
                      //   only (no axis form, like dot); sqdist<Acc>/dist<Acc> = leading TYPE =
                      //   acc+result; dtype<Acc>{}/into(dest) compose same as dot. Also methods:
                      //   a.sqdist(b), a.dist(b).
a.normalize_();  auto u = normalize(a);// in place a/=norm(a) (floating) / out-of-place unit vector.
                      //   normalize static->stack, dynamic->heap; zero vector -> NaN (no epsilon)
a.normalize_<1>(); normalize<-1>(a);   // ...over NAMED AXES (keepdim broadcast); axes distinct, ANY order
a.normalize(axis<1>{}); a.normalize<1>();  // ...as a METHOD too (either spelling), and every
                      //   spelling takes into(y): normalize(a,axis<1>{},into(y)); a.normalize<1>(into(y)).
                      //   y matches a's shape EXACTLY on every spelling (no broadcast leeway):
                      //   static mismatch = compile error, dynamic = _TNY_CHECK (#434)
auto c = cross(a,b);  a.cross_(b);     // 3D cross product (rank-1, length 3): new / in place (a=a×b).
                      //   Into a separate slot: cross(a,b,into(slot)) -- the slot may be a slice
                      //   of a bigger array, cross(a,b,into(N(i,all))). (no crossto_ spelling.)
// --- axis reductions -> a lower-rank TENSOR (named axes removed; negatives wrap).
//   Same rule: accumulate in reduce_type, result element type = the tensor's type
//   (mean over an integer tensor is the exception: DOUBLE, like the scalar mean).
//   sum<Acc,Axes...>(a) makes Acc accumulator AND result (leading TYPE = accumulator,
//   leading int = axis -> never collide).
sum<0>(a); mean<0,2>(a); max<1>(a); min<-1>(a); prod<0>(a); sum<double,0>(a);
//   VALUE FORM (numpy `axis=`): sum(a, axis<0,2>{}) == sum<0,2>(a); sum<double>(a, axis<0>{})
//   == sum<double,0>(a). Deduced -> no `.template` on a dependent receiver.
//   static result -> stack (host+device); any dynamic -> heap (HOST ONLY: allocates)
sum<0>(a, into(buf)); mean(a, axis<1>{}, into(buf));  // into(dest) too -> copies the lower-rank
                      //   result into buf (any spelling: <Axes>, <Acc,Axes>, or the axis<...> value form)
sum<0>(a, keepdims); sum(a, axis<0>{}, keepdims);  // keepdims: reduced axis kept as size-1 (numpy
                      //   keepdims=True), broadcasts back over a. sum/prod/max/min/mean/sqnorm/norm.
                      //   Axes distinct, in ANY order (sorted internally, #275/#371) — with or
                      //   without keepdims: sum<2,0>(a, keepdims) == sum<0,2>(a, keepdims).
sum(a, dtype<double>{}, axis<0>{}, keepdims, into(buf));  // ...and every trailing keyword composes,
                      //   any subset/order (dtype/axis/keepdims/into) — see the `_TNY_RED_TAGGED`
                      //   generic entry point in math.h, next to `sum<Acc,Axes...>(a, keepdims,
                      //   into(buf))`'s explicit-template twin (`_TNY_RED_AXIS_CORE`).
sum(a, axis<>{});     // an EMPTY axis list reduces over NO axis (numpy's axis=(), #398): each cell
                      //   aggregates its OWN element alone -> a's shape back, as an OWNED copy.
                      //   DIFFERENT from `sum(a)` — no axis argument at all is the full, EVERY-axis
                      //   reduction (a scalar). sqnorm/norm are Σaᵢ²/√Σaᵢ² over the NAMED axes, so
                      //   over none they are the elementwise a² / |a| (not a plain copy); the other
                      //   keywords compose as usual (keepdims has no reduced axis to keep).

// --- nd-peel: iterate a SUBSET of axes, each yielding a lower-rank view ---
for (auto line : peel<0,1>(t)) f(line);   // peel axes 0,1; each `line` is a view. The range-for is
                                          //   INCREMENTAL: it advances the pointer + reuses the cell
                                          //   mapping (O(1)/step), not an O(#peeled) decode per cell.
auto s = peel_at<0,1>(t, i);               // the i-th peeled sub-view — random access (grid-stride style)
for (auto v : peel<0,1>(t).subrange(lo,hi)) f(v);  // a [lo,hi) chunk: seed the cursor once at lo, then
                                          //   O(1)/step. Split [0,size()) across threads/blocks — each
                                          //   sweeps its chunk incrementally (CPU threads / device blocks).
for (auto [m, v] : peel<0,1>(t).enumerate()) g(m[0], m[1], v);  // ALSO yield the peeled multi-index m
                                          //   (m[d] = coord of peeled axis d) — for a per-axis table
                                          //   axtab[d][m[d]] or a write-by-coord. OPT-IN: the bare range's
                                          //   cell stays LEAN (no coord words); enumerate composes with
                                          //   subrange (.enumerate().subrange(lo,hi)). Or it.index(d) on the
                                          //   raw iterator. Peeled axes vary row-major (last listed fastest).

// --- nd-peel: zip-peel 2 or 3 tensors in LOCK-STEP (#327) ---
for (auto [a,b,c] : peel_zip<0>(x,y,z)) f(a,b,c);  // one cs::tuple<ViewX,ViewY,ViewZ> per step —
                      //   the "triangle's three vertex tensors" idiom. DISTINCT name from peel
                      //   (not an overload): 1 tensor -> a view, 2+ -> a tuple is a silent
                      //   return-type bifurcation on arity (mirrors python's own zip() being its
                      //   own name). Operands may differ in shape if BROADCAST-compatible (numpy
                      //   right-align, same rule a+b uses); Axes... name axes in the BROADCAST
                      //   rank's numbering (max of the operands' own ranks). Operands may also
                      //   differ in INDEX TYPE: the cells carry `_offset_int_t` over the whole
                      //   operand pack (#362) — wide enough, and on mixed signedness SIGNED
                      //   enough, for every operand, so a flipped operand zipped with an
                      //   unsigned-indexed one steps backwards instead of wrapping. Decodes fresh
                      //   each step (no incremental cursor yet — a perf follow-up, not #327's scope).
for (auto [a,b] : peel_zip(x, y, axis<0>{})) f(a,b);   // value form: axis<...> TRAILING (after
                      //   every positional tensor — unlike slice_along/peel_at's LEADING tag,
                      //   which disambiguates a second variadic pack there; peel_zip's tensor
                      //   args are each fixed-arity, so a trailing tag is unambiguous)
for (auto [m, cell] : peel_zip<0>(x,y).enumerate()) g(m[0], cell);  // (multi_index, tuple) per
                      //   step, same shape as the single-tensor peel's enumerate
for (auto cell : peel_zip<0>(x,y).subrange(lo,hi)) f(cell);   // a [lo,hi) chunk

// --- scan_/scan: sequential fold along ONE axis, batched over the rest (#254) ---
scan_<0>(t, 0.0, sum_op{});           // carry=init, then carry=f(carry,x); x=carry for each
                      //   element along axis 0 (increasing order), batched (peeled) over
                      //   every OTHER axis (reuses peel's own incremental cursor there — the
                      //   sequential part is inherent, the batching isn't). f is a device-safe
                      //   functor (lambda-free, like map_/zip_with_'s own convention): Carry
                      //   operator()(Carry carry, T x) const. The new carry doubles as the new
                      //   element -- a 1-D Felzenszwalb min-plus sweep (carry=min(carry+w,x))
                      //   is exactly this shape (examples/distance_transform.cpp's hand-written
                      //   twin). Reverse sweep composes with flip, no separate direction flag:
                      //   scan_<0>(t.flip<0>(), init, f); (scan_ has both lvalue and rvalue
                      //   overloads, so a temporary flip() view binds fine -- it mutates the
                      //   same underlying storage as a named view would).
                      //   Value form: scan_(t, axis<0>{}, init, f) — single-axis tag right
                      //   after t (matches the issue's own free-function sketch), not trailing
                      //   like peel_zip's (scan_'s only other args, init/f, are fixed-arity).
                      //   This LEADING placement is a known, still-open deviation from the
                      //   trailing-bag rule (see "Keyword arguments" below), tracked as an
                      //   open question in #348 -- NOT changed here. The axis<...> vs Int<k>()
                      //   VOCABULARY choice is unaffected and settled (see "Static vs runtime
                      //   values" above): a singleton axis<...> tag is correct here because
                      //   init is itself an arbitrary Carry value it must stay distinct from.
auto y = scan<0>(x, 0.0, sum_op{});   // out-of-place: fresh dense copy, scanned (static->stack,
                      //   dynamic->heap host-only, built on clone()); x itself untouched.
scan<0>(x, 0.0, sum_op{}, into(dest)); // into(dest): no fresh allocation beyond the copy; dest&.
                      //   dest must match x's shape EXACTLY (checked -- unlike copy_'s own
                      //   broadcast rule, since scan_ then walks dest's own axis numbering).

// --- nd-peel: peel the FIRST N axes (arbitrary batch rank) ---
for (auto v : peel_front<N>(t)) f(v);      // v is (*spatial, C); N = #batch dims. Incremental (as above);
                                           //   .subrange(lo,hi) for a threaded/block chunk.
auto v = peel_front_at<N>(t, i);            // the i-th — random access, for a grid-stride loop (i += nthreads,
                                            //   whose stride the odometer can't express — keep this entry point)
auto nb = size_front<N>(t);                 // #cells peel_front<N> yields (product of the peeled
                                            //   extents), computed directly — no range materialised

// --- dynamic-rank / dynamic-value host boundary (dynamic.h) ---
auto at = as_anyrank(data, shape, stride, ndim);    // -> anyrank: rank-erased carrier; WRAPS the
                                             //   shape/stride arrays (1-D tensor views), NO copy; HOST only (default)
auto ac = as_anyrank(data, shape, stride, ndim, copy_meta);  // COPIES into an inline TNY_MAX_RANK (default
                                             //   32) store -> trivially copyable, device-passable
auto ag = as_anyrank<storage::gpu_view>(dptr, shape, stride, ndim);  // DEVICE data: fixed()/peel_front yield
                                             //   gpu_view cells (Space param; default storage::view = host).
                                             //   from_dlpack<T[,Space]>/dispatch_dlpack<Space> set+check it vs m->device
auto af = as_anyrank(data, shape, stride, ndim, anyshape<etc,-1,-1,3>{});  // STATIC TRAILING shape baked into
                                             //   the TYPE (anyshape<etc,...>: `etc` = the erased batch, the dims after
                                             //   it = the static tail). fixed()/peel_front()/the peel_front<-Sr>()
                                             //   iterator hand out cells with those inner EXTENTS already folded — no
                                             //   per-call recast. The runtime trailing dims are debug-checked vs the
                                             //   tag ONCE here (at the boundary, next to the producer), then trusted.
                                             //   Bare `anyshape<etc>` (empty tail) == today's fully-dynamic carrier,
                                             //   byte-identical. Also on the copy_meta form (…, copy_meta,
                                             //   anyshape<etc,-1,-1,3>{}) and from_dlpack<T, anyshape<etc,-1,-1,3>>(m).
                                             //   A trailing LAYOUT tag folds the inner STRIDES too (like recast's 2nd
                                             //   arg): ,ccontiguous{} bakes a contiguous inner block (folds strides;
                                             //   fully-static tail -> EBO cell), checked vs the runtime strides at the
                                             //   boundary — subsumes dispatch_layout for the "input is contiguous"
                                             //   precondition. Default keep_strides = strides stay runtime (#209).
auto af2 = from_dlpack<T, anyshape<etc,-1,-1,3>>(m, ccontiguous{});  // ...and straight off DLPack (layout by value)
auto af3 = as_anyrank(data, shape, stride, ndim, anyshape<3,etc,5>{});  // static leading HEAD too (#219):
                                             //   anyshape<A,B,etc,C,D> = (A,B,*middle,C,D). e.g. anyshape<3,etc,5> =
                                             //   (C_in=3, *spatial, C_out=5). The Head folds in fixed()/dispatch_rank
                                             //   (full-rank window); peel_front<-Sr> stays trailing (the Head is peeled
                                             //   into the batch — inert there). Head EXTENTS fold; head STRIDES stay
                                             //   runtime (a leading stride spans the dynamic middle). Empty Head (etc
                                             //   first) == the trailing-only carrier, byte-identical.
dispatch_rank(at, [&](auto v){ kernel(v); });  // instantiates kernel once per TOTAL rank
dispatch_rank<narrow_index>(at, f);           // ...+ int32 offsets when the span fits (rank OUTER, width INNER;
                                              //   only the leaf doubles). Default Narrow=false == plain dispatch.
dispatch_index<Idx2=int32_t>(v, f);           // the primitive: narrow ONE fixed view's offset width when
                                              //   index_fits, else keep it; f instantiated for both widths.
                                              //   Batch: for (cell : at.peel_front<-Sr>()) dispatch_index(cell, f);
dispatch_layout(v, f);                        // LAYOUT sibling: runtime-classify a dyn view's strides as
                                              //   ccontiguous / fcontiguous / (else) dynamic_strides and hand f
                                              //   the RETYPED view — so recast<shape<-1,c,c>>() folds the inner
                                              //   strides SAFELY (no "I promise"). C-order arm is the win (inner
                                              //   strides fold); opt-in (triples instantiations), don't default it.
auto v3 = at.fixed<3>();                      // or force a known rank
dispatch_value<1,2,3>(D, [&](auto d){ kern<d.value>(v); });  // runtime value -> static
// BATCH idiom (one kernel per Sr, not per total rank): peel the runtime batch
// dims, keep the trailing Sr "interesting" dims static. NB the arg is NEGATIVE
// (keep the last |N|), like the tensor's peel_front — anyrank asserts N<0.
for (auto cell : at.peel_front<-Sr>()) kernel<Sr>(cell);  // Sr=2 -> peel_front<-2>; cell is rank-Sr.
                                              //   The range-for is INCREMENTAL (advances the pointer +
                                              //   reuses the cell mapping, O(1)/batch-cell, not an
                                              //   O(#batch) decode). For a CPU thread / device block that
                                              //   owns a chunk: for (cell : at.peel_front<-Sr>().subrange(lo,hi)).
for (auto [m, cell] : at.peel_front<-Sr>().enumerate()) f(m[0], cell);  // ALSO yield the BATCH multi-index
                                              //   m (m[d] = coord of batch axis d, m.rank() runtime, m.linear()
                                              //   the flat batch idx) — for a per-batch-axis table param[d][m[d]].
                                              //   OPT-IN (bare cell stays lean); m is a VIEW of the live odometer
                                              //   (don't store past the body). Or it.index(d)/nbatch()/linear()
                                              //   on the raw iterator. Composes with .subrange(lo,hi).
auto cell = at.peel_front_at<-Sr>(i);         // i-th — random access (grid-stride i += nthreads, whose
                                              //   stride the odometer can't express — keep this path)
auto cell = at.peel_front_at(i, shape<-1,c,c>{});  // FUSED peel+recast: cell has the target trailing shape
                                              //   DIRECTLY (static inner EXTENTS fold, -1 stays dynamic), no
                                              //   separate recast. == peel_front_at<-Sr>(i).recast<shape<-1,c,c>>()
                                              //   with Sr=NewE::rank(). value-form -> no `.template`. STRIDES:
                                              //   keeps runtime (layout_stride) by default — an anyrank has no
                                              //   static stride info; add a layout (,ccontiguous{}) to fold them
                                              //   (debug-checked promise) or dispatch_layout for a proven fold.
auto nb = at.size_front<-Sr>();               // flattened batch count (no range built), same NEGATIVE arg
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

Methods that take a compile-time **selector** as an explicit `<...>` template
argument also have a **deduced value-form twin**, so on a **type-dependent**
receiver (inside a kernel/template) you avoid the `x.template method<...>()`
disambiguator (a deduced call needs no `.template`; the explicit `<...>` form
does). Three selector vocabularies:

- **`axis<...>`** — the numpy-like axis selector (`axis: int | list[int]`), a
  value tag sibling to `shape<...>` (in `alias.h`). **The rule (restated by
  #348): `axis<...>` marks any named-axis argument — a LIST or a SINGLETON —
  in a call that also carries another value-form argument**, so the selector
  stays visually and typewise distinct from a positional integer or an
  arbitrary value it sits next to. List example: `peel(t, axis<0,1>{})` ==
  `peel<0,1>(t)`, `peel_at(t, i, axis<0,1>{})`, `t.slice_along(axis<0,2>{}, i,
  slice(1,4))` == `t.slice_along<0,2>(...)`, `t.squeeze(axis<0,2>{})` ==
  `t.squeeze<0,2>()`, likewise `unsqueeze`/`flip`/`permute`. SINGLETON example:
  `t.index_select(idx, axis<Axis>{})` and `scan_(t, axis<0>{}, init, f)` each
  name exactly one axis, but still reach for `axis<...>`, not `Int<k>()` —
  `idx`/`init` are themselves arbitrary values (an index tensor, an
  application-defined `Carry`) that a bare `Int<k>()` (which converts
  implicitly to a runtime integer) could be mistaken for; `axis<Axis>{}` is a
  distinct, non-converting type, so misuse is a compile error naming the
  mistake instead of silently swapped arguments. The reductions follow the
  same rule for their singleton form: `sum(a, axis<0>{})` == `sum<0>(a)`. (An
  earlier draft of this section scoped `axis<...>` to list ops only — that was
  never quite right, since the reductions' own singleton usage predates this
  wording; `index_select`/`scan_` were consistent with the rule above all
  along, so no code changed here, only this paragraph.)
  Being a single distinct-typed arg it also disambiguates `slice_along`'s two packs.
  An EMPTY list `axis<>{}` names no axis, so every axis-list op treats it as the
  IDENTITY (numpy's `axis=()`): `squeeze`/`unsqueeze`/`flip` return the same shape as a
  view (#369 — they used to fall through to the DEFAULTED single-axis form and
  silently drop every singleton / insert at axis 0), matching what
  `slice_along(axis<>{})`, `peel(t, axis<>{})` and `_keepdims<>` already did. The
  no-ARGUMENT `squeeze()`/`unsqueeze()`/`flip()` are a DIFFERENT thing and keep their own
  meanings — an empty template pack can't be told from a defaulted one in C++, so
  only the `axis<...>` spelling can carry "zero axes named". `permute` needs a full
  permutation and so already rejected it above rank 0. The REDUCTIONS follow the
  same rule (#398): `sum(a, axis<>{})` reduces over no axis and gives `a`'s shape
  back (numpy's `np.sum(a, axis=())`), while a call with NO axis keyword at all is
  the full every-axis reduction. A call site whose axis keyword is OPTIONAL must
  therefore spell "not supplied" as `_kw::unset`, never as `axis<>` — overloading
  one type for both is what made `sum(a, axis<>{})` silently reduce everything;
  `_is_empty_axis<X>` (alias.h) is the "explicitly empty" test.
- **`dtype<T>`** — a value tag for an element/accumulator type `T` (`alias.h`,
  next to `axis`), numpy's `dtype=` namesake: `empty(shape<3,3>{}, dtype<double>{})`
  == `empty<double>(shape<3,3>{})`, likewise `zeros`/`ones`/`full`/`arange` and
  `a.to(dtype<float>{})` == `a.to<float>()`. Also reused as the reduction
  accumulator: `sum(a, dtype<double>{})` == `sum<double>(a)`, and composes freely
  with `axis<...>`/`keepdims`/`into(dest)` in any subset/order (the generic
  trailing keyword bag, `tny::_kw` in `kwargs.h` — see `_TNY_RED_TAGGED` in
  math.h): `sum(a, dtype<double>{}, axis<0>{}, keepdims, into(buf))`.
- **`Int<k>()` / `shape<...>{}` / a layout tag** — for PURE single-selector ops,
  where the selector is the ONLY value-form argument in the call (nothing else
  it could be confused with) — or a variadic `Int<>`-per-axis pack, for
  `permute`: `t.squeeze(Int<1>())` (one axis), `t.permute(Int<2>(),Int<0>(),Int<1>())`,
  `t.reshape(Int<6>(),Int<-1>())`, `t.recast(shape<3,3>{})`,
  `t.is_contiguous(ccontiguous{})`. `t.unfold(Int<Axis>(), size, step)` fits
  here too even though `size`/`step` follow it: those are plain integer-like
  arguments of the very same kind as the selector, not an arbitrary/keyword
  value it needs to stay distinct from — the distinguishing question is "is
  there another *value-form* argument this could be mistaken for," not "is
  this the only argument at all."

The **reductions** `sum`/`mean`/`max`/`min`/`prod` also take the `axis<...>` value
form — `sum(a, axis<0,2>{})` == `sum<0,2>(a)`, and `sum<double>(a, axis<0>{})`
keeps the leading TYPE as the accumulator (the value axis arg and the type arg
never collide). `peel_front<N>` / `size_front<N>` (a COUNT, not an axis list) stay
template-only.

### Keyword arguments (the trailing-bag design rule)

Several call sites accept a run of **keyword-like value tags** after their
required positional arguments — `dtype<T>{}`, `axis<...>{}`, a layout tag
(`ccontiguous`/`fcontiguous`), `storage_c<O>{}`, `keepdims`, `into(dest)`. This
is a deliberate API-wide **design rule**, not an implementation detail the
caller needs to reach for: **keywords are trailing** (after every required
positional argument, in fixed order), **order-free among themselves**, and
**one of each kind per call** (matched by distinct TYPE, so nothing is
positional/ambiguous — passing two of the same kind, or a kind a call site
doesn't recognise, is a compile error naming the mistake, not silent
misbehaviour or a wall of overload-resolution noise). So `empty(shape<3,3>{},
dtype<double>{}, fcontiguous{})`, `zeros(shape, storage_c<storage::pinned>{},
dtype<double>{})`, and `sum(a, dtype<double>{}, axis<0>{}, keepdims,
into(buf))` all compose their keywords in **any subset, any order** — the
factory/`wrap`/`make_*` family (`storage.h`/`layout.h`/`tensor.h`) and the
reduction family (`math.h`) both build on the same primitive:
`tny::_kw` (`kwargs.h`) — `find_t`/`has`/`count`/`get` ask questions of the
trailing pack, the `_TNY_KW_CHECK(...)` macro is the ONE guard every call site
opens with (over `accepts<Ps...>::known/unique`), and `_kw::resolve` is the ONE
copy of the explicit-arg-beats-tag-beats-default rule each per-keyword reader
(`dtype_arg_t`/`storage_arg`/`layout_arg_t`) is a one-line alias over. **A
keyword's "not supplied" sentinel must be `_kw::unset`, never a degenerate value
of the keyword's own type** — `axis<>` doubling as both "no axis keyword" and "an
explicitly empty axis list" is what made `sum(a, axis<>{})` silently reduce over
everything (#398). Where a selector still needs an EXPLICIT
`<...>` template argument (an axis list too big to spell as a value in a
generic context, or the `<Acc, Axes...>` reduction split — C++17 has no
universal template parameter to unify "leading type = accumulator" with
"leading int = axis"), that split stays; it is the keyword BAG past it that
is generic.

This trailing-placement rule is orthogonal to the `axis<...>` vs `Int<k>()`
vocabulary choice described above — one governs WHERE a tag goes, the other
WHICH tag to use. `scan_`/`scan` are the one shipped exception to the
placement rule: `scan_(t, axis<0>{}, init, f)` puts its `axis<...>` tag
LEADING, not trailing, even though `init`/`f` are fixed-arity like
`index_select`'s `idx`. That placement question is still open, tracked in
`#348` — it is a breaking argument-order change to a shipped API and needs a
maintainer decision, unlike the vocabulary question above (already settled by
this paragraph's restated rule, no code change required).

## How the hard parts work (so you don't re-derive them)

- **Broadcasting** (`math.h`, `_md::bzip`): numpy-style — operands are aligned
  from the **right** (`bc_ext`/`bc_str` right-align each operand into the result
  rank; a shorter operand's missing leading axes are extent 1 / stride 0), so the
  result rank is `bc_rank = max(rankA, rankB)`. A dimension of extent 1 gets
  stride 0 (it is stretched). The result extent per axis is computed at compile
  time by `bc1`/`bcast_extents` (`dynamic_extent` if either operand is dynamic).
  In-place `a.op_(b)` needs `rankB ≤ rankA` (can't grow the destination).
  Out-of-place: a fully static result → `storage::stack` (host+device); any dynamic →
  `storage::heap` (host only). The SFINAE keys on `bcast_extents<...>::rank_dynamic()`,
  **not** on instantiating a stack tensor (that would fire the "stack needs static
  shape" `static_assert`). The result's offset **index type** is `_bcast_index_t` of the
  two operands' — the SAME `_offset_int_t` rule the engines decode in, over those two:
  the WIDER for a same-signedness pair (so a mixed-width broadcast, int32 view + int64
  view, yields an int64 result — lossless, and it stops the engine truncating the wide
  operand's strides to a narrow result width), a SIGNED type wide enough for both ranges
  for a mixed-signedness one. It was a `sizeof`-only pick (`_wider_int_t`), which is not
  "holds every value either operand can name" once signedness disagrees (#347): a
  signed-narrow + unsigned-WIDER pair (int16+uint32) resolved UNSIGNED — handing back a
  result that breaks teeny's signed-index contract (`flip`/a negative slice step both
  `static_assert` signed; `index_fits` is a signed reach) — and at EQUAL width the tie
  kept the FIRST, so int32+uint32 resolved to int32, which cannot hold the uint32
  operand's upper half. Only mixed-signedness pairs move (unreachable through teeny's own
  vocabulary — every teeny shape is signed), so this is a latent-hazard fix.
  `bzip_` itself decodes its offsets in `_offset_int_t<C::index_type, Ia, Ib>` — a type
  that holds every extent/stride value ALL THREE participants can produce (#346).
  Out-of-place the result already carries the wider of the two operands (no-op), but
  IN-PLACE (`c` is `a`) and `into(dest)` hand it a destination whose index type is fixed
  by the caller's own tensor and may be NARROWER than an operand's; taking the
  destination's alone truncated a wide rhs's strides — silently, since `bzip_`'s
  `static_cast<I>`s suppress even the narrowing diagnostic that caught #342. The pick is
  **signedness-aware, not `sizeof`-only**: an all-signed or all-unsigned set keeps the
  plain widest (the identical type, so those call sites are untouched), but a MIXED set
  decodes in a SIGNED type wide enough for both sides — at least the widest participant,
  and past the widest UNSIGNED one's range (twice its size, capped at 64 bits). A width-
  only pick can choose an unsigned type over a signed participant carrying a NEGATIVE
  stride, and `static_cast<uint32_t>(-1)` = 4294967295 zero-extends into the pointer
  offset instead of stepping backwards (a flipped view + an unsigned-indexed operand →
  a write off the front of the buffer). All of this is internal only: no tensor's own
  type changes, each tensor's offsets fit its own index type by construction, and
  `data()[off]` takes any integer, so nothing is narrowed back. The 64-bit cap is the
  one contract-backed step (no signed type holds all of `uint64_t`) — a reachable
  offset must fit `ptrdiff_t` anyway, and teeny's reach contract is signed throughout
  (`index_fits`). `_wider_int_t` stays the pure WIDTH pick, but it is now ONLY the width
  half `_offset_int_t` is built from — nothing else should reach for it (#347).
  The two-operand REDUCTION engine (`zipreduce_decode_`, behind `dot`/`sqdist`/`dist`)
  decodes in that SAME `_offset_int_t`, just over TWO participants instead of three
  (`_offset_int_t<Ia, Ib>` — it writes no tensor, only a scalar accumulator in the
  caller's reduce type, so there is no destination index type in play). It first took
  the first operand's index type alone (truncating the second's: a hard clang error, a
  silently wrong offset under g++, #342), then a width-only pick (#342's
  fix) — which still zero-extended a flipped operand's negative stride whenever the
  OTHER operand was unsigned and wider, i.e. `dot(flipped_int16_view, uint32_view)`
  segfaulted (#355). For an all-signed or all-unsigned pair `_offset_int_t<Ia,Ib>` IS
  the plain widest (`_widest_int` of two left-folds to exactly `_wider_int_t`, same tie
  rule), so every non-mixed instantiation is byte-identical. `zipreduce_static_` (the
  #255 static unroll) is exempt by construction, not by luck: it is gated on BOTH
  operands being fully static AND `ccontiguous`, it addresses `data()[Lin]` with a
  compile-time `cs::size_t` linear index, and it never touches an `index_type` at all —
  and a `ccontiguous` mapping's strides are products of extents, so a negative stride
  (which needs teeny's `strides<...>` layout anyway) cannot reach it.
  The SAME `_offset_int_t` is what the other multi-tensor engines decode in (#353) —
  every engine that walks more than one tensor uses it, so there is one rule, not six:
  `scalo_` (scalar rhs, `c(i)=op(a(i),s)`) and `unaryo_` (`c(i)=uop(a(i))`) over
  `<C::index_type, A::index_type>`, `allclose_` over `<A::index_type,
  B::index_type>`, and — the one ITERATION engine in the family — `peel_zip_range`
  (`iterate.h`) over its whole operand pack, `_offset_int_t` being variadic because
  its rule is stated over a participant SET (#362; the 2- and 3-tensor `peel_zip`
  forms need no separate spelling). `peel_zip` predates the chain by a day and so
  was never audited into it: it picked `cs::common_type_t` instead, which applies
  the usual arithmetic conversions, so at EQUAL width the UNSIGNED type wins
  (`common_type_t<int32_t,uint32_t>` is `uint32_t`) — not even a width mistake, the
  one shape a `sizeof` pick could not make. It is also the one site where the
  decode type is the CELL's type as well, and that second half is a bug in its own
  right: a `peel_zip` cell is a VIEW of its operand, not a fresh allocation, so
  unlike a broadcast RESULT its kept-axis strides can legitimately be NEGATIVE, and
  an unsigned index type cannot represent them even where the base pointer comes
  out right. One
  substitution fixes both halves. `scalo_`/`unaryo_` are only reachable narrow through a caller's
  `into(dest)` (their allocating producers build `c` from `a`'s own extents type) and
  were not even silent — their initializers lacked the `static_cast<I>`s, so g++ warned
  and clang rejected them; `allclose_` casts, so it was silent like `bzip_`. `scalo_`/
  `unaryo_` also took their loop BOUNDS from `a` and their strides from `c` without
  checking the two agree, unlike `bzip_`'s `_TNY_CHECK` — a mis-shaped `into(dest)`
  wrote out of bounds unguarded (#357: `a.mul(2.0, into(y))` with an 8x8 `a` and a
  2x2 `y` stored 64 elements through a 4-element buffer). They now carry the same
  guard, in the ENGINE rather than the wrapper (`scmp` calls `scalo_` directly), in
  both halves: a `static_assert` (`ext_static_eq`, the same one `dot`/`scan`'s
  `into(dest)` use) when both shapes are fully static, and a per-axis `_TNY_CHECK`
  otherwise. Both halves live in ONE helper, `_md::check_into_same_shape` (#363
  folded `scalo_`'s, `unaryo_`'s and `scan`'s three copies into it; it compares in
  `_md::ext_cmp_t`, at least 64 bits on every platform, where `scan`'s copy used to
  compare as `long` — 32-bit on LLP64). EXACT equality, NOT `bzip_`'s broadcast-tolerant
  `== ce[r] || == 1`: those engines index source and destination with the SAME
  counter and never substitute stride 0, so an extent 1 against an extent n is a
  mismatch, not a stretch. Zero cost for every other caller — the allocating
  producers build the destination from `a`'s own extents type, the in-place `unary`
  passes `c` as both arguments, and under `-DNDEBUG` the object code is byte-
  identical. NB `bzip_`'s own dest check stays runtime-only even for a fully-static
  mis-shaped `into(dest)`: `bzip`'s static gate (`bc_static_ok_r`) compares the two
  OPERANDS with each other, never either against `c`. Not a silent write (the
  `_TNY_CHECK` fires), just a later diagnosis than the two single-source engines now
  give — a candidate for a follow-up, out of #357's scope.
  **The compute type `Cv` (#379)** is the OTHER type every one of these engines carries,
  and it is about ARITHMETIC, not addressing: each element is widened to `Cv` before the
  op and the result cast back to the destination's element type after it. There are two
  suppliers. The IN-PLACE callers (`a.add_(b)`, `a.mul_(2.0)`, `a.exp_()`, …) leave it
  unspecified, which resolves to the DESTINATION's compute type — and there the
  destination IS the lhs operand, so that is a source type. Every OUT-OF-PLACE producer
  names it from its OPERANDS: the comparisons pass `Rc` =
  `compute_type_t<promote_t<Ta,Tb>>` (so `a < b` compares the values, not their bool
  cast), and the `into(dest)` entry points (`oop_to`/`oops_to`/`uop_to`) pass exactly
  what their allocating twin's `promote_t` destination would have carried. `into(dest)`
  is the ONE path where the caller picks the destination's element type, so it is the one
  place where "the destination's compute type" is NOT a source type — and taking it from
  there ran the WHOLE computation in the destination's type, operands, scalar rhs and
  axpy coefficient alike: `a.mul(0.5, into(int_y))` multiplied by `int(0.5)` == 0 (all
  zeros), `exp(a, into(int_y))` exponentiated the truncated input, and
  `a.add(b, 0.5, into(int_y))` left `y` == `a`. Silent, and the exact opposite of what
  every doc promises (source precision throughout, the RESULT cast to `dest`). Naming
  `Cv` from the operands restores the invariant that `x.op(y, into(dest))` is numerically
  indistinguishable from `dest.copy_(x.op(y))` — which also means the operands' own
  promotion rule still governs: two int tensors divide as INTS into a double dest, since
  the dest's dtype is a cast target, never a promotion input. `scalo_`/`unaryo_`'s stores
  had no explicit cast (their `Cv` used to BE `C`'s own type); they now cast to `Ce` like
  `bzip_` always did. Zero change for every other caller — the allocating producers' dest
  IS the promoted type, so the default resolves to the same type and the object code is
  byte-identical (verified: the numeric-reference tests `test_pull`/`test_posdef`/
  `test_distance_l1` compile to identical binaries; the only instruction-level change
  anywhere in the suite is `test_into`'s one double-sources-into-a-float-dest sum, which
  moves from `addss` to `addsd`). `scan`'s `into(dest)` remains the DELIBERATE exception:
  it `copy_`s into `dest` FIRST, so the whole recurrence runs in `dest`'s precision — a
  sequential carry has no "final result only" to cast (`docs/api-ux-review.md`'s F4-e).
  **`Cv` alone is not quite the whole invariant, though — a follow-up review of this same
  PR caught the gap:** the allocating twin doesn't store its result AT `Cv`, it stores at
  `promote_t<Ta,Tb>`, THEN copies that into `dest`. Those two types coincide for every
  element type except `half`/`bfloat16`, whose `compute_type_t` is `float` — so a
  half/bfloat16-sourced twin rounds float -> half -> dest, while a naive `Cv`-only
  `into(dest)` went straight float -> dest and skipped that rounding stop:
  `half(1.3).add(1.5, into(double_y))` gave 2.7998046875 (one rounding) where the twin's
  2.80078125 (two roundings) is what the invariant promises. `oop_to`/`oops_to`/`uop_to`
  (and `_cross3`'s `into` overload) now wrap `op` in `_round_to<Cv, Rt, Op>` — `Rt` being
  the allocating twin's own element type — which casts the result to `Rt` and then **back
  to `Cv`**, leaving the engine's writer to make the final cast to `dest`'s type. Both
  steps matter: `Rt` is the twin's rounding stop, and the widen back to `Cv` is what the
  twin's own `copy_` does when it reads that stored `Rt` back. Handing the writer a bare
  `Rt` instead is not just a different rounding, it does not COMPILE when `Rt` and `dest`
  are the two DIFFERENT 16-bit floats — `half`/`bfloat16` convert only from arithmetic
  types and only to `float`, so `half -> bfloat16` would need two user-defined
  conversions. (That is exactly how this shipped once and was caught in review:
  `half_a.add(b, into(bfloat16_y))` became a compile error while its twin compiled fine.)
  `Rt == Cv` for every non-16-bit-float type, so both casts vanish there (verified: object
  code for every int/float/double combination is unchanged, both compilers); it only
  changes half/bfloat16-SOURCED `into` calls, making them match their twin exactly instead
  of being one rounding more precise. `_cross3`'s `Rt` defaults to `void` = NO rounding
  stop, so the allocating `cross` and the in-place `cross_` keep their single store cast —
  only the `into` overload names it.
  **The same non-conversion, in the REDUCTIONS (#406).** A FULL reduction's `into(dest)`
  (`_TNY_RED_TAGGED`'s rank-0 branch, and `_TNY_RED_BINARY_TAGGED`'s — the binary twin
  `dot`/`sqdist`/`dist` share) writes
  a SCALAR, so no engine and no `Cv` is involved — but it still had to get that scalar
  into the cell's element type, and did it with one direct `static_cast`, which is
  ill-formed for exactly the pair above: `sum(half_a, into(bfloat16_cell))` did not
  compile at all. They now go through `_md::reduce_cast<To>` (math.h, next to
  `reduce_to`, the tensor-valued twin an AXIS reduction uses), which casts via
  `compute_type_t<From>` — `float` for either 16-bit float, `From` itself for everything
  else, so the added stop is a no-op for every other pair. No rounding question here, in
  contrast to `_round_to` above: the reduction has ALREADY cast its accumulator down to
  the result element type (`_reduce_result_t`) before this cast sees it, so the widen to
  `float` is exact and `dest` lands where it always did. The AXIS reductions never had
  the bug — their `into` goes through `out.dest.copy_(r)`, and `copy_` widens to the
  destination's compute type on the way in.
  **Contiguous linear fast path (#161, #175):** contiguous elementwise ops replace the
  per-element mixed-radix decode with a flat `for(i) cp[i]=…` loop that auto-vectorizes.
  Two flavours by whether a second array is in play:
  (a) OUT-OF-PLACE (`oop`/`oops`/`uop_out`/`oop_cmp`/`oops_cmp`) thread `Restrict=true`
  into `bzip_`/`scalo_`/`unaryo_`; when the writer is `w_set` AND every operand has the
  result's rank+extents and is C-contiguous, `cp=op(ap[i],…)` with `cp` `_TNY_RESTRICT`-
  qualified (the fresh result can't alias the operands; sources stay un-restricted so
  `a+a` is safe).
  (b) IN-PLACE SINGLE-ARRAY ops — `scal_` (scalar rhs: `add_`/`mul_`/compound/`fill_`/
  `zero_`), the in-place `unary` (`neg_`/`exp_`/`map_`), and `iota_` — are one-pointer
  read-modify-writes, so they take a fast path with NO restrict (nothing to alias).
  `scal_` and `unary` are ORDER-INDEPENDENT, so they gate on `is_dense()` (dense in ANY
  axis order — C/F/permuted; excludes stride-0 overlap + negative strides, offsets are
  exactly [0,numel)) and walk the physical block linearly — a transposed in-place op
  still vectorizes. `iota_` is order-DEPENDENT so it keeps the exact-C-order
  `is_contiguous()` gate. `scal_`'s is also gated to `w_set` so atomic scalars keep the
  decode; `unary` runs `check_dest_no_overlap` (like `scal_`/`iota_`).
  **`check_dest_no_overlap` covers every writing engine (#364).** A destination axis
  with extent>1 AND stride 0 aliases: many indices name one element. `bzip`, `scal_`,
  `iota_` and the in-place `unary` have checked for it all along; `scalo_`/`unaryo_`
  did not, and their one un-guarded reachable caller is a user-supplied `into(dest)`
  (their allocating producers `oops`/`uop_out` build a fresh dense destination, and
  the in-place `unary` checks before delegating). So the SAME mistake aborted with a
  diagnostic through `a.add(a, into(y))` and passed silently through
  `a.mul(2.0, into(y))` / `exp(a, into(y))`, writing 8 results into 2 slots. Both
  engines now call it next to #357's extent guard. NOT an OOB class (every write is
  inside the destination's own buffer) — a silently-wrong-result one, and the message
  now names both symptoms, since a read-modify-write DOUBLE-COUNTS while these two
  do a plain store and so DISCARD all but the last writer. Destination only, so a
  broadcasting RHS and an overlapping SOURCE read by `into(dest)` stay legal; the
  check is `_TNY_CHECK`, so under `-DNDEBUG` the object code is byte-identical.
  The ONLY case left on the decode is an in-place op with a TENSOR rhs (`add_(b)`/
  `copy_`): `b` may overlap the destination, so it can't vectorize (restrict = UB, a
  plain loop = assumed-overlap). Those stay byte-for-byte unchanged.
  **Static unroll (#218, #255):** a DIFFERENT mechanism from the contiguous linear
  fast path above — when the operand SHAPE(S) are fully static (not just
  contiguous), the per-element decode folds to a compile-time function of the
  linear index, so the whole loop unrolls via an `index_sequence` fold instead of
  a runtime loop at all (no back-edge, no runtime `%`/`/`). `axreduce`'s
  `reduce_axes_static_` (#218, axis reductions: `sum<0>`/`mean<1>`/…) and
  `zipreduce_static_` (#255, `dot`/`sqdist` — the two-operand case, gated on BOTH
  operands static AND C-contiguous so the same linear index addresses matching
  elements in each) both do this; each falls straight back to the runtime-decode
  engine the moment its own gate fails (any dynamic extent, or — for `zipreduce_`
  — a non-C-contiguous or mismatched-layout operand).
  **The unroll is for SMALL fixed shapes only, and is CAPPED (#343).** Both folds
  emit ONE argument per element, so an uncapped large static shape is a compile-time
  disaster: clang hard-errors past 256 fold arguments ("exceeded expression nesting
  limit of 256", its default `-fbracket-depth` — so `dot` on two `shape<16,17>`
  operands did not compile AT ALL), and g++ grows superlinearly (~1 min for a
  `shape<64,64>` reduction; `shape<128,128>` was not attempted). Both gates now go through
  `_md::_unrollable<E>()` (`math.h`, next to `_static_numel`) = fully static AND
  `numel <= TNY_MAX_STATIC_UNROLL` (`defines.h`, **256** — exactly clang's limit,
  and ~2 orders of magnitude above the 3-27-element shapes the unroll targets).
  Above the cap a static shape simply takes the same decode path a dynamic one
  does — identical results, so the cap is a pure compile-time/binary-size guard.
  Lower it (`-DTNY_MAX_STATIC_UNROLL=64`) to buy compile time back; do NOT raise it
  past 256 (clang breaks). `tests/test_static_unroll.cpp` pins the gate, both sides
  of the boundary, and that the two engines agree numerically.
- **The gather** (`tensor.h` `_slice_range`, `iterate.h` `gather_peel`): ALL
  view-making ops — `operator()` slicing, `slice_along`, `peel` — route through
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
  teeny never calls `cs::submdspan` anymore. The static stride pack is read
  through **pack folds** — `static_stride(r)` (select by sum), `ndyn()`,
  `slot(r)` — never through a `static constexpr` array member, so the pack is
  device-visible (golden rule 3) and a compile-time `r` still collapses to an
  immediate. Three call sites read it at a RUNTIME index and all three are on the
  device path: `mapping::stride(rank_type)` (`layout.h`), the gather's
  `fold_mapping` (`axis.h`), and `anyrank`'s `_build_cell` (`dynamic.h`).
- **EBO**: `tensor : private Layout::mapping<Extents>`. `mapping()` returns
  `*this`. Do not add non-static data members besides `store_`. A class privately
  inheriting TWO OR MORE empty bases for this trick (e.g. `strides<S...>::mapping`,
  `layout.h`) needs `_TNY_EMPTY_BASES` (`defines.h`, `__declspec(empty_bases)` on
  MSVC, no-op elsewhere) on the derived class — MSVC only auto-folds the FIRST
  empty base by default (#295). **Not fixable for CCCL's own `ccontiguous`/
  `fcontiguous` mappings**: `cs::layout_right`/`layout_left::mapping` stores
  extents as a *member* tagged `_CCCL_NO_UNIQUE_ADDRESS`, and CCCL deliberately
  disables that attribute on MSVC (kernel-launch data-corruption history) — so
  the sizeof-exact guarantee (a fully-static view/stack tensor == just its data)
  holds on every compiler for teeny's own `strides<...>` layout, but not on real
  MSVC for the contiguous layouts. This is CCCL's own upstream tradeoff, not a
  teeny bug — `test_tensor.cpp`'s `sizeof` assertions are `#if`'d out on MSVC
  for exactly this reason; don't try to "fix" that guard.
- **The "type pack + deduced pack" MSVC trap (#334):** confirmed once so far,
  at `peel_zip`'s `zip_oe_` (`iterate.h`). A function template that is only
  ever DECLARED — never defined, used purely for `decltype()` extraction —
  fails on real MSVC with `error C2672` when its template parameter list mixes
  a TYPE parameter pack with a FURTHER deduced pack, e.g.
  `template <class... Es, cs::size_t... A> ... f(cs::index_sequence<A...>);`
  — even though GCC and Clang accept the identical code. Fix: bundle the
  offending pack into a single tag type (`es_list<Es...>` in `iterate.h`) and
  extract it via a class-template PARTIAL SPECIALIZATION's `::value`, so the
  declared-only template itself only ever has ONE genuine pack (the deduced
  one). This is a SIBLING of, not the same as, the NTTP-pack-call quirk
  documented at `_red_ext_v` in `tensor.h` (a function CALL can't be used
  inline as a non-type template argument pack element on MSVC either) — both
  converge on the same "route through a class-template `::value`" fix shape,
  which is why it's easy to conflate them, but they are two distinct defects.
  `_red_ext_v`/`reduced_ext_` and `index_select`'s `_repl_ext_v` use that same
  `::value` shape for the OTHER (NTTP-pack-call) defect, not this one —
  `reduced_ext_` in fact still ships an UNBUNDLED leading non-type pack
  (`long... Axes`) alongside its deduced `cs::size_t... D` pack and compiles
  fine on MSVC, which is exactly the evidence this trap is specific to a TYPE
  pack, not any two-pack combination. Write any new helper of this shape (a
  pack-of-packs decltype-extraction template mixing a TYPE pack specifically)
  through the tag-bundling pattern from the start, rather than
  discovering the failure on a real-MSVC CI round-trip.
- **MSVC traps — five more recurring authoring rules** (found across the #267
  CI-onboarding fix marathon; each is a rule a future contributor can violate,
  not a one-off patch, so write NEW code to these from the start):
  - **Named trait, not an inline fold, inside `enable_if_t<...>`.** MSVC fails
    to resolve an overload set partitioned by a fold expression written
    directly inside `enable_if_t<...>` (`operator()` overload resolution,
    #268/#297). Give the fold a name (`_all_index`/`_all_ic` exist for
    exactly this) and gate on the named trait instead. **The same rule covers
    a constexpr CALL that expands a pack** (`storage_arg<O, Dflt, Tags...>()`),
    and there it is worse than a bad diagnostic: `/permissive-` MSVC cannot
    tell two parameter lists that differ ONLY in such an inline condition
    apart, so it MERGES the two overloads into one template and rejects the
    second as a redefinition (`C2995`) with duplicated default template
    arguments (`C2572`) — which is how the whole `empty`/`full`/`zeros`/`ones`
    family collapsed to `void` the moment #316 turned `/permissive-` on. Fix
    (`_fac_storage`/`_fac_on_stack`/`_fac_allocates`, `tensor.h`): route the
    call through a class template's `static constexpr ... value` and give each
    HALF of the split its OWN trait name, so the two lists differ by a plain
    template-id. A condition with no pack in it is fine inline — the pack is
    the part MSVC chokes on. NB each factory splits TWICE: once on the `T`-led
    entry point, once on the BACKEND-led one right below it (#373), and BOTH
    halves of both pairs must gate on the named traits. #373 reproduced the
    bug verbatim in its new overloads because that branch predated this fix —
    which is exactly why this is an authoring rule and not a one-off patch.
  - **Floor a pack-derived array to size 1.** `x[sizeof...(D)]` is a
    zero-length array — a GCC/Clang extension MSVC rejects (`C2466`) — the
    moment a rank-0 operand makes the pack empty. Always
    `x[sizeof...(D) ? sizeof...(D) : 1]` (#313/#314, `math.h`/`layout.h`'s
    rank-0 engines).
  - **Inside `tensor`'s body, route a qualified `E::rank()` /
    `E::static_extent(d)` through `_shape_rank`/`_shape_static_extent`** — even
    when `E` is a template parameter unrelated to `tensor`'s own private EBO
    base (`tensor::rank()`'s own private-access trip, #294/#297, is this same
    rule; #315/#318 is where it was named and fixed at every OTHER site).
    Demonstrated cost: `index_select` (#326) put raw `Ei::rank()`/
    `Ei::static_extent(0)`/`DstE::static_extent(A)` straight back in one
    commit after this rule was established (#366) — nothing told the author
    not to until now. Latent, not confirmed failing: it only misfires for a
    `strides<...>`-layout receiver, which nothing in `test_index_select.cpp`
    exercises, so real-Windows CI is currently green on it regardless.
  - **`long` is 32-bit on Windows (LLP64).** Anything that must hold a value
    past `2^31` (an extent, stride, or offset) needs `cs::int64_t`, not `long`
    (#320/#321).
  - **Teeny headers `#undef min`/`max`/`interface` OUTSIDE the include guard**
    (`defines.h`), which only keeps them SUPPRESSED (teeny never restores
    them) if every teeny header includes `<cuda/std/...>` *before*
    `<teeny/defines.h>` — CCCL's own headers restore the Windows definitions
    at the end of each of theirs (#328/#329). A new header written
    include-order-backwards silently breaks
    this. Also: always write `numeric_limits<T>::min()`/`max()` parenthesized,
    the standard workaround for the same macro collision.
  - **A scoped-enum template argument can land in an integral NTTP slot on
    `/permissive-` MSVC** (#316), where the standard says the candidate is
    discarded. Two shapes, both real: (1) `from_dlpack<T, Space>(m)` written
    with DEPENDENT arguments inside a template also matched the fixed-rank
    `from_dlpack<T, cs::size_t R, storage Space>` (`Space` bound to `R`) and
    went ambiguous — `dispatch_dlpack`/`dispatch_dlpack_dtype` therefore call
    `_dl::import_anyrank<T, Space>` straight, one candidate on every compiler;
    with non-dependent arguments MSVC gets it right, so the public
    `from_dlpack<T, storage::gpu_view>(m)` spelling is fine. (2) `empty<T,
    storage::O>(shape)` — the shape drags `cuda::std::empty(const T (&)[N])` in
    by ADL, and with EXACTLY TWO explicit arguments the arity matches, so MSVC
    hard-errors (C3411) binding the enum to `N`. Three or more explicit
    arguments (teeny's own internal `empty<T, storage::gpu, Layout>` calls) miss
    the arity and are unaffected. There is no library-side fix — ADL cannot be
    turned off — so tests qualify `tny::empty<...>` and `docs/cuda-compat.md`
    steers users to the keyword spelling `empty<T>(e, storage_c<O>{})`. Watch
    for it whenever a new API takes a `storage`/enum non-type parameter in the
    second template slot.

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
6. **Update the docs IN THE SAME change** (docs drift is a bug, not a follow-up):
   the header **doc-comment** (feeds the autodoc API page) AND every hand-written
   `docs/*.md` page that describes the touched API (`cheatsheet.md`, `reference.md`,
   `structure.md`, `indexing.md`, … — grep the docs for the affected names) AND the
   `CLAUDE.md` cheat-sheet. Updating only `CLAUDE.md` is not enough — the docs site
   is separate and drifts silently.
7. Commit with a focused message.

## Documentation style

Docs are for **humans who are not necessarily C++ template-metaprogramming
experts**. They want to know how to call teeny well — correct, beautiful
call-site code, and (where it matters) why that call compiles to fast code.
Apply this to every hand-written `docs/*.md` page, tutorial, and example —
not just the header doc-comments:

1. **teeny vocabulary over mdspan vocabulary.** Lead with teeny's own spelling
   — `shape()`/`strides()`/`rank()`/`numel()`/`is_contiguous()` — not the raw
   mdspan escape hatch (`extent()`/`extents()`/`mapping()`). The mdspan
   spellings are the *interop* path (`docs/mdspan-vs-teeny.md`), not the
   primary one; don't let them leak into pages that aren't specifically about
   interop.
2. **Value sugar before templates, as tabs, in this order.** When an API has
   more than one equivalent spelling, show them as `=== "..."` tabs, in this
   fixed order: (1) the value-tag sugar (`axis<...>{}`, `dtype<T>{}`, a layout
   tag, …) — what a caller should reach for first; (2) `Int<k>()`/
   integral-constant values; (3) the explicit `<...>` template form last — the
   escape hatch for when the sugar can't be deduced (a bare `none`, say).
3. **Plain language.** No agentic/internal-monologue phrasing, and no
   unexplained internal engine names (`_md`, `bzip`, `_offset`, …) — those
   belong in code comments and this file's internals section, not in
   `docs/*.md`. Say what a reader needs in order to use the library, not what
   teeny's implementation does internally.
4. **Leanness bar.** If an example or tutorial snippet doesn't read as short
   and natural, that's a signal: either a nicer teeny spelling already exists
   and the example should use it, or teeny itself is missing sugar — file an
   `enhancement` issue for the gap rather than shipping an awkward example.

## Testing

`make run-test` builds and runs every `tests/test_*.cpp`, printing `PASS`/`FAIL`.
Tests return non-zero on failure (the return code is the failing check number).
`test_cuda` compiles against a malloc-backed fake CUDA runtime in
`tests/fakecuda/`. `test_pull`, `test_distance_l1`, `test_posdef` port real
kernels and validate them numerically against hand-written references — when you
change math or layout code, these are the ones that catch regressions.

**Never commit build output.** `make` writes compiled binaries into `BUILDDIR`
(default `./build`; a second-compiler run is usually `make BUILDDIR=./build-clang`
or similar). `.gitignore` ignores `build*/`, so every conventional build dir is
covered — but if you point `BUILDDIR` somewhere else, ignore that path too and
**`git status` before every commit**: a stray `build*/` or a compiled test binary
(extensionless ELF, so `*.out`/`*.exe` won't catch it) in the staged set means the
gitignore was bypassed — never `git add -A` past it.

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
