# API UX review (#336)

A deliberate pass over `docs/cheatsheet.md` / `docs/reference.md`, checking
every public entry point against three questions: does its **name** match the
closest numpy/pytorch/xtensor prior art (or diverge for a stated reason)? Does
its **argument order / keyword placement** follow teeny's own stated rules? Is
it **symmetric** with sibling ops in the same family? This is the output
`#336` asks for: each finding below is either filed as a follow-up issue or
marked intentional with a one-line rationale. Nothing here is a mandate to
rename anything — the point is that every divergence from prior art, or from
teeny's own conventions, is *deliberate and written down* rather than an
accident of whichever call site was written first.

## Naming

| # | Finding | Disposition |
|---|---|---|
| F1-a | `into(dest)` has no numpy/pytorch lexical match. Of the **seven** registered keyword kinds (`tny::_kw::is_keyword`: `axis`, `dtype`, `keepdims`, `ccontiguous`/`fcontiguous`, `storage_c`, `into_t`), only `axis`/`dtype`/`keepdims` reuse numpy's own spelling verbatim; `storage_c` has no numpy/pytorch match either (closest is torch's `device=`), and the layout tags don't match numpy's `order='C'` spelling. | **Intentional.** Rationale: pytorch's closest analog to `into` is the positional `out=` kwarg, which reads as ordinary keyword syntax in python; teeny has no keyword syntax, so a same-named `out(dest)` would still be a bare positional call, no clearer than `into(dest)` — and `into` reads better as English at a C++ call site (`a.add(b, into(y))` vs `a.add(b, out(y))`). `storage_c`/the layout tags diverge from numpy for a different, unrelated reason (numpy has no separate memory-space/layout-tag concept to match at all — see F1-e for the same "no applicable prior art" shape). No action needed for any of the three. |
| F1-b | `wrap()` and `as_tensor()` are two public names that, for the `wrap(any_mdspan)` overload specifically, are extremely close in behavior — the default (no-tag) call is still codegen-identical to `as_tensor(md)` (`-O2` instruction streams match byte-for-byte; g++ even tail-calls one into the other), though since #370/#393 `wrap(md, tags...)` is no longer a literal one-line forwarder in source (it now carries the same `storage_view_of`/keyword-bag machinery as `wrap`'s pointer-form siblings). | **Filed as #351** (documentation). A rationale *is* already stated in passing (`tensor.h`'s doc-comment and `docs/reference.md`/`docs/tensors.md` both note "`as_tensor` is what teeny's own view-producing ops call internally"; `docs/cheatsheet.md` only notes "same thing," without that detail) — but that only explains why `as_tensor` exists as an internal name, not why it's *also* public and behaviorally identical to `wrap` in the default case. #351 asks for either a real public-facing rationale (the two do diverge on other overloads — `wrap` takes a raw pointer + extents/layout/storage tags, `as_tensor` only takes an already-built `cs::mdspan`/`submdspan` result) or, if the `any_mdspan` overload really is fully redundant, deprecating `wrap(any_mdspan)` in favor of calling `as_tensor` directly there. Filed rather than fixed here since it needs a decision from the library owner. |
| F1-c | `clamp` matches pytorch's name, not numpy's `clip`; `minimum`/`maximum`/`allclose` happen to match both. No stated policy on which prior art teeny defaults to when the two disagree. | Tracked under **#337** (naming policy) — this is a second concrete instance of the question #337 already raises, not a new issue. |
| F1-d | `sqnorm`/`sqdist` have no numpy/pytorch/xtensor precedent (neither library spells "squared norm" as one word), unlike their siblings `norm`/`dist`/`cross`/`normalize` in the same doc section, which all do. | Tracked under **#337** — same pattern as F1-c: a teeny-invented name sitting beside prior-art-modeled siblings. |
| F1-e | `recast`/`reindex` are teeny-invented, sitting beside pytorch-modeled `to<T2>()` in the same "Structure" doc section. | **Intentional, not filed.** Neither operation (retype static extents in place / narrow the offset-index width, both no-copy) has a numpy/pytorch/xtensor equivalent — those libraries don't expose a static/dynamic shape distinction or an index-width choice at all, so there is no prior art to diverge from. Noted here for completeness per #336's own instruction to record the "no applicable prior art" case, not left silent. |
| F1-f | `subsample` (built on `slice_along` + `slice`) is a second teeny-invented name in the same indexing section that also contains `unfold`/`index_select`. | **Intentional, low severity.** `subsample` is pure sugar (a fixed composition of `slice_along`+`slice`, #258) with no single-call numpy/pytorch equivalent to name it after (numpy's strided-slice spelling, `a[::k]`, isn't a named function). No action needed. |
| F1-h | The view op that binds named axes to a scalar index or a slice was called `take_along` — one underscore away from numpy's `take_along_axis` / pytorch's `take_along_dim`, which name a data-dependent gather driven by an index *array*. Same-family prior art already exists in teeny under the right name (`index_select`, mirroring `torch.index_select`), so the old name pointed a numpy/pytorch reader at the wrong operation. | **Renamed to `slice_along` in #423** (clean rename, no compatibility alias — teeny is pre-1.0 and has no deprecation convention). `slice_along` says what it does (bind axes by index/slice, always an affine view) and matches its real prior art, pytorch's `select`/`narrow` generalised to several axes. The `take_along*` name is now left free for the gather family it belongs to. |
| F1-g | The whole `peel`/`peel_at`/`peel_zip`/`peel_front`/`scan_`/`scan` family is uniformly, deliberately invented — zero numpy/pytorch precedent anywhere in it. | **Positive finding, no action.** `docs/structure.md` already analogizes `peel_zip` explicitly to python's `zip()`, so at least that part of the invented vocabulary is documented as intentional rather than silently divergent. `scan_`/`scan` are not analogized to any prior art there (the closest real-world equivalent, JAX's `lax.scan`, isn't currently mentioned anywhere in the docs) — worth adding as a one-line cross-reference in `docs/structure.md`'s `scan_` section, but that's a doc-polish gap, not a UX defect on its own, so recorded here rather than filed. |

## Argument order & keyword placement

The stated rule (`CLAUDE.md`): a keyword tag is **leading** when the call
still has an *open variadic pack* after it (so the tag disambiguates that pack
from another one — `slice_along`, `subsample`, `peel_zip`/`peel_at`'s tensor
args are fixed-arity so *their* tag is trailing, see table); a keyword tag is
**trailing** when the receiver's only other arguments are fixed-arity
(`index_select`, `peel_zip`'s axis tag).

| Call site | Other args | Rule predicts | Actual | OK? |
|---|---|---|---|---|
| `slice_along(axis<...>{}, i, slice...)` | open pack (per-axis slice/index args) | leading | leading | yes |
| `subsample(axis<...>{}, k, starts...)` | open pack (`starts...`) | leading | leading | yes |
| `index_select(idx, axis<Axis>{})` | fixed (`idx`) | trailing | trailing | yes |
| `peel_zip(x, y, axis<0>{})` | fixed (tensor operands) | trailing | trailing | yes |
| `scan_(t, init, f, axis<0>{})` | fixed (`init`, `f`) | trailing | trailing | yes (**fixed in #348**) |

`scan_` used to lead with its tag (`scan_(t, axis<0>{}, init, f)`), a real
deviation from the stated rule — `init`/`f` are fixed-arity, so the rule
predicts trailing — that nothing rationalized. That, plus a second sub-finding
(`index_select`/`scan_` use `axis<Axis>{}` for a single axis, where CLAUDE.md's
*then*-stated taxonomy scoped `axis<...>` to axis-**list** ops only), were
**filed as #348**. **Both halves are now fixed:**

- The **vocabulary** sub-finding turned out to be a documentation bug, not a
  code defect. Re-reading the taxonomy claim against the library's own flagship
  keyword family shows CLAUDE.md's wording, not `index_select`/`scan_`'s
  spelling, was what was wrong: the reductions already used `axis<...>` as a
  singleton (`sum(a, axis<0>{})`) before this review was even written, so "list
  tag" was never an accurate description of `axis<...>`'s actual scope.
  CLAUDE.md's "Static vs runtime values" section now states the restated rule
  directly: `axis<...>` marks any named-axis argument — list or singleton — in
  a call that *also* carries another value-form argument, so the selector stays
  visually/typewise distinct from a positional integer or an arbitrary value it
  sits next to (`scan_`'s `init` is exactly such a value; `Int<k>()` converts
  implicitly to a runtime integer and would blend in where `axis<0>{}`, a
  non-converting distinct type, cannot); `Int<k>()` is reserved for pure
  single-selector ops where the selector is the *only* value-form argument in
  the call. Under that rule `index_select` was consistent all along and needed
  **no code change**.
- The **placement** sub-finding was fixed by moving `scan_`/`scan`'s tag to
  trailing, so the table row above now reads "yes". It is a **breaking
  argument-order change** to a shipped API, taken (with the maintainer's
  sign-off) as a clean break: there is no deprecated leading-form alias, and a
  call that still passes the tag leading fails on a `static_assert` naming the
  new order. While there, `scan`'s two trailing keywords (`axis<...>{}` and
  `into(dest)`) were put on the generic keyword bag (`_kw`), so they compose in
  any subset and any order exactly like the reduction family's do.

`unfold(Int<Axis>(), size, step)` was initially flagged as a second violation
but doesn't hold up: `Int<Axis>()` is a single-axis **selector** (like
`flip`/`squeeze`'s own `Int<k>()` form), not a **keyword tag** in the sense
the leading/trailing rule governs — CLAUDE.md's own taxonomy treats selectors
and keyword tags as two different vocabularies, and this review's own
argument-order table conflated them for this one row. Not filed.

The sweep also found the review's one clear-cut **documentation bug**, not a
design inconsistency: `docs/indexing.md` claimed `index_select`'s value form
"leads with an `axis<...>{}` selector, like `slice_along`" — backwards; the
actual placement (confirmed against `tensor.h` and consistent with
`cheatsheet.md`/`reference.md`/`CLAUDE.md`) is trailing, for the reason in the
table above. **Fixed directly in this change** (`docs/indexing.md`) since it
was a factual contradiction across docs, not an open design question.

## Symmetry across op families

| # | Finding | Disposition |
|---|---|---|
| F3-a | `flip` has no multi-axis form, unlike `squeeze`/`unsqueeze` (both gained `<Axes...>` + `axis<...>{}` under #275) and unlike numpy's `np.flip(a, axis=(0,1))`. | **Filed as #349, fixed there**: `flip<Ax0,Ax1,...>()` + the `axis<...>{}` twin (with the empty-list no-op the other axis-list ops got in #369). It needed no `_sorted_axes` fold in the end — reversing one axis never disturbs another, so the axis list is simply a *set*: one pass negates every named axis's stride and moves the base pointer to its last element along each, and any order of the same axes yields the identical view type. |
| F3-b | `allclose` has no method form (`a.allclose(b)`) and doesn't compose `dtype<Acc>{}`/`into(dest)`, unlike its closest siblings `dot`/`sqdist`/`dist`, which do both. | **Filed as #350, now fixed.** `a.allclose(b, rtol, atol)` and the `dtype<Acc>{}`/`into(dest)` bag now mirror `dot`/`sqdist`/`dist`'s shape. One shape difference remains, and is inherent rather than an omission: `allclose`'s two tolerances are optional POSITIONAL arguments, which C++17 cannot default ahead of a trailing keyword pack, so the bag comes after any prefix of `(rtol, atol)` — `allclose(a, b, 0.1, into(cell))` — instead of directly after the operands. |
| F3-c | `normalize<Axes...>`'s result is always full-shape (unlike `sum`/`prod`/`max`/`min`/`mean`/`sqnorm`/`norm`, where `keepdims` is an explicit, user-facing opt-in on the *result* shape) — internally, `normalize` always computes its norm with keepdims-broadcast semantics, with no tag to turn that off. | **Intentional, not filed.** `normalize` divides the source by a per-slice norm, broadcast back over that same slice; this only works in general (e.g. `normalize<1>` on a `(3,4)` source producing a `(3,)` norm) if the norm keeps its reduced axis as size-1 so the divide can broadcast it back — a non-keepdims norm isn't broadcast-compatible with the numerator for a non-leading reduced axis. (A leading-axis case like `normalize<0>` would technically still broadcast either way, but the operation needs one uniform rule, and only the keepdims one generalizes to every axis.) So this isn't a style choice on `normalize`'s own output shape (which was never a keepdims/non-keepdims question to begin with) — it's that the *internal* norm has no reason to expose a keepdims toggle nobody could safely turn off. Recorded here as the one-line rationale #336 asks for so it doesn't read as an accidental omission. |
| F3-d | The `iterate.h` family (`peel`/`peel_at`/`peel_zip`/`peel_front`/`scan_`/`scan`) is free-function-only, with no method twins — every math/structure op in the library is explicitly "also a method" (`CLAUDE.md`'s own promise for `add`/`exp`/`minimum`/etc.). | **Intentional, not filed.** These read (this review's own characterization, not yet a claim `docs/structure.md` itself makes — see F1-g) as iteration/control-flow constructs closer to python's `zip()` or JAX's `lax.scan` (both free functions in their own ecosystems, not methods) than to data-transform ops; a `t.peel<0>()` method form would also read oddly since `peel` operates over N tensors symmetrically in `peel_zip`'s case, with no privileged receiver. Not generalized as a stated family-wide rule anywhere in the docs today — worth a short note in `docs/structure.md` at some point (folding in F1-g's `lax.scan` cross-reference gap too), but that's a doc polish, not a UX defect, so no issue filed. |
| F3-e | The core reduction family (`sum`/`prod`/`max`/`min`/`mean`/`sqnorm`/`norm`) — full/axis/`<Acc>`/keyword-bag forms, method and free-function — is fully uniform. | **Positive finding, no action.** Confirmed by direct comparison across all seven; this is the family the rest of the API should be measured against. |

## Prior-art fidelity

| # | Finding | Disposition |
|---|---|---|
| F4-a | Float/int promotion (`compute_type`, the float-narrows-not-widens rule) diverges from C++'s own usual arithmetic conversions, and from numpy/pytorch's widen-on-mixed-width rule. | **Positive finding, no action.** Already the model example of a *documented* intentional divergence: a worked example (fenced code block in `docs/math.md`) plus an explicit escape hatch (`-DTNY_STD_PROMOTION`, the one-sentence rule + build-flag row in `docs/reference.md`). Cited here as the bar the rest of this review is held to. |
| F4-b | `flip`'s missing multi-axis form is also a fidelity gap against `np.flip`. | Same finding as F3-a — cross-referenced, not double-filed. |
| F4-c | Negative-axis handling (numpy-style, count from the back) for **compile-time** axis arguments (`Int<k>()`/explicit `<Axis>` template args) across every op checked (`unsqueeze`/`squeeze`, `permute`, `slice_along`, `peel`, `extent(Int<-1>())`). | **Positive finding for the compile-time case, no action** — confirmed uniform there. Scoped deliberately: the **runtime** overloads (`extent(index_type d)`, `shape(d)`, `stride(d)`) do NOT wrap a negative `d` — they cast straight to `cs::size_t` with no normalization step, so a negative runtime axis argument is undefined behavior, not a wrapped negative index. This isn't a new finding to file (CLAUDE.md itself only ever claims the wrap for the static/`Int<k>()` form), just a scoping correction so this row isn't read as "all axis arguments negative-wrap," which isn't true. |
| F4-d | `keepdims` is uniform (explicit opt-in) across the seven true axis-reductions, correctly N/A on the three binary ops with no axis form (`dot`/`sqdist`/`dist`), and silently different (always-on, no tag) on `normalize`. | The `normalize` case is F3-c, already given its rationale above; the reduction family itself is the F3-e positive finding. No new issue. |
| F4-e | `scan`'s `into(dest)` casts `dest`'s dtype **first**, so the whole recurrence runs in `dest`'s own precision — every other `into(dest)` in the library (reductions, `index_select`) computes in source precision and casts only the **final** result. | **Intentional, documented at the call site** (added during #340's own review round) but not cross-referenced from the general "Keyword arguments" design-rule section in `CLAUDE.md`. Recorded here as the explicit rationale #336 asks for: `scan_`'s carry is inherently sequential and stateful (each step depends on the previous), so "cast only the final result" isn't an available option the way it is for a one-shot reduction — the precision the recurrence runs in **is** the precision of every intermediate carry, not just the output. No new issue; a `CLAUDE.md` cross-reference is a candidate for a future doc-polish pass, not a UX defect. **Correction (#379).** The "every other `into(dest)`" half of this row was true of the reductions and `index_select`, which is all it was checked against — but NOT of the elementwise/unary/scalar/axpy family, which at the time this review was written took its compute type from the destination and so ran the entire computation (operands, scalar right-hand side, axpy coefficient) in `dest`'s type: `a.mul(0.5, into(int_y))` yielded all zeros. That was a bug, not a second deliberate exception, and it is fixed in #379 — the elementwise family now computes in the operands' precision and casts only the result, like everybody else. So the row's *rule* was right, its *scope claim* was premature: `scan` is the sole exception **now**, and for the reason given above; before #379 it was one of two, and the other one was silent and wrong. |
| F4-f | `mean(int_tensor) -> double` and `normalize(int_tensor) -> double` both mirror numpy's own integer-mean-is-float64 rule. | **Positive finding, no action.** Worth distinguishing explicitly from F3-c/F4-e above: this is teeny *correctly matching* prior art, not another undocumented special case — recorded so it doesn't get miscounted as a gotcha. |

## Summary

Of the concrete gaps found, three were filed as sub-issues of #336: **#348**
(axis-tag placement on `scan_` + the `index_select`/`scan_` vocabulary split —
`unfold` was initially suspected but turned out not to be an instance, see
above), **#349** (`flip` multi-axis form), **#350** (`allclose` method form +
keyword composition). Both of #348's sub-findings are now fixed: the vocabulary
split turned out to be a CLAUDE.md wording bug rather than a code defect and was
fixed by restating the taxonomy (no code change), and the placement gap
(`scan_`'s tag was leading, not trailing) was closed by moving `scan_`/`scan`'s
tag to trailing — a breaking argument-order change, taken as a clean break with
no deprecated alias. One needed a documentation decision and is filed as
**#351** (`wrap`/`as_tensor` naming rationale). Two findings
(`clamp`/`clip`, `sqnorm`/`sqdist`) feed the already-open naming-policy
question in **#337** rather than standing alone. One documentation bug
(`docs/indexing.md`'s backwards claim about `index_select`'s tag placement)
is fixed directly in this change. Everything else resolved to either an
intentional divergence with a stated one-line rationale (`into`/`out` naming,
`normalize`'s forced `keepdims`, the `iterate.h` family's free-function-only
shape, `scan`'s `into(dest)` precision timing) or a positive finding recorded
as evidence of what "done well" looks like elsewhere in the API (the
promotion table, negative-axis handling, `keepdims` uniformity, the core
reduction family, `mean`/`normalize`'s numpy-faithful int promotion).
