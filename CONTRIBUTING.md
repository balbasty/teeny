# Contributing to teeny

teeny is a **header-only C++17 tensor library for host + CUDA**, built on CCCL's
`cuda::std::mdspan`. It stays small on purpose: one tensor type, short
single-purpose headers, and *only* what mdspan lacks. Please keep it that way —
**prefer deleting code to adding it**.

This file is the human-facing process guide. The deep design rules (golden rules,
how the hard parts work, the add-a-feature checklist) live in
[`CLAUDE.md`](CLAUDE.md); read it before changing core code.

---

## Issue-based workflow

Work is tracked in issues. Before writing code:

1. **File an issue** describing the change (or claim an existing one). One issue =
   one coherent change.
2. **Break up big work into sub-issues.** A multi-part effort gets an umbrella
   issue with a checklist, and each part its own issue that references the parent
   (`part of #NN`). Land the parts as separate PRs; close the umbrella when the
   last one merges.
3. **Triage with a label** (see below) — every issue should carry at least one.
4. **Reference the issue** from your branch and PR, and close it from the PR body
   with `Closes #NN` (a squash-merge then auto-closes it). Use `part of #NN` when
   a PR advances but doesn't finish an issue.

Don't leave finished work untracked and don't leave merged issues open.

### Labels (triage)

| label | for |
|---|---|
| `bug` | incorrect behaviour, UB, crashes |
| `enhancement` | new feature or API |
| `perf` | efficiency / codegen, correctness unaffected |
| `maintainability` | refactors, dedupe, internal cleanup |
| `documentation` | docs, comments, guidelines |
| `blocked` | needs hardware, a design decision, or human input to proceed |
| `good first issue` | small, well-scoped, low-context |

`bug` / `enhancement` / `documentation` are GitHub defaults; add the rest once.

---

## Branches, commits, PRs

- **One branch and one PR per task — never bundle.** A branch fixes exactly ONE
  thing (one issue, one feature, one refactor). If you notice an unrelated bug or
  improvement while working, resist the urge to fix it here — file an issue or note
  it, and give it its OWN branch. A PR that touches several unrelated concerns is
  hard to review, hard to revert, hard to bisect, and hard to reason about; split
  it. Do this even when a workflow hands you a shared branch: rebase each concern
  onto its own branch rather than stacking them.
- **Name the branch for the task** under your author prefix (`claude/…` for
  Claude-authored work, `<user>/…` otherwise), descriptive and referencing the
  issue where there is one: `claude/fix-59-anyrank-device-guard`,
  `claude/hardened-bounds-checks`, `claude/docs-assignment-semantics`,
  `claude/ci-nvcc-compile`. (The `fix:`/`feat:`/… prefix belongs on the commit
  *subject*, below.) Never commit straight to `main`.
- **Commit subjects** use a conventional prefix, then a concise imperative
  summary:

  ```text
  fix:      a bug / UB / crash
  feat:     a new feature or API surface
  refactor: behaviour-preserving restructuring (dedupe, extract, rename)
  perf:     an efficiency change (behaviour unchanged)
  docs:     docs / comments / this file
  test:     tests only
  chore:    build, CI, tooling
  ```

  Explain the *why* in the body, not just the *what*. Reference the issue.
- **One PR per branch/task.** Title mirrors the single change; body says what
  changed, why, how it was verified, and links the issue (`Closes #NN`). If you
  catch yourself writing "and also…" in a PR body, that "also" belongs on its own
  branch.
- **Squash-merge**, then delete the branch.

There is a PR template (`.github/pull_request_template.md`) — fill it in.

---

## Build & test gate (must pass before you open a PR)

Everything must compile and run on **both** compilers at `-std=c++17`:

```sh
make CXX=g++ run-test && make CXX=clang++ run-test      # 30+ tests, all PASS
make CXX=g++ run-examples                               # the example kernels
```

- Add a `tests/test_<feature>.cpp` for anything new, mixing `static_assert`
  (compile-time shape/stride checks) and runtime asserts, and wire it into the
  Makefile's `TESTS` list. A test returns non-zero (the failing check number) on
  failure.
- For host code paths, run the **sanitizers** on the affected tests:

  ```sh
  g++ -std=c++17 -Iinclude -Iexternal/cccl/libcudacxx/include \
      -fsanitize=address,undefined -g tests/test_<feature>.cpp -o /tmp/t && /tmp/t
  ```

- If you touch slicing / broadcasting / layout folds, also build once with
  `-DTNY_NO_NEGATIVE_INDEX` — the compile-time fold and the runtime gather must
  agree under every flag (a divergence is UB).

CI (`.github/workflows/`) runs the g++ and clang++ test matrix on every PR, plus
a macOS (AppleClang) and Windows (MSVC) CMake+CTest build — you don't need
local access to either platform to be confident a PR is portable.

---

## Coding style

**Device-safety by construction.** No virtuals, exceptions, RTTI, or host-only
calls in device code.

- `_TNY_API` = host **and** device; `_TNY_HOST` = host-only. Anything that
  allocates (heap / CUDA storage, out-of-place ops on dynamic shapes) is
  `_TNY_HOST`; element access, in-place math, views, and static-shape out-of-place
  ops are `_TNY_API`. An `_TNY_API` function must never call a `_TNY_HOST` one on
  the device path.
- **Engines are lambda-free** (index-sequence folds + tiny functors) so they
  instantiate under `nvcc` *without* `--extended-lambda`. Keep them that way.

**mdspan does the heavy lifting.** Before adding machinery, check whether
`cuda::std::mdspan`/`submdspan`/`extents` already does it. Alias it in `alias.h`
rather than reinventing layouts or offset math.

**Single source of truth.** If the same logic must hold in two places (e.g. a
runtime path and its compile-time fold), factor it into one shared helper — a
silent divergence there is how UB creeps in.

### Naming & layout

- `snake_case` for functions, members, variables, and `storage`/enum values.
- `PascalCase` for public type aliases (`Int<>`, `Long<>`, `UInt<>`); `shape<…>`,
  `strides<…>`, `rank<N>` for the vocabulary types.
- `_`-prefixed names are **internal**. Namespaces: `tny` is public; internal code
  lives in `tny::_md` (compute/iteration engines), `tny::_detail` (view/mapping
  builders and host helpers), `tny::_dl` (DLPack), and bare `_name` directly in
  `tny` for vocabulary that public templates must see unqualified. Put new helpers
  in the matching bucket.
- Keep each header **small and single-purpose**. Declare a member in `tensor.h`;
  define structural members inline there and math members in `math.h`. Mind
  declaration-before-use.
- Match the surrounding code's comment density and idiom. Comments explain *why*
  and how the tricky bits work — not what a line obviously does.

### Review

Core-path changes (slicing/gather, broadcasting, layout folds, device `to`,
storage) should get a skeptical diff review that verifies **element-identity**
against the previous behaviour (a brute-force old-vs-new / fold-vs-runtime probe
for the UB-sensitive ones) before merge. Correctness beats brevity beats cleverness.

---

Thanks for keeping teeny teeny.
