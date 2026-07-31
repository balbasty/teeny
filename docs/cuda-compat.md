# CUDA & CCCL compatibility

teeny is built on **CCCL** (NVIDIA's CUDA C++ Core Libraries — the home of
`cuda::std::mdspan`). Two facts shape everything here:

1. teeny needs `cuda::std::mdspan`, which CCCL gained in the **2.2** release.
2. Each CCCL release supports a bounded range of CUDA toolkits (`nvcc`), and each
   CUDA toolkit *bundles* a particular CCCL — sometimes one **without** mdspan.

For a **host build** (g++/clang++, no CUDA) none of this matters — the bundled
CCCL below just works. The nuance is only for **CUDA device builds / wheels**.

## The short version

**You don't need to configure anything.** teeny vendors a CCCL that works for
host builds and for CUDA 11.1–12.9, and automatically uses the toolkit's own CCCL
when you compile with a newer CUDA (13+). Concretely:

| You compile with… | CCCL used |
|---|---|
| a host compiler, or **`nvcc` ≤ 12.x** | the **vendored** CCCL (v2.8.2) |
| **`nvcc` ≥ 13** | the toolkit's **bundled** CCCL (a 3.x) |

The **Makefile** (and plain compiler flags) selects this automatically from the
compiler version — nothing to set. Under **CMake**, teeny uses an installed CCCL
if `find_package(CCCL)` finds one (as a CUDA-13 toolkit provides) and otherwise
the vendored 2.8.2. To force a specific CCCL for any compiler, override the include
path — see [Overriding the CCCL](#overriding-the-cccl).

## Why a vendored CCCL at all

teeny vendors **CCCL v2.8.2** (a git submodule under `external/cccl`) — the last
release of the 2.x line:

- **2.x has mdspan**, so teeny builds against it.
- **2.8.2 spans the widest `nvcc` range** of any mdspan-capable CCCL: **CUDA 11.1
  through 12.9** — one copy for every CUDA-11/12 toolkit, including the older ones
  whose *bundled* CCCL predates mdspan (see the table below).
- **3.x drops CUDA 11** (its floor is CUDA 12.0), so it can't be the single
  vendored copy without losing CUDA-11 support.

**CUDA 13** dropped CCCL 2.x, but every CUDA-13 toolkit **bundles a 3.x that works
with teeny** — so a CUDA-13 build simply uses that bundled copy, no vendoring
needed. teeny's source is the same for both CCCL lines.

!!! note "Why GitHub shows the submodule as a commit, not `v2.8.2`"
    The submodule pin (`external/cccl @ 207d20f`) **is** exactly CCCL's tagged
    `v2.8.2` release — but a git submodule always records a commit hash in the
    tree, never a ref name, so GitHub (and `git`) have no tag to display. This
    isn't fixable: it's how submodules work in the git object model.

    `.gitmodules` has a `branch` key, but it names a **branch** to track for
    `git submodule update --remote` — CCCL only tags releases (`v2.8.2` has no
    corresponding branch), so setting `branch = v2.8.2` would point
    `--remote` at a branch that doesn't exist rather than clarify anything.
    Pinning by tag, and updating deliberately by re-running
    `git -C external/cccl checkout v2.8.3 && git add external/cccl` when a new
    release lands, is the correct workflow here.

## CCCL ↔ `nvcc` support matrix

Which CUDA toolkits each relevant CCCL release supports, and whether it has the
`cuda::std::mdspan` teeny needs:

| CCCL release | Supported CUDA (`nvcc`) | `cuda::std::mdspan`? |
|---|---|---|
| ≤ 2.1 | 11.1 – 12.x | ✗ (not yet) |
| 2.2 – 2.8.2 | **11.1 – 12.9** | **✓** |
| 3.0 – 3.4.0 (latest) | **12.0 – 13.x** | ✓ (but no CUDA 11) |

No single CCCL covers both CUDA 11 and CUDA 13: 2.8.2 tops out at 12.9 and 3.x
starts at 12.0. That's why teeny uses 2.8.2 by default and the toolkit's 3.x on
CUDA 13 (the two lines overlap on CUDA 12, which can use either).

## CCCL bundled with each CUDA toolkit

Each `nvcc` toolkit ships a CCCL; older toolkits shipped one that predates mdspan.
Approximate mapping:

| CUDA toolkit | Bundled CCCL | Bundled has mdspan? |
|---|---|---|
| 11.8 | ~1.8 | ✗ |
| 12.0 | ~2.0 | ✗ |
| 12.2 | ~2.1 | ✗ |
| 12.3 | ~2.2 | ✓ |
| 12.6 | ~2.5 | ✓ |
| 12.9 | 2.8.2 | ✓ |
| 13.0 | 3.0 | ✓ |
| 13.3 | ~3.3 | ✓ |

On CUDA 11.8 or 12.0–12.2 the bundled CCCL has no mdspan, so teeny uses the
vendored 2.8.2 instead — which is exactly what the [auto-selection](#the-short-version)
does. From CUDA 12.3 up the bundled copy would work; the vendored one is still
used through 12.x for a single, consistent version, and the bundled 3.x is used on
CUDA 13.

## Host compilers

teeny is header-only C++17, so any C++17-conformant host compiler works. CCCL sets
the practical minimums:

| Host compiler | Minimum | teeny CI |
|---|---|---|
| **g++** | 7 (first C++17) | tested on 11–14 |
| **clang++** | 9 | tested on 18 |
| **AppleClang** (macOS) | 9 | tested on `macos-latest` |
| **MSVC** (Windows) | 2019 (v19.20) | tested on `windows-latest`, currently failing — [#296](https://github.com/balbasty/teeny/issues/296) |

For a **CUDA device build**, the host compiler must also be one your `nvcc`
supports — each CUDA release caps the host g++/clang/MSVC version (e.g. CUDA 12.6
allows g++ ≤ 13, CUDA 13.0 allows g++ ≤ 15). See NVIDIA's system requirements for
the exact per-toolkit host-compiler ranges.

Linux, macOS, and Windows are all exercised in CI: Linux via `make` (g++/clang++),
macOS and Windows via the CMake + CTest build (each `tests/test_*.cpp` is a CTest
target). MSVC used to fail to compile teeny at all (a compiler-specific defect in
`operator()`'s overload resolution, [#268](https://github.com/balbasty/teeny/issues/268),
fixed). Fixing it let compilation get far enough to uncover further pre-existing,
independent MSVC-only defects that #268 had been masking: empty-base-optimization
not applying to a mapping with 2+ private empty bases ([#295](https://github.com/balbasty/teeny/issues/295),
fixed — needed `__declspec(empty_bases)`, see the EBO note below) and some
axis-reduction overloads failing to resolve ([#296](https://github.com/balbasty/teeny/issues/296),
open). The Windows CI job stays `continue-on-error` until #296 is fixed too, so
only Linux and macOS are proven-portable today.

**A permanent, documented MSVC limitation from #295's investigation:** teeny's
`sizeof`-exact guarantee (a fully-static view/stack tensor is exactly the size
of its data) holds on every compiler for teeny's own `strides<...>` layout, but
**not** on real MSVC for the `ccontiguous`/`fcontiguous` layouts — CCCL's own
`layout_right`/`layout_left::mapping` stores extents as a member tagged
`_CCCL_NO_UNIQUE_ADDRESS`, and CCCL deliberately disables that attribute on
MSVC (a documented kernel-launch data-corruption history), so the member
always occupies real space there. This is CCCL's own upstream tradeoff, not a
teeny defect, and isn't something `__declspec(empty_bases)` can fix (it only
folds bases that are already empty). `tests/test_tensor.cpp`'s affected
`sizeof` assertions are `#if`'d out on real MSVC for this reason.

**One call-site spelling to avoid on MSVC.** With `using namespace tny`, prefer
the keyword spelling

```cpp
auto g = empty<float>(shape<2,3>{}, storage_c<storage::gpu>{});   // portable
```

over `empty<float, storage::gpu>(shape<2,3>{})`. Because the shape argument
comes from `cuda::std`, argument-dependent lookup also considers
`cuda::std::empty(const T (&)[N])`, and that overload takes exactly two template
parameters — so the two-argument spelling matches its shape. Every other
compiler simply drops the candidate (a `storage` value cannot be an array
bound); MSVC in conformance mode reports an error instead. The keyword spelling
passes only one explicit template argument and is unaffected, and the qualified
`tny::empty<float, storage::gpu>(shape<2,3>{})` — no argument-dependent lookup —
works as well. Nothing else in the factory family is affected: `zeros`, `ones`,
`full`, and `arange` have no `cuda::std` namesake.

## Choosing a toolkit for your target hardware

For CUDA wheels, `nvcc`'s version — *not* CCCL — sets the reachable GPU
architectures and the minimum driver:

- **CUDA 11.x (e.g. `nvcc` 11.8)** reaches down to Kepler/Maxwell and older drivers
  (roughly **r450+** with CUDA 11 minor-version compatibility) — the toolkit for
  maximum backward reach.
- **CUDA 12.x** needs a newer driver (roughly **r525+**) and drops the oldest archs,
  but targets Hopper/Ada.
- **CUDA 13.x** raises the offline-compile floor to **`sm_75` (Turing)** — Maxwell,
  Pascal, and Volta are dropped — needs driver **~r580+**, and targets up to
  Blackwell.
- **Gencode:** embed PTX for the *oldest* arch you support plus SASS for the common
  ones, e.g. `-gencode arch=compute_52,code=sm_52 … -gencode
  arch=compute_80,code=compute_80` (a trailing `code=compute_XX` PTX entry lets the
  driver JIT for newer GPUs you didn't list).

### A wheel matrix for wide coverage

To cover as many GPUs and drivers as possible, build one wheel per CUDA major:

| Wheel | Built with `nvcc` | Reaches | Min driver |
|---|---|---|---|
| **cu11** | 11.8 | Kepler/Maxwell → Ada (via PTX JIT) | ~r450+ |
| **cu12** | a 12.x (e.g. 12.6) | Volta → Hopper/Ada | ~r525+ |
| **cu13** | a 13.x (e.g. 13.0) | Turing → Blackwell (`sm_75+`) | ~r580+ |

Each wheel compiles the same teeny source; only the `nvcc` version and `-gencode`
targets differ. `cu13` is **additive** — it reaches the newest archs but not
pre-Turing GPUs, which is what `cu11` is for.

## Overriding the CCCL

To point teeny at a specific CCCL (e.g. to pin an exact 3.x version, or use one
you keep elsewhere), set the include path — this overrides the auto-selection:

```sh
# Make (and plain compiler flags):
make CCCL_INC=/path/to/cccl/libcudacxx/include run-test

# CMake:
cmake -DTEENY_CCCL_INCLUDE=/path/to/cccl/libcudacxx/include …
#   (or install CCCL and let find_package(CCCL) pick it up)
```

## Device builds, spelled out

Building via the Makefile (`make CXX=nvcc …`) or CMake needs no CCCL flags — the
[auto-selection](#the-short-version) handles it. With a raw `nvcc` command you add
the vendored CCCL include for CUDA 11/12 and omit it for CUDA 13 (which supplies a
compatible CCCL itself). Otherwise the invocations are identical — only `-arch`
differs:

```sh
# CUDA 11  — vendored CCCL, oldest archs
nvcc -std=c++17 -arch=sm_52 -I include -I external/cccl/libcudacxx/include \
     --compile tests/nvcc_smoke.cu

# CUDA 12  — vendored CCCL
nvcc -std=c++17 -arch=sm_70 -I include -I external/cccl/libcudacxx/include \
     --compile tests/nvcc_smoke.cu

# CUDA 13  — bundled CCCL, no CCCL -I needed
nvcc -std=c++17 -arch=sm_75 -I include \
     --compile tests/nvcc_smoke.cu
```

## CI

`.github/workflows/nvcc-compile.yaml` compiles the device smoke translation unit
across the whole supported range — **`nvcc` 11.8** (`sm_52`), **12.6** (`sm_70`),
and **13.0** (`sm_75`) — so both ends of the CUDA range stay green.

### `nvcc` checks device code you never call

Worth knowing when you build a `.cu` file against teeny: for a function marked
host **and** device, `nvcc` generates and checks its device version as soon as the
function is instantiated — **whether or not any kernel calls it**. A host-only call
site in a `.cu` translation unit is enough to surface a device-side error.

Two practical consequences:

- Compiling a `.cu` file is a real device check even before you write a kernel, so
  the smoke translation unit above catches device problems from plain host code.
- The same code in a `.cpp` file, or under `clang`'s CUDA front end (which defers
  these diagnostics until the function is genuinely reachable from a kernel), will
  *not* report them. If a `.cu` build fails where the `.cpp` build passed, this
  asymmetry — not your call site — is usually why.
