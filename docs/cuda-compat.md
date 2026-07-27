# CUDA & CCCL compatibility

teeny is built on **CCCL** (NVIDIA's CUDA C++ Core Libraries — the home of
`cuda::std::mdspan`). Everything else here follows from two facts:

1. teeny needs `cuda::std::mdspan`, which CCCL only gained in the **2.x** line.
2. Each CCCL release supports a bounded range of CUDA toolkits (`nvcc`), and each
   CUDA toolkit *bundles* a particular CCCL — often one **without** mdspan.

So teeny **vendors its own CCCL** (a git submodule under `external/cccl`) rather
than relying on whatever the toolkit ships. This page explains the constraints
and the strategy, so you can pick a CCCL/`nvcc` pair for your target hardware.

## What teeny vendors

The submodule is pinned to **CCCL v2.8.2** — the **last** release of the 2.x
line. That choice is deliberate:

- **2.x has mdspan** (added in the 2.2/2.3 timeframe), so teeny builds against it.
- **2.8.2 spans the widest `nvcc` range** of any mdspan-capable CCCL: **CUDA
  11.1 through 12.9**. One vendored copy covers every toolkit teeny targets today.
- **3.x drops CUDA 11.** CCCL 3.0 raised its floor to **CUDA 12.0**, so adopting
  a 3.x would forfeit the CUDA-11 wheels (see the strategy section). We stay on
  2.8.2 until CUDA 13 / CCCL 3.x actually needs building.

teeny is source-compatible with 2.8.2 as-is (a single rank-0 `stride()` guard in
`tensor.h` was the only adjustment 2.8.2's stricter, spec-conformant mdspan
required). The full test suite passes on both g++ and clang against it.

!!! note "You are not stuck on the vendored copy"
    Every build entry point can point at a *different* CCCL — e.g. a newer 3.x
    when you build a CUDA-13 wheel:

    - **Make:** `make CCCL_INC=/path/to/cccl/libcudacxx/include run-test`
    - **CMake:** `-DTEENY_CCCL_INCLUDE=/path/to/cccl/libcudacxx/include`
      (or install CCCL and let `find_package(CCCL)` pick it up)
    - **Plain flags:** just swap the `-I .../libcudacxx/include` path.

## CCCL ↔ `nvcc` support matrix

Which CUDA toolkits each relevant CCCL release supports, and whether it has the
`cuda::std::mdspan` teeny needs:

| CCCL release | Supported CUDA (`nvcc`) | `cuda::std::mdspan`? |
|---|---|---|
| ≤ 2.1 | 11.1 – 12.x | ✗ (not yet) |
| 2.2 – 2.8.2 | **11.1 – 12.9** | **✓** |
| 3.0+ | **12.0** – 13.x | ✓ (but no CUDA 11) |

**Takeaway:** the 2.2 – 2.8.2 window is the only one that has mdspan *and* still
supports CUDA 11. 2.8.2 is its newest point, so that is what teeny vendors.

## CCCL bundled with each CUDA toolkit

Each `nvcc` toolkit ships a CCCL. Historically that bundled copy lagged, which is
exactly why teeny does not depend on it. Approximate mapping (the toolkit's
*bundled* CCCL — not what teeny uses):

| CUDA toolkit | Bundled CCCL | Bundled has mdspan? |
|---|---|---|
| 11.8 | ~1.8 (pre-unification libcu++) | ✗ |
| 12.0 | ~2.0 | ✗ |
| 12.2 | ~2.1 | ✗ |
| 12.3 | ~2.2 | ✓ |
| 12.4 | ~2.3 | ✓ |
| 12.5 | ~2.4 | ✓ |
| 12.6 | ~2.5 | ✓ |
| 12.8 | ~2.8 | ✓ |
| 12.9 | 2.8.2 | ✓ |

The lesson: on CUDA 11.8 or 12.0–12.2, the *bundled* CCCL cannot build teeny at
all — you **must** supply a newer CCCL. Vendoring 2.8.2 makes that automatic and
identical across every toolkit, so the host and device builds see byte-identical
headers regardless of which `nvcc` is installed.

## Choosing a toolkit for your target hardware

For a **header-only host build**, none of this matters: any C++17 g++/clang++ plus
the vendored CCCL works. The nuance is only for **CUDA device builds / wheels**,
where `nvcc`'s version — *not* CCCL — sets the reachable GPU architectures and the
minimum driver.

Rules of thumb:

- **The compute-capability floor is `nvcc`'s, not CCCL's.** CCCL 2.8.2 is happy
  from CUDA 11.1 up; how *old* a GPU or driver you reach is decided by which
  `nvcc` you compile with and the `-gencode` targets you pass.
- **CUDA 11.x (e.g. `nvcc` 11.8)** reaches down to Kepler/Maxwell and older
  drivers (roughly **r450+** with CUDA 11 enhanced/minor-version compatibility).
  This is the toolkit for maximum backward reach.
- **CUDA 12.x** requires a newer driver (roughly **r525+**) and drops the oldest
  archs, but targets Hopper/Ada and current toolchains.
- **Gencode:** embed PTX for the *oldest* arch you support plus SASS for the
  common ones, e.g. `-gencode arch=compute_52,code=sm_52 ... -gencode
  arch=compute_80,code=compute_80` (a trailing `code=compute_XX` PTX entry lets
  the driver JIT for newer GPUs you didn't list).

### Recommended wheel matrix (downstream: fastfields)

`fastfields` ships wheels that cover as many GPUs and drivers as possible. The
recommended build matrix, all on **vendored CCCL 2.8.2**:

| Wheel | Built with `nvcc` | Reaches | Min driver |
|---|---|---|---|
| **cu11** | 11.8 | Kepler/Maxwell → Ada (via PTX JIT) | ~r450+ |
| **cu12** | a 12.x (e.g. 12.6) | Volta → Hopper/Ada | ~r525+ |

Two wheels, one CCCL. The cu11 wheel exists purely to reach old
archs/drivers that CUDA 12 dropped; both compile the *same* teeny headers against
the *same* vendored CCCL, so there is no source divergence to maintain — only the
`nvcc`/`-gencode` differ.

When CUDA 13 (and a required CCCL 3.x) ships, add a third leg that points its
build at a vendored **3.x** via `CCCL_INC` / `TEENY_CCCL_INCLUDE`, selected by
CUDA major version. Until then, 2.8.2 alone is correct and simplest.

## CI

`.github/workflows/nvcc-compile.yaml` proves both ends of the supported range: it
compiles the device smoke TU with **`nvcc` 11.8** (the floor) and **`nvcc` 12.6**
(a recent 12.x), each with its matching max host g++, all against the vendored
CCCL 2.8.2. That guards the CUDA-11 floor from regressing.
