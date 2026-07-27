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
  11.1 through 12.9**. One vendored copy covers every CUDA-11/12 toolkit.
- **3.x drops CUDA 11.** CCCL 3.0 raised its floor to **CUDA 12.0**, so it can't
  be the *default* vendored copy without forfeiting the CUDA-11 wheels. It is
  instead selected **per leg** for the CUDA-13 build (below), where 2.8.2 no
  longer reaches.

**CUDA 13** (released Aug 2025, now at 13.3) needs a **CCCL 3.x** — 2.8.2 caps at
CUDA 12.9. teeny is source-clean on 3.x as well (it built on CCCL 3.3.0 before
the 2.8.2 pin, and the rank-0 `stride()` guard 2.8.2 needs is inert there), so a
CUDA-13 build just points the include knob at a 3.x — see
[Building for CUDA 13](#building-for-cuda-13). Verified against **CCCL 3.4.0**
(latest 3.x): the full host suite is 51/51 on g++ and clang++.

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
| 3.0 – 3.4.0 (latest) | **12.0 – 13.x** | ✓ (but no CUDA 11) |

**Takeaways:** the 2.2 – 2.8.2 window is the only one that has mdspan *and* still
supports CUDA 11 — 2.8.2 is its newest point, so that is teeny's default vendored
copy. There is **no single CCCL that covers both CUDA 11 and CUDA 13**: 2.8.2 tops
out at 12.9, and 3.x starts at 12.0. So a CUDA-13 build selects a 3.x for that leg
(the two lines overlap on CUDA 12, so 12.x can use either).

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
| 13.0 | 3.0 | ✓ |
| 13.3 | ~3.3 | ✓ |

The lesson: on CUDA 11.8 or 12.0–12.2, the *bundled* CCCL cannot build teeny at
all — you **must** supply a newer CCCL. (CUDA 12.3+ and all of CUDA 13 bundle an
mdspan-capable CCCL, so there the bundled copy *would* work — but teeny still
vendors its own for a pinned, byte-identical version across toolchains.) Vendoring
2.8.2 makes that automatic and identical across every CUDA-11/12 toolkit; the
CUDA-13 leg swaps in a pinned 3.x the same way (below).

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
- **CUDA 13.x** raises the offline-compile floor to **`sm_75` (Turing)** — Maxwell,
  Pascal, and Volta are dropped — needs driver **~r580+**, and targets up to
  Blackwell. It also requires a **CCCL 3.x** (2.8.2 stops at 12.9).
- **Gencode:** embed PTX for the *oldest* arch you support plus SASS for the
  common ones, e.g. `-gencode arch=compute_52,code=sm_52 ... -gencode
  arch=compute_80,code=compute_80` (a trailing `code=compute_XX` PTX entry lets
  the driver JIT for newer GPUs you didn't list).

### Recommended wheel matrix (downstream: fastfields)

`fastfields` ships wheels that cover as many GPUs and drivers as possible. The
recommended build matrix:

| Wheel | Built with `nvcc` | CCCL | Reaches | Min driver |
|---|---|---|---|---|
| **cu11** | 11.8 | vendored 2.8.2 | Kepler/Maxwell → Ada (via PTX JIT) | ~r450+ |
| **cu12** | a 12.x (e.g. 12.6) | vendored 2.8.2 | Volta → Hopper/Ada | ~r525+ |
| **cu13** | a 13.x (e.g. 13.0) | a 3.x (e.g. 3.4.0) | Turing → Blackwell (`sm_75+`) | ~r580+ |

The `cu11`/`cu12` wheels share the vendored 2.8.2 and differ only in
`nvcc`/`-gencode`. `cu13` is **additive**, not a replacement: it reaches the
newest archs (Blackwell) but *cannot* reach pre-Turing GPUs, which is exactly what
`cu11` is for. Its only source difference is the CCCL line — selected with the
include knob, no `#ifdef`s in teeny.

### Building for CUDA 13

CUDA 13 shipped in Aug 2025 (now at 13.3). Because it needs a CCCL 3.x, point the
include knob at one instead of the vendored 2.8.2 — everything else is unchanged:

```sh
# Fetch a CCCL 3.x once (or use one already installed / bundled with CUDA 13):
git clone --depth 1 --branch v3.4.0 https://github.com/NVIDIA/cccl.git /opt/cccl-3x

# Host suite against CCCL 3.x (proves source compatibility):
make CCCL_INC=/opt/cccl-3x/libcudacxx/include run-test          # 51/51 on g++ and clang++

# Device compile with nvcc 13 (Turing floor):
nvcc -std=c++17 -arch=sm_75 -I include \
     -I /opt/cccl-3x/libcudacxx/include --compile tests/nvcc_smoke.cu
```

CMake is the same one knob: `-DTEENY_CCCL_INCLUDE=/opt/cccl-3x/libcudacxx/include`
(or install CCCL 3.x and let `find_package(CCCL)` find it). teeny is source-clean
on both CCCL lines — the rank-0 `stride()` guard 2.8.2 requires is inert on 3.x —
so no code changes are needed to move a build between them.

## CI

`.github/workflows/nvcc-compile.yaml` compiles the device smoke TU across the whole
supported range, each leg against the CCCL that CUDA version needs:

| Leg | `nvcc` | host g++ | CCCL | `-arch` |
|---|---|---|---|---|
| floor | 11.8 | g++-11 | vendored 2.8.2 | `sm_52` |
| mid | 12.6 | g++-13 | vendored 2.8.2 | `sm_70` |
| ceiling | 13.0 | g++-14 | 3.4.0 (checked out for the leg) | `sm_75` |

The 2.x legs use the pinned submodule; the CUDA-13 leg checks out CCCL `v3.4.0`
over it first. That guards both ends — the CUDA-11 floor and the CUDA-13
ceiling — from regressing.
