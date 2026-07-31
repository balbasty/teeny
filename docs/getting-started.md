# Getting started

## Requirements

- **C++17** (the floor set by CCCL, which refuses to build lower).
- A host compiler — tested on **g++ 13** and **clang++ 18** — and/or **`nvcc`**.
- **CCCL** (libcudacxx), vendored under `external/cccl`.

teeny is header-only: there is nothing to build or link. Add two include paths:

```sh
-I path/to/teeny/include
-I path/to/teeny/external/cccl/libcudacxx/include
```

### With CMake

teeny ships a `CMakeLists.txt` that exposes it as an `INTERFACE` (header-only)
target. Linking it inherits the include directories, the C++17 requirement, and
the CCCL dependency, so there is nothing to compile.

```cmake
# In-tree (vendored, or as a submodule):
add_subdirectory(teeny)
target_link_libraries(app PRIVATE teeny::teeny)

# Or, after `find_package(teeny CONFIG)` on an installed copy:
target_link_libraries(app PRIVATE teeny::teeny)
```

CCCL (libcudacxx) is resolved in this order: an explicit include dir you point at
with `-DTEENY_CCCL_INCLUDE=/path/to/cccl/include` (the knob for a per-CUDA choice),
then an installed CCCL via `find_package(CCCL)`, then the vendored submodule under
`external/cccl`. See [CUDA & CCCL compatibility](cuda-compat.md).

## Hello, tensor

```cpp
#include <teeny/teeny.h>
#include <cstdio>
using namespace tny;

int main() {
    double buf[6] = {1, 2, 3, 4, 5, 6};
    auto m = wrap(buf, shape<2, 3>{});     // a 2×3 view over `buf`
    m(1, 2) = 60;                          // write through the view
    std::printf("%g  sum=%g\n", m(1, -1), m.sum());  // -1 = last column
    return 0;
}
```

```sh
g++ -std=c++17 -I include -I external/cccl/libcudacxx/include hello.cpp -o hello
```

## Building the tests and examples

```sh
make run-test        # build + run every tests/test_*.cpp, printing PASS/FAIL
make run-examples    # build + run the standalone example algorithms
make CXX=clang++ run-test    # pick a compiler
```

The suite also has one target per build variant, each of which cleans first and
rebuilds everything under its own flags: `make negindex`
(`-DTNY_NO_NEGATIVE_INDEX`), `make hardened` (`-DTNY_HARDENED`), `make release`
(`-O2 -DNDEBUG`), `make cxx23` (`-std=c++23`).

## One include, one namespace

```cpp
#include <teeny/teeny.h>  // the whole library
using namespace tny;      // the public namespace
```

`teeny/teeny.h` pulls in everything. The CUDA compiler and runtime are detected
automatically, in which case tensors can allocate and work with CUDA memory (device
or pinned host). When CUDA is not available, these variants are not included. You do
not need your own `__CUDACC__` guards. Define `TNY_NO_CUDA` to force the CUDA
support off.

!!! tip "Compile flags worth knowing"
    | flag | effect |
    |---|---|
    | `-DTNY_STD_PROMOTION` | use standard C++ float promotion instead of teeny's lower-width-wins rule (see [Math](math.md)) |
    | `-DTNY_NO_NEGATIVE_INDEX` | drop python-style negative-index wrapping from `operator()` for the tightest codegen (kernels that guarantee non-negative indices) |
    | `-DTNY_PORTABLE_HALF` | force the portable software `half`/`bfloat16` even under `nvcc` (see [Half precision](half.md)) |
    | `-DNDEBUG` | strip the debug shape/precondition checks (they are host-only and already compiled out on device) |
