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

## Hello, tensor

```cpp
#include <teeny/teeny.h>
#include <cstdio>
using namespace tny;

int main() {
    double buf[6] = {1, 2, 3, 4, 5, 6};
    auto m = view(buf, shape<2, 3>{});     // a 2×3 view over `buf`
    m(1, 2) = 60;                          // write through the view
    std::printf("%g  sum=%g\n", m(1, -1),  // -1 = last column
                                (double)sum(m));
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

## One include, one namespace

```cpp
#include <teeny/teeny.h>  // the whole library
using namespace tny;      // the public namespace
```

`teeny/teeny.h` pulls in everything, including `teeny/cuda.h`. That header adds
CUDA **memory** (device / pinned owning tensors) and self-detects the CUDA
runtime (`__has_include` / `__CUDACC__`): it is a no-op on a host compiler with
no CUDA toolkit, so you never have to include or guard it yourself. Define
`TNY_NO_CUDA` to force it off.

!!! tip "Compile flags worth knowing"
    - `-DTNY_STD_PROMOTION` — use standard C++ float promotion instead of teeny's
      lower-width-wins rule (see [Math](math.md)).
    - `-DTNY_NO_NEGATIVE_INDEX` — drop python-style negative-index wrapping from
      `operator()` for the tightest codegen (kernels that guarantee non-negative
      indices).
    - `-DTNY_PORTABLE_HALF` — force the portable software `half`/`bfloat16` even
      under `nvcc` (see [Half precision](half.md)).
    - `-DNDEBUG` — strip the debug shape/precondition checks (they are host-only
      and already compiled out on device).
