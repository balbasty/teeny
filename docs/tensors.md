# Tensors & ownership

There is **one tensor type** in teeny. Everything in this guide — indexing,
slicing, math, views, dispatch — is defined on it and works the same regardless
of *where the data lives*. The only thing that changes between the variants is
**who owns the memory**.

So start with the variant that owns nothing — the **view** — learn the whole API
on it, and then read the short sections on the owning variants, which add only
allocation and lifetime.

## `view` — the one to learn first

A **view** is a non-owning window onto memory you already have (a `std::vector`,
a stack array, a `cudaMalloc` pointer, a DLPack tensor). It stores just a pointer
and a mapping, is **trivially copyable**, and can be passed straight into a
CUDA `__global__` kernel by value.

```cpp
double buf[6] = {1,2,3,4,5,6};
auto m = view(buf, shape<2,3>{});   // a 2×3 view over buf — no allocation, no copy
m(1,2) = 60;                        // writes go straight to buf
```

Everything works on a view:

```cpp
m(0,-1);                    // element access (negative index = from the back)
m(all, slice(0,2));         // a sub-view (still no copy)
m.permute<1,0>();           // transpose (a view)
m.add_(other);              // in-place math
auto s = sum(m);            // reductions
for (auto row : peel<0>(m)) work(row);   // iterate a subset of axes
```

`view` is the default ownership, so the bare type name means a view:

```cpp
tensor<double, shape<2,3>>          // == a view
view_t<double, shape<2,3>>          // the explicit alias
```

Factories:

| factory | makes |
|---|---|
| `view(ptr, extents)` | a C-order view |
| `view<layout_left>(ptr, extents)` | an F-order view |
| `view_strided<Sx,Sy,...>(ptr, extents)` | a view with **compile-time strides** (may be negative) |
| `as_tensor(any_mdspan)` | wrap a `submdspan`/`mdspan` result as a view |
| `make_view(ptr, extents)` | same as `view`, deducing the extents type |

!!! note "A view never allocates or frees"
    Copying a view copies the pointer, not the data. The memory's lifetime is
    the caller's problem — exactly what you want inside a kernel.

## Owning variants — what each adds

The owning tensors have the **same API as a view**; they just *also* hold the
storage and free it when they die. Pick one by where the memory should live.

=== "`local` — stack"

    An inline array. Requires a **fully static shape** (so its size is known at
    compile time). A `local` is *exactly* `sizeof` its data — no pointer, no
    heap, host **and** device.

    ```cpp
    auto m = local<double, shape<3,3>>();   // 9 doubles on the stack
    static_assert(sizeof(m) == 9*sizeof(double));
    m.fill_(0.0);
    ```

    Use it for kernel-local scratch (a small matrix, an accumulator).

=== "`owned` — heap (host)"

    Host memory via `new[]`/`delete[]`. **Move-only** (no accidental deep
    copies), host-only. Works with dynamic shapes.

    ```cpp
    auto h = owned<double, shape<-1,3>>(shape<-1,3>{n});   // n×3 on the heap
    auto g = make_heap<double>(shape<-1,3>{n});            // same, deducing E
    ```

=== "`device` / `host` / `pinned` — CUDA"

    From the opt-in header `#include <teeny/cuda.h>` (needs the CUDA runtime).
    Move-only owning tensors in device / page-locked / pinned memory.

    ```cpp
    #include <teeny/cuda.h>
    auto d = device<float, shape<-1,3,3>>(shape<-1,3,3>{n});  // cudaMalloc'd
    my_kernel<<<grid, block>>>(d.view());                     // pass a view in
    ```

The **creation factories** build an owning tensor and fill it in one step
(static shape → `local`, dynamic shape → `owned`):

```cpp
auto z = zeros<double>(shape<3,3>{});      // stack, all zeros
auto o = ones<float>(shape<-1,4>{n});      // heap, all ones
auto f = full<int>(shape<8>{}, 7);         // filled with 7
auto a = arange<long>(10);                 // [0,1,…,9]
```

## Getting a view from an owning tensor

Inside a kernel you almost always want a **view** (trivially copyable). Every
owning tensor hands one out:

```cpp
auto d = device<float, shape<-1,3,3>>(shape<-1,3,3>{n});
auto v = d.view();       // a view over d's memory — pass THIS to the kernel
```

## Side note: the `own` template parameter

The variants above are all the same class template with a different final
argument:

```cpp
template <class T, class Extents, class Layout = layout_right, own O = own::view>
struct tensor;
```

`own` is `{ view, stack, heap, device, host, pinned }`. **You rarely name it
directly** — use the aliases (`view`/`local`/`owned`/…) and factories
(`make_*`, `zeros`/`ones`/`full`) instead; they exist precisely so you don't
have to spell out the ownership. The parameter is there so that one class, one
set of algorithms, covers every memory space.
