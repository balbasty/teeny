# Tensors & ownership

There is one tensor type in teeny. Indexing, slicing, math, views, and dispatch
are all defined on it and work the same regardless of where the data lives. The
only thing that changes between variants is **who owns the memory**.

Learn the whole API on the **view** (owns nothing), then read the short sections
on the owning variants, which add only allocation and lifetime.

## `view` — start here

A view is a non-owning window onto memory you already have (a `std::vector`, a
stack array, a `cudaMalloc` pointer, a DLPack tensor). It holds a pointer and a
mapping, is trivially copyable, and passes into a CUDA `__global__` kernel by
value.

```cpp
double buf[6] = {1,2,3,4,5,6};
auto m = view(buf, shape<2,3>{});   // 2×3 view over buf — no allocation, no copy
m(1,2) = 60;                        // writes go straight to buf
```

Everything works on a view:

```cpp
m(0,-1);                    // element access (negative index counts from the back)
m(all, slice(0,2));         // sub-view (still no copy)
m.permute<1,0>();           // transpose (a view)
m.add_(other);              // in-place math
auto s = sum(m);            // reductions
for (auto row : peel<0>(m)) work(row);   // iterate a subset of axes
```

`view` is the default ownership, so the bare type name is a view:

```cpp
tensor<double, shape<2,3>>   // == a view
view_t<double, shape<2,3>>   // the explicit alias
```

Factories:

| factory | makes |
|---|---|
| `view(ptr, extents)` | C-order view |
| `view<layout_left>(ptr, extents)` | F-order view |
| `view_strided<Sx,Sy,...>(ptr, extents)` | view with compile-time strides (may be negative) |
| `as_tensor(any_mdspan)` | wrap an `mdspan`/`submdspan` result as a view |
| `make_view(ptr, extents)` | same as `view`, deducing the extents type |

Copying a view copies the pointer, not the data; memory lifetime is the caller's.

## Owning variants

Owning tensors have the same API as a view and also hold the storage, freeing it
when they die. Pick one by where the memory should live.

=== "`local` — stack"

    Inline array. Requires a fully static shape (size known at compile time). A
    `local` is exactly `sizeof` its data — no pointer, no heap, host and device.
    Use for kernel-local scratch (a small matrix, an accumulator).

    ```cpp
    auto m = local<double, shape<3,3>>{};   // 9 doubles on the stack
    static_assert(sizeof(m) == 9*sizeof(double));
    m.fill_(0.0);
    ```

=== "`owned` — heap (host)"

    Host memory via `new[]`/`delete[]`. Move-only, host-only, works with dynamic
    shapes.

    ```cpp
    auto h = owned<double, shape<-1,3>>(shape<-1,3>{n});   // n×3 on the heap
    auto g = make_heap<double>(shape<-1,3>{n});            // same, deducing E
    ```

=== "`device` / `host` / `pinned` — CUDA"

    From `#include <teeny/cuda.h>` (needs the CUDA runtime). Move-only owning
    tensors in device / page-locked / pinned memory.

    ```cpp
    #include <teeny/cuda.h>
    auto d = device<float, shape<-1,3,3>>(shape<-1,3,3>{n});  // cudaMalloc'd
    my_kernel<<<grid, block>>>(d.view());                     // pass a view in
    ```

Creation factories build and fill an owning tensor in one step (static shape →
`local`, dynamic → `owned`):

```cpp
auto z = zeros<double>(shape<3,3>{});   // stack, all zeros
auto o = ones<float>(shape<-1,4>{n});   // heap, all ones
auto f = full<int>(shape<8>{}, 7);      // filled with 7
auto a = arange<long>(10);              // [0,1,…,9] (1-D heap)
```

## Getting a view from an owning tensor

Inside a kernel you want a view (trivially copyable). Every owning tensor hands
one out:

```cpp
auto d = device<float, shape<-1,3,3>>(shape<-1,3,3>{n});
auto v = d.view();   // view over d's memory — pass THIS to the kernel
```

## The `own` template parameter

The variants are one class template with a different final argument:

```cpp
template <class T, class Extents, class Layout = layout_right, own O = own::view>
struct tensor;
```

`own` is `{ view, stack, heap, device, host, pinned }`. Rarely named directly —
use the aliases (`view`/`local`/`owned`) and factories (`make_*`,
`zeros`/`ones`/`full`) instead. The parameter lets one class and one set of
algorithms cover every memory space.
