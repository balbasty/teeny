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
auto m = wrap(buf, shape<2,3>{});  // 2×3 view over buf — no allocation, no copy
m(1,2) = 60;                       // writes go straight to buf
```

Everything works on a view:

```cpp
m(0,-1);                             // element access (negative index counts from the back)
m(all, slice(0,2));                  // sub-view (still no copy)
m.permute(axis<1,0>{});              // transpose (a view)
m.add_(other);                       // in-place math
auto s = m.sum();                    // reductions (also free functions: sum(m))
for (auto row : peel(m, axis<0>{})) work(row);  // iterate a subset of axes
```

`view` is the default ownership, so the bare type name is a view:

```cpp
tensor<double, shape<2,3>>  // == a view
view<double, shape<2,3>>  // the explicit alias
```

Factories:

| factory | makes |
|---|---|
| `wrap(ptr, shape)` | C-order view |
| `wrap<fcontiguous>(ptr, shape)` / `wrap(ptr, shape, fcontiguous{})` | F-order view — template or value-tag spelling (the value form deduces the layout, no `.template` on a dependent receiver) |
| `wrap(ptr, shape, strides<Sx,Sy,...>{})` | view with compile-time strides (may be negative) |
| `wrap(ptr, shape, {s0,s1,...})` | view with runtime strides (`dynamic_strides`) |
| `wrap<S...>(ptr, shape, {dyn...})` | mixed static/runtime strides (`dynamic_stride` slots) |
| `as_tensor(any_mdspan)` / `wrap(any_mdspan)` | wrap a raw `mdspan`/`submdspan` result as a view — both spellings, same result; `as_tensor` is what teeny's own view-producing ops (`permute`/`flip`/…) call internally |
| `make_view(ptr, shape)` | an alias of `wrap` in the `make_*` family (same result) |

`wrap` already deduces the shape type from its argument, so `make_view` is a plain
synonym that exists for symmetry with `make_local` / `make_heap` / `make_gpu`. Use
whichever reads better.

Copying a view copies the pointer, not the data; memory lifetime is the caller's.

!!! warning "`a = b` rebinds a view; `a(ellipsis) = b` copies elements"
    On a **named** view, `a = b` is ordinary C++ assignment: `a` *rebinds* to point
    at `b`'s memory — nothing is copied. To copy elements (numpy's `a[:] = b`), put a
    slice on the left: `operator()` returns a *temporary* view, and assigning into a
    temporary copies (with broadcasting).

    ```cpp
    a = b;             // rebind: `a` now views `b`'s memory (nothing copied)
    a(ellipsis) = b;   // copy: write b's elements into a's buffer (== a.copy_(b))
    a(0, all) = b;     // copy into a sub-region; a(0, all) = 5.0 fills it
    ```

    The copy accepts **any** right-hand side, including another view of the same type
    (`y(all) = x(all)`), and broadcasts `b` numpy-style (right-aligned; `b`'s rank may
    be ≤ `a`'s). The rebind is an `&`-qualified defaulted assignment, so it binds only
    to named lvalues — every temporary-view assignment routes to the deep copy, and the
    view stays **trivially copyable** (passable into a CUDA kernel by value).

    Two caveats: an *owning* `local`/`owned` copies its elements on `a = b` as usual
    (it holds storage, not a pointer); and if a destination slice **overlaps its own
    source**, `clone()` the source first — the copy has no overlap check. Since `a = b`
    needs `a` and `b` to be the *same* type, use `a(ellipsis) = b` (which broadcasts)
    to copy across differing shapes.

## Owning variants

Owning tensors have the same API as a view and also hold the storage, freeing it
when they die. Pick one by where the memory should live.

=== "`local` — stack"

    Inline array. Requires a fully static shape (size known at compile time). A
    `local` is exactly `sizeof` its data — no pointer, no heap, host and device.
    Use for kernel-local scratch (a small matrix, an accumulator). A static shape
    also lets every index computation fold to compile-time constants — see
    [Performance](performance.md).

    ```cpp
    auto m = local<double, shape<3,3>>{};       // 9 doubles on the stack
    static_assert(sizeof(m) == 9*sizeof(double));
    m.fill_(0.0);
    auto m2 = make_local<double>(shape<3,3>{});  // same, deducing E from the shape
    ```

=== "`owned` — heap (host)"

    Host memory via `new[]`/`delete[]`. Move-only, host-only, works with dynamic
    shapes.

    ```cpp
    auto h = owned<double, shape<-1,3>>(shape<-1,3>{n});  // n×3 on the heap
    auto g = make_heap<double>(shape<-1,3>{n});           // same, deducing E
    ```

=== "`gpu` / `pinned` / `mapped` — CUDA"

    Move-only owning tensors in device (`gpu`), page-locked host (`pinned`,
    pytorch's `pin_memory`), or device-mapped zero-copy host (`mapped`) memory.
    These live in `teeny/cuda.h`, which `teeny/teeny.h` already includes when a
    CUDA toolkit is present — you do not include it yourself.

    ```cpp
    auto d = gpu<float, shape<-1,3,3>>(shape<-1,3,3>{n});  // cudaMalloc'd
    auto e = make_gpu<float>(shape<-1,3,3>{n});            // same, deducing E
    auto p = make_pinned<float>(shape<-1,3,3>{n});         // page-locked host memory
    my_kernel<<<grid, block>>>(d.view());                  // pass a view in
    ```

Creation factories build and fill an owning tensor in one step. Ownership is
deduced from the shape — a static shape gives a `local`, a dynamic one gives an
`owned`:

```cpp
auto z = zeros<double>(shape<3,3>{});  // stack, all zeros
auto o = ones<float>(shape<-1,4>{n});  // heap, all ones
auto f = full<int>(shape<8>{}, 7);     // filled with 7
auto a = arange<long>(10);             // [0,1,…,9] (1-D heap)
```

You can override the deduced ownership by naming a backend, and override the
default `ccontiguous` layout with a layout template argument:

```cpp
auto h = make_heap<double>(shape<3,3>{});               // force HEAP for a static shape
auto e = empty<double, storage::heap>(shape<3,3>{});    // same, via empty
auto f = make_local<double, fcontiguous>(shape<3,3>{}); // stack, F-order (column-major)
auto g = zeros<float, storage_deduce, fcontiguous>(shape<-1,4>{n});  // F-order zeros
```

### Composable keywords {: #factories }

Every factory above (`empty`/`zeros`/`ones`/`full`/`arange`, and `wrap`'s
memory-space tag) also takes its `dtype<T>{}`/`storage_c<O>{}`/layout-tag
arguments as **value-tag keywords** instead of explicit template arguments —
and, unlike the template forms above, these compose in **any subset, any
order**:

```cpp
auto e2 = empty(shape<3,3>{}, dtype<double>{}, storage_c<storage::heap>{});
auto g2 = zeros(shape<-1,4>{n}, fcontiguous{}, dtype<float>{});   // order doesn't matter
auto h2 = make_heap(shape<3,3>{}, dtype<double>{});               // make_* forwards its keywords too
```

This is the same **keyword-argument design rule** used by [reductions](math.md#keyword-arguments)
(`sum(a, dtype<double>{}, axis<0>{}, keepdims, into(buf))`): keywords are
trailing, order-free among themselves, and one of each kind per call —
`tny::_kw` (`kwargs.h`) is the shared primitive behind both.

## Getting a view from an owning tensor

Inside a kernel you want a view (trivially copyable). Every owning tensor hands
one out:

```cpp
auto d = make_gpu<float>(shape<-1,3,3>{n});  // deduces E — no repeated shape
auto v = d.view();  // view over d's memory — pass THIS to the kernel
```

## The `storage` template parameter

The variants are one class template with a different final argument:

```cpp
template <class T, class Shape, class Layout = ccontiguous, storage O = storage::view>
struct tensor;
```

`storage` is `{ view, stack, heap, gpu, pinned, mapped, gpu_view, pinned_view,
mapped_view }`. The `*_view` kinds are non-owning views that remember their memory
space: `gpu_view` is a view of *device* memory (what slicing or permuting a `gpu`
tensor yields), and `pinned_view` / `mapped_view` keep the page-locked space of a
`pinned` / `mapped` source (so DLPack can label them `kDLCUDAHost`). The `storage`
parameter is rarely named directly — use the aliases (`view`/`local`/`owned`) and
factories (`make_*`, `zeros`/`ones`/`full`) instead. It lets one class and one set
of algorithms cover every memory space.
