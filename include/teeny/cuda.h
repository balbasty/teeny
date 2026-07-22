#ifndef TNY_MD_CUDA
#define TNY_MD_CUDA
// CUDA memory-space storage for tensor (gpu / pinned / mapped owning
// modes). Safe to include ALWAYS — the body activates ONLY when the CUDA runtime
// is reachable: compiling with nvcc, or <cuda_runtime.h> is on the include path
// (auto-detected via __has_include). Otherwise this header is empty, so teeny.h
// pulls it in unconditionally and users never have to think about it. Define
// TNY_NO_CUDA to force it off even when the runtime is available.
#if !defined(TNY_NO_CUDA) && (defined(__CUDACC__) || (defined(__has_include) && __has_include(<cuda_runtime.h>)))
#  define TNY_HAS_CUDA 1
#endif

#ifdef TNY_HAS_CUDA
#include <cuda_runtime.h>
#include <cuda/std/cstddef>
#include <teeny/defines.h>
#include <teeny/storage.h>
#include <teeny/tensor.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/* ------------------------------------------------------------------ *
 *     CUDA allocator policies                                        *
 * ------------------------------------------------------------------ */

/** @brief Device (GPU) memory (`cudaMalloc`). Not host-dereferenceable. */
struct cuda_gpu_alloc {
    template <class T> _TNY_HOST static T * allocate(cs::size_t n) {
        void * p = nullptr; cudaMalloc(&p, n * sizeof(T)); _TNY_CHECK(p, "cudaMalloc failed"); return static_cast<T *>(p);
    }
    template <class T> _TNY_HOST static void deallocate(T * p) { cudaFree(p); }
};

/** @brief Page-locked ("pinned") host memory (`cudaMallocHost`). */
struct cuda_pinned_alloc {
    template <class T> _TNY_HOST static T * allocate(cs::size_t n) {
        void * p = nullptr; cudaMallocHost(&p, n * sizeof(T)); _TNY_CHECK(p, "cudaMallocHost failed"); return static_cast<T *>(p);
    }
    template <class T> _TNY_HOST static void deallocate(T * p) { cudaFreeHost(p); }
};

/** @brief Page-locked + device-mapped (zero-copy) host memory (`cudaHostAlloc`). */
struct cuda_mapped_alloc {
    template <class T> _TNY_HOST static T * allocate(cs::size_t n) {
        void * p = nullptr; cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped); _TNY_CHECK(p, "cudaHostAlloc failed"); return static_cast<T *>(p);
    }
    template <class T> _TNY_HOST static void deallocate(T * p) { cudaFreeHost(p); }
};

/* ------------------------------------------------------------------ *
 *     Storage specializations for the CUDA `own` modes               *
 * ------------------------------------------------------------------ */

template <class T, cs::size_t N>
struct storage<T, own::gpu, N> : owning_storage<T, cuda_gpu_alloc> {
    using owning_storage<T, cuda_gpu_alloc>::owning_storage;
};
template <class T, cs::size_t N>
struct storage<T, own::pinned, N> : owning_storage<T, cuda_pinned_alloc> {
    using owning_storage<T, cuda_pinned_alloc>::owning_storage;
};
template <class T, cs::size_t N>
struct storage<T, own::mapped, N> : owning_storage<T, cuda_mapped_alloc> {
    using owning_storage<T, cuda_mapped_alloc>::owning_storage;
};

/* ------------------------------------------------------------------ *
 *     Factories                                                      *
 * ------------------------------------------------------------------ */

/** @brief Owning tensor in device (GPU) memory (move-only). `gpu<T,E>(extents)`. */
template <class T, class Shape, class Layout = cs::layout_right>
using gpu = tensor<T, Shape, Layout, own::gpu>;
/** @brief Owning tensor in page-locked ("pinned") host memory (move-only).
 *         `pinned<T,E>(extents)` — pytorch's `pin_memory`. */
template <class T, class Shape, class Layout = cs::layout_right>
using pinned = tensor<T, Shape, Layout, own::pinned>;
/** @brief Owning tensor in mapped (zero-copy) host memory (move-only). `mapped<T,E>(extents)`. */
template <class T, class Shape, class Layout = cs::layout_right>
using mapped = tensor<T, Shape, Layout, own::mapped>;

/* --- functional factories (deduce the Shape type from the argument;
 *     `T` defaults to `float`, like the host factories) --- */
template <class T = float, class Layout = cs::layout_right, class Shape>
_TNY_HOST auto make_gpu(Shape e)    { return tensor<T, Shape, Layout, own::gpu>(e); }
template <class T = float, class Layout = cs::layout_right, class Shape>
_TNY_HOST auto make_pinned(Shape e) { return tensor<T, Shape, Layout, own::pinned>(e); }
template <class T = float, class Layout = cs::layout_right, class Shape>
_TNY_HOST auto make_mapped(Shape e) { return tensor<T, Shape, Layout, own::mapped>(e); }

/* ------------------------------------------------------------------ *
 *     Memory-backend `to` — the CUDA half of pytorch's `.to`         *
 *                                                                    *
 *  The member `x.to<T2>()` (tensor.h) does the dtype half on the      *
 *  host; these free functions add the MEMORY-SPACE move (a member     *
 *  can't reach cuda.h, which is included after tensor.h).            *
 * ------------------------------------------------------------------ */
namespace _detail {
// A dense, row-major HOST copy of `x` with element type `E2`. A host-accessible
// source (view/stack/heap/pinned/mapped) is read + converted directly; an owning
// `gpu` source is downloaded raw into a host buffer that mirrors its LAYOUT (so
// F-order / strided gpu tensors are not silently transposed), then densified on
// the host. `Force=true` on the inner `.to` guarantees an OWNING dense buffer,
// never a borrow of the local (which would dangle at return).
template <class E2, class T, class Shape, class Layout, own O>
_TNY_HOST auto dense_host(const tensor<T, Shape, Layout, O> & x) {
    using Ts = cs::remove_cv_t<T>;
    if constexpr (O == own::gpu) {
        auto tmp = make_heap<Ts, Layout>(x.extents());   // host heap mirroring x's layout (C/F/strided)
        cudaMemcpy(tmp.data(), x.data(),
                   static_cast<cs::size_t>(tmp.mapping().required_span_size()) * sizeof(Ts),
                   cudaMemcpyDeviceToHost);              // copy the full device span, layout preserved
        return tmp.template to<E2, true>();              // densify to row-major on the host
    } else {
        return x.template to<E2, true>();                // host-accessible: read + convert into a dense OWNING copy
    }
}
} // namespace _detail

/**
 * @brief Move `x` to memory space `Space` (`own::gpu`/`pinned`/`mapped`/`heap`),
 *        optionally converting the element type to `ET` — the memory-backend half
 *        of pytorch's `.to`. `ET` defaults to the source type.
 *
 *            auto d = to<own::gpu>(h);           // upload host -> device
 *            auto e = to<own::gpu, half>(h);     // convert to half AND upload
 *            auto c = to<own::heap>(d);          // download device -> host
 *
 * **No-op when already there:** if the source is already an owning tensor of
 * element type `ET` in space `Space`, and `Force` is false, this returns a *view*
 * of it (no copy, borrows the source) — like any teeny view op. Pass `Force =
 * true` to always materialise a fresh owning copy (a force-clone into a space):
 *
 *            auto v = to<own::gpu>(g);           // g is already gpu<T> -> a view, no copy
 *            auto k = to<own::gpu, void, true>(g);  // forced: a fresh gpu copy
 *
 * An **owning** `gpu` source is downloaded via `cudaMemcpy` (any dense layout —
 * C, F, or strided — is preserved and densified on the host). `Space == stack`
 * needs a static shape.
 *
 * @warning **A view carries no memory space.** Every source that is not `own::gpu`
 * — including a *view obtained by slicing/permuting a `gpu` tensor* (those are
 * `own::view`, not `own::gpu`) — is treated as **host-accessible** and read on the
 * host. Passing a device-memory view therefore dereferences a device pointer on
 * the host (UB / segfault on real CUDA). Materialise or `clone()` such a view on
 * the device first, or pass the owning `gpu` tensor. Distinguishing host vs device
 * views in the type system is tracked by #15.
 *
 * @note The no-copy branch returns a **borrow** of `x`, so it must outlive the
 * result — same lifetime rule as `view()`/`permute()`/slicing. Calling it on a
 * temporary lvalue would dangle; the rvalue overload below forces a copy instead.
 */
template <own Space, class ET = void, bool Force = false, class T, class Shape, class Layout, own O>
_TNY_HOST auto to(const tensor<T, Shape, Layout, O> & x) {
    static_assert(Space != own::view, "to<Space>: Space must be an owning memory space, not own::view");
    using Tb = cs::remove_cv_t<T>;
    using E2 = cs::conditional_t<cs::is_same<ET, void>::value, Tb, ET>;
    if constexpr (!Force && cs::is_same<E2, Tb>::value && O == Space) {
        return tensor<const Tb, Shape, Layout, own::view>(x.data(), x.mapping());  // already there -> borrow
    } else if constexpr (Space == own::stack) {
        auto host = _detail::dense_host<E2>(x);
        tensor<E2, Shape, cs::layout_right, own::stack> dst{};   // static shape
        cudaMemcpy(dst.data(), host.data(), static_cast<cs::size_t>(dst.numel()) * sizeof(E2), cudaMemcpyHostToHost);
        return dst;
    } else {
        auto host = _detail::dense_host<E2>(x);
        tensor<E2, Shape, cs::layout_right, Space> dst(x.extents());
        const cudaMemcpyKind kind = (Space == own::gpu) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
        cudaMemcpy(dst.data(), host.data(), static_cast<cs::size_t>(dst.numel()) * sizeof(E2), kind);
        return dst;
    }
}

/** @brief Rvalue overload of `to<Space>`: a temporary source cannot be borrowed
 *  (the no-copy branch would dangle — and for a `gpu` temporary would point at
 *  freed device memory), so this always **forces a fresh owning copy**. */
template <own Space, class ET = void, bool Force = false, class T, class Shape, class Layout, own O>
_TNY_HOST auto to(tensor<T, Shape, Layout, O> && x) {
    return to<Space, ET, /*Force=*/true>(x);   // x is a named lvalue here -> the const& overload, copy branch
}

_TNY_NAMESPACE_END(tny)

#endif // TNY_HAS_CUDA

#endif // TNY_MD_CUDA
