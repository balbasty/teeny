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

_TNY_NAMESPACE_END(tny)

#endif // TNY_HAS_CUDA

#endif // TNY_MD_CUDA
