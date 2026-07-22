#ifndef TNY_MD_CUDA
#define TNY_MD_CUDA
// CUDA memory-space storage for tensor (device / page-locked / pinned owning
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

/** @brief Device memory (`cudaMalloc`). Not host-dereferenceable. */
struct cuda_device_alloc {
    template <class T> _TNY_HOST static T * allocate(cs::size_t n) {
        void * p = nullptr; cudaMalloc(&p, n * sizeof(T)); _TNY_CHECK(p, "cudaMalloc failed"); return static_cast<T *>(p);
    }
    template <class T> _TNY_HOST static void deallocate(T * p) { cudaFree(p); }
};

/** @brief Page-locked host memory (`cudaMallocHost`). */
struct cuda_host_alloc {
    template <class T> _TNY_HOST static T * allocate(cs::size_t n) {
        void * p = nullptr; cudaMallocHost(&p, n * sizeof(T)); _TNY_CHECK(p, "cudaMallocHost failed"); return static_cast<T *>(p);
    }
    template <class T> _TNY_HOST static void deallocate(T * p) { cudaFreeHost(p); }
};

/** @brief Pinned + mapped host memory (`cudaHostAlloc`). */
struct cuda_pinned_alloc {
    template <class T> _TNY_HOST static T * allocate(cs::size_t n) {
        void * p = nullptr; cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped); _TNY_CHECK(p, "cudaHostAlloc failed"); return static_cast<T *>(p);
    }
    template <class T> _TNY_HOST static void deallocate(T * p) { cudaFreeHost(p); }
};

/* ------------------------------------------------------------------ *
 *     Storage specializations for the CUDA `own` modes               *
 * ------------------------------------------------------------------ */

template <class T, cs::size_t N>
struct storage<T, own::device, N> : owning_storage<T, cuda_device_alloc> {
    using owning_storage<T, cuda_device_alloc>::owning_storage;
};
template <class T, cs::size_t N>
struct storage<T, own::host, N> : owning_storage<T, cuda_host_alloc> {
    using owning_storage<T, cuda_host_alloc>::owning_storage;
};
template <class T, cs::size_t N>
struct storage<T, own::pinned, N> : owning_storage<T, cuda_pinned_alloc> {
    using owning_storage<T, cuda_pinned_alloc>::owning_storage;
};

/* ------------------------------------------------------------------ *
 *     Factories                                                      *
 * ------------------------------------------------------------------ */

/** @brief Owning tensor in CUDA device memory (move-only). `device<T,E>(extents)`. */
template <class T, class Extents, class Layout = cs::layout_right>
using device = tensor<T, Extents, Layout, own::device>;
/** @brief Owning tensor in page-locked host memory (move-only). `host<T,E>(extents)`. */
template <class T, class Extents, class Layout = cs::layout_right>
using host = tensor<T, Extents, Layout, own::host>;
/** @brief Owning tensor in pinned/mapped host memory (move-only). `pinned<T,E>(extents)`. */
template <class T, class Extents, class Layout = cs::layout_right>
using pinned = tensor<T, Extents, Layout, own::pinned>;

/* --- functional factories (deduce the Extents type from the argument) --- */
template <class T, class Layout = cs::layout_right, class Extents>
_TNY_HOST auto make_device(Extents e) { return tensor<T, Extents, Layout, own::device>(e); }
template <class T, class Layout = cs::layout_right, class Extents>
_TNY_HOST auto make_host(Extents e)   { return tensor<T, Extents, Layout, own::host>(e); }
template <class T, class Layout = cs::layout_right, class Extents>
_TNY_HOST auto make_pinned(Extents e) { return tensor<T, Extents, Layout, own::pinned>(e); }

_TNY_NAMESPACE_END(tny)

#endif // TNY_HAS_CUDA

#endif // TNY_MD_CUDA
