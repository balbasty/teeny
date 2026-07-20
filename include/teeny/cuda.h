#ifndef TNY_MD_CUDA
#define TNY_MD_CUDA
// Opt-in CUDA memory-space storage for md::tensor. Needs the CUDA runtime, so
// this header is NOT pulled in by <teeny/md.h>; include it explicitly when you
// want device / page-locked / pinned owning tensors.
#include <cuda_runtime.h>
#include <cuda/std/cstddef>
#include <teeny/_core/defines.h>
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
        void * p = nullptr; cudaMalloc(&p, n * sizeof(T)); return static_cast<T *>(p);
    }
    template <class T> _TNY_HOST static void deallocate(T * p) { cudaFree(p); }
};

/** @brief Page-locked host memory (`cudaMallocHost`). */
struct cuda_host_alloc {
    template <class T> _TNY_HOST static T * allocate(cs::size_t n) {
        void * p = nullptr; cudaMallocHost(&p, n * sizeof(T)); return static_cast<T *>(p);
    }
    template <class T> _TNY_HOST static void deallocate(T * p) { cudaFreeHost(p); }
};

/** @brief Pinned + mapped host memory (`cudaHostAlloc`). */
struct cuda_pinned_alloc {
    template <class T> _TNY_HOST static T * allocate(cs::size_t n) {
        void * p = nullptr; cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped); return static_cast<T *>(p);
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

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_CUDA
