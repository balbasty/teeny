// Minimal malloc-backed fake of the CUDA runtime, so teeny/cuda.h can be
// compiled AND run on a host without a CUDA toolkit. It validates the storage
// structure (allocate / move / free / factories) -- NOT real GPU behaviour.
#ifndef TNY_FAKE_CUDA_RUNTIME
#define TNY_FAKE_CUDA_RUNTIME
#include <cstdlib>
#include <cstddef>
#include <cstring>

typedef int cudaError_t;
enum { cudaSuccess = 0 };
enum { cudaHostAllocDefault = 0, cudaHostAllocMapped = 2 };

// Copy directions (all just memcpy here — fake "device" memory is host malloc).
typedef enum {
    cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4,
} cudaMemcpyKind;
static inline cudaError_t cudaMemcpy(void * dst, const void * src, std::size_t n, cudaMemcpyKind) {
    std::memcpy(dst, src, n); return cudaSuccess;
}

static inline cudaError_t cudaMalloc(void ** p, std::size_t n)              { *p = std::malloc(n); return cudaSuccess; }
static inline cudaError_t cudaFree(void * p)                                { std::free(p); return cudaSuccess; }
static inline cudaError_t cudaMallocHost(void ** p, std::size_t n)          { *p = std::malloc(n); return cudaSuccess; }
static inline cudaError_t cudaFreeHost(void * p)                            { std::free(p); return cudaSuccess; }
static inline cudaError_t cudaHostAlloc(void ** p, std::size_t n, unsigned) { *p = std::malloc(n); return cudaSuccess; }

#endif // TNY_FAKE_CUDA_RUNTIME
