#ifndef TNY__CORE_DEFINES
#define TNY__CORE_DEFINES

#ifdef __CUDACC__
#   define _TNY_HOST         __host__
#   define _TNY_DEVICE       __device__
#   define _TNY_HOSTDEVICE   __host__ __device__
#else
#   define _TNY_HOST
#   define _TNY_DEVICE
#   define _TNY_HOSTDEVICE
#endif // defined(__CUDACC__)


#define _TNY_API _TNY_HOSTDEVICE

// Debug-only precondition check (shape mismatches etc.). Active on the host in
// non-NDEBUG builds; compiled out on the device and under NDEBUG so `_TNY_API`
// code stays device-safe and release-fast.
#if !defined(__CUDA_ARCH__) && !defined(NDEBUG)
#   include <cassert>
#   define _TNY_CHECK(cond, msg) assert((cond) && (msg))
#else
// Compiled out on device / under NDEBUG, but still *consume* `cond` in an
// UNEVALUATED context: `sizeof` keeps any parameter pack in `cond` expanded, so
// `_TNY_CHECK` remains a valid operand inside a fold expression (e.g. recast's
// per-axis validation). Zero runtime cost, no side effects.
#   define _TNY_CHECK(cond, msg) ((void)sizeof((cond) ? 0 : 0))
#endif

#define _TNY_NAMESPACE_BEGIN(NAME) namespace NAME {
#define _TNY_NAMESPACE_END(NAME)   }

// Max rank of the rank-erased `anyrank` carrier's inline shape/stride store.
// Default 32 = numpy's classic NPY_MAXDIMS (PyTorch allows up to 64); override
// with -DTNY_MAX_RANK=N. Larger = a bigger inline carrier (2*N*sizeof(offset_t)).
#ifndef TNY_MAX_RANK
#   define TNY_MAX_RANK 32
#endif

#endif // TNY__CORE_DEFINES
