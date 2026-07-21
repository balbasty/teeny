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

#ifdef __CUDA_ARCH__
#       define _TNY_HOSTEXCEPT
#else
#       define _TNY_HOSTEXCEPT noexcept
#endif

// Debug-only precondition check (shape mismatches etc.). Active on the host in
// non-NDEBUG builds; compiled out on the device and under NDEBUG so `_TNY_API`
// code stays device-safe and release-fast.
#if !defined(__CUDA_ARCH__) && !defined(NDEBUG)
#   include <cassert>
#   define _TNY_CHECK(cond, msg) assert((cond) && (msg))
#else
#   define _TNY_CHECK(cond, msg) ((void)0)
#endif

#define _TNY_NAMESPACE_BEGIN(NAME) namespace NAME {
#define _TNY_NAMESPACE_END(NAME)   }

#endif // TNY__CORE_DEFINES
