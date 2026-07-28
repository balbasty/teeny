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

// MSVC-only: the Microsoft ABI applies empty-base-optimization to only the
// FIRST eligible empty base of a class by default -- a class with two or more
// private empty bases (teeny's usual EBO-for-a-mapping trick) stays non-empty
// unless the DERIVED class itself is tagged __declspec(empty_bases), which
// opts every base back into the standard-conforming layout. No-op (and no
// warning) on clang/g++/nvcc, which already fold every empty base for free.
#if defined(_MSC_VER) && !defined(__clang__)
#   define _TNY_EMPTY_BASES __declspec(empty_bases)
#else
#   define _TNY_EMPTY_BASES
#endif

// Restrict qualifier for a non-aliasing pointer. Used by the out-of-place
// elementwise engines to mark a freshly-allocated DESTINATION that provably
// cannot alias its (const) operands, unlocking the auto-vectorized linear write.
// `__restrict__` is accepted by g++, clang++, and nvcc (device code included);
// real MSVC (cl.exe) only recognizes the single-underscore `__restrict` (#311)
// -- clang-cl defines both __clang__ and _MSC_VER, so it stays on the more
// portable `__restrict__` spelling like every other compiler branch here.
#if defined(_MSC_VER) && !defined(__clang__)
#   define _TNY_RESTRICT __restrict
#else
#   define _TNY_RESTRICT __restrict__
#endif

// Portable full-unroll hint for a SMALL STATIC-trip-count loop (a downstream
// static-C kernel's packed-index / per-axis inner loop) so it folds to immediates
// instead of runtime imul/cmov. clang & nvcc honour `#pragma unroll`; **gcc silently
// ignores it** and needs `#pragma GCC unroll N`. Place immediately before the `for`.
// No-op on unknown compilers. Public spelling — downstream kernels use it too.
//   NB: gcc's `#pragma GCC unroll` needs a LITERAL count, so a per-count macro
//   (`TNY_UNROLL_N(N)`) can't take a *template-parameter* count on gcc (the exact
//   static-C case) — hence a single full-unroll spelling with a generous fixed
//   count. For a partial unroll, write the compiler pragma directly.
#if defined(__clang__) || defined(__CUDACC__)
#   define TNY_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#   define TNY_UNROLL _Pragma("GCC unroll 16")
#else
#   define TNY_UNROLL
#endif

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

// Opt-in element-access bounds check. teeny follows `mdspan`: `operator()` /
// `operator[]` / `at` are UNCHECKED by default (an out-of-range index is UB, as
// with `mdspan`'s subscript). Define `-DTNY_HARDENED` to turn on a per-index
// bounds check in the CHECKED accessors; the `u`-accessors always skip it, and it
// is always off on the device. Gated on TNY_HARDENED, NOT NDEBUG, so it is a
// deliberate opt-in that survives an optimized `-DNDEBUG` release build.
#if defined(TNY_HARDENED) && !defined(__CUDA_ARCH__)
#   include <cassert>
#   define _TNY_BOUND(cond, msg) assert((cond) && (msg))
#else
#   define _TNY_BOUND(cond, msg) ((void)sizeof((cond) ? 0 : 0))
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
