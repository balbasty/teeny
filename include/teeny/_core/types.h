#ifndef TNY__CORE_TYPES
#define TNY__CORE_TYPES
#include <cuda/std/cstddef>  // size_t, ptrdiff_t
#include <cuda/std/cstdint>  // (u)int*_t, uintptr_t
#include <teeny/_core/defines.h>
// #include <teeny/_core/half.h>

_TNY_NAMESPACE_BEGIN(tny)

using cuda::std::uint8_t;
using cuda::std::int8_t;
using cuda::std::uint16_t;
using cuda::std::int16_t;
using cuda::std::uint32_t;
using cuda::std::int32_t;
using cuda::std::uint64_t;
using cuda::std::int64_t;
using cuda::std::intptr_t;
using cuda::std::uintptr_t;
using cuda::std::size_t;
using cuda::std::ptrdiff_t;
using float32_t = float;
using float64_t = double;

_TNY_NAMESPACE_END(tny)

#endif // TNY__CORE_TYPES
