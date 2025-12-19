#ifndef TNY__CORE_HALF
#define TNY__CORE_HALF
#include <teeny/_core/defines.h>

_TNY_NAMESPACE_BEGIN(tny)
#   ifdef __CUDACC__
#       include <cuda_fp16.h>
        using half = ::half;
#   else
#       include <half/half.hpp>
        using half = half_float::half;
#   endif
_TNY_NAMESPACE_END(tny)

#endif // TNY__CORE_HALF
