#ifndef MINITEN__CORE_HALF
#define MINITEN__CORE_HALF
#include <miniten/_core/defines.h>

NAMESPACE_BEGIN(miniten)
#   ifdef __CUDACC__
#       include <cuda_fp16.h>
        using half = ::half;
#   else
#       include <half/half.hpp>
        using half = half_float::half;
#   endif
NAMESPACE_END(miniten)

#endif // MINITEN__CORE_HALF
