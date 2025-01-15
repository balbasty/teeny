#ifndef MINITEN_HALF_H
#define MINITEN_HALF_H
#include "defines.h"

namespace miniten {
#   ifdef __CUDACC__
#       include <cuda_fp16.h>
        using half = ::half;
#   else
#       include "../half/half.hpp"
        using half = half_float::half;
#   endif
} // namespace miniten

#endif // MINITEN_HALF_H
