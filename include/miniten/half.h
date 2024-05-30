#ifndef MINITEN_HALF_H
#define MINITEN_HALF_H
#include "defines.h"

#ifdef __CUDACC__
#include <cuda_fp16.h>

namespace miniten {

    using half = ::half;

} // namespace miniten

#else // !defined(__CUDACC__)

namespace miniten {

class half {
protected:
    unsigned short value_as_int;
};

} // namespace miniten

#endif // defined(__CUDACC__)
#endif // MINITEN_HALF_H
