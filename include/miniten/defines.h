#ifndef MINITEN_DEFINES_H
#define MINITEN_DEFINES_H

#ifdef __CUDACC__
#   define MINITEN_HOST         __host__
#   define MINITEN_DEVICE       __device__
#   define MINITEN_HOSTDEVICE   __host__ __device__
#else
#   define MINITEN_HOST
#   define MINITEN_DEVICE
#   define MINITEN_HOSTDEVICE
#endif // defined(__CUDACC__)

#endif // MINITEN_DEFINES_H
