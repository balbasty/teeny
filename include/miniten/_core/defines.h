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

#define _MH_  MINITEN_HOST
#define _MD_  MINITEN_DEVICE
#define _MHD_ MINITEN_HOSTDEVICE

#define _CPP98 199711L
#define _CPP11 201103L
#define _CPP14 201402L
#define _CPP17 201703L
#define _CPP20 202002L
#define _CPP23 202302L

#endif // MINITEN_DEFINES_H
