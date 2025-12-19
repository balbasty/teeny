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

#define _TNYDEF_H   _TNY_HOST
#define _TNYDEF_D   _TNY_DEVICE
#define _TNYDEF_I   inline
#define _TNYDEF_S   static
#define _TNYDEF_C   const
#define _TNYDEF_CX  constexpr
#define _TNYDEF_

#define _TNYDEF_ALL(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,...) \
        _TNYDEF_##A _TNYDEF_##B _TNYDEF_##C _TNYDEF_##D _TNYDEF_##E                  \
        _TNYDEF_##F _TNYDEF_##G _TNYDEF_##H _TNYDEF_##I _TNYDEF_##J                  \
        _TNYDEF_##K _TNYDEF_##L _TNYDEF_##M _TNYDEF_##N _TNYDEF_##O                  \
        _TNYDEF_##P _TNYDEF_##Q _TNYDEF_##R _TNYDEF_##S _TNYDEF_##T                  \
        _TNYDEF_##U _TNYDEF_##V _TNYDEF_##W _TNYDEF_##X _TNYDEF_##Y _TNYDEF_##Z
#define _TNYDEF(...) _TNYDEF_ALL(__VA_ARGS__,,,,,,,,,,,,,,,,,,,,,,,,,,)

#define _CPP98 199711L
#define _CPP11 201103L
#define _CPP14 201402L
#define _CPP17 201703L
#define _CPP20 202002L
#define _CPP23 202302L

#define _TNY_NAMESPACE_BEGIN(NAME) namespace NAME {
#define _TNY_NAMESPACE_END(NAME)   }

#endif // TNY__CORE_DEFINES
