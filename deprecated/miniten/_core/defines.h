#ifndef MINITEN__CORE_DEFINES
#define MINITEN__CORE_DEFINES

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

#define MINIDEF_H   MINITEN_HOST
#define MINIDEF_D   MINITEN_DEVICE
#define MINIDEF_I   inline
#define MINIDEF_S   static
#define MINIDEF_C   const
#define MINIDEF_CX  constexpr
#define MINIDEF_

#define MINIDEF_ALL(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,...) \
        MINIDEF_##A MINIDEF_##B MINIDEF_##C MINIDEF_##D MINIDEF_##E \
        MINIDEF_##F MINIDEF_##G MINIDEF_##H MINIDEF_##I MINIDEF_##J \
        MINIDEF_##K MINIDEF_##L MINIDEF_##M MINIDEF_##N MINIDEF_##O \
        MINIDEF_##P MINIDEF_##Q MINIDEF_##R MINIDEF_##S MINIDEF_##T \
        MINIDEF_##U MINIDEF_##V MINIDEF_##W MINIDEF_##X MINIDEF_##Y MINIDEF_##Z
#define MINIDEF(...) MINIDEF_ALL(__VA_ARGS__,,,,,,,,,,,,,,,,,,,,,,,,,,)

#define _CPP98 199711L
#define _CPP11 201103L
#define _CPP14 201402L
#define _CPP17 201703L
#define _CPP20 202002L
#define _CPP23 202302L

#define NAMESPACE_BEGIN(NAME) namespace NAME {
#define NAMESPACE_END(NAME)   }

#endif // MINITEN__CORE_DEFINES
