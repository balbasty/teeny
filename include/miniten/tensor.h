#ifndef MINITEN_TENSOR_H
#define MINITEN_TENSOR_H
#include "defines.h"
#include "types.h"
#include "tuple.h"

namespace miniten {

template <typename scalar_t>
struct AnyTensor {};

template <uint64_t ndim, typename scalar_t>
struct AnyBatchedTensor: public AnyTensor<scalar_t>
{
    static constexpr uint64_t ndim = ndim;
    MINITEN_HOSTDEVICE inline virtual StaticVector<ndim, uint64_t> shape()   const = 0;
    MINITEN_HOSTDEVICE inline virtual StaticVector<ndim, uint64_t> strides() const = 0;
    MINITEN_HOSTDEVICE inline virtual uint64_t numel() const { return this->shape().prod(); };
    MINITEN_HOSTDEVICE inline virtual uint64_t len()   const { return this->shape()[0]; };
};

template <typename scalar_t, typename StaticShape>
struct StaticallyShapedTensor: public AnyBatchedTensor<StaticShape::Length, scalar_t>
{
    using scalar_t;
    using offset_t;
    using static_shape = StaticShape;
    static constexpr uint64_t ndim = StaticShape::Length;

    MINITEN_HOSTDEVICE inline virtual StaticVector<ndim, uint64_t> shape() const
    {
        VecMeta2Static<StaticShape> tmp;
        StaticVector<ndim, uint64_t> shp(
            reinterpret_cast
        )
        for (uint64_t d=0; d<ndim; ++d)
        {
            shp[d] = StaticShape::Get
        }
    }

};

// utility

template <typename T>
struct VecMeta2Static {};

template <typename T, T N1, T... N>
struct VecMeta2Static< meta::Vector<T, N1, N...> >
{
    using ThisVector = meta::Vector<T, N1, N...>;
    using NextVector = meta::Vector<T, N...>
    using ThisType   = VecMeta2Static<ThisVector>;

    VecMeta2Static() { fill(data); }

    MINITEN_HOSTDEVICE static void fill(T * data)
    {
        *data = N1;
        VecMeta2Static<NextVector>::fill(data+1);
    }

    T data[meta::Count<N...>::Value+1];
};

template <typename T, T N1>
struct VecMeta2Static< meta::Vector<T, N1> >
{
    using ThisVector = meta::Vector<T, N1>;
    using NextVector = meta::Vector<T>
    using ThisType   = VecMeta2Static<ThisVector>;

    VecMeta2Static() { fill(data); }

    MINITEN_HOSTDEVICE static void fill(T * data)
    {
        *data = N1;
    }

    T data[1];
};

template <T>
struct VecMeta2Static< meta::Vector<T> >
{
    using ThisVector = meta::Vector<T>;
    using NextVector = meta::Vector<T>
    using ThisType   = VecMeta2Static<ThisVector>;

    VecMeta2Static() {}

    MINITEN_HOSTDEVICE static void fill(T * data)
    {}
};


} // namespace miniten

#endif // MINITEN_TENSOR_H