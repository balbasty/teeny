#ifndef TNY__XARRAY_IMPL
#define TNY__XARRAY_IMPL
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <teeny/core.h>
#include <teeny/statix.h>
#include <teeny/_xarray/decl.h>

_TNY_NAMESPACE_BEGIN(tny)

using cuda::std::tuple;
using cuda::std::tuple_size;
using cuda::std::tuple_element;
using cuda::std::conditional_t;
using cuda::std::is_same;


template <size_t>
struct xarray_numel: public statix::csize<size_t> {};


template <class T, class values>
struct _xarray_tuple;

template <class T, class values>
using xarray_tuple = typename _xarray_tuple<T, values>::type;

template <class T, class values>
struct _xarray_tuple {
    using type = xarray_tuple<T, statix::as_tuple<values>>;
};

template <class T, class X0, class... X>
struct _xarray_tuple<T, tuple<X0, X...>> {
    using type = statix::cat<
        tuple<conditional_t<
            is_same<X0, statix::cnone>::value,
            T,
            statix::as_carray<X0, T>
        >>,
        xarray_tuple<T, tuple<X...>>
    >;
};

template <class T>
struct _xarray_tuple<T, tuple<>> {
    using type = tuple<>;
};

template <class T>
struct xarray_base: public tuple<> {
    using this_type  = xarray_base<T>;
    using value_type = T;
};

template <class T, class values>
struct xarray:
    public xarray_tuple<T, values>,
    public xarray_base<T>
{
public:
    using this_type  = xarray<T, values>;
    using base_type  = xarray_base<T>;
    using tuple_type = xarray_tuple<T, values>;

private:
    template <class Index> access            = _xarray::access<this_type, Index>;
    template <class Index> access_type       = _xarray::access_type<this_type, Index>;
    template <class Index> access_const_type = _xarray::access_const_type<this_type, Index>;

    using front_type       = statix::front<tuple_type>;
    using front_type_const = statix::front<const tuple_type>;
    using back_type        = statix::back<tuple_type>;
    using back_type_const  = statix::back<const tuple_type>;

public:

    using tuple_type::size;

    template <class Index> _TNYDEF(H,D,I)
    access_const_type<Index> at(const Index & index) const {
        return access<Index>::at(*this, index);
    }

    template <class Index> _TNYDEF(H,D,I)
    access_type<Index> at(const Index & index) {
        return access<Index>::at(*this, index);
    }

    template <class Index> _TNYDEF(H,D,I)
    access_const_type<Index> operator[](const Index & index) const {
        return access<Index>::bracket(*this, index);
    }

    template <class Index> _TNYDEF(H,D,I)
    access_type<Index> operator[](const Index & index) {
        return access<Index>::bracket(*this, index);
    }

    _TNYDEF(H,D,I)
    auto front() const {
        return at(csize<0>());
    }

    _TNYDEF(H,D,I)
    auto front() {
        return at(csize<0>());
    }

    _TNYDEF(H,D,I)
    auto back() const {
        return at(csize<size()-1>());
    }

    _TNYDEF(H,D,I)
    auto back() {
        return at(csize<size()-1>());
    }

};

_TNY_NAMESPACE_END(tny)

#endif // TNY__XARRAY_IMPL
