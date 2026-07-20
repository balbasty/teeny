// !!! WORK IN PROGRESS
#ifndef TNY__XARRAY_ITER
#define TNY__XARRAY_ITER
#include <cuda/std/type_traits>
#include <teeny/core.h>
#include <teeny/statix.h>
#include <teeny/_xarray/decl.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

using cuda::std::enable_if_t;

struct xarray_iterator_base {};

template <class T, class I>
struct xarray_iterator: public xarray_iterator_base
{
    using xarray_type = T;
    using index_type  = I;
    using this_type   = xarray_iterator<xarray_type, index_type>;
    using next_type   = xarray_iterator<xarray_type, add<index_type, cptrdiff<1>>>;
    using prev_type   = xarray_iterator<xarray_type, add<index_type, cptrdiff<-1>>>;
    using value_type  = front<xarray_type>;
    using pointer     = value_type*;
    using reference   = value_type&;

    xarray_iterator(xarray<T...> & t): _xarray(&t) {}
    xarray_iterator(const this_type & other): _xarray(other._xarray) {}

    enable_if_t<
        0 <= index_type() && index_type() < statix::size<xarray_type>::value,
        value_type &
    > operator*() {
        return _xarray->at(index_type());
    }

    enable_if_t<
        0 <= index_type() && index_type() < statix::size<xarray_type>::value,
        const value_type &
    > operator*() const {
        return _xarray->at(index_type());
    }

    auto operator++() const {
        return xarray_iterator<xarray_type, next_type>(*_xarray);
    }

    auto operator--() const {
        return xarray_iterator<xarray_type, prev_type>(*_xarray);
    }

    template <class J>
    auto operator+(J) const {
        using next_type = add<index_type, J>;
        return xarray_iterator<xarray_type, next_type>(*_xarray);
    }

    template <class J>
    auto operator-(J) const {
        using prev_type = sub<index_type, J>;
        return xarray_iterator<xarray_type, prev_type>(*_xarray);
    }

    template <class OTHER>
    bool operator==(const OTHER & other) const {
        return _xarray == other._xarray && index_type() == other::index_type();
    }

    template <class OTHER>
    bool operator<(const OTHER & other) const {
        return _xarray == other._xarray && index_type() < other::index_type();
    }

    template <class OTHER>
    bool operator!=(const OTHER & other) const {
        return !(*this == other);
    }

    template <class OTHER>
    bool operator>(const OTHER & other) const {
        return other < *this;
    }

    template <class OTHER>
    bool operator<=(const OTHER & other) const {
        return !(other > *this);
    }

    template <class OTHER>
    bool operator>=(const OTHER & other) const {
        return !(other < *this);
    }

protected:
    xarray_type * _xarray;
};

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif // TNY__XARRAY_ITER
