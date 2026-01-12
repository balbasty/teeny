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

struct tuple_iterator_base {};

template <class T, class I>
struct tuple_iterator: public tuple_iterator_base
{
    using tuple_type = T;
    using index_type = I;
    using this_type  = tuple_iterator<tuple_type, index_type>;
    using next_type  = tuple_iterator<tuple_type, add<index_type, cptrdiff<1>>>;
    using prev_type  = tuple_iterator<tuple_type, add<index_type, cptrdiff<-1>>>;
    using value_type = front<tuple_type>;
    using pointer    = value_type*;
    using reference  = value_type&;

    tuple_iterator(tuple<T...> & t): _tuple(&t) {}
    tuple_iterator(const this_type & other): _tuple(other._tuple) {}

    enable_if_t<
        0 <= index_type() && index_type() < statix::size<tuple_type>::value,
        value_type &
    > operator*() {
        return _tuple->at(index_type());
    }

    enable_if_t<
        0 <= index_type() && index_type() < statix::size<tuple_type>::value,
        const value_type &
    > operator*() const {
        return _tuple->at(index_type());
    }

    auto operator++() const {
        return tuple_iterator<tuple_type, next_type>(*_tuple);
    }

    auto operator--() const {
        return tuple_iterator<tuple_type, prev_type>(*_tuple);
    }

    template <class J>
    auto operator+(J) const {
        using next_type = add<index_type, J>;
        return tuple_iterator<tuple_type, next_type>(*_tuple);
    }

    template <class J>
    auto operator-(J) const {
        using prev_type = sub<index_type, J>;
        return tuple_iterator<tuple_type, prev_type>(*_tuple);
    }

    template <class OTHER>
    bool operator==(const OTHER & other) const {
        return _tuple == other._tuple && index_type() == other::index_type();
    }

    template <class OTHER>
    bool operator<(const OTHER & other) const {
        return _tuple == other._tuple && index_type() < other::index_type();
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
    tuple_type * _tuple;
};

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif
