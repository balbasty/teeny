#ifndef MINITEN_XARRAY_H
#define MINITEN_XARRAY_H
#include <cuda/std/tuple>
#include <miniten/core.h>

NAMESPACE_BEGIN(miniten)

using cuda::std::tuple;
using cuda::std::tuple_size_v;
using cuda::std::tuple_element;


NAMESPACE_BEGIN(_xarray)

template <
    class XArray, class Index,
    class StaticValue = typename tuple_element<Index::Value, typename XArray::static_values>::type
>
struct access {
    using typename XArray::base_type;
    using typename XArray::value_type;
    using const_return_type = value_type;
    using return_type       = value_type;

    static value_type at(const XArray & x, const Index & index) {
        return static_cast<value_type>(StaticValue::value);
    }

    static value_type bracket(const XArray & x, const Index & index) {
        return static_cast<value_type>(StaticValue::value);
    }
};

template <class XArray, class Index, class StaticValue>
struct access<XArray, Index, None> {
    using typename XArray::base_type;
    using typename XArray::const_reference;
    using const_return_type = const_reference;
    using return_type       = reference;

    static const_reference at(const XArray & x, const Index & index) {
        return x.base_type::at(index);
    }

    static reference at(XArray & x, const Index & index) {
        return x.base_type::at(index);
    }

    static const_reference bracket(const XArray & x, const Index & index) {
        return x.base_type::operator[](index);
    }

    static reference bracket(XArray & x, const Index & index) {
        return x.base_type::operator[](index);
    }
};

template <class XArray, class Index>
using access_type = typename access<XArray, Index>::return_type;

template <class XArray, class Index>
using access_const_type = typename access<XArray, Index>::const_return_type;

NAMESPACE_END(_xarray)

template <class ItemType, class StaticValues>
struct xarray: public array<ItemType, tuple_size_v<StaticValues>> {
public:
    using this_type = xarray<ItemType, StaticValues>;
    using base_type = array<ItemType, tuple_size_v<StaticValues>>;
    using static_values = StaticValues;
    using typename base_type::value_type;
    using typename base_type::size_type;
    using typename base_type::difference_type;
    using typename base_type::reference;
    using typename base_type::const_reference;
    using typename base_type::pointer;
    using typename base_type::const_pointer;
    using static_front_type = typename tuple_element<0, StaticValues>::type;
    using static_back_type  = typename tuple_element<
        tuple_size_v<StaticValues>-1, StaticValues
    >::type;
    using front_type       = conditional_t<is_none_v<static_front_type>, value_type, reference>;
    using front_type_const = conditional_t<is_none_v<static_front_type>, value_type, const_reference>;
    >::type;
    using back_type       = conditional_t<is_none_v<static_back_type>, value_type, reference>;
    using back_type_const = conditional_t<is_none_v<static_back_type>, value_type, const_reference>;

private:
    template <class Index> access            = _xarray::access<this_type, Index>;
    template <class Index> access_type       = _xarray::access_type<this_type, Index>;
    template <class Index> access_const_type = _xarray::access_const_type<this_type, Index>;

public:

    template <class Index> MINIDEF(H,D,I)
    access_const_type<Index> at(const Index & index) const {
        return access<Index>::at(*this, index);
    }

    template <class Index> MINIDEF(H,D,I)
    access_type<Index> at(const Index & index) {
        return access<Index>::at(*this, index);
    }

    template <class Index> MINIDEF(H,D,I)
    access_const_type<Index> operator[](const Index & index) const {
        return access<Index>::bracket(*this, index);
    }

    template <class Index> MINIDEF(H,D,I)
    access_type<Index> operator[](const Index & index) {
        return access<Index>::bracket(*this, index);
    }

    MINIDEF(H,D,I)
    front_type_const front() const {
        return at(SizeT<0>());
    }

    MINIDEF(H,D,I)
    front_type front() {
        return at(SizeT<0>());
    }

    MINIDEF(H,D,I)
    back_type_const back() const {
        return at(SizeT<size()-1()>());
    }

    MINIDEF(H,D,I)
    back_type back() {
        return at(SizeT<size()-1()>());
    }

};

NAMESPACE_END(miniten)

#endif // MINITEN_XARRAY_H
