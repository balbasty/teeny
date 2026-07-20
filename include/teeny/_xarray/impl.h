#ifndef TNY__XARRAY_IMPL
#define TNY__XARRAY_IMPL
#include <cuda/std/tuple>
#include <cuda/std/utility>       // index_sequence, make_index_sequence
#include <cuda/std/type_traits>
#include <teeny/core.h>
#include <teeny/statix.h>
#include <teeny/_xarray/decl.h>
#include <teeny/_xarray/access.h>

_TNY_NAMESPACE_BEGIN(tny)

/* ------------------------------------------------------------------ *
 *     xarray_num_dynamic  (count of `cnone` slots)                   *
 * ------------------------------------------------------------------ */

// Primary: normalize any pack-like `values` to a `statix::tuple`.
template <class values>
struct _xarray_num_dynamic {
    using type = xarray_num_dynamic<statix::as_tuple<values> >;
};

template <class X0, class... X>
struct _xarray_num_dynamic<statix::tuple<X0, X...> > {
    using type = statix::csize<
        (statix::is_cnone<X0>::value ? size_t(1) : size_t(0))
        + xarray_num_dynamic<statix::tuple<X...> >::value
    >;
};

template <>
struct _xarray_num_dynamic<statix::tuple<> > {
    using type = statix::csize<0>;
};

/* ------------------------------------------------------------------ *
 *     dynamic_ordinal  (count of `cnone` strictly before index I)    *
 * ------------------------------------------------------------------ */

// Primary: normalize any pack-like `values` to a `statix::tuple`.
template <class values, size_t I>
struct _dynamic_ordinal {
    using type = dynamic_ordinal<statix::as_tuple<values>, I>;
};

template <class X0, class... X, size_t I>
struct _dynamic_ordinal<statix::tuple<X0, X...>, I> {
    using type = statix::csize<
        (statix::is_cnone<X0>::value ? size_t(1) : size_t(0))
        + dynamic_ordinal<statix::tuple<X...>, I - 1>::value
    >;
};

template <class X0, class... X>
struct _dynamic_ordinal<statix::tuple<X0, X...>, 0> {
    using type = statix::csize<0>;
};

template <>
struct _dynamic_ordinal<statix::tuple<>, 0> {
    using type = statix::csize<0>;
};

/* ------------------------------------------------------------------ *
 *     xarray_tuple  (storage = one `T` per dynamic slot)             *
 * ------------------------------------------------------------------ */

template <class T, size_t> using _xarray_leaf = T;   // ignore the index

template <class T, class Seq>
struct _xarray_ntuple;

template <class T, size_t... I>
struct _xarray_ntuple<T, cuda::std::index_sequence<I...> > {
    using type = cuda::std::tuple<_xarray_leaf<T, I>...>;
};

template <class T, class values>
struct _xarray_tuple {
    using type = typename _xarray_ntuple<
        T,
        cuda::std::make_index_sequence<xarray_num_dynamic<values>::value>
    >::type;
};

/* ------------------------------------------------------------------ *
 *     Convenience: all-dynamic descriptors                           *
 * ------------------------------------------------------------------ */

/** @brief A `values` pack of `N` dynamic (`cnone`) slots. */
template <size_t N>
using dynamic_values = statix::ntuple<N, statix::cnone>;

/** @brief A fully-dynamic xarray of `N` elements of type `T`. */
template <class T, size_t N>
using dynarray = xarray<T, dynamic_values<N> >;

/* ------------------------------------------------------------------ *
 *     xarray                                                         *
 * ------------------------------------------------------------------ */

/** @brief Common (values-independent) typedefs for every `xarray<T,.>`. */
template <class T>
struct xarray_base {
    using value_type = T;
};

/**
 * @brief Hybrid one-dimensional array (see `decl.h`).
 *
 * Storage is `xarray_tuple<T, values>` -- a plain tuple holding exactly the
 * dynamic elements -- so static elements cost no memory. Element access is
 * routed through `xarray_access`: a reference for dynamic elements, a prvalue
 * for static ones.
 *
 * The type is trivially copyable (it declares no special members), so it can
 * be passed into a `__global__` kernel by value.
 */
template <class T, class values>
struct xarray:
    public xarray_tuple<T, values>,
    public xarray_base<T>
{
public:
    using this_type   = xarray<T, values>;
    using base_type   = xarray_base<T>;
    using tuple_type  = xarray_tuple<T, values>;      // cuda::std::tuple<T,...,T>
    using values_type = statix::as_tuple<values>;     // tuple<cvalue|cnone, ...>
    using value_type  = T;

private:
    template <class Index>
    using access = xarray_access<this_type, Index>;

    template <class Index>
    using if_static = cuda::std::enable_if_t<
        statix::is_static_index<Index>::value, bool>;

public:
    /* --- size ----------------------------------------------------- */

    /** @brief Total number of elements (static + dynamic). */
    _TNYDEF(H,D,S,CX) size_t size() noexcept
    { return statix::size<values_type>::value; }

    /** @brief Number of dynamic (runtime-stored) elements. */
    _TNYDEF(H,D,S,CX) size_t num_dynamic() noexcept
    { return xarray_num_dynamic<values>::value; }

    /** @brief True if the array has no elements. */
    _TNYDEF(H,D,S,CX) bool empty() noexcept
    { return size() == 0; }

    /* --- element access at a static index ------------------------- *
     *  Constrained to static indices so a stray runtime `int` fails    *
     *  cleanly (and leaves room for a future runtime overload).        */

    /** @brief Access the element at a compile-time index. */
    template <class Index, if_static<Index> = true>
    _TNYDEF(H,D,I) typename access<Index>::type
    at(Index index) noexcept
    { return access<Index>::at(*this, index); }

    template <class Index, if_static<Index> = true>
    _TNYDEF(H,D,I) typename access<Index>::const_type
    at(Index index) const noexcept
    { return access<Index>::at(*this, index); }

    /** @brief Access the element at a compile-time index. */
    template <class Index, if_static<Index> = true>
    _TNYDEF(H,D,I) typename access<Index>::type
    operator[](Index index) noexcept
    { return access<Index>::at(*this, index); }

    template <class Index, if_static<Index> = true>
    _TNYDEF(H,D,I) typename access<Index>::const_type
    operator[](Index index) const noexcept
    { return access<Index>::at(*this, index); }

    /* --- first / last element ------------------------------------- *
     *  Templated on the return type so the body/return-type are only  *
     *  instantiated on use (harmless for an empty xarray otherwise).  */

    template <class R = typename access<statix::csize<0> >::type>
    _TNYDEF(H,D,I) R front() noexcept
    { return at(statix::csize<0>()); }

    template <class R = typename access<statix::csize<0> >::const_type>
    _TNYDEF(H,D,I) R front() const noexcept
    { return at(statix::csize<0>()); }

    template <class R = typename access<statix::cptrdiff<-1> >::type>
    _TNYDEF(H,D,I) R back() noexcept
    { return at(statix::cptrdiff<-1>()); }

    template <class R = typename access<statix::cptrdiff<-1> >::const_type>
    _TNYDEF(H,D,I) R back() const noexcept
    { return at(statix::cptrdiff<-1>()); }
};

_TNY_NAMESPACE_END(tny)

#endif // TNY__XARRAY_IMPL
