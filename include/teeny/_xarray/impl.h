#ifndef TNY__XARRAY_IMPL
#define TNY__XARRAY_IMPL
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <teeny/core.h>
#include <teeny/statix.h>
#include <teeny/_xarray/decl.h>
#include <teeny/_xarray/access.h>

_TNY_NAMESPACE_BEGIN(tny)

using cuda::std::conditional_t;

/* ------------------------------------------------------------------ *
 *     xarray_tuple                                                   *
 *                                                                    *
 *  Materialize the storage tuple from a `values` pack:               *
 *   - dynamic slot (`cnone`)         -> `T`             (real leaf)  *
 *   - static  slot (`cvalue<T,X>`)   -> `carray<T,X>`   (empty leaf) *
 *                                                                    *
 *  Uses `statix::tuple` throughout (NOT `cuda::std::tuple`) so that   *
 *  `statix::cat` / `statix::as_tuple` resolve and the recursion      *
 *  terminates.                                                       *
 * ------------------------------------------------------------------ */

// Primary: normalize any pack-like `values` to a `statix::tuple`.
template <class T, class values>
struct _xarray_tuple {
    using type = xarray_tuple<T, statix::as_tuple<values>>;
};

template <class T, class X0, class... X>
struct _xarray_tuple<T, statix::tuple<X0, X...>> {
    using type = statix::cat<
        statix::tuple<conditional_t<
            statix::is_cnone<X0>::value,
            T,
            statix::as_carray<X0, T>
        >>,
        xarray_tuple<T, statix::tuple<X...>>
    >;
};

template <class T>
struct _xarray_tuple<T, statix::tuple<>> {
    using type = statix::tuple<>;
};

/* ------------------------------------------------------------------ *
 *     xarray_num_dynamic                                             *
 *                                                                    *
 *  Count the dynamic (`cnone`) slots in a `values` pack.             *
 * ------------------------------------------------------------------ */

// Primary: normalize any pack-like `values` to a `statix::tuple`.
template <class values>
struct _xarray_num_dynamic {
    using type = xarray_num_dynamic<statix::as_tuple<values>>;
};

template <class X0, class... X>
struct _xarray_num_dynamic<statix::tuple<X0, X...>> {
    using type = statix::csize<
        (statix::is_cnone<X0>::value ? size_t(1) : size_t(0))
        + xarray_num_dynamic<statix::tuple<X...>>::value
    >;
};

template <>
struct _xarray_num_dynamic<statix::tuple<>> {
    using type = statix::csize<0>;
};

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
 * Inherits its storage from `xarray_tuple<T, values>` (a `statix::tuple`),
 * so static elements cost no storage. Element access is routed through
 * `xarray_access`, which yields a reference for dynamic elements and a
 * prvalue for static ones.
 */
template <class T, class values>
struct xarray:
    public xarray_tuple<T, values>,
    public xarray_base<T>
{
public:
    using this_type   = xarray<T, values>;
    using base_type   = xarray_base<T>;
    using tuple_type  = xarray_tuple<T, values>;
    using values_type = statix::as_tuple<values>;
    using value_type  = T;

private:
    template <class Index>
    using access = xarray_access<this_type, Index>;

public:
    /* --- constructors --------------------------------------------- */

    // Inherit the storage-tuple constructors (default, copy, per-leaf...).
    using tuple_type::tuple_type;

    /* --- size ----------------------------------------------------- */

    /** @brief Total number of elements (static + dynamic). */
    _TNYDEF(H,D,S,CX) size_t size() noexcept
    { return statix::size<tuple_type>::value; }

    /** @brief Number of dynamic (runtime-stored) elements. */
    _TNYDEF(H,D,S,CX) size_t num_dynamic() noexcept
    { return xarray_num_dynamic<values>::value; }

    /** @brief True if the array has no elements. */
    _TNYDEF(H,D,S,CX) bool empty() noexcept
    { return size() == 0; }

    /* --- element access at a static index ------------------------- */

    /** @brief Access the element at a compile-time index. */
    template <class Index>
    _TNYDEF(H,D,I) typename access<Index>::type
    at(Index index) noexcept
    { return access<Index>::at(*this, index); }

    template <class Index>
    _TNYDEF(H,D,I) typename access<Index>::const_type
    at(Index index) const noexcept
    { return access<Index>::at(*this, index); }

    /** @brief Access the element at a compile-time index. */
    template <class Index>
    _TNYDEF(H,D,I) typename access<Index>::type
    operator[](Index index) noexcept
    { return access<Index>::at(*this, index); }

    template <class Index>
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
