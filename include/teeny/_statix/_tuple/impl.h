#ifndef TNY__STATIX__TUPLE_IMPL
#define TNY__STATIX__TUPLE_IMPL
#include <cuda/std/tuple>
#include <teeny/core.h>
#include <teeny/_statix/_tuple/decl.h>
#include <teeny/_statix/_pack/decl.h>
#include <teeny/_statix/_packapi/decl.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

/* ------------------------------------------------------------------ *
 *     Pack API                                                       *
 * ------------------------------------------------------------------ */

template <class... X>               struct _is_tuple     <tuple<X...>>      { using type = ctrue; };
template <class... X>               struct _as_pack      <tuple<X...>>      { using type = pack<X...>; };
template <class... X>               struct _as_tuple     <pack<X...>>       { using type = tuple<X...>; };
template <class... X>               struct _empty_like   <tuple<X...>>      { using type = tuple<>; };
template <class M, class... X>      struct _like_from    <tuple<X...>, M>   { using type = as_tuple<M>; };

/* ------------------------------------------------------------------ *
 *     Repeat                                                         *
 * ------------------------------------------------------------------ */

template <size_t N, class T>
struct _ntuple {
    using type = append<ntuple<N-1, T>, T>;
};

template <class T>
struct _ntuple<0, T> {
    using type = tuple<>;
};

/* ------------------------------------------------------------------ *
 *     Tuple                                                          *
 * ------------------------------------------------------------------ */

struct tuple_base {};

template <class... T>
struct _simple_tuple: public tuple_base {
protected:
    struct _scalar_type {};
public:
    using this_type = _simple_tuple<T...>;
    static constexpr _scalar_type value = _scalar_type();
};

template <class T0, class... T>
struct _simple_tuple<T0, T...>: public tuple_base {
protected:
    using _scalar_type = T0;
public:
    using value_type  = T0;
    using this_type   = _simple_tuple<T0, T...>;
    static constexpr _scalar_type value = T0();
};

template <class... T>
struct tuple:
    public _simple_tuple<T...>,
    public cuda::std::tuple<T...>
{
public:
    /** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
     **     Static types and values                                   **
     ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/
    using this_type = tuple<T...>;
    using base_type = tuple_base;
    using std_type  = cuda::std::tuple<T...>;

private:
    using _simple_type = _simple_tuple<T...>;
    using _scalar_type = typename _simple_tuple<T...>::_scalar_type;
    template <class index> using _wrap_index = statix::wrap_index<size<this_type>, index>;
    template <class index> using _as_carray_index = statix::as_index_carray<index, size<this_type>::value>;
    template <class index> using enable_if_static = enable_if_t<is_static_index<index>::value, bool>;
public:
    /** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
     **     Constructors                                              **
     ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/

    using std_type::tuple;
    using std_type::operator=;
    using std_type::swap;

    /** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
     **     Constexpr methods                                         **
     ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/

    /** @brief Array length */
    _TNYDEF(H,D,CX) size_t size() const noexcept { return statix::size<this_type>::value; }

    /** @brief True if array has no elements */
    _TNYDEF(H,D,CX) bool   empty() const noexcept { return size() == 0; }

    /** @brief Implicit conversion to value_type (only if size == 1) */
    // NOTE: when size != 1, this converts to a useless structure type.
    //       when size == 1, it converts to the scalae value.
    _TNYDEF(H,D,CX) operator _scalar_type() const { return front(); }

    /* @brief Access static element at static index */
    template <
        class index,
        enable_if_t<
            is_static_index<index>::value &&
            statix::size<_as_carray_index<index>>::value == 1 &&
            is_carray<at<this_type, index>>::value,
            bool
        > = true
    >
    _TNYDEF(H,D,CX) auto at(index) const noexcept
    {
        return statix::at<this_type, index>();
    }

    /* @brief Access dynamic element at static index */
    template <
        class index,
        enable_if_t<
            is_static_index<index>::value &&
            statix::size<_as_carray_index<index>>::value == 1 &&
            !is_carray<statix::at<this_type, index>>::value,
            bool
        > = true
    >
    _TNYDEF(H,D) auto at(index) const noexcept
    {
        constexpr auto index_value = statix::wrap_index<statix::size<this_type>, index()>::value;
        return cuda::std::get<index_value>(*this);
    }

    /* @brief Access element at static index or indices */
    template <class index, enable_if_static<index> = true>
    _TNYDEF(H,D,CX) auto operator[] (index i) const noexcept
    { return at(i); }

    /* @brief Access first element(s) */
    template <class numel, enable_if_static<numel> = true>
    _TNYDEF(H,D,CX) auto front(numel) const noexcept
    {
        using return_type = statix::head<this_type,numel::value>;
        return return_type(*reinterpret_cast<const return_type*>(this));
    }

    /* @brief Access first element */
    template <enable_if_t<(statix::size<this_type>::value > 0), bool> = true>
    _TNYDEF(H,D,CX) auto front() const noexcept
    { return cuda::std::get<0>(*this); }

    /* @brief Access last element(s) */
    template <class numel, enable_if_static<numel> = true>
    _TNYDEF(H,D) auto back(numel) const noexcept
    { static_assert(false, "not implemented yet"); }

    /* @brief Access last element */
    template <enable_if_t<(statix::size<this_type>::value > 0), bool> = true>
    _TNYDEF(H,D,CX) auto back() const noexcept
    { return cuda::std::get<statix::size<this_type>::value - 1>(*this); }

    template <class index, enable_if_static<index> = true>
    _TNYDEF(H,D,CX) auto erase(index) const noexcept
    { static_assert(false, "not implemented yet"); }

    /* @brief Access first element(s) */
    template <class numel, enable_if_static<numel> = true>
    _TNYDEF(H,D,CX) auto pop_front(numel) const noexcept
    { static_assert(false, "not implemented yet"); }

    /* @brief Access first element */
    template <enable_if_t<(statix::size<this_type>::value > 0), bool> = true>
    _TNYDEF(H,D,CX) auto pop_front() const noexcept
    { static_assert(false, "not implemented yet"); }

    /* @brief Access last element(s) */
    template <class numel, enable_if_static<numel> = true>
    _TNYDEF(H,D,CX) auto pop_back(numel) const noexcept
    { static_assert(false, "not implemented yet"); }

    /* @brief Access last element */
    template <enable_if_t<(statix::size<this_type>::value > 0), bool> = true>
    _TNYDEF(H,D,CX) auto pop_back() const noexcept
    { static_assert(false, "not implemented yet"); }

    /* @brief Assign element(s) at static index (or indices) */
    template <class index, class... others>
    _TNYDEF(H,D,CX) typename enable_if_t<
        is_static_index<index>::value,
        statix::erase<this_type,index>
    >::type set(index, others...) const noexcept
    { static_assert(false, "not implemented yet"); }

    template <class index, class... others>
    _TNYDEF(H,D,CX) auto set_front(others...) const noexcept
    { static_assert(false, "not implemented yet"); }

    template <class index, class... others>
    _TNYDEF(H,D,CX) auto set_back(others...) const noexcept
    { static_assert(false, "not implemented yet"); }

    template <class index, class... others>
    _TNYDEF(H,D,CX) typename enable_if_t<
        is_static_index<index>::value,
        statix::erase<this_type,index>
    >::type insert(index, others...) const noexcept
    { static_assert(false, "not implemented yet"); }

    template <class index, class... others>
    _TNYDEF(H,D,CX) auto prepend(others...) const noexcept
    { static_assert(false, "not implemented yet"); }

    template <class index, class... others>
    _TNYDEF(H,D,CX) auto append(others...) const noexcept
    { static_assert(false, "not implemented yet"); }

    template <class index, class... others>
    _TNYDEF(H,D,CX) auto extend(others...) const noexcept
    { static_assert(false, "not implemented yet"); }

};

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif // TNY__STATIX__TUPLE_IMPL
