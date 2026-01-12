/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file implements a compile-time "vector of values"                  **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef TNY__STATIX__CARRAY_IMPL
#define TNY__STATIX__CARRAY_IMPL
#include <cuda/std/type_traits>
#ifndef __CUDA_ARCH__
#include <stdexcept>
#endif
#include <teeny/core.h>
#include <teeny/_statix/_carray/decl.h>     // carray, crepeat
#include <teeny/_statix/_packapi/decl.h>    // cat
#include <teeny/_statix/_math/decl.h>       // count_values
#include <teeny/_statix/_index/decl.h>      // is_static_index

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

using cuda::std::enable_if_t;

/* ------------------------------------------------------------------ *
 *     crepeat                                                        *
 * ------------------------------------------------------------------ */

template <size_t N, typename T, T Val>
struct _crepeat {
    using type = cat<crepeat<N-1, T, Val>, carray<T, Val>>;
};

template <typename T, T Val>
struct _crepeat<0, T, Val> {
    using type = carray<T>;
};

/* ------------------------------------------------------------------ *
 *     crange                                                         *
 * ------------------------------------------------------------------ */

template <ptrdiff_t stop, ptrdiff_t start, ptrdiff_t step,
          bool is_empty = (step > 0 ? start >= stop : start <= stop)>
struct __crange {
    using type = cat<
        carray<ptrdiff_t, start>,
        crange<start + step, stop, step>
    >;
};

template <ptrdiff_t stop, ptrdiff_t step>
struct __crange<stop, stop, step, false> {
    using type = carray<ptrdiff_t>;
};


template <ptrdiff_t stop, ptrdiff_t start, ptrdiff_t step>
struct _crange {
    using type = typename __crange<stop, start, step>::type;
};

/* ------------------------------------------------------------------ *
 *     carray                                                         *
 * ------------------------------------------------------------------ */

template <typename T>
struct carray_base {
    using this_type  = carray_base<T>;
    using value_type = T;
};

template <typename T, T... X>
struct _simple_carray: public carray_base<T> {
protected:
    struct _scalar_type {};
public:
    using value_type  = T;
    using this_type   = _simple_carray<T, X...>;
    static constexpr _scalar_type value = _scalar_type();
};

template <typename T, T X0>
struct _simple_carray<T, X0>: public carray_base<T> {
protected:
    using _scalar_type = T;
public:
    using value_type  = T;
    using this_type   = _simple_carray<T, X0>;
    static constexpr _scalar_type value = X0;
};

/* --- A compile-time tuple of values with the same type ------------ */
template <typename T, T... X>
struct carray: public _simple_carray<T, X...> {
private:

    template <class index>
    using enable_if_static = enable_if_t<is_static_index<index>::value, bool>;

    using _simple_type = _simple_carray<T, X...>;
    using _scalar_type = typename _simple_type::_scalar_type;

public:
    /** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
     **     Static types and values                                   **
     ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/

    using this_type  = carray<T, X...>;
    // using next_type  = statix::erase_head<this_type>;
    using value_type = T;

    static constexpr _scalar_type value = _simple_type::value;

    /** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
     **     Constexpr methods                                         **
     ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/

    /** @brief Array length */
    _TNYDEF(H,D,CX) size_t size() const noexcept { return statix::size<this_type>::value; }

    /** @brief True if array has no elements */
    _TNYDEF(H,D,CX) bool   empty()  const noexcept { return size() == 0; }

    /** @brief Implicit conversion to value_type (only if size == 1) */
    // NOTE: when size != 1, this converts to a useless structure type.
    //       when size == 1, it converts to the scalar value.
    _TNYDEF(H,D,CX) operator _scalar_type() const noexcept { return value; }

    /** @brief Access element at static index (or indices) */
    template <class index, enable_if_static<index> = true>
    _TNYDEF(H,D,CX) auto at(index) const noexcept
    { return statix::at<this_type,index>::value; }

    /** @brief Access element at static index or indices */
    template <class index, enable_if_static<index> = true>
    _TNYDEF(H,D,CX) auto operator[] (index) const noexcept
    { return statix::at<this_type,index>::value; }

    /** @brief Access first element(s) */
    template <class numel, enable_if_static<numel> = true>
    _TNYDEF(H,D,CX) auto front(numel) const noexcept
    { return statix::head<this_type,numel::value>::value; }

    /** @brief Access first element */
    template <enable_if_t<(statix::size<this_type>::value > 0), bool> = true>
    _TNYDEF(H,D,CX) auto front() const noexcept
    { return statix::head<this_type>::value; }

    /** @brief Access last element(s) */
    template <class numel, enable_if_static<numel> = true>
    _TNYDEF(H,D,CX) auto back(numel) const noexcept
    { return statix::tail<this_type,numel::value>::value; }

    /** @brief Access last element */
    template <enable_if_t<(statix::size<this_type>::value > 0), bool> = true>
    _TNYDEF(H,D,CX) auto back() const noexcept
    { return statix::tail<this_type>::value ; }

    /** @brief Erase an element */
    template <class index, enable_if_static<index> = true>
    _TNYDEF(H,D,CX) auto erase(index) const noexcept
    { return statix::erase<this_type,index>(); }

    /** @brief Erase first element(s) */
    template <class numel, enable_if_static<numel> = true>
    _TNYDEF(H,D,CX) auto pop_front(numel) const noexcept
    { return statix::erase_head<this_type,numel::value>(); }

    /** @brief Erase first element */
    template <enable_if_t<(statix::size<this_type>::value > 0), bool> = true>
    _TNYDEF(H,D,CX) auto pop_front() const noexcept
    { return statix::erase_head<this_type>(); }

    /** @brief Erase last element(s) */
    template <class numel, enable_if_static<numel> = true>
    _TNYDEF(H,D,CX) auto pop_back(numel) const noexcept
    { return statix::erase_tail<this_type,numel::value>(); }

    /** @brief Erase last element */
    template <enable_if_t<(statix::size<this_type>::value > 0), bool> = true>
    _TNYDEF(H,D,CX) auto pop_back() const noexcept
    { return statix::erase_tail<this_type>(); }

    /** @brief Assign element(s) at static index (or indices) */
    template <class index, class... others>
    _TNYDEF(H,D,CX) typename enable_if_t<
        is_static_index<index>::value,
        statix::erase<this_type,index>
    >::type set(index, others...) const noexcept
    { return statix::set<this_type,index,others...>(); }

    /** @brief Assign first element(s) */
    template <class index, class... others>
    _TNYDEF(H,D,CX) auto set_front(others...) const noexcept
    { return statix::set_front<this_type,others...>(); }

    /** @brief Assign last element(s) */
    template <class index, class... others>
    _TNYDEF(H,D,CX) auto set_back(others...) const noexcept
    { return statix::set_back<this_type,others...>(); }

    /** @brief Insert element(s) at static index (or indices) */
    template <class index, class... others>
    _TNYDEF(H,D,CX) typename enable_if_t<
        is_static_index<index>::value,
        statix::erase<this_type,index>
    >::type insert(index, others...) const noexcept
    { return statix::insert<this_type,index,others...>(); }

    /** @brief Prepend element(s) */
    template <class index, class... others>
    _TNYDEF(H,D,CX) auto prepend(others...) const noexcept
    { return statix::prepend<this_type,others...>(); }

    /** @brief Append element(s) */
    template <class index, class... others>
    _TNYDEF(H,D,CX) auto append(others...) const noexcept
    { return statix::append<this_type,others...>(); }

    /** @brief Extend with element(s) */
    template <class index, class... others>
    _TNYDEF(H,D,CX) auto extend(others...) const noexcept
    { return statix::extend<this_type,others...>(); }

    /** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
     **     Runtime methods                                           **
     ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **/

    _TNYDEF(H,D) value_type at(ptrdiff_t index) const _TNY_HOSTEXCEPT
    {
        if (index < 0)
            index += size();
        if (index < 0 || index >= size())
#ifdef __CUDA_ARCH__
            return value_type();
#else
            throw std::out_of_range("carray::at(): index out of range");
#endif
        if (index == 0)
            return front();
        return pop_front().at(index - 1);
    }

    _TNYDEF(H,D) value_type operator[] (ptrdiff_t index) const _TNY_HOSTEXCEPT
    { return at(index); }

};

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif /// TNY__STATIX__CARRAY_IMPL
