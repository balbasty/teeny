#ifndef TNY__STATIX__PACKAPI_IMPL2
#define TNY__STATIX__PACKAPI_IMPL2
#include <cuda/std/type_traits>
#include <teeny/core.h>
#include <teeny/_statix/_packapi/decl.h>
#include <teeny/_statix/_packapi/impl1.h>
#include <teeny/_statix/_pack/decl.h>       // pack
#include <teeny/_statix/_carray/decl.h>     // cptrdiff

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

using cuda::std::conditional_t;

/* ================================================================== *
 *                                                                    *
 * Implementation                                                     *
 * We specialize some of the helpers class in container-agnostic ways *
 *                                                                    *
 * ================================================================== */

/* ------------------------------------------------------------------ *
 * cat                                                                *
 * ------------------------------------------------------------------ */
/* Containers only need to implement _cat2<CONTAINER,CONTAINER>       */

template <class A>                          struct _cat<A>          { using type = A; };
template <class A, class M>                 struct _cat<A,M>        { using type = typename _cat2<A,like_from<A,M>>::type; };
template <class A, class M0, class... M>    struct _cat<A,M0,M...>  { using type = cat<cat<A,M0>, M...>; };

/* ------------------------------------------------------------------ *
 * get                                                                *
 * ------------------------------------------------------------------ *
 * _get only needs to be implemented for cptrdiff indices             *
 * because _get_wrap converts all indices (even slices) to cptrdiff   */

// Multiple indices -> concatenate outputs
template <class A, ptrdiff_t I0, ptrdiff_t...I>
struct _get<A, cptrdiff<I0, I...>> {
    using type = cat<get<A, cptrdiff<I0>>, get<A, cptrdiff<I...>>>;
};

// No indices -> empty container
template <class A>
struct _get<A, cptrdiff<>> {
    using type = empty_like<A>;
};

// Single index -> delete all preceding elements -> fallback to Get<0>
template <class A, ptrdiff_t I>
struct _get<A, cptrdiff<I>> {
    static_assert(I >= 0 && I < size<A>::value, "at out of bounds");
    using type = head<erase_head<A, static_cast<size_t>(I)>>;
};

template <class A>
struct _get<A, cptrdiff<0>> {
    static_assert(size<A>::value > 0, "get<...,0> on empty container");
    using type = typename _head<A>::type;
};

template <class A, ptrdiff_t I>
struct _at<A, cptrdiff<I>> {
    static_assert(I >= 0 && I < size<A>::value, "at out of bounds");
    using type = front<erase_head<A, static_cast<size_t>(I)>>;
};

template <class A>
struct _at<A, cptrdiff<0>> {
    static_assert(size<A>::value > 0, "at<...,0> on empty container");
    using type = typename _front<A>::type;
};

/* ------------------------------------------------------------------ *
 * erase                                                              *
 * ------------------------------------------------------------------ *
 * Containers must implement _erase_head                              */

// Helper to re-number indices after one element has been deleted.
template <ptrdiff_t J, ptrdiff_t... I>
struct _erase_update_index {};

template <ptrdiff_t J, ptrdiff_t I0, ptrdiff_t... I>
struct _erase_update_index<J, I0, I...> {
    using type = cat<
        typename _erase_update_index<J, I0>::type,
        typename _erase_update_index<J, I...>::type
    >;
};

template <ptrdiff_t J, ptrdiff_t I0>
struct _erase_update_index<J, I0> {
    using type = cptrdiff<((I0 < J) ? I0 : (I0 - 1))>;
};

template <ptrdiff_t J>
struct _erase_update_index<J, J> {
    using type = cptrdiff<>;
};

template <ptrdiff_t J>
struct _erase_update_index<J> {
    using type = cptrdiff<>;
};

// Multiple indices -> recursive deletion
template <class A, ptrdiff_t I0, ptrdiff_t...I>
struct _erase<A, cptrdiff<I0, I...>> {
private:
    using erased_first = erase<A, cptrdiff<I0>>;
    using new_index    = typename _erase_update_index<I0, I...>::type;
public:
    using type = erase<erased_first, new_index>;
};

// No indices -> return input
template <class A>
struct _erase<A, cptrdiff<>> {
    using type = A;
};

// Single index -> concatenate left and right containers
template <class A, ptrdiff_t I>
struct _erase<A, cptrdiff<I>>
{
    using type = cat<
        head        <A, static_cast<size_t>(I)>,
        erase_head  <A, static_cast<size_t>(I)+1>
    >;
};

template <class A>
struct _erase<A, cptrdiff<0>>
{
    using type = typename _erase_head<A>::type;
};


/* ------------------------------------------------------------------ *
 * insert                                                             *
 * ------------------------------------------------------------------ */
// Container do not need to implement anything

template <class A, ptrdiff_t I, class... M>
struct _insert<A, cptrdiff<I>, M...>
{
    using type = cat<head<A,I>, cat<M...>, erase_head<A,I>>;
};

/* ------------------------------------------------------------------ *
 * set_from                                                           *
 * ------------------------------------------------------------------ */
// Container do not need to implement anything

// Multiple containers -> contatenate them
template <class A, class I>
struct _set_from<A, I>
{
    using type = A;
};
template <class A, class I, class M0, class... M>
struct _set_from<A, I, M0, M...>
{
    using type = set_from<A, I, cat<like_from<A,M0>, M...>>;
};

// Single container -> recursive call
template <class A, class I, class M>
struct _set_from<A, I, M>
{
    static_assert(
        (size<M>::value == size<I>::value) ||
        (size<M>::value == 1),
        "Set requires indices and values to be broadcastable"
    );
    using type =
        conditional_t<
        size<M>::value == 0,
            A,
        conditional_t<
        size<M>::value == size<I>::value,
            set_from<
                set_from<A, head<I>, head<M>>,
                erase_head<I>, erase_head<M>
            >,
        // Otherwise size<M> == 1
            set_from<
                set_from<A, head<I>, M>,
                erase_head<I>, M
            >
        >
    >;
};

template <class A, ptrdiff_t I, class M>
struct _set_from<A, cptrdiff<I>, M>
{
    // Do not static_assert here so that we can use a "wrong" type in
    // the second branch of the conditional above (we won't ever enter
    // the branch when it's wrong).
    using type = cat<head<A, I>, M, erase_head<A, I+1>>;
};

/* ------------------------------------------------------------------ *
 * set_head / set_tail                                                *
 * ------------------------------------------------------------------ */

// HEAD
template <class A, class M0, class... M>
struct _set_head<A, M0, M...>
{
private:
    static constexpr auto start = size<M0>::value;
    static constexpr auto stop  = start + sum_sizes<M...>::value;
    using slice = simple_slice<start, stop>;
    using head  = set_head<A, M0>;
public:
    using type  = set_from<head, slice, M...>;
};

template <class A, class M>
struct _set_head<A, M>{
private:
    static constexpr auto stop = size<M>::value;
    using slice = simple_slice<0, stop>;
public:
    using type  = set_from<A, slice, M>;
};

template <class A>
struct _set_head<A> {
  using type = A;
};

// TAIL
template <class A, class M0, class... M>
struct _set_tail<A, M0, M...>
{
private:
    static constexpr auto length = size<A>::value;
    static constexpr auto stop   = sum_sizes<M...>::value;
    static constexpr auto start  = stop + size<M0>::value;
    using size0 = size<M0>;
    using slice = simple_slice<length-stop, length-start>;
    using tail  = set_tail<A, M...>;
public:
    using type  = set_from<tail, slice, M0>;
};

template <class A, class M>
struct _set_tail<A,M>{
private:
    static constexpr auto sizeA = size<A>::value;
    static constexpr auto sizeM = size<M>::value;
    using slice = simple_slice<sizeA-sizeM, sizeA>;
public:
    using type  = set_from<A, slice, M>;
};

template <class A>
struct _set_tail<A>
{
  using type = A;
};

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif // TNY__STATIX__PACKAPI_IMPL2
