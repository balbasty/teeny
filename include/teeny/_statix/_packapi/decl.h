/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file declares a compile-time API for accessing and                 **
 ** modifying pack-like meta types (pack, tuple, carray).                   **
 **                                                                         **
 ** value_type<X>           The type of elements in a carray                **
 ** size<P>                 Size of the pack [csize]                        **
 ** empty_like<P>           An empty version of P                           **
 ** reversed<P>             Reversed version of P                           **
 ** like_from<P, M>         Same metatype as P with the content of M        **
 ** like<P, M...>           Same metatype as P with M... as content         **
 **                                                                         **
 ** as_carray<P,[T]>        Convert to a carray with same content           **
 ** as_tuple<P>             Convert to a tuple with same content            **
 ** as_pack<P>              Convert to a pack with same content             **
 **                                                                         **
 ** is_carray<P>            ctrue if P is a carray                          **
 ** is_tuple<P>             ctrue if P is a tuple                           **
 ** is_pack<P>              ctrue if P is a pack                            **
 **                                                                         **
 ** get<P,I>                Get elements at I                               **
 ** get_index<P,I>          Same as get, but I is a concrete integral       **
 ** head<P,N=1>             Get first N elements                            **
 ** tail<P,N=1>             Get last N elements                             **
 **                                                                         **
 ** at<P,I>                 Get element at I                                **
 ** at_index<P,I>           Same as Get, but I is a concrete integral       **
 ** front<P>                Get first element                               **
 ** back<P>                 Get last element                                **
 **                                                                         **
 ** erase<P,I>              Erase elements at I                             **
 ** erase_index<P,I>        Same as erase, but I is a concrete integral     **
 ** erase_head<P,N=1>       Erase first N elements                          **
 ** erase_tail<P,N=1>       Erase last N elements                           **
 **                                                                         **
 ** set_from<P,I,M...>      Set elements at I, copied from cat<M...>        **
 ** set_head<P,M...>        Set first N elements, copied from cat<M...>     **
 ** set_tail<P,M...>        Set last N elements, copied from cat<M...>      **
 **                                                                         **
 ** set<P,I,M...>           Set elements at I, copied from pack<M...>       **
 ** set_index<P,I,M>        Same as set, but I is a concrete integral       **
 ** set_front<P,M...>       Set first N elements, copied from pack<M...>    **
 ** set_back<P,M...>        Set last N elements, copied from pack<M...>     **
 **                                                                         **
 ** insert<P,I,M...>        Insert elements at I, copied from cat<M...>     **
 ** insert_index<P,I,M>     Same as insert, but I is an integral            **
 ** prextend<P,M...>        Same as cat<empty_like<P>, M..., P>             **
 ** extend<P,M...>          Same as cat<P, M...>                            **
 **                                                                         **
 ** insert_values<P,I,M...> Insert elements at I, copied from pack<M...>    **
 ** insert_index<P,I,M...>  Same as insert_values, but I is an integral     **
 ** prepend<P,M...>         Same as cat<empty_like<P>, pack<M...>, P>       **
 ** append<P,M...>          Same as cat<P, pack<M...>>                      **
 **                                                                         **
 ** cat<P, M...>            Same as extend<P, M...>                         **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 ** Notes                                                                   **
 ** -----                                                                   **
 ** {get,head,tail} return the same metatype as the input.                  **
 ** That is, `head<tuple<int, float, bool>> = tuple<int>`.                  **
 ** This is to accomodate vectors of indices, e.g.,                         **
 ** `get<tuple<int, float, bool>, clong<0, 1>> = tuple<int, float>`.        **
 ** It also provides a consistent behaviour between vectors and tuples.     **
 **                                                                         **
 ** In contrast, {at,front,back} return the indexed element in              **
 ** the case of tuple and pack. However, it still returns a single-element. **
 ** array in the case of carray.                                            **
 ** The actual value can then be accessed from its `::value` field.         **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef TNY__STATIX__PACKAPI_DECL
#define TNY__STATIX__PACKAPI_DECL
#include <cuda/std/type_traits>
#include <teeny/_core/defines.h>
#include <teeny/_statix/_pack/decl.h>   // pack
#include <teeny/_statix/_index/decl.h>  // wrap_index, simple_slice

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

using _PDT = ptrdiff_t;
using _SZT = size_t;

/* ------------------------------------------------------------------ *
 * Convert
 * ------------------------------------------------------------------ *
 * Define as_pack first so we can use it to implement fallbacks that
 * go to/from pack.
 * ------------------------------------------------------------------ */

template <class A, class T>   struct _as_carray;
template <class A>            struct _as_tuple;
template <class A>            struct _as_pack;
template <class A>            struct _value_type;

/**
 * @brief Get the type of elements in a carray-like type A
 *
 * @tparam A   Input type
 *
 * @example value_type<cint<1, 2, 3>> = int
 * @example value_type<cfalse>        = bool
 */
template <class A>
using value_type = typename _value_type<A>::type;

/**
 * @brief Convert a pack-like type to a pack
 *
 * @tparam A   Input type
 *
 * @example as_pack<tuple<int, float>> = pack<int, float>
 * @example as_pack<cint<1, 2, 3>>     = pack<cint<1>, cint<2>, cint<3>>
 */
template <class A>
using as_pack   = typename _as_pack<A>::type;

/**
 * @brief Convert a pack-like type to a tuple
 *
 * @tparam A   Input type
 *
 * @example as_tuple<pack<int, float>> = tuple<int, float>
 * @example as_tuple<cint<1, 2, 3>>    = tuple<cint<1>, cint<2>, cint<3>>
 */
template <class A>
using as_tuple  = typename _as_tuple<A>::type;

/**
 * @brief Convert a pack-like type to a carray
 *
 * @tparam A   Input type
 * @tparam T   Element type (defaults to value_type<A>)
 *
 * @example as_carray<pack<cint<1>, cint<2>, cint<3>>> = cint<1, 2, 3>
 * @example as_carray<cint<1, 2, 3>, long>             = clong<1, 2, 3>
 */
template <class A, class T = value_type<A>>
using as_carray = typename _as_carray<A,T>::type;

/* ------------------------------------------------------------------ *
 * Construct
 * ------------------------------------------------------------------ */

template <class A, class M>     struct _like_from;
template <class A>              struct _empty_like;
template <class A>              struct _reversed;

/**
 * @brief Construct a pack-like type with the same metatype as A
 *        but with the content of M
 *
 * @tparam A   Input type (provides the metatype)
 * @tparam M   Content type
 *
 * @example like_from<tuple<int, float>, pack<bool, double>> = tuple<bool, double>
 * @example like_from<cint<1, 2>, pack<clong<3>, clong<4>>>  = cint<3, 4>
 */
template <class A, class M>
using like_from = typename _like_from<A,M>::type;

/**
 * @brief Construct a pack-like type with the same metatype as A
 *        but with M... as content
 *
 * @tparam A     Input type (provides the metatype)
 * @tparam M...  Content types
 *
 * @example like<tuple<int, float>, bool, double> = tuple<bool, double>
 * @example like<cint<1, 2>, clong<3>, clong<4>>  = cint<3, 4>
 */
template <class A, class... M>
using like = like_from<A, pack<M...>>;

/**
 * @brief Construct an empty pack-like type with the same metatype as A
 *
 * @tparam A   Input type (provides the metatype)
 *
 * @example empty_like<tuple<int, float>> = tuple<>
 * @example empty_like<cint<1, 2, 3>>     = cint<>
 */
template <class A>
using empty_like = typename _empty_like<A>::type;

/**
 * @brief Construct a pack-like type with the same metatype as A
 *        but with the content reversed
 *
 * @tparam A   Input type
 *
 * @example reversed<tuple<int, float, bool>> = tuple<bool, float, int>
 * @example reversed<cint<1, 2, 3>>           = cint<3, 2, 1>
 */
template <class A>
using reversed = typename _reversed<A>::type;

/* ------------------------------------------------------------------ *
 * Cat
 * ------------------------------------------------------------------ */

template <class A, class M> struct _cat2;; // MUST BE IMPLEMENTED BY EACH PACK-LIKE TYPE
template <class... M>       struct _cat;

/**
 * @brief Concatenate multiple pack-like types into one
 *
 * @tparam M...   Input types to concatenate
 *
 * @example cat<tuple<int>, tuple<float, bool>> = tuple<int, float, bool>
 * @example cat<cint<1, 2>, cint<3>>             = cint<1, 2, 3>
 */
template <class... M>
using cat = typename _cat<M...>::type;

/* ------------------------------------------------------------------ *
 * Size
 * ------------------------------------------------------------------ */

template <class... A> struct _size;
template <class... A> struct _sum_sizes;

/**
 * @brief Get the size of a pack-like type (or multiple types)
 *
 * @tparam A...   Input type(s)
 *
 * @example size<tuple<int, float, bool>> = csize<3>
 * @example size<cint<1, 2, 3, 4>>        = csize<4>
 * @example size<cint<1, 2>, cint<3, 4>>  = csize<2, 2>
 */
template <class... A>
using size = typename _size<A...>::type;

/**
 * @brief Get the sum of sizes of multiple pack-like types
 *
 * @tparam A...   Input type(s)
 *
 * @example sum_sizes<tuple<int, float>, tuple<bool>> = csize<3>
 * @example sum_sizes<cint<1, 2>, cint<3, 4>>         = csize<4>
 */
template <class... A>
using sum_sizes = typename _sum_sizes<A...>::type;

/* ------------------------------------------------------------------ *
 * Get
 * ------------------------------------------------------------------ */

template <class A>           struct _head;
template <class A, class I>  struct _get;
template <class A>           struct _front;
template <class A, class I>  struct _at;
template <class A, class I>  using  _get_wrap = typename _get<A, wrap_index<size<A>, I>>::type;

/**
 * @brief Get elements at indices I from pack-like type A
 *
 * @tparam A   Input type
 * @tparam I   Indices to get
 *
 * @example get<tuple<int, float, bool>, cptrdiff<0, 2>> = tuple<int, bool>
 * @example get<cint<1, 2, 3>, cptrdiff<1>>              = cint<2>
 */
template <class A, class I>
using get       = _get_wrap<A, I>;

/**
 * @brief Get elements at concrete integral indices I from pack-like type A
 *
 * get_index<A, I...>
 *
 * @tparam A   Input type
 * @tparam I   Concrete integral indices to get
 *
 * @example get_index<tuple<int, float, bool>, 0, 2> = tuple<int, bool>
 * @example get_index<cint<1, 2, 3>, 1>              = cint<2>
 */
template <class A, _PDT... I>
using get_index = get<A, cptrdiff<I...>>;

/**
 * @brief Get first N elements from pack-like type A
 *
 * @tparam A   Input type
 * @tparam N   Number of elements to get (default: 1)
 *
 * @example head<tuple<int, float, bool>, 2> = tuple<int, float>
 * @example head<cint<1, 2, 3>, 2>            = cint<1, 2>
 */
template <class A, _SZT N=1>
using head      = get<A, simple_slice<0, N>>;

/**
 * @brief Get last N elements from pack-like type A
 *
 * @tparam A   Input type
 * @tparam N   Number of elements to get (default: 1)
 *
 * @example tail<tuple<int, float, bool>, 2> = tuple<float, bool>
 * @example tail<cint<1, 2, 3>, 2>           = cint<2, 3>
 */
template <class A, _SZT N=1>
using tail      = get<A, simple_slice<size<A>::value-N, size<A>::value>>;

/**
 * @brief Get element at index I from pack-like type A
 *
 * @tparam A   Input type
 * @tparam I   Index to get
 *
 * @example at<tuple<int, float, bool>, cptrdiff<1>> = float
 * @example at<cint<1, 2, 3>, cptrdiff<0>>           = cint<1>
 */
template <class A, class I>
using at        = typename _at<A, wrap_index<size<A>, I>>::type;

/**
 * @brief Get element at concrete integral index I from pack-like type A
 *
 * @tparam A   Input type
 * @tparam I   Concrete integral index to get
 *
 * @example at_index<tuple<int, float, bool>, 1> = float
 * @example at_index<cint<1, 2, 3>, 0>           = cint<1>
 */
template <class A, _PDT I>
using at_index  = at<A, cptrdiff<I>>;

/**
 * @brief Get first element from pack-like type A
 *
 * @tparam A   Input type
 *
 * @example front<tuple<int, float, bool>> = int
 * @example front<cint<1, 2, 3>>           = cint<1>
 */
template <class A>
using front     = at<A, cptrdiff<0>>;

/**
 * @brief Get last element from pack-like type A
 *
 * @tparam A   Input type
 *
 * @example back<tuple<int, float, bool>> = bool
 * @example back<cint<1, 2, 3>>           = cint<3>
 */
template <class A>
using back      = at<A, cptrdiff<-1>>;

/* ------------------------------------------------------------------ *
 * Delete
 * ------------------------------------------------------------------ */

template <class A>           struct _erase_head;
template <class A, class I>  struct _erase;
template <class A, class I>  using  _erase_wrap = typename _erase<A, wrap_index<size<A>, I>>::type;

/**
 * @brief Erase elements at indices I from pack-like type A
 *
 * @tparam A   Input type
 * @tparam I   Indices to erase
 *
 * @example erase<tuple<int, float, bool>, cptrdiff<0>> = tuple<float, bool>
 * @example erase<cint<1, 2, 3>, cptrdiff<1, 2>>        = cint<1>
 */
template <class A, class I>
using erase        = _erase_wrap<A, I>;

/**
 * @brief Erase elements at concrete integral indices I from pack-like type A
 *
 * @tparam A   Input type
 * @tparam I   Concrete integral indices to erase
 *
 * @example erase_index<tuple<int, float, bool>, 0> = tuple<float, bool>
 * @example erase_index<cint<1, 2, 3>, 1, 2>        = cint<1>
 */
template <class A, _PDT I>
using erase_index  = erase<A, cptrdiff<I>>;

/**
 * @brief Erase first N elements from pack-like type A
 *
 * @tparam A   Input type
 * @tparam N   Number of elements to erase (default: 1)
 *
 * @example erase_head<tuple<int, float, bool>, 2> = tuple<bool>
 * @example erase_head<cint<1, 2, 3>, 2>           = cint<3>
 */
template <class A, _SZT N=1>
using erase_head   = erase<A, simple_slice<0, N>>;

/**
 * @brief Erase last N elements from pack-like type A
 *
 * @tparam A   Input type
 * @tparam N   Number of elements to erase (default: 1)
 *
 * @example erase_tail<tuple<int, float, bool>, 2> = tuple<int>
 * @example erase_tail<cint<1, 2, 3>, 2>           = cint<1>
 */
template <class A, _SZT N=1>
using erase_tail   = erase<A, simple_slice<size<A>::value-N, size<A>::value>>;

/* ------------------------------------------------------------------ *
 * Insert
 * ------------------------------------------------------------------ */

/*
 * NOTE
 *      For append/prepend, I used to fallback to InsertFrom, but
 *      got some weird bugs in weird places. I now fallback to Cat
 *      and it seems to have solved it.
 */

template <class A, class I, class... M> struct _insert;
template <class A, class I, class... M> using  _insert_wrap  = typename _insert<A, wrap_index<size<A>, I>, M...>::type;

/**
 * @brief Insert elements at indices I into pack-like type A,
 *        copied from cat<M...>
 *
 * @tparam A    Input type
 * @tparam I    Indices to insert at
 * @tparam M... Elements to insert (as types)
 *
 * @example insert<tuple<int, bool>, cptrdiff<1>, tuple<float>> = tuple<int, float, bool>
 * @example insert<cint<1, 3>, cptrdiff<1>, cint<2>>            = cint<1, 2, 3>
 */
template <class A, class I, class... M>
using insert             = _insert_wrap<A, I, M...>;

/**
 * @brief Insert elements at concrete integral indices I into pack-like type A,
 *        copied from cat<M...>
 *
 * @tparam A    Input type
 * @tparam I    Concrete integral indices to insert at
 * @tparam M... Elements to insert (as types)
 *
 * @example insert_index_from<tuple<int, bool>, 1, tuple<float>> = tuple<int, float, bool>
 * @example insert_index_from<cint<1, 3>, 1, cint<2>>            = cint<1, 2, 3>
 */
template <class A, _PDT   I, class... M>
using insert_index        = insert<A, cptrdiff<I>, M...>;

/**
 * @brief Prepend elements to pack-like type A,
 *        copied from cat<M...>
 *
 * @tparam A    Input type
 * @tparam M... Elements to prepend (as types)
 *
 * @example prextend<tuple<float, bool>, tuple<int>> = tuple<int, float, bool>
 * @example prextend<cint<2, 3>, cint<1>>            = cint<1, 2, 3>
 */
template <class A, class... M>
using prextend     = cat<empty_like<A>, cat<M...>, A>;
/**
 * @brief Append elements to pack-like type A,
 *        copied from cat<M...>
 *
 * @tparam A    Input type
 * @tparam M... Elements to append (as types)
 *
 * @example extend<tuple<int, float>, tuple<bool>> = tuple<int, float, bool>
 * @example extend<cint<1, 2>, cint<3>>            = cint<1, 2, 3>
 */
template <class A, class... M>
using extend              = cat<A, M...>;

/**
 * @brief Insert elements at indices I into pack-like type A,
 *        copied from pack<M...>
 *
 * @tparam A    Input type
 * @tparam I    Indices to insert at
 * @tparam M... Elements to insert (as types)
 *
 * @example insert_values<tuple<int, bool>, cptrdiff<1>, float> = tuple<int, float, bool>
 * @example insert_values<cint<1, 3>, cptrdiff<1>, cint<2>>     = cint<1, 2, 3>
 */
template <class A, class I, class... M>
using insert_values       = insert<A, I, pack<M...>>;

/**
 * @brief Insert elements at concrete integral indices I into pack-like type A,
 *        copied from pack<M...>
 *
 * @tparam A    Input type
 * @tparam I    Concrete integral indices to insert at
 * @tparam M... Elements to insert (as types)
 *
 * @example insert_index_values<tuple<int, bool>, 1, pack<float>> = tuple<int, float, bool>
 * @example insert_index_values<cint<1, 3>, 1, pack<c
 */
template <class A, _PDT   I, class... M>
using insert_index_values = insert<A, cptrdiff<I>, pack<M...>>;

/**
 * @brief Prepend elements to pack-like type A,
 *        copied from pack<M...>
 *
 * @tparam A    Input type
 * @tparam M... Elements to prepend (as types)
 *
 * @example prepend<tuple<float, bool>, float> = tuple<float, float, bool>
 * @example prepend<cint<2, 3>, cint<1>>        = cint<1, 2, 3>
 */
template <class A, class... M>
using prepend             = cat<like<A, M...>, A>;

/**
 * @brief Append elements to pack-like type A,
 *        copied from pack<M...>
 *
 * @tparam A    Input type
 * @tparam M... Elements to append (as types)
 *
 * @example append<tuple<int, float>, bool> = tuple<int, float, bool>
 * @example append<cint<1, 2>, cint<3>>      = cint<1, 2, 3>
 */
template <class A, class... M>
using append          = cat<A, pack<M...>>;

/* ------------------------------------------------------------------ *
 * Assign
 * ------------------------------------------------------------------ */

template <class A, class I, class... M>  struct _set_from;
template <class A, class I, class... M>  using  _set_from_wrap = typename _set_from<A, wrap_index<size<A>, I>, M...>::type;
template <class A, class... M>           struct _set_head;
template <class A, class... M>           struct _set_tail;

/**
 * @brief Set elements at indices I in pack-like type A,
 *        copied from cat<M...>
 *
 * @tparam A    Input type
 * @tparam I    Indices to set
 * @tparam M... Elements to set (as types)
 *
 * @example set_from<tuple<int, float, bool>, cptrdiff<1>, tuple<double>> = tuple<int, double, bool>
 * @example set_from<cint<1, 2, 3>, cptrdiff<0, 2>, cint<4, 5>>           = cint<4, 2, 5>
 */
template <class A, class I, class... M>
using set_from   = _set_from_wrap<A, I, M...>;

/**
 * @brief Set first N elements in pack-like type A,
 *        copied from cat<M...>
 *
 * @tparam A    Input type
 * @tparam M... Elements to set (as types)
 *
 * @example set_head<tuple<int, float, bool>, tuple<double, char>> = tuple<double, char, bool>
 * @example set_head<cint<1, 2, 3>, cint<4, 5>>                    = cint<4, 2, 3>
 */
template <class A, class... M>
using set_head   = typename _set_head<A, M...>::type;

/**
 * @brief Set last N elements in pack-like type A,
 *        copied from cat<M...>
 *
 * @tparam A    Input type
 * @tparam M... Elements to set (as types)
 *
 * @example set_tail<tuple<int, float, bool>, tuple<double, char>> = tuple<int, double, char>
 * @example set_tail<cint<1, 2, 3>, cint<4, 5>>                    = cint<1, 4, 5>
 */
template <class A, class... M>
using set_tail   = typename _set_tail<A, M...>::type;

/**
 * @brief Set elements at indices I in pack-like type A,
 *        copied from pack<M...>
 *
 * @tparam A    Input type
 * @tparam I    Indices to set
 * @tparam M... Elements to set (as types)
 *
 * @example set<tuple<int, float, bool>, cptrdiff<1>, double>    = tuple<int, double, bool>
 * @example set<cint<1, 2, 3>, cptrdiff<0, 2>, cint<4>, cint<5>> = cint<4, 2, 5>
 */
template <class A, class I, class... M>
using set        = set_from<A, I, pack<M... >>;

/**
 * @brief Set elements at concrete integral indices I in pack-like type A,
 *        copied from pack<M...>
 *
 * @tparam A    Input type
 * @tparam I    Concrete integral indices to set
 * @tparam M... Elements to set (as types)
 *
 * @example set_index<tuple<int, float, bool>, 1, double>    = tuple<int, double, bool>
 * @example set_index<cint<1, 2, 3>, 0, 2, cint<4>, cint<5>> = cint<4, 2, 5>
 */
template <class A, _PDT I, class M>
using set_index  = set<A, cptrdiff<I>, M>;

/**
 * @brief Set first N elements in pack-like type A,
 *        copied from pack<M...>
 *
 * @tparam A    Input type
 * @tparam M... Elements to set (as types)
 *
 * @example set_front<tuple<int, float, bool>, double, char> = tuple<double, char, bool>
 * @example set_front<cint<1, 2, 3>, cint<4>, cint<5>>       = cint<4, 2, 3>
 */
template <class A, class... M>
using set_front  = set<A, simple_slice<0, size<pack<M...>>::value>, M...>;

/**
 * @brief Set last N elements in pack-like type A,
 *        copied from pack<M...>
 *
 * @tparam A    Input type
 * @tparam M... Elements to set (as types)
 *
 * @example set_back<tuple<int, float, bool>, double, char> = tuple<int, double, char>
 * @example set_back<cint<1, 2, 3>, cint<4>, cint<5>>       = cint<1, 4, 5>
 */
template <class A, class... M>
using set_back   = set<A, simple_slice<size<A>::value-size<pack<M...>>::value, size<A>::value>, M...>;

/* ------------------------------------------------------------------ *
 * Apply type_traits
 * ------------------------------------------------------------------ */

template <class A> struct _apply_add_const;
template <class A> struct _apply_add_volatile;
template <class A> struct _apply_add_cv;
template <class A> struct _apply_add_lvalue_reference;
template <class A> struct _apply_add_rvalue_reference;
template <class A> struct _apply_add_pointer;
template <class A> struct _apply_remove_const;
template <class A> struct _apply_remove_volatile;
template <class A> struct _apply_remove_cv;
template <class A> struct _apply_remove_reference;
template <class A> struct _apply_remove_pointer;
template <class A> struct _apply_remove_cvref;
template <class A> struct _apply_decay;
template <class A> struct _apply_sizeof;

template <class A> using apply_add_const            = typename _apply_add_const            <A>::type;
template <class A> using apply_add_volatile         = typename _apply_add_volatile         <A>::type;
template <class A> using apply_add_cv               = typename _apply_add_cv               <A>::type;
template <class A> using apply_add_lvalue_reference = typename _apply_add_lvalue_reference <A>::type;
template <class A> using apply_add_rvalue_reference = typename _apply_add_rvalue_reference <A>::type;
template <class A> using apply_add_pointer          = typename _apply_add_pointer          <A>::type;
template <class A> using apply_remove_const         = typename _apply_remove_const         <A>::type;
template <class A> using apply_remove_volatile      = typename _apply_remove_volatile      <A>::type;
template <class A> using apply_remove_cv            = typename _apply_remove_cv            <A>::type;
template <class A> using apply_remove_reference     = typename _apply_remove_reference     <A>::type;
template <class A> using apply_remove_pointer       = typename _apply_remove_pointer       <A>::type;
template <class A> using apply_remove_cvref         = typename _apply_remove_cvref         <A>::type;
template <class A> using apply_decay                = typename _apply_decay                <A>::type;
template <class A> using apply_sizeof               = typename _apply_sizeof               <A>::type;

/* ------------------------------------------------------------------ *
 * Test
 * ------------------------------------------------------------------ */

template <class A> struct _is_carray;
template <class A> struct _is_tuple;
template <class A> struct _is_pack;

template <class A> using is_carray = typename _is_carray <A>::type;
template <class A> using is_tuple  = typename _is_tuple  <A>::type;
template <class A> using is_pack   = typename _is_pack   <A>::type;

/* ------------------------------------------------------------------ *
 * Conversion sugar
 * ------------------------------------------------------------------ */

template <class T> using as_cbool    = as_carray<T, bool>;
template <class T> using as_csize    = as_carray<T, size_t>;
template <class T> using as_cptrdiff = as_carray<T, ptrdiff_t>;

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif // TNY__STATIX__PACKAPI_DECL
