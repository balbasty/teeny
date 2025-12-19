/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file defines a compile-time API for accessing and                  **
 ** modifying pack-like meta types (pack, tuple, carray).                   **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef TNY__STATIX__PACKAPI_IMPL1
#define TNY__STATIX__PACKAPI_IMPL1
#include <cuda/std/type_traits>
#include <teeny/_core/defines.h>
#include <teeny/_statix/_packapi/decl.h>
#include <teeny/_statix/_pack/decl.h>   // pack
#include <teeny/_statix/_index/decl.h>  // wrap_index, simple_slice

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

using cuda::std::remove_cvref_t;

/* ------------------------------------------------------------------ */

#define _TNY_PACKAPI_FORWARD(NAME) \
    template <class A> struct NAME   <         A* >  { using type =          typename NAME<A>::type  *; }; \
    template <class A> struct NAME   <         A& >  { using type =          typename NAME<A>::type  &; }; \
    template <class A> struct NAME   <         A&&>  { using type =          typename NAME<A>::type &&; }; \
    template <class A> struct NAME   <const    A  >  { using type =    const typename NAME<A>::type   ; }; \
    template <class A> struct NAME   <volatile A  >  { using type = volatile typename NAME<A>::type   ; };

#define _TNY_PACKAPI_FORWARD2(NAME) \
    template <class A, class M> struct NAME   <         A* , M>  { using type =          typename NAME<A,M>::type  *; }; \
    template <class A, class M> struct NAME   <         A& , M>  { using type =          typename NAME<A,M>::type  &; }; \
    template <class A, class M> struct NAME   <         A&&, M>  { using type =          typename NAME<A,M>::type &&; }; \
    template <class A, class M> struct NAME   <const    A  , M>  { using type =    const typename NAME<A,M>::type   ; }; \
    template <class A, class M> struct NAME   <volatile A  , M>  { using type = volatile typename NAME<A,M>::type   ; };

#define _TNY_PACKAPI_DONT_FORWARD(NAME) \
    template <class A> struct NAME   <         A* >  { using type = typename NAME<A>::type; }; \
    template <class A> struct NAME   <         A& >  { using type = typename NAME<A>::type; }; \
    template <class A> struct NAME   <         A&&>  { using type = typename NAME<A>::type; }; \
    template <class A> struct NAME   <const    A  >  { using type = typename NAME<A>::type; }; \
    template <class A> struct NAME   <volatile A  >  { using type = typename NAME<A>::type; };

/* ------------------------------------------------------------------ *
 * Convert
 * ------------------------------------------------------------------ */

template <class A>           struct _as_pack     { static_assert(false, "as_pack not implemented for this type"); };
template <class A>           struct _as_tuple    { using type = as_tuple<as_pack<A>>; };
template <class A, class T>  struct _as_carray   { using type = as_carray<as_pack<A>,T>; };
template <class A>           struct _value_type  { using type = value_type<remove_cvref_t<front<A>>>; };

_TNY_PACKAPI_FORWARD(_as_pack)
_TNY_PACKAPI_FORWARD(_as_tuple)
_TNY_PACKAPI_FORWARD2(_as_carray)

/* ------------------------------------------------------------------ *
 * Construct
 * ------------------------------------------------------------------ */

template <class A, class M>  struct _like_from   { using type = like_from<A, as_pack<M>>; };
template <class A>           struct _empty_like  { using type = like_from<A,pack<>>; };
template <class A>           struct _reversed    { using type = like_from<A,reversed<as_pack<A>>>; };

_TNY_PACKAPI_FORWARD2(_like_from)
_TNY_PACKAPI_FORWARD(_empty_like)
_TNY_PACKAPI_FORWARD(_reversed)

/* ------------------------------------------------------------------ *
 * Cat
 * ------------------------------------------------------------------ */

template <class A, class M> struct _cat2 { using type = like_from<A, cat<as_pack<A>, as_pack<M>>>; }; // MUST BE IMPLEMENTED BY PACK
template <class... M>       struct _cat  { using type = pack<>; }; // Fully implemented in impl2

/* ------------------------------------------------------------------ *
 * Size
 * ------------------------------------------------------------------ */

template <class... A>           struct _size               { static_assert(false, "fully specialized"); };
template <>                     struct _size<>             { using type = csize<>; };
template <class A>              struct _size<A>            { using type = size<as_pack<A>>; };
template <class A, class... B>  struct _size<A, B...>      { using type = cat<size<A>, size<B...>>; };

_TNY_PACKAPI_DONT_FORWARD(_size)

template <class... A>           struct _sum_sizes          { static_assert(false, "specialized"); };
template <>                     struct _sum_sizes<>        { using type = csize<0>; };
template <class A, class... B>  struct _sum_sizes<A, B...> { using type = csize<size<A>::value + sum_sizes<B...>::value>; };

/* ------------------------------------------------------------------ *
 * Get
 * ------------------------------------------------------------------ */

template <class A>              struct _head            { using type = like_from<A,head<as_pack<A>>>; }; // MUST BE IMPLEMENTED IN PACK
template <class A, class I>     struct _get             { static_assert(false, "Index must be a cptrdiff"); }; // Fully implemented in impl2

template <class A>              struct _front           { using type = front<as_pack<A>>; }; // MUST BE IMPLEMENTED IN PACK
template <class A, class I>     struct _at              { static_assert(false, "Index must be a cptrdiff"); }; // Fully implemented impl2

_TNY_PACKAPI_FORWARD(_head)
_TNY_PACKAPI_DONT_FORWARD(_front)

/* ------------------------------------------------------------------ *
 * Delete
 * ------------------------------------------------------------------ */

template <class A>              struct _erase_head           { using type = like_from<A,erase_head<as_pack<A>>>; };
template <class A, class I>     struct _erase                { static_assert(false, "Index must be a cptrdiff"); }; // Fully implemented in impl2

_TNY_PACKAPI_FORWARD(_erase_head)

/* ------------------------------------------------------------------ *
 * Insert
 * ------------------------------------------------------------------ */

template <class A, class I, class... M>  struct _insert { static_assert(size<I>::value == 1, "Insert does not support vector of indices"); }; // Fully implemented in impl2

/* ------------------------------------------------------------------ *
 * Assign
 * ------------------------------------------------------------------ */

template <class A, class I, class... M>     struct _set_from    { static_assert(false, "Index must be a cptrdiff"); }; // Fully implemented in impl2
template <class A, class... M>              struct _set_head    { static_assert(false, "Index must be a cptrdiff"); }; // Fully implemented in impl2
template <class A, class... M>              struct _set_tail    { static_assert(false, "Index must be a cptrdiff"); }; // Fully implemented in impl2

/* ------------------------------------------------------------------ *
 * Apply modifiers
 * ------------------------------------------------------------------ */


template <class A> struct _apply_add_const              { using type = like_from<A, apply_add_const            <as_pack<A>>>; };
template <class A> struct _apply_add_volatile           { using type = like_from<A, apply_add_volatile         <as_pack<A>>>; };
template <class A> struct _apply_add_cv                 { using type = like_from<A, apply_add_cv               <as_pack<A>>>; };
template <class A> struct _apply_add_lvalue_reference   { using type = like_from<A, apply_add_lvalue_reference <as_pack<A>>>; };
template <class A> struct _apply_add_rvalue_reference   { using type = like_from<A, apply_add_rvalue_reference <as_pack<A>>>; };
template <class A> struct _apply_add_pointer            { using type = like_from<A, apply_add_pointer          <as_pack<A>>>; };
template <class A> struct _apply_remove_const           { using type = like_from<A, apply_remove_const         <as_pack<A>>>; };
template <class A> struct _apply_remove_volatile        { using type = like_from<A, apply_remove_volatile      <as_pack<A>>>; };
template <class A> struct _apply_remove_cv              { using type = like_from<A, apply_remove_cv            <as_pack<A>>>; };
template <class A> struct _apply_remove_reference       { using type = like_from<A, apply_remove_reference     <as_pack<A>>>; };
template <class A> struct _apply_remove_pointer         { using type = like_from<A, apply_remove_pointer       <as_pack<A>>>; };
template <class A> struct _apply_remove_cvref           { using type = like_from<A, apply_remove_cvref         <as_pack<A>>>; };
template <class A> struct _apply_decay                  { using type = like_from<A, apply_decay                <as_pack<A>>>; };
template <class A> struct _apply_sizeof                 { using type =              apply_sizeof               <as_pack<A>>; };

/* ------------------------------------------------------------------ *
 * Test
 * ------------------------------------------------------------------ */

template <class A> struct _is_carray { using type = cfalse; };
template <class A> struct _is_tuple  { using type = cfalse; };
template <class A> struct _is_pack   { using type = cfalse; };

_TNY_PACKAPI_DONT_FORWARD(_is_carray)
_TNY_PACKAPI_DONT_FORWARD(_is_tuple)
_TNY_PACKAPI_DONT_FORWARD(_is_pack)

/* ------------------------------------------------------------------ */

#undef _TNY_PACKAPI_FORWARD
#undef _TNY_PACKAPI_FORWARD2
#undef _TNY_PACKAPI_DONT_FORWARD

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif // TNY__STATIX__PACKAPI_IMPL1
