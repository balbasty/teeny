/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file implements the pack API for vectors                           **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef TNY__STATIX__CARRAY_PACKAPI
#define TNY__STATIX__CARRAY_PACKAPI
#include <cuda/std/type_traits>
#include <teeny/_core/defines.h>
#include <teeny/_statix/_carray/decl.h>
#include <teeny/_statix/_packapi/decl.h>
#include <teeny/_statix/_pack/decl.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

using cuda::std::is_same;

/* ------------------------------------------------------------------ *
 *     Pack API                                                       *
 * ------------------------------------------------------------------ */

/* --- cat ---------------------------------------------------------- */

template <class RIGHT, class T>
struct _cat2<carray<T>, RIGHT> {
    static_assert(is_carray<RIGHT>::value,             "L/R should be vectors");
    static_assert(is_same<value_type<RIGHT>,T>::value, "L/R should have same item type");
    using type = RIGHT;
};

template <class T, T X0, T... X>
struct _cat2<carray<T,X0>, carray<T, X...>> {
    using type = carray<T,X0,X...>;
};

template <class RIGHT, class T, T X0, T... X>
struct _cat2<carray<T,X0,X...>, RIGHT> {
    using type = cat<carray<T,X0>, cat<carray<T,X...>, RIGHT>>;
};

/* --- others ------------------------------------------------------- */

template <class T, T... X>          struct _value_type  <carray<T, X...>>     { using type = T; };
template <class T, T X0, T... X>    struct _head        <carray<T, X0, X...>> { using type = carray<T, X0>; };
template <class T>                  struct _head        <carray<T>>           { using type = carray<T>; };
template <class T, T X0, T... X>    struct _front       <carray<T, X0, X...>> { using type = carray<T, X0>; };
template <class T>                  struct _front       <carray<T>>           { using type = carray<T>; };
template <class T, T... X>          struct _is_carray   <carray<T, X...>>     { using type = ctrue; };
template <class T, T... X, class M> struct _like_from   <carray<T, X...>, M>  { using type = as_carray<M, T>; };

/* --- as_carray ---------------------------------------------------- */

template <typename T, typename U, T X0, T... X>
struct _as_carray<carray<T, X0, X...>, U>
{
    using type = cat<
        carray<U, static_cast<U>(X0)>,
        as_carray<carray<T, X...>, U>
    >;
};

template <typename T, typename U, T X0>
struct _as_carray<carray<T, X0>, U>
{
    using type = carray<U, static_cast<U>(X0)>;
};

template <typename T, typename U>
struct _as_carray<carray<T>, U>
{
    using type = carray<U>;
};

/* --- as_pack ------------------------------------------------------ */

template <class T, T X0, T... X>
struct _as_pack<carray<T, X0, X...>> {
    using type = cat<
        pack<carray<T, X0>>,
        as_pack<carray<T, X...>>
    >;
};

template <class T, T X0>
struct _as_pack<carray<T, X0>>
{
    using type = pack<carray<T, X0>>;
};

template <class T>
struct _as_pack<carray<T>>
{
    using type = pack<carray<T>>;
};

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif /// TNY__STATIX__CARRAY_PACKAPI
