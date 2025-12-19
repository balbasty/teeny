#ifndef MINITEN__TUPLE_PACKAPI
#define MINITEN__TUPLE_PACKAPI
#include <miniten/core.h>
#include <miniten/disp.h>
#include <miniten/meta.h>
#include <miniten/_tuple/decl.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

/* ------------------------------------------------------------------ *
 *     Pack API                                                       *
 * ------------------------------------------------------------------ */

template <class RIGHT>
struct _Cat2<Tuple<>, RIGHT> {
    static_assert(IsTuple<RIGHT>::Value, "L/R must be tuples");
    using Type = RIGHT;
};

template <class X0, class... X>
struct _Cat2<Tuple<X0>, Tuple<X...>> {
    using Type = Tuple<X0,X...>;
};

template <class RIGHT, class X0, class... X>
struct _Cat2<Tuple<X0,X...>, RIGHT> {
    static_assert(IsTuple<RIGHT>::Value, "L/R must be tuples");
    using Type = Cat<Tuple<X0>, Cat<Tuple<X...>, RIGHT>>;
};

template <class... X>               struct _IsTuple<  Tuple<X...>>      { using Type = True; };
template <class... X>               struct _AsPack<   Tuple<X...>>      { using Type = Pack<X...>; };
template <class M, class... X>      struct _LikeFrom< Tuple<X...>, M>   { using Type = AsTuple<M>; };

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif /// MINITEN__TUPLE_PACKAPI
