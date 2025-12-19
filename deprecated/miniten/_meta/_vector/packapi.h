/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file implements the pack API for vectors                           **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef MINITEN__META__VECTOR_PACKAPI
#define MINITEN__META__VECTOR_PACKAPI
#include <miniten/_core/defines.h>
#include <miniten/_meta/_vector/decl.h>
#include <miniten/_meta/_packapi/decl.h>
#include <miniten/_meta/_pack/decl.h>       // Pack
#include <miniten/_meta/traits.h>           // IsSame

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

/* ------------------------------------------------------------------ *
 *     Helpers forward decl                                           *
 * ------------------------------------------------------------------ */

NAMESPACE_BEGIN(_vector)
template <class VECTOR, class T> struct _Cast;
template <class VECTOR, class T> using   Cast = typename _Cast<VECTOR, T>::Type;
template <class VECTOR>          struct _AsPack;
template <class VECTOR>          using   AsPack = typename _AsPack<VECTOR>::Type;
NAMESPACE_END(_vector)

/* ------------------------------------------------------------------ *
 *     Pack API                                                       *
 * ------------------------------------------------------------------ */

/* --- Cat ---------------------------------------------------------- */

template <class RIGHT, class T>
struct _Cat2<Vector<T>, RIGHT> {
    static_assert(IsVector<RIGHT>::Value, "L/R should be vectors");
    static_assert(IsSame<ItemType<RIGHT>,T>::Value, "L/R should have same item type");
    using Type = RIGHT;
};

template <class T, T X0, T... X>
struct _Cat2<Vector<T,X0>, Vector<T, X...>> {
    using Type = Vector<T,X0,X...>;
};

template <class RIGHT, class T, T X0, T... X>
struct _Cat2<Vector<T,X0,X...>, RIGHT> {
    using Type = Cat<Vector<T,X0>, Cat<Vector<T,X...>, RIGHT>>;
};

/* --- Others ------------------------------------------------------- */

template <class T, T... X>          struct _ItemType<Vector<T, X...>>           { using Type = T; };
template <class T, T X0, T... X>    struct _GetFirstValue<Vector<T, X0, X...>>  { using Type = Vector<T, X0>; };
template <class T, T... X>          struct _IsVector<Vector<T, X...>>           { using Type = True; };
template <class T, class U, T... X> struct _AsVector<Vector<T, X...>, U>        { using Type = _vector::Cast<Vector<T, X...>, U>; };
template <class T, T... X>          struct _AsPack<Vector<T, X...>>             { using Type = _vector::AsPack<Vector<T, X...>>; };
template <class T, T... X, class M> struct _LikeFrom<Vector<T, X...>, M>        { using Type = AsVector<M, T>; };

/* ------------------------------------------------------------------ *
 *     Helpers implementation                                         *
 * ------------------------------------------------------------------ */

NAMESPACE_BEGIN(_vector)

/* --- Cast --------------------------------------------------------- */

template <class VECTOR, typename U>
struct _Cast {};

template <typename T, typename U, T X0, T... X>
struct _Cast<Vector<T, X0, X...>, U>
{
    using Type = Cat<
        Vector<U, static_cast<U>(X0)>,
        Cast<Vector<T, X...>, U>
    >;
};

template <typename T, typename U, T X0>
struct _Cast<Vector<T, X0>, U>
{
    using Type = Vector<U, static_cast<U>(X0)>;
};

template <typename T, typename U>
struct _Cast<Vector<T>, U>
{
    using Type = Vector<U>;
};

/* --- AsPack ------------------------------------------------------- */

template <class VECTOR>
struct _AsPack {};

template <class T, T X0, T... X>
struct _AsPack<Vector<T, X0, X...>>
{
    using Type = Cat<
        Pack<Vector<T, X0>>,
        AsPack<Vector<T, X...>>
    >;
};

template <class T, T X0>
struct _AsPack<Vector<T, X0>>
{
    using Type = Pack<Vector<T, X0>>;
};

template <class T>
struct _AsPack<Vector<T>>
{
    using Type = Pack<>;
};

NAMESPACE_END(_vector)

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif /// MINITEN__META__VECTOR_PACKAPI
