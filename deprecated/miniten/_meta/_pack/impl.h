/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file implements a compile-time "pack of types"                     **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef MINITEN__META__PACK_IMPL
#define MINITEN__META__PACK_IMPL
#include <miniten/_core/defines.h>
#include <miniten/_core/types.h>
#include <miniten/disp.h>
#include <miniten/_meta/traits.h>
#include <miniten/_meta/_math/decl.h>
#include <miniten/_meta/_pack/decl.h>
#include <miniten/_meta/_vector/decl.h>
#include <miniten/_meta/_packapi/decl.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

/* ------------------------------------------------------------------ *
 *     Pack Specialization                                            *
 * ------------------------------------------------------------------ */

struct PackBase {};

/**
 * @brief Pack of types.
 *
 * Pack<T...>
 *
 * @tparam T  Wrapped Type(s).
 */
template <class... X>
struct Pack: public PackBase {
    using ThisType = Pack<X...>;
    static constexpr size_t Length = CountTypes<X...>::Value;
};

template <class X>
struct Pack<X>: public PackBase {
    using Type     = X;
    using ThisType = Pack<X>;
    static constexpr size_t Length = 1;
};

/* ------------------------------------------------------------------ *
 *     Repeat                                                         *
 * ------------------------------------------------------------------ */

template <long N, class T>
struct _NPack {
    using Type = Append<NPack<N-1, T>, T>;
};

template <class T>
struct _NPack<0, T> {
    using Type = Pack<>;
};

/* ------------------------------------------------------------------ *
 *     Pack API                                                       *
 * ------------------------------------------------------------------ */

/* --- Cat ---------------------------------------------------------- */

template <class RIGHT>
struct _Cat2<Pack<>, RIGHT> {
    static_assert(IsPack<RIGHT>::Value, "L/R must be packs");
    using Type = RIGHT;
};

template <class X0, class... X>
struct _Cat2<Pack<X0>, Pack<X...>> {
    using Type = Pack<X0,X...>;
};

template <class RIGHT, class X0, class... X>
struct _Cat2<Pack<X0,X...>, RIGHT> {
    static_assert(IsPack<RIGHT>::Value, "L/R must be packs");
    using Type = Cat<Pack<X0>, Cat<Pack<X...>, RIGHT>>;
};

NAMESPACE_BEGIN(_pack)
template <class PACK>                 struct _Reversed;
template <class PACK>                 using   Reversed      = typename _Reversed<PACK>::Type;
template <class PACK, typename U>     struct _AsVector;
template <class PACK, typename U>     using   AsVector      = typename _AsVector<PACK,U>::Type;
template <long>                       struct  ApplyFn;
template <class PACK, class APPLY>    struct _Apply;
template <class PACK, class APPLY>    using   Apply         = typename _Apply<PACK, APPLY>::Type;
template <class PACK>                 struct _ApplySizeOf;
template <class PACK>                 using   ApplySizeOf   = typename _ApplySizeOf<PACK>::Type;
NAMESPACE_END(_pack)

template <class... X>               struct _Length<Pack<X...>>             { using Type = CountTypes<X...>; };
template <class... X>               struct _EmptyLike<Pack<X...>>          { using Type = Pack<>; };
template <class... X>               struct _IsPack<Pack<X...>>             { using Type = True; };
template <class... X>               struct _Reversed<Pack<X...>>           { using Type = _pack::Reversed<Pack<X...>>; };
template <class U, class... X>      struct _AsVector<Pack<X...>, U>        { using Type = _pack::AsVector<Pack<X...>, U>; };
template <class... X>               struct _AsPack<Pack<X...>>             { using Type = Pack<X...>; };
template <class M, class... X>      struct _LikeFrom<Pack<X...>, M>        { using Type = AsPack<M>; };
template <class X0, class... X>     struct _GetFirst<Pack<X0, X...>>       { using Type = Pack<X0>; };
template <class X0, class... X>     struct _DelFirst<Pack<X0, X...>>       { using Type = Pack<X...>; };
template <class X0, class... X>     struct _GetFirstValue<Pack<X0, X...>>  { using Type = X0; };
template <class... X>               struct _ApplyAddConst<Pack<X...>>      { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<0>>; };
template <class... X>               struct _ApplyAddConstPtr<Pack<X...>>   { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<1>>; };
template <class... X>               struct _ApplyAddConstRef<Pack<X...>>   { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<2>>; };
template <class... X>               struct _ApplyAddPtr<Pack<X...>>        { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<3>>; };
template <class... X>               struct _ApplyAddRef<Pack<X...>>        { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<4>>; };
template <class... X>               struct _ApplyAddRValueRef<Pack<X...>>  { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<5>>; };
template <class... X>               struct _ApplyRemoveConst<Pack<X...>>   { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<6>>; };
template <class... X>               struct _ApplyRemovePtr<Pack<X...>>     { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<7>>; };
template <class... X>               struct _ApplyRemoveRef<Pack<X...>>     { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<8>>; };
template <class... X>               struct _ApplyRemoveCV<Pack<X...>>      { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<9>>; };
template <class... X>               struct _ApplyDecay<Pack<X...>>         { using Type = _pack::Apply<Pack<X...>, _pack::ApplyFn<10>>; };
template <class... X>               struct _ApplySizeOf<Pack<X...>>        { using Type = _pack::ApplySizeOf<Pack<X...>>; };

/* ------------------------------------------------------------------ *
 *     Pack API Implementation                                        *
 * ------------------------------------------------------------------ */

NAMESPACE_BEGIN(_pack)

/* --- Revert ------------------------------------------------------- */

template <class PACK>
struct _Reversed {};

template <class X0, class... X>
struct _Reversed<Pack<X0, X...>> {
    using Type = Cat< Reversed<Pack<X...>>, Pack<X0> >;
};

template <class X0>
struct _Reversed<Pack<X0>> {
    using Type = Pack<X0>;
};

template <>
struct _Reversed<Pack<>> {
    using Type = Pack<>;
};

/* --- AsVector ----------------------------------------------------- */

template <class PACK, typename U>
struct _AsVector {};

template <typename U, class X0, class... X>
struct _AsVector<Pack<X0, X...>, U>
{
    using Type = Cat<
        Vector<U, static_cast<U>(X0())>,
        AsVector<Pack<X...>, U>
    >;
};

template <typename U, class X0>
struct _AsVector<Pack<X0>, U>
{
    using Type = Vector<U, static_cast<U>(X0())>;
};

template <typename U>
struct _AsVector<Pack<>, U>
{
    using Type = Vector<U>;
};

/* --- Apply -------------------------------------------------------- */

template <long>  struct ApplyFn {};
template <>      struct ApplyFn<0>  { template <class T> using Type = AddConst<T>; };
template <>      struct ApplyFn<1>  { template <class T> using Type = AddConstPtr<T>; };
template <>      struct ApplyFn<2>  { template <class T> using Type = AddConstRef<T>; };
template <>      struct ApplyFn<3>  { template <class T> using Type = AddPtr<T>; };
template <>      struct ApplyFn<4>  { template <class T> using Type = AddRef<T>; };
template <>      struct ApplyFn<5>  { template <class T> using Type = AddRValueRef<T>; };
template <>      struct ApplyFn<6>  { template <class T> using Type = RemoveConst<T>; };
template <>      struct ApplyFn<7>  { template <class T> using Type = RemovePtr<T>; };
template <>      struct ApplyFn<8>  { template <class T> using Type = RemoveRef<T>; };
template <>      struct ApplyFn<9>  { template <class T> using Type = RemoveCV<T>; };
template <>      struct ApplyFn<10> { template <class T> using Type = Decay<T>; };

template <class PACK, class APPLY>
struct _Apply {};

template <class APPLY, class X0, class... X>
struct _Apply<Pack<X0, X...>, APPLY>
{
    using Type = Cat<
        Pack<typename APPLY::template Type<X0>>,
        Apply<Pack<X...>, APPLY>
    >;
};

template <class APPLY, class X0>
struct _Apply<Pack<X0>, APPLY>
{
    using Type = Pack<typename APPLY::template Type<X0>>;
};

template <class APPLY>
struct _Apply<Pack<>, APPLY>
{
    using Type = Pack<>;
};

template <class PACK>
struct _ApplySizeOf
{};

template <class X0, class... X>
struct _ApplySizeOf<Pack<X0, X...>>
{
    using Type = Cat<
        ApplySizeOf<Pack<X0>>,
        ApplySizeOf<Pack<X...>>
    >;
};

template <class X0>
struct _ApplySizeOf<Pack<X0>>
{
    using Type = SizeT<sizeof(X0)>;
};

template <>
struct _ApplySizeOf<Pack<>>
{
    using Type = SizeT<>;
};

NAMESPACE_END(_pack)

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif /// MINITEN__META__PACK_IMPL
