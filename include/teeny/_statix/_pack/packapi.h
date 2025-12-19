/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file implements a compile-time "pack of types"                     **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef TNY__META__PACK_PACKAPI
#define TNY__META__PACK_PACKAPI
#include <cuda/std/tuple>
#include <teeny/core.h>
#include <teeny/_statix/_math/decl.h>   // count_types
#include <teeny/_statix/_pack/decl.h>
#include <teeny/_statix/_carray/decl.h>
#include <teeny/_statix/_packapi/decl.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

/* --- Cat ---------------------------------------------------------- */

template <class RIGHT>
struct _cat2<pack<>, RIGHT> {
    static_assert(is_pack<RIGHT>::value, "L/R must be packs");
    using type = RIGHT;
};

template <class X0, class... X>
struct _cat2<pack<X0>, pack<X...>> {
    using type = pack<X0,X...>;
};

template <class RIGHT, class X0, class... X>
struct _cat2<pack<X0,X...>, RIGHT> {
    static_assert(is_pack<RIGHT>::value, "L/R must be packs");
    using type = typename _cat2<pack<X0>, typename _cat2<pack<X...>, RIGHT>::type>::type;
};

_TNY_NAMESPACE_BEGIN(_pack)
template <long>                     struct  apply_fn;
template <class PACK, class APPLY>  struct _apply;
template <class PACK, class APPLY>  using   apply          = typename _apply<PACK, APPLY>::type;
template <class PACK>               struct _apply_sizeof;
template <class PACK>               using   apply_sizeof   = typename _apply_sizeof<PACK>::type;
_TNY_NAMESPACE_END(_pack)

template <class... X>               struct _size        <pack<X...>>      { using type = count_types<X...>; };
template <class... X>               struct _empty_like  <pack<X...>>      { using type = pack<>; };
template <class... X>               struct _is_pack     <pack<X...>>      { using type = ctrue; };
template <class... X>               struct _as_pack     <pack<X...>>      { using type = pack<X...>; };
template <class M, class... X>      struct _like_from   <pack<X...>, M>   { using type = as_pack<M>; };
template <class X0, class... X>     struct _front       <pack<X0, X...>>  { using type = X0; };
template <class X0, class... X>     struct _head        <pack<X0, X...>>  { using type = pack<X0>; };
template <>                         struct _head        <pack<>>          { using type = pack<>; };
template <class X0, class... X>     struct _erase_head  <pack<X0, X...>>  { using type = pack<X...>; };
template <>                         struct _erase_head  <pack<>>          { using type = pack<>; };

template <class... X>               struct _apply_add_const             <pack<X...>> { using type = _pack::apply<pack<X...>, _pack::apply_fn< 0>>; };
template <class... X>               struct _apply_add_volatile          <pack<X...>> { using type = _pack::apply<pack<X...>, _pack::apply_fn< 1>>; };
template <class... X>               struct _apply_add_cv                <pack<X...>> { using type = _pack::apply<pack<X...>, _pack::apply_fn< 2>>; };
template <class... X>               struct _apply_add_lvalue_reference  <pack<X...>> { using type = _pack::apply<pack<X...>, _pack::apply_fn< 3>>; };
template <class... X>               struct _apply_add_rvalue_reference  <pack<X...>> { using type = _pack::apply<pack<X...>, _pack::apply_fn< 4>>; };
template <class... X>               struct _apply_add_pointer           <pack<X...>> { using type = _pack::apply<pack<X...>, _pack::apply_fn< 5>>; };
template <class... X>               struct _apply_remove_const          <pack<X...>> { using type = _pack::apply<pack<X...>, _pack::apply_fn< 6>>; };
template <class... X>               struct _apply_remove_volatile       <pack<X...>> { using type = _pack::apply<pack<X...>, _pack::apply_fn< 7>>; };
template <class... X>               struct _apply_remove_cv             <pack<X...>> { using type = _pack::apply<pack<X...>, _pack::apply_fn< 8>>; };
template <class... X>               struct _apply_remove_reference      <pack<X...>> { using type = _pack::apply<pack<X...>, _pack::apply_fn< 9>>; };
template <class... X>               struct _apply_remove_pointer        <pack<X...>> { using type = _pack::apply<pack<X...>, _pack::apply_fn<10>>; };
template <class... X>               struct _apply_remove_cvref          <pack<X...>> { using type = _pack::apply<pack<X...>, _pack::apply_fn<11>>; };
template <class... X>               struct _apply_decay                 <pack<X...>> { using type = _pack::apply<pack<X...>, _pack::apply_fn<12>>; };
template <class... X>               struct _apply_sizeof                <pack<X...>> { using type = _pack::apply_sizeof<pack<X...>>; };

// tuple
template <class... X>               struct _is_tuple     <cuda::std::tuple<X...>>      { using type = ctrue; };
template <class... X>               struct _as_pack      <cuda::std::tuple<X...>>      { using type = pack<X...>; };
template <class... X>               struct _as_tuple     <pack<X...>>                  { using type = cuda::std::tuple<X...>; };
template <class... X>               struct _empty_like   <cuda::std::tuple<X...>>      { using type = cuda::std::tuple<>; };
template <class M, class... X>      struct _like_from    <cuda::std::tuple<X...>, M>   { using type = as_tuple<M>; };

/* --- reversed ----------------------------------------------------- */

template <class X0, class... X>
struct _reversed<pack<X0, X...>> {
    using type = cat<reversed<pack<X...>>, pack<X0> >;
};

template <class X0>
struct _reversed<pack<X0>> {
    using type = pack<X0>;
};

template <>
struct _reversed<pack<>> {
    using type = pack<>;
};

/* --- as_carray ---------------------------------------------------- */

template <typename U, class X0, class... X>
struct _as_carray<pack<X0, X...>, U>
{
    using type = cat<
        as_carray<pack<X0>, U>,
        as_carray<pack<X...>, U>
    >;
};

template <typename U, class T, T... X>
struct _as_carray<pack<carray<T, X...>>, U>
{
    using type = as_carray<carray<T, X...>, U>;
};

template <typename U>
struct _as_carray<pack<>, U>
{
    using type = carray<U>;
};

/* --- apply -------------------------------------------------------- */

_TNY_NAMESPACE_BEGIN(_pack)

template <long>  struct apply_fn {};
template <>      struct apply_fn<0>  { template <class T> using type = cuda::std::add_const_t            <T>; };
template <>      struct apply_fn<1>  { template <class T> using type = cuda::std::add_volatile_t         <T>; };
template <>      struct apply_fn<2>  { template <class T> using type = cuda::std::add_cv_t               <T>; };
template <>      struct apply_fn<3>  { template <class T> using type = cuda::std::add_lvalue_reference_t <T>; };
template <>      struct apply_fn<4>  { template <class T> using type = cuda::std::add_rvalue_reference_t <T>; };
template <>      struct apply_fn<5>  { template <class T> using type = cuda::std::add_pointer_t          <T>; };
template <>      struct apply_fn<6>  { template <class T> using type = cuda::std::remove_const_t         <T>; };
template <>      struct apply_fn<7>  { template <class T> using type = cuda::std::remove_volatile_t      <T>; };
template <>      struct apply_fn<8>  { template <class T> using type = cuda::std::remove_cv_t            <T>; };
template <>      struct apply_fn<9>  { template <class T> using type = cuda::std::remove_reference_t     <T>; };
template <>      struct apply_fn<10> { template <class T> using type = cuda::std::remove_pointer_t       <T>; };
template <>      struct apply_fn<11> { template <class T> using type = cuda::std::remove_cvref_t         <T>; };
template <>      struct apply_fn<12> { template <class T> using type = cuda::std::decay_t                <T>; };

template <class PACK, class APPLY>
struct _apply {};

template <class APPLY, class X0, class... X>
struct _apply<pack<X0, X...>, APPLY>
{
    using type = cat<
        pack<typename APPLY::template type<X0>>,
        apply<pack<X...>, APPLY>
    >;
};

template <class APPLY, class X0>
struct _apply<pack<X0>, APPLY>
{
    using type = pack<typename APPLY::template type<X0>>;
};

template <class APPLY>
struct _apply<pack<>, APPLY>
{
    using type = pack<>;
};

template <class PACK>
struct _apply_sizeof
{};

template <class X0, class... X>
struct _apply_sizeof<pack<X0, X...>>
{
    using type = cat<
        apply_sizeof<pack<X0>>,
        apply_sizeof<pack<X...>>
    >;
};

template <class X0>
struct _apply_sizeof<pack<X0>>
{
    using type = csize<sizeof(X0)>;
};

template <>
struct _apply_sizeof<pack<>>
{
    using type = csize<>;
};

_TNY_NAMESPACE_END(_pack)

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif /// TNY__META__PACK_PACKAPI
