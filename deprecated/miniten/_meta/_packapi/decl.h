/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file declares a compile-time API for accessing and                 **
 ** modifying pack-like meta types (Pack, Tuple, Vector).                   **
 **                                                                         **
 ** ItemType<V>          -> The type of elements in a vector                **
 ** Length<P>            -> Length of the pack [SizeT]                      **
 ** EmptyLike<P>         -> An empty version of P                           **
 ** Reversed<P>          -> Reversed version of P                           **
 ** LikeFrom<P, M>       -> Same metatype as P with the content of M        **
 ** Like<P, M...>        -> Same metatype as P with M... as content         **
 **                                                                         **
 ** AsVector<P,[T]>      -> Convert to a meta::Vector with same content     **
 ** AsTuple<P>           -> Convert to a meta::Tuple with same content      **
 ** AsPack<P>            -> Convert to a meta::Pack with same content       **
 **                                                                         **
 ** IsVector<P>          -> True if P is a meta::Vector                     **
 ** IsTuple<P>           -> True if P is a meta::Tuple                      **
 ** IsPack<P>            -> True if P is a meta::Pack                       **
 **                                                                         **
 ** Get<P,I>             -> Get elements at I                               **
 ** GetIndex<P,I>        -> Same as Get, but I is a concrete integral       **
 ** GetFirst<P,N=1>      -> Get first N elements                            **
 ** GetLast<P,N=1>       -> Get last N elements                             **
 **                                                                         **
 ** GetValue<P,I>        -> Get element at I                                **
 ** GetIndexValue<P,I>   -> Same as Get, but I is a concrete integral       **
 ** GetFirstValue<P>     -> Get first element                               **
 ** GetLastValue<P>      -> Get last element                                **
 **                                                                         **
 ** Del<P,I>             -> Delete elements at I                            **
 ** DelIndex<P,I>        -> Same as Del, but I is a concrete integral       **
 ** DelFirst<P,N=1>      -> Delete first N elements                         **
 ** DelLast<P,N=1>       -> Delete last N elements                          **
 **                                                                         **
 ** SetFrom<P,I,M...>    -> Set elements at I, copied from Cat<M...>        **
 ** SetFirstFrom<P,M...> -> Set first N elements, copied from Cat<M...>     **
 ** SetLastFrom<P,M....> -> Set last N elements, copied from Cat<M...>      **
 **                                                                         **
 ** Set<P,I,M...>        -> Set elements at I, copied from Tuple<M...>      **
 ** SetIndex<P,I,M>      -> Same as Set, but I is a concrete integral       **
 ** SetFirst<P,M...>     -> Set first N elements, copied from Tuple<M...>   **
 ** SetLast<P,M...>      -> Set last N elements, copied from Tuple<M...>    **
 **                                                                         **
 ** InsertFrom<P,I,M...> -> Insert elements at I, copied from Cat<M...>     **
 ** InsertIndexFrom<P,I,M> -> Same as Insert, but I is a concrete integral  **
 ** PrependFrom<P,M...>  -> Same as Cat<M..., P>                            **
 ** AppendFrom<P,M...>   -> Same as Cat<P, M...>                            **
 **                                                                         **
 ** Insert<P,I,M...>     -> Insert elements at I, copied from Tuple<M...>   **
 ** InsertIndex<P,I,M>   -> Same as Insert, but I is a concrete integral    **
 ** Prepend<P,M...>      -> Same as Cat<EmptyLike<P>, M..., P>              **
 ** Append<P,M...>       -> Same as Cat<P, M...>                            **
 **                                                                         **
 ** Extend<P, M...>      -> Same as AppendFrom<P,M...>                      **
 ** Cat<P, M...>         -> Same as AppendFrom<P,M...>                      **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 ** Notes                                                                   **
 ** -----                                                                   **
 ** Get/GetIndex/GetFirst/GetLast return the same metatype as the           **
 ** input. That is, `GetFirst<Tuple<int, float, bool>> = Tuple<int>`.       **
 ** This is to accomodate vectors of indices, e.g.,                         **
 ** `Get<Tuple<int, float, bool>, Long<0, 1>> = Tuple<int, float>`.         **
 ** It also provides a consitent behaviour between vectors and tuples.      **
 **                                                                         **
 ** In contrast, GetValue/GetIndexValue/GetFirstValue/GetLastValue          **
 ** return the indexed element in the case of Tuple and Pack. However,      **
 ** it returns a single-element vector in the case of Vector. The actual    **
 ** value can then be accessed from its `::Value` field.                    **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef MINITEN__META__PACKAPI_DECL
#define MINITEN__META__PACKAPI_DECL
#include <miniten/_core/defines.h>
#include <miniten/_meta/_pack/decl.h>   // Pack
#include <miniten/_meta/_index/decl.h>  // WrapIndex, SimpleSlice
#include <miniten/_meta/traits.h>       // RemoveRef, RemoveConst

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

using PDT = ptrdiff_t;
using SZT = size_t;

/*
 * Define AsPack first so we can use it to implement fallbacks that
 * go to/from Pack.
*/
template <class A>                          struct _AsPack          {};
template <class A>                          using   AsPack          = typename _AsPack<A>::Type;

/* ------------------------------------------------------------------ *
 * Construct
 * ------------------------------------------------------------------ */

template <class A, class M>                 struct _LikeFrom        {}; // MUST BE IMPLEMENTED
template <class A, class M>                 using   LikeFrom        = typename _LikeFrom<A,M>::Type;
template <class A, class... M>              using   Like            = LikeFrom<A, Pack<M...>>;
template <class A>                          struct _EmptyLike       { using Type = LikeFrom<A,typename _EmptyLike<AsPack<A>>::Type>; };
template <class A>                          using   EmptyLike       = typename _EmptyLike<A>::Type;
template <class A>                          struct _Reversed        { using Type = LikeFrom<A,typename _Reversed<AsPack<A>>::Type>; };
template <class A>                          using   Reversed        = typename _Reversed<A>::Type;

template <class A>                          struct  _EmptyLike<A&>          { using Type = EmptyLike<A>&; };
template <class A>                          struct  _EmptyLike<A&&>         { using Type = EmptyLike<A>&&; };
template <class A>                          struct  _EmptyLike<const A>     { using Type = const EmptyLike<A>; };
template <class A>                          struct  _Reversed<A&>           { using Type = Reversed<A>&; };
template <class A>                          struct  _Reversed<A&&>          { using Type = Reversed<A>&&; };
template <class A>                          struct  _Reversed<const A>      { using Type = const Reversed<A>; };
template <class A, class M>                 struct  _LikeFrom<A&,M>         { using Type = LikeFrom<A,M>&; };
template <class A, class M>                 struct  _LikeFrom<A&&,M>        { using Type = LikeFrom<A,M>&&; };
template <class A, class M>                 struct  _LikeFrom<const A,M>    { using Type = const LikeFrom<A,M>; };

/* ------------------------------------------------------------------ *
 * Cat
 * ------------------------------------------------------------------ */

template <class A, class M>                 struct _Cat2            { using Type = LikeFrom<A, typename _Cat2<AsPack<A>,AsPack<M>>::Type>; }; // MUST BE IMPLEMENTED
template <class... M>                       struct _Cat             { using Type = Pack<>; }; // Fully implemented later
template <class... M>                       using   Cat             = typename _Cat<M...>::Type;

/* ------------------------------------------------------------------ *
 * Length
 * ------------------------------------------------------------------ */

template <class... A>                       struct  _Length          {};
template <class... A>                       using    Length          = typename _Length<A...>::Type;
template <>                                 struct  _Length<>        { using Type = SizeT<>; };
template <class A>                          struct  _Length<A>       { using Type = Length<AsPack<A>>; };
template <class A, class... B>              struct  _Length<A, B...> { using Type = Cat<Length<A>, Length<B...>>; };
template <class A>                          struct  _Length<A&>      { using Type = Length<A>; };
template <class A>                          struct  _Length<A&&>     { using Type = Length<A>; };
template <class A>                          struct  _Length<const A> { using Type = Length<A>; };

template <class... A>                       struct _SumLength        {};
template <class... A>                       using   SumLength        = typename _SumLength<A...>::Type;
template <>                                 struct _SumLength<>      { using Type = SizeT<0>; };
template <class A, class... B>              struct _SumLength<A, B...> {
    using _LA = Length<A>;
    using _LB = typename _SumLength<B...>::Type;
    using Type = SizeT<_LA::Value + _LB::Value>;
};

/* ------------------------------------------------------------------ *
 * Get
 * ------------------------------------------------------------------ */

template <class A>                          struct _GetFirst        { using Type = LikeFrom<A,typename _GetFirst<AsPack<A>>::Type>; };
template <class A, class I>                 struct _Get             {}; // Fully implemented later
template <class A>                          struct _Get<A,Z_0>      { using Type = typename _GetFirst<A>::Type; };
template <class A, class I>                 using  _GetWrap         = typename _Get<A, WrapIndex<Length<A>, I>>::Type;
template <class A, class I>                 using   Get             = _GetWrap<A, I>;
template <class A, PDT... I>                using   GetIndex        = Get<A, PtrDiff<I...>>;
template <class A, SZT N=1>                 using   GetFirst        = Get<A, SimpleSlice<0, N>>;
template <class A, SZT N=1>                 using   GetLast         = Get<A, SimpleSlice<Length<A>::Value-N, Length<A>::Value>>;

template <class A>                          struct _GetFirstValue   { using Type = typename _GetFirstValue<AsPack<A>>::Type; }; // MUST BE IMPLEMENTED IN PACK & VECTOR
template <class A, class I>                 struct _GetValue        { static_assert(Length<I>::Value == 1, "GetValue does not support vector of indices"); }; // Fully implemented later
template <class A>                          struct _GetValue<A,Z_0> { using Type = typename _GetFirstValue<A>::Type; };
template <class A, class I>                 using   GetValue        = typename _GetValue<A, WrapIndex<Length<A>, I>>::Type;
template <class A, PDT I>                   using   GetIndexValue   = GetValue<A, PtrDiff<I>>;
template <class A>                          using   GetFirstValue   = GetValue<A, PtrDiff<0>>;
template <class A>                          using   GetLastValue    = GetValue<A, PtrDiff<-1>>;

template <class A>                          struct  _GetFirst<A&>               { using Type = GetFirst<A>&; };
template <class A>                          struct  _GetFirst<A&&>              { using Type = GetFirst<A>&&; };
template <class A>                          struct  _GetFirst<const A>          { using Type = const GetFirst<A>; };

template <class A>                          struct  _GetFirstValue<A&>          { using Type = GetFirstValue<A>; };
template <class A>                          struct  _GetFirstValue<A&&>         { using Type = GetFirstValue<A>; };
template <class A>                          struct  _GetFirstValue<const A>     { using Type = GetFirstValue<A>; };

/* ------------------------------------------------------------------ *
 * Delete
 * ------------------------------------------------------------------ */

template <class A>                          struct _DelFirst        { using Type = LikeFrom<A,typename _DelFirst<AsPack<A>>::Type>; };
template <class A, class I>                 struct _Del             {}; // Fully implemented later
template <class A>                          struct _Del<A,Z_0>      { using Type = typename _DelFirst<A>::Type; };
template <class A, class I>                 using  _DelWrap         = typename _Del<A, WrapIndex<Length<A>, I>>::Type;
template <class A, class I>                 using   Del             = _DelWrap<A, I>;
template <class A, PDT I>                   using   DelIndex        = Del<A, PtrDiff<I>>;
template <class A, SZT N=1>                 using   DelFirst        = Del<A, SimpleSlice<0, N>>;
template <class A, SZT N=1>                 using   DelLast         = Del<A, SimpleSlice<Length<A>::Value-N, Length<A>::Value>>;

template <class A>                          struct  _DelFirst<A&>          { using Type = DelFirst<A>&; };
template <class A>                          struct  _DelFirst<A&&>         { using Type = DelFirst<A>&&; };
template <class A>                          struct  _DelFirst<const A>     { using Type = const DelFirst<A>; };

/* ------------------------------------------------------------------ *
 * Insert
 * ------------------------------------------------------------------ */

/*
 * NOTE
 *      For append/prepend, I used to fallback to InsertFrom, but
 *      got some weird bugs in weird places. I now fallback to Cat
 *      and it seems to have solved it.
 */

template <class A, class I, class... M>     struct _InsertFrom      { static_assert(Length<I>::Value == 1, "Insert does not support vector of indices"); }; // Fully implemented later
template <class A, class I, class... M>     using  _InsertFromWrap  = typename _InsertFrom<A, WrapIndex<Length<A>, I>, M...>::Type;
template <class A, class I, class... M>     using   InsertFrom      = _InsertFromWrap<A, I, M...>;
template <class A, PDT   I, class... M>     using   InsertIndexFrom = InsertFrom<A, PtrDiff<I>, M...>;
template <class A, class... M>              using   PrependFrom     = Cat<EmptyLike<A>, Cat<M...>, A>;
template <class A, class... M>              using   AppendFrom      = Cat<A, M...>;
template <class A, class I, class... M>     using   Insert          = InsertFrom<A, I, Pack<M... >>;
template <class A, PDT   I, class... M>     using   InsertIndex     = Insert<A, PtrDiff<I>, M...>;
template <class A, class... M>              using   Prepend         = Cat<Like<A, M...>, A>;
template <class A, class... M>              using   Append          = Cat<A, Pack<M...>>;
template <class A, class... M>              using   Extend          = Cat<A, M...>;

/* ------------------------------------------------------------------ *
 * Assign
 * ------------------------------------------------------------------ */

template <class A, class I, class... M>     struct _SetFrom         {}; // Fully implemented later
template <class A, class I, class... M>     using  _SetFromWrap     = typename _SetFrom<A, WrapIndex<Length<A>, I>, M...>::Type;
template <class A, class I, class... M>     using   SetFrom         = _SetFromWrap<A, I, M...>;
template <class A, class... M>              struct _SetFirstFrom    {};
template <class A, class... M>              using   SetFirstFrom    = typename _SetFirstFrom<A, M...>::Type;
template <class A, class M0, class... M>    struct _SetFirstFrom<A,M0,M...>
                                                                    { using _Length0 = Length<M0>;
                                                                      using _LengthM = SumLength<M...>;
                                                                      using _Slice  = SimpleSlice<
                                                                        _Length0::Value,
                                                                        _Length0::Value + _LengthM::Value
                                                                      >;
                                                                      using _Front  = SetFirstFrom<A, M0>;
                                                                      using Type    = SetFrom<_Front, _Slice, M...>; };
template <class A, class M>                 struct _SetFirstFrom<A,M>{using _Length = Length<M>;
                                                                      using _Slice  = SimpleSlice<0, _Length::Value>;
                                                                      using Type    = SetFrom<A, _Slice, M>; };
template <class A>                          struct _SetFirstFrom<A> { using Type    = A; };
template <class A, class... M>              struct _SetLastFrom     {};
template <class A, class... M>              using   SetLastFrom     = typename _SetLastFrom<A, M...>::Type;
template <class A, class M0, class... M>    struct _SetLastFrom<A,M0,M...>
                                                                    { using _LengthM = SumLength<M...>;
                                                                      using _Length0 = Length<M0>;
                                                                      using _LengthA = Length<A>;
                                                                      using _Slice   = SimpleSlice<
                                                                        _LengthA::Value-_LengthM::Value-_Length0::Value,
                                                                        _LengthA::Value-_LengthM::Value
                                                                      >;
                                                                      using _Back    = SetLastFrom<A, M...>;
                                                                      using Type     = SetFrom<_Back, _Slice, M0>; };
template <class A, class M>                 struct _SetLastFrom<A,M>{ using _LengthM = Length<M>;
                                                                      using _LengthA = Length<A>;
                                                                      using _Slice   = SimpleSlice<_LengthA::Value-_LengthM::Value, _LengthA::Value>;
                                                                      using Type     = SetFrom<A, _Slice, M>; };
template <class A>                          struct _SetLastFrom<A>  { using Type     = A; };
template <class A, class I, class... M>     using   Set             = SetFrom<A, I, Pack<M... >>;
template <class A, PDT I, class M>          using   SetIndex        = Set<A, PtrDiff<I>, M>;
template <class A, class... M>              using   SetFirst        = Set<A, SimpleSlice<0, Length<Pack<M...>>::Value>, M...>;
template <class A, class... M>              using   SetLast         = Set<A, SimpleSlice<Length<A>::Value-Length<Pack<M...>>::Value, Length<A>::Value>, M...>;

/* ------------------------------------------------------------------ *
 * Metadata
 * ------------------------------------------------------------------ */

template <class A>                          struct _ItemType        { using Type = typename _ItemType<GetFirstValue<A>>::Type; };
template <class A>                          using   ItemType        = typename _ItemType<A>::Type;

template <class A>                          struct  _ItemType<A&>          { using Type = _ItemType<RemoveRef<A>>; };
template <class A>                          struct  _ItemType<A&&>         { using Type = _ItemType<RemoveRef<A>>; };
template <class A>                          struct  _ItemType<const A>     { using Type = _ItemType<RemoveConst<A>>; };

/* ------------------------------------------------------------------ *
 * Convert
 * ------------------------------------------------------------------ */

template <class A, class T = ItemType<A>>   struct _AsVector        { using Type = typename _AsVector<AsPack<A>>::Type; };
template <class A, class T = ItemType<A>>   using   AsVector        = typename _AsVector<A,T>::Type;
template <class A>                          struct _AsTuple         { using Type = typename _AsTuple<AsPack<A>>::Type; };
template <class A>                          using   AsTuple         = typename _AsTuple<A>::Type;

template <class A>                          struct  _AsVector<A&>          { using Type = AsVector<A>&; };
template <class A>                          struct  _AsVector<A&&>         { using Type = AsVector<A>&&; };
template <class A>                          struct  _AsVector<const A>     { using Type = const AsVector<A>; };
template <class A>                          struct  _AsTuple<A&>           { using Type = AsTuple<A>&; };
template <class A>                          struct  _AsTuple<A&&>          { using Type = AsTuple<A>&&; };
template <class A>                          struct  _AsTuple<const A>      { using Type = const AsTuple<A>; };
template <class A>                          struct  _AsPack<A&>            { using Type = AsPack<A>&; };
template <class A>                          struct  _AsPack<A&&>           { using Type = AsPack<A>&&; };
template <class A>                          struct  _AsPack<const A>       { using Type = const AsPack<A>; };

/* ------------------------------------------------------------------ *
 * Apply modifiers
 * ------------------------------------------------------------------ */

template <class A>                          struct _ApplyAddConst    { using Type = LikeFrom<A,typename _ApplyAddConst<AsPack<A>>::Type>; };
template <class A>                          using   ApplyAddConst    = typename _ApplyAddConst<A>::Type;
template <class A>                          struct _ApplyAddRef      { using Type = LikeFrom<A,typename _ApplyAddRef<AsPack<A>>::Type>; };
template <class A>                          using   ApplyAddRef      = typename _ApplyAddRef<A>::Type;
template <class A>                          struct _ApplyAddConstRef { using Type = LikeFrom<A,typename _ApplyAddConstRef<AsPack<A>>::Type>; };
template <class A>                          using   ApplyAddConstRef = typename _ApplyAddConstRef<A>::Type;
template <class A>                          struct _ApplyAddPtr      { using Type = LikeFrom<A,typename _ApplyAddPtr<AsPack<A>>::Type>; };
template <class A>                          using   ApplyAddPtr      = typename _ApplyAddPtr<A>::Type;
template <class A>                          struct _ApplyAddConstPtr { using Type = LikeFrom<A,typename _ApplyAddConstPtr<AsPack<A>>::Type>; };
template <class A>                          using   ApplyAddConstPtr = typename _ApplyAddConstPtr<A>::Type;
template <class A>                          struct _ApplyAddRValueRef{ using Type = LikeFrom<A,typename _ApplyAddRValueRef<AsPack<A>>::Type>; };
template <class A>                          using   ApplyAddRValueRef= typename _ApplyAddRValueRef<A>::Type;
template <class A>                          struct _ApplyRemoveRef   { using Type = LikeFrom<A,typename _ApplyRemoveRef<AsPack<A>>::Type>; };
template <class A>                          using   ApplyRemoveRef   = typename _ApplyRemoveRef<A>::Type;
template <class A>                          struct _ApplyRemovePtr   { using Type = LikeFrom<A,typename _ApplyRemovePtr<AsPack<A>>::Type>; };
template <class A>                          using   ApplyRemovePtr   = typename _ApplyRemoveRef<A>::Type;
template <class A>                          struct _ApplyRemoveConst { using Type = LikeFrom<A,typename _ApplyRemoveConst<AsPack<A>>::Type>; };
template <class A>                          using   ApplyRemoveConst = typename _ApplyRemoveConst<A>::Type;
template <class A>                          struct _ApplyRemoveCV    { using Type = LikeFrom<A,typename _ApplyRemoveCV<AsPack<A>>::Type>; };
template <class A>                          using   ApplyRemoveCV    = typename _ApplyRemoveCV<A>::Type;
template <class A>                          struct _ApplyDecay       { using Type = LikeFrom<A,typename _ApplyDecay<AsPack<A>>::Type>; };
template <class A>                          using   ApplyDecay       = typename _ApplyDecay<A>::Type;
template <class A>                          struct _ApplySizeOf      { using Type = LikeFrom<A,typename _ApplySizeOf<AsPack<A>>::Type>; };
template <class A>                          using   ApplySizeOf      = typename _ApplySizeOf<A>::Type;

template <class A>                          struct  _ApplyAddConst<A&>              { using Type = ApplyAddConst<A>&; };
template <class A>                          struct  _ApplyAddConst<A&&>             { using Type = ApplyAddConst<A>&&; };
template <class A>                          struct  _ApplyAddConst<const A>         { using Type = const ApplyAddConst<A>; };
template <class A>                          struct  _ApplyAddRef<A&>                { using Type = ApplyAddRef<A>&; };
template <class A>                          struct  _ApplyAddRef<A&&>               { using Type = ApplyAddRef<A>&&; };
template <class A>                          struct  _ApplyAddRef<const A>           { using Type = const ApplyAddRef<A>; };
template <class A>                          struct  _ApplyAddConstRef<A&>           { using Type = ApplyAddConstRef<A>&; };
template <class A>                          struct  _ApplyAddConstRef<A&&>          { using Type = ApplyAddConstRef<A>&&; };
template <class A>                          struct  _ApplyAddConstRef<const A>      { using Type = const ApplyAddConstRef<A>; };
template <class A>                          struct  _ApplyAddPtr<A&>                { using Type = ApplyAddPtr<A>&; };
template <class A>                          struct  _ApplyAddPtr<A&&>               { using Type = ApplyAddPtr<A>&&; };
template <class A>                          struct  _ApplyAddPtr<const A>           { using Type = const ApplyAddPtr<A>; };
template <class A>                          struct  _ApplyAddConstPtr<A&>           { using Type = ApplyAddConstPtr<A>&; };
template <class A>                          struct  _ApplyAddConstPtr<A&&>          { using Type = ApplyAddConstPtr<A>&&; };
template <class A>                          struct  _ApplyAddConstPtr<const A>      { using Type = const ApplyAddConstPtr<A>; };
template <class A>                          struct  _ApplyAddRValueRef<A&>          { using Type = ApplyAddRValueRef<A>&; };
template <class A>                          struct  _ApplyAddRValueRef<A&&>         { using Type = ApplyAddRValueRef<A>&&; };
template <class A>                          struct  _ApplyAddRValueRef<const A>     { using Type = const ApplyAddRValueRef<A>; };
template <class A>                          struct  _ApplyRemoveRef<A&>             { using Type = ApplyRemoveRef<A>&; };
template <class A>                          struct  _ApplyRemoveRef<A&&>            { using Type = ApplyRemoveRef<A>&&; };
template <class A>                          struct  _ApplyRemoveRef<const A>        { using Type = const ApplyRemoveRef<A>; };
template <class A>                          struct  _ApplyRemovePtr<A&>             { using Type = ApplyRemovePtr<A>&; };
template <class A>                          struct  _ApplyRemovePtr<A&&>            { using Type = ApplyRemovePtr<A>&&; };
template <class A>                          struct  _ApplyRemovePtr<const A>        { using Type = const ApplyRemovePtr<A>; };
template <class A>                          struct  _ApplyRemoveConst<A&>           { using Type = ApplyRemoveConst<A>&; };
template <class A>                          struct  _ApplyRemoveConst<A&&>          { using Type = ApplyRemoveConst<A>&&; };
template <class A>                          struct  _ApplyRemoveConst<const A>      { using Type = const ApplyRemoveConst<A>; };
template <class A>                          struct  _ApplyRemoveCV<A&>              { using Type = ApplyRemoveCV<A>&; };
template <class A>                          struct  _ApplyRemoveCV<A&&>             { using Type = ApplyRemoveCV<A>&&; };
template <class A>                          struct  _ApplyRemoveCV<const A>         { using Type = const ApplyRemoveCV<A>; };
template <class A>                          struct  _ApplyDecay<A&>                 { using Type = ApplyDecay<A>&; };
template <class A>                          struct  _ApplyDecay<A&&>                { using Type = ApplyDecay<A>&&; };
template <class A>                          struct  _ApplyDecay<const A>            { using Type = const ApplyDecay<A>; };
template <class A>                          struct  _ApplySizeOf<A&>                { using Type = ApplySizeOf<A>&; };
template <class A>                          struct  _ApplySizeOf<A&&>               { using Type = ApplySizeOf<A>&&; };
template <class A>                          struct  _ApplySizeOf<const A>           { using Type = const ApplySizeOf<A>; };

/* ------------------------------------------------------------------ *
 * Test
 * ------------------------------------------------------------------ */

template <class A>                          struct _IsVector        { using Type = False; };
template <class A>                          using   IsVector        = typename _IsVector<A>::Type;
template <class A>                          struct _IsTuple         { using Type = False; };
template <class A>                          using   IsTuple         = typename _IsTuple<A>::Type;
template <class A>                          struct _IsPack          { using Type = False; };
template <class A>                          using   IsPack          = typename _IsPack<A>::Type;

template <class A>                          struct  _IsVector<A&>          { using Type = IsVector<RemoveRef<A>>; };
template <class A>                          struct  _IsVector<A&&>         { using Type = IsVector<RemoveRef<A>>; };
template <class A>                          struct  _IsVector<const A>     { using Type = IsVector<RemoveConst<A>>; };
template <class A>                          struct  _IsTuple<A&>           { using Type = IsTuple<RemoveRef<A>>; };
template <class A>                          struct  _IsTuple<A&&>          { using Type = IsTuple<RemoveRef<A>>; };
template <class A>                          struct  _IsTuple<const A>      { using Type = IsTuple<RemoveConst<A>>; };
template <class A>                          struct  _IsPack<A&>            { using Type = IsPack<RemoveRef<A>>; };
template <class A>                          struct  _IsPack<A&&>           { using Type = IsPack<RemoveRef<A>>; };
template <class A>                          struct  _IsPack<const A>       { using Type = IsPack<RemoveConst<A>>; };

/* ------------------------------------------------------------------ *
 * Operations
 * ------------------------------------------------------------------ */

// template <class A, class B, class Op>       struct _ApplyDeclType {>; };
// template <class A, class B, class Op>       using   ApplyDeclType = typename _ApplyDeclType<A,B,Op>::Type {};


/* ------------------------------------------------------------------ *
 * Conversion sugar
 * ------------------------------------------------------------------ */

template <class T> using AsBool     = AsVector<T, bool>;
template <class T> using AsPtrDiff  = AsVector<T, ptrdiff_t>;
template <class T> using AsSizeT    = AsVector<T, size_t>;

NAMESPACE_END(meta)
NAMESPACE_END(miniten)

#endif // MINITEN__META__PACKAPI_DECL
