/***********************************************************************
 * This file declares a compile-time API for accessing and
 * modifying pack-like meta types (Pack, Tuple, Vector)
 *
 * ItemType<V>          -> The type of elements in a vector
 * Length<P>            -> Length of the pack [SizeT]
 * EmptyLike<P>         -> An empty version of P
 * Reversed<P>          -> Reversed version of P
 * LikeFrom<P, M>       -> Same metatype as P with the content of M
 * Like<P, M...>        -> Same metatype as P with M... as content
 *
 * AsVector<P,[T]>      -> Convert to a meta::Vector with same content
 * AsTuple<P>           -> Convert to a meta::Tuple with same content
 * AsPack<P>            -> Convert to a meta::Pack with same content
 *
 * IsVector<P>          -> True if P is a meta::Vector
 * IsTuple<P>           -> True if P is a meta::Tuple
 * IsPack<P>            -> True if P is a meta::Pack
 *
 * Get<P,I>             -> Get elements at I
 * GetIndex<P,I>        -> Same as Get, but I is a concrete integral
 * GetFirst<P,N=1>      -> Get first N elements
 * GetLast<P,N=1>       -> Get last N elements
 *
 * GetValue<P,I>        -> Get element at I
 * GetIndexValue<P,I>   -> Same as Get, but I is a concrete integral
 * GetFirst<P>          -> Get first element
 * GetLast<P>           -> Get last element
 *
 * Del<P,I>             -> Delete elements at I
 * DelIndex<P,I>        -> Same as Del, but I is a concrete integral
 * DelFirst<P,N=1>      -> Delete first N elements
 * DelLast<P,N=1>       -> Delete last N elements
 *
 * SetFrom<P,I,M...>    -> Set elements at I, copied from Cat<M...>
 * SetFirstFrom<P,M...> -> Set first `M::Length` elements, copied from Cat<M...>
 * SetLastFrom<P,M....> -> Set last `M::Length` elements, copied from Cat<M...>
 *
 * Set<P,I,M...>        -> Set elements at I, copied from Tuple<M...>
 * SetIndex<P,I,M>      -> Same as Set, but I is a concrete integral
 * SetFirst<P,M...>     -> Set first `M::Length` elements, copied from Tuple<M...>
 * SetLast<P,M...>      -> Set last `M::Length` elements, copied from Tuple<M...>
 *
 * InsertFrom<P,I,M...> -> Insert elements at I, copied from Cat<M...>
 * InsertIndexFrom<P,I,M> -> Same as Insert, but I is a concrete integral
 * PrependFrom<P,M...>  -> Same as Cat<M..., P>
 * AppendFrom<P,M...>   -> Same as Cat<P, M...>
 *
 * Insert<P,I,M...>     -> Insert elements at I, copied from Tuple<M...>
 * InsertIndex<P,I,M>   -> Same as Insert, but I is a concrete integral
 * InsertFirst<P,M...>  -> Same as Cat<EmptyLike<P>, M..., P>
 * InsertLast<P,M...>   -> Same as Cat<P, M...>
 *
 * Extend<P, M...>      -> Same as AppendFrom<P,M...>
 * Cat<P, M...>         -> Same as AppendFrom<P,M...>
 *
 * NOTE
 *      Get/GetIndex/GetFirst/GetLast return the same metatype as the
 *      input. That is, `GetFirst<Tuple<int, float, bool>> = Tuple<int>`.
 *      This is to accomodate vectors of indices, e.g.,
 *      `Get<Tuple<int, float, bool>, Long<0, 1>> = Tuple<int, float>`.
 *      It also provides a consitent behaviour between vectors and tuples.
 *
 *      In contrast, GetValue/GetIndexValue/GetFirstValue/GetLastValue
 *      return the indexed element in the case of Tuple and Pack. However,
 *      it returns a single-element vector in the case of Vector. The actual
 *      value can then be accessed from its `::Value` field.
 ***********************************************************************/
#ifndef MINITEN_META_PACKAPI_H
#define MINITEN_META_PACKAPI_H
#include "pack.h"
#include "tuple.h"
#include "vector.h"
#include "index.h"

namespace miniten {
namespace meta {

template <class A>                          struct  _Length         {}; // MUST BE IMPLEMENTED
template <class A>                          using    Length         = typename _Length<A>::Type;

// ---------------------------------------------------------------------
// Construct
// ---------------------------------------------------------------------

template <class A>                          struct _EmptyLike       {}; // MUST BE IMPLEMENTED
template <class A>                          using   EmptyLike       = typename _EmptyLike<A>::Type;
template <class A>                          struct _Reversed        {}; // MUST BE IMPLEMENTED
template <class A>                          using   Reversed        = typename _Reversed<A>::Type;
template <class A, class M>                 struct _LikeFrom        {}; // MUST BE IMPLEMENTED
template <class A, class M>                 using   LikeFrom        = typename _LikeFrom<A,M>::Type;
template <class A, class... M>              using   Like            = LikeFrom<A, Pack<M...>>;

// ---------------------------------------------------------------------
// Cat
// ---------------------------------------------------------------------

template <class A, class M>                 struct _Cat2            {}; // MUST BE IMPLEMENTED
template <class... M>                       struct _Cat             { using Type = Pack<>; }; // Fully implemented later
template <class... M>                       using   Cat             = typename _Cat<M...>::Type;

// ---------------------------------------------------------------------
// Get
// ---------------------------------------------------------------------

template <class A>                          struct _GetFirst        {}; // MUST BE IMPLEMENTED
template <class A, class I>                 struct _Get             {}; // Fully implemented later
template <class A>                          struct _Get<A,Z0>       { using Type = typename _GetFirst<A>::Type; };
template <class A, class I>                 using  _GetWrap         = typename _Get<A, WrapIndex<Length<A>, I>>::Type;
template <class A, class I>                 using   Get             = _GetWrap<A, I>;
template <class A, ptrdiff_t... I>          using   GetIndex        = Get<A, PtrDiff<I...>>;
template <class A, size_t N=1>              using   GetFirst        = Get<A, SimpleSlice<0, N>>;
template <class A, size_t N=1>              using   GetLast         = Get<A, SimpleSlice<Length<A>::Value-N, Length<A>::Value>>;

template <class A>                          struct _GetFirstValue   {}; // MUST BE IMPLEMENTED
template <class A, class I>                 struct _GetValue        { static_assert(Length<I>::Value == 1, "GetValue does not support vector of indices"); }; // Fully implemented later
template <class A>                          struct _GetValue<A,Z0>  { using Type = typename _GetFirstValue<A>::Type; };
template <class A, class I>                 using   GetValue        = typename _GetValue<A, WrapIndex<Length<A>, I>>::Type;
template <class A, ptrdiff_t I>             using   GetIndexValue   = GetValue<A, PtrDiff<I>>;
template <class A>                          using   GetFirstValue   = GetValue<A, PtrDiff<0>>;
template <class A>                          using   GetLastValue    = GetValue<A, PtrDiff<-1>>;

// ---------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------

template <class A>                          struct _DelFirst        {}; // MUST BE IMPLEMENTED
template <class A, class I>                 struct _Del             {}; // Fully implemented later
template <class A>                          struct _Del<A,Z0>       { using Type = typename _DelFirst<A>::Type; };
template <class A, class I>                 using  _DelWrap         = typename _Del<A, WrapIndex<Length<A>, I>>::Type;
template <class A, class I>                 using   Del             = _DelWrap<A, I>;
template <class A, ptrdiff_t I>             using   DelIndex        = Del<A, PtrDiff<I>>;
template <class A, size_t N=1>              using   DelFirst        = Del<A, SimpleSlice<0, N>>;
template <class A, size_t N=1>              using   DelLast         = Del<A, SimpleSlice<Length<A>::Value-N, Length<A>::Value>>;

// ---------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------

// NOTE: For append/prepend, I used to fallback to InsertFrom, but
//       got some weird bugs in weird places. I now fallback to Cat
//       and it seems to have solved it.

template <class A, class I, class... M>     struct _InsertFrom      { static_assert(Length<I>::Value == 1, "Insert does not support vector of indices"); }; // Fully implemented later
template <class A, class I, class... M>     using  _InsertFromWrap  = typename _InsertFrom<A, WrapIndex<Length<A>, I>, M...>::Type;
template <class A, class I, class... M>     using   InsertFrom      = _InsertFromWrap<A, I, M...>;
template <class A, ptrdiff_t I, class... M> using   InsertIndexFrom = InsertFrom<A, PtrDiff<I>, M...>;
template <class A, class... M>              using   PrependFrom     = Cat<EmptyLike<A>, Cat<M...>, A>;
template <class A, class... M>              using   AppendFrom      = Cat<A, M...>;
template <class A, class I, class... M>     using   Insert          = InsertFrom<A, I, Pack<M... >>;
template <class A, ptrdiff_t I, class... M> using   InsertIndex     = Insert<A, PtrDiff<I>, M...>;
template <class A, class... M>              using   Prepend         = Cat<Like<A, M...>, A>;
template <class A, class... M>              using   Append          = Cat<A, Pack<M...>>;
template <class A, class... M>              using   Extend          = Cat<A, M...>;

// ---------------------------------------------------------------------
// Assign
// ---------------------------------------------------------------------

template <class A, class I, class... M>     struct _SetFrom         {}; // Fully implemented later
template <class A, class I, class... M>     using  _SetFromWrap     = typename _SetFrom<A, WrapIndex<Length<A>, I>, M...>::Type;
template <class A, class I, class... M>     using   SetFrom         = _SetFromWrap<A, I, M...>;
template <class A, class... M>              struct _SetFirstFrom    { using _AllM   = Cat<Pack<>, M...>;
                                                                      using _Length = Length<_AllM>;
                                                                      using _Slice  = SimpleSlice<0, _Length::Value>;
                                                                      using Type    = SetFrom<A, _Slice, _AllM>; };
template <class A, class... M>              using   SetFirstFrom    = typename _SetFirstFrom<A, M...>::Type;
template <class A, class... M>              struct _SetLastFrom     { using _AllM    = Cat<Pack<>, M...>;
                                                                      using _LengthM = Length<_AllM>;
                                                                      using _LengthA = Length<A>;
                                                                      using _Slice   = SimpleSlice<_LengthA::Value-_LengthM::Value, _LengthA::Value>;
                                                                      using Type     = SetFrom<A, _Slice, _AllM>; };
template <class A, class... M>              using   SetLastFrom     = typename _SetLastFrom<A, M...>::Type;
template <class A, class I, class... M>     using   Set             = SetFrom<A, I, Pack<M... >>;
template <class A, ptrdiff_t I, class M>    using   SetIndex        = Set<A, PtrDiff<I>, M>;
template <class A, class... M>              using   SetFirst        = Set<A, SimpleSlice<0, Length<Pack<M...>>::Value>, M...>;
template <class A, class... M>              using   SetLast         = Set<A, SimpleSlice<Length<A>::Value-Length<Pack<M...>>::Value, Length<A>::Value>, M...>;

// ---------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------

template <class A>                          struct _ItemType        { using Type = typename _ItemType<GetFirstValue<A>>::Type; };
template <class A>                          using   ItemType        = typename _ItemType<A>::Type;

// ---------------------------------------------------------------------
// Convert
// ---------------------------------------------------------------------

template <class A, class T = ItemType<A>>   struct _AsVector        {};
template <class A, class T = ItemType<A>>   using   AsVector        = typename _AsVector<A,T>::Type;
template <class A>                          struct _AsTuple         {};
template <class A>                          using   AsTuple         = typename _AsTuple<A>::Type;
template <class A>                          struct _AsPack          {};
template <class A>                          using   AsPack          = typename _AsPack<A>::Type;

// ---------------------------------------------------------------------
// Apply modifiers
// ---------------------------------------------------------------------

template <class A>                          struct _ApplyAddConst    {};
template <class A>                          using   ApplyAddConst    = typename _ApplyAddConst<A>::Type;
template <class A>                          struct _ApplyAddRef      {};
template <class A>                          using   ApplyAddRef      = typename _ApplyAddRef<A>::Type;
template <class A>                          struct _ApplyAddConstRef {};
template <class A>                          using   ApplyAddConstRef = typename _ApplyAddConstRef<A>::Type;
template <class A>                          struct _ApplyAddPtr      {};
template <class A>                          using   ApplyAddPtr      = typename _ApplyAddPtr<A>::Type;
template <class A>                          struct _ApplyAddConstPtr {};
template <class A>                          using   ApplyAddConstPtr = typename _ApplyAddConstPtr<A>::Type;
template <class A>                          struct _ApplyAddRValueRef{};
template <class A>                          using   ApplyAddRValueRef= typename _ApplyAddRValueRef<A>::Type;
template <class A>                          struct _ApplyRemoveRef   {};
template <class A>                          using   ApplyRemoveRef   = typename _ApplyRemoveRef<A>::Type;
template <class A>                          struct _ApplyRemovePtr   {};
template <class A>                          using   ApplyRemovePtr   = typename _ApplyRemoveRef<A>::Type;
template <class A>                          struct _ApplyRemoveConst {};
template <class A>                          using   ApplyRemoveConst = typename _ApplyRemoveConst<A>::Type;
template <class A>                          struct _ApplyRemoveCV    {};
template <class A>                          using   ApplyRemoveCV    = typename _ApplyRemoveCV<A>::Type;
template <class A>                          struct _ApplyDecay       {};
template <class A>                          using   ApplyDecay       = typename _ApplyDecay<A>::Type;

// ---------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------

template <class A>                          struct _IsVector        { using Type = False; };
template <class A>                          using   IsVector        = typename _IsVector<A>::Type;
template <class A>                          struct _IsTuple         { using Type = False; };
template <class A>                          using   IsTuple         = typename _IsTuple<A>::Type;
template <class A>                          struct _IsPack          { using Type = False; };
template <class A>                          using   IsPack          = typename _IsPack<A>::Type;

} // namespace meta
} // namespace miniten

#endif // MINITEN_META_PACKAPI_H
