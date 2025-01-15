#ifndef MINITEN_META_PACKAPI_IMPL_H
#define MINITEN_META_PACKAPI_IMPL_H
#include "packapi.h"
#include "pack.h"
#include "tuple.h"
#include "vector.h"
#include "index.h"

namespace miniten {
namespace meta {

// =====================================================================
//
// Implementation
// We specialize some of the helpers class in container-agnostic ways
//
// =====================================================================

// ---------------------------------------------------------------------
// Cat
// ---------------------------------------------------------------------
// Containers only need to implement _Cat2<CONTAINER,CONTAINER>

template <>                                 struct _Cat<>           { using Type = Pack<>; };
template <class A>                          struct _Cat<A>          { using Type = A; };
template <class A, class M>                 struct _Cat<A,M>        { using Type = typename _Cat2<A,LikeFrom<A,M>>::Type; };
template <class A, class M0, class... M>    struct _Cat<A,M0,M...>  { using Type = Cat<Cat<A,M0>, M...>; };

// ---------------------------------------------------------------------
// Get
// ---------------------------------------------------------------------
// Containers only need to implement _Get<CONTAINER, PtrDiff<0>>

// Multiple indices -> concatenate outputs
template <class A, ptrdiff_t I0, ptrdiff_t...I>
struct _Get<A, PtrDiff<I0, I...>> {
    using Type = Cat<Get<A, PtrDiff<I0>>, Get<A, PtrDiff<I...>>>;
};

// No indices -> empty container
template <class A>
struct _Get<A, PtrDiff<>> {
    using Type = EmptyLike<A>;
};

// Single index -> delete all preceding elements -> fallback to Get<0>
template <class A, ptrdiff_t I>
struct _Get<A, PtrDiff<I>> {
    using Type = GetFirst<DelFirst<A, static_cast<size_t>(I)>>;
};

template <class A, ptrdiff_t I>
struct _GetValue<A, PtrDiff<I>> {
    using Type = GetFirstValue<DelFirst<A, static_cast<size_t>(I)>>;
};

// ---------------------------------------------------------------------
// Del
// ---------------------------------------------------------------------
// Containers only need to implement _Del<CONTAINER, PtrDiff<0>>

// Helper to re-number indices after one element has been deleted.
template <ptrdiff_t J, ptrdiff_t... I>
struct _Del_UpdateIndex {};

template <ptrdiff_t J, ptrdiff_t I0, ptrdiff_t... I>
struct _Del_UpdateIndex<J, I0, I...> {
    using Type = Cat<
        typename _Del_UpdateIndex<J, I0>::Type,
        typename _Del_UpdateIndex<J, I...>::Type
    >;
};

template <ptrdiff_t J, ptrdiff_t I0>
struct _Del_UpdateIndex<J, I0> {
    using Type = PtrDiff<((I0 < J) ? I0 : (I0 - 1))>;
};

template <ptrdiff_t J>
struct _Del_UpdateIndex<J, J> {
    using Type = PtrDiff<>;
};

template <ptrdiff_t J>
struct _Del_UpdateIndex<J> {
    using Type = PtrDiff<>;
};

// Multiple indices -> recursive deletion
template <class A, ptrdiff_t I0, ptrdiff_t...I>
struct _Del<A, PtrDiff<I0, I...>> {
private:
    using DelFirst = Del<A, PtrDiff<I0>>;
    using NewIndex = typename _Del_UpdateIndex<I0, I...>::Type;
public:
    using Type = Del<DelFirst, NewIndex>;
};

// No indices -> return input
template <class A>
struct _Del<A, PtrDiff<>> {
    using Type = A;
};

// Single index -> concatenate left and right containers
template <class A, ptrdiff_t I>
struct _Del<A,PtrDiff<I>>
{
    using Type = Cat<
        GetFirst<A, static_cast<size_t>(I)>,
        DelFirst<A, static_cast<size_t>(I)+1>
    >;
};


// ---------------------------------------------------------------------
// InsertFrom
// ---------------------------------------------------------------------
// Container do not need to implement anything

template <class A, ptrdiff_t I, class... M>
struct _InsertFrom<A, PtrDiff<I>, M...>
{
    using Type = Cat<
        GetFirst<A,I>,
        Cat<M...>,
        DelFirst<A,I>
    >;
};

// ---------------------------------------------------------------------
// SetFrom
// ---------------------------------------------------------------------
// Container do not need to implement anything

// Multiple containers -> contatenate them
template <class A, class I, class M0, class... M>
struct _SetFrom<A, I, M0, M...>
{
    using Type = SetFrom<A, I, Cat<M0, M...>>;
};

// Single container -> recursive call
template <class A, class I, class M>
struct _SetFrom<A, I, M>
{
    static_assert(
        (Length<M>::Value == Length<I>::Value) ||
        (Length<M>::Value == 1),
        "Set requires indices and values to be broadcastable"
    );
    using Type = SwitchCase<
        IsZero<Length<M>>,
            A,
        IsEqual<Length<M>, Length<I>>,
            SetFrom<
                SetFrom<A, GetFirst<I>, GetFirst<M>>,
                DelFirst<I>, DelFirst<M>
            >,
        // Otherwise Length<M> == 1
            SetFrom<
                SetFrom<A, GetFirst<I>, M>,
                DelFirst<I>, M
            >
    >;
};

template <class A, ptrdiff_t I, class M>
struct _SetFrom<A, PtrDiff<I>, M>
{
    // Do not static_assert here so that we can use a "wrong" type in
    // the second branch of the conditional above (we won't ever enter
    // the branch when it's wrong).
    using Type = Cat<GetFirst<A, I>, M, DelFirst<A, I+1>>;
};

} // namespace meta
} // namespace miniten

#endif // MINITEN_META_PACKAPI_IMPL_H
