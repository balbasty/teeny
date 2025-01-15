/***********************************************************************
 * This file implements metaprogramming math
 *
 * IsPositiveValue<T,X>         IsPositive<X>           -> True | False
 * IsNegativeValue<T,X>         IsNegative<X>           -> True | False
 * IsZeroValue<T,X>             IsZero<X>               -> True | False
 * IsNonPositiveValue<T, X>     IsNonPositive<X>        -> True | False
 * IsNonNegativeValue<T, X>     IsNonNegative<X>        -> True | False
 * IsNonZeroValue<T, X>         IsNonZero<X>            -> True | False
 *
 * ProdValues<T, X...>          Prod<Vector>            -> Scalar<T,Result>
 * SumValues<T, X...>           Sum<Vector>             -> Scalar<T,Result>
 * MinValues<T, X...>           Min<Vector>             -> Scalar<T,Result>
 * MaxValues<T, X...>           Max<Vector>             -> Scalar<T,Result>
 * CountValues<T, X...>                                 -> SizeT<Result>
 * CountTypes<T...>                                     -> SizeT<Result>
 ***********************************************************************/
#ifndef MINITEN_META_MATH_IMPL_H
#define MINITEN_META_MATH_IMPL_H

#include "math.h"
#include "traits.h"
#include "vector.h"
#include "packapi.h"

namespace miniten {
namespace  meta {


#define _PROD(A,B) (A * B)
#define _SUM(A,B)  (A + B)
#define _MIN(A,B)  (A <= B ? A : B)
#define _MAX(A,B)  (A >= B ? A : B)
#define _OR(A,B)   (A | B)
#define _AND(A,B)  (A & B)
#define _GT(A,B)   (A > B)
#define _GE(A,B)   (A >= B)
#define _LT(A,B)   (A < B)
#define _LE(A,B)   (A <= B)
#define _EQ(A,B)   (A == B)
#define _NE(A,B)   (A != B)

/// ---------------------------------------------------------------- ///
///     Sign                                                         ///
/// ---------------------------------------------------------------- ///

#define MAKE_MATHTESTUNARY(NAME,TEST) \
    template <class T, T... X>       struct _Is##NAME##Value                {}; \
    template <class T>               struct _Is##NAME##Value<T>             { using Type = Bool<>; }; \
    template <class T, T X>          struct _Is##NAME##Value<T, X>          { using Type = ConditionalB<_##TEST(X,0), True, False>; }; \
    template <class T, T X0, T... X> struct _Is##NAME##Value<T, X0, X...>   { using Type = Cat<Is##NAME##Value<T,X0>, Is##NAME##Value<T,X...>>; }; \
    template <class X>               struct _Is##NAME                       { using Type = Is##NAME<AsVector<X>>; }; \
    template <class T, T... X>       struct _Is##NAME<Vector<T, X...>>      { using Type = Is##NAME##Value<T, X...>; };

MAKE_MATHTESTUNARY(Positive,     GT)
MAKE_MATHTESTUNARY(Negative,     LT)
MAKE_MATHTESTUNARY(Zero,         EQ)
MAKE_MATHTESTUNARY(NonZero,      NE)
MAKE_MATHTESTUNARY(NonePositive, LE)
MAKE_MATHTESTUNARY(NonNegative,  GE)


#define MAKE_MATHTESTBINARY(NAME,TEST) \
    template <class T, T X, T... Y>             struct _Is##NAME##Value                         {}; \
    template <class T, T X>                     struct _Is##NAME##Value<T, X>                   { using Type = Bool<>; }; \
    template <class T, T X, T Y>                struct _Is##NAME##Value<T, X, Y>                { using Type = ConditionalB<_##TEST(X,Y), True, False>; }; \
    template <class T, T X, T Y0, T... Y>       struct _Is##NAME##Value<T, X, Y0, Y...>         { using Type = Cat<Is##NAME##Value<T,X,Y0>, Is##NAME##Value<T,X,Y...>>; }; \
    template <class X, class Y>                 struct _Is##NAME                                { \
    private: \
        using _XV = AsVector<X>; \
        using _YV = AsVector<Y>; \
        using _O0 = _Is##NAME<GetFirst<_XV>, GetFirst<_YV>>; \
        using _O1 = SwitchCase< \
            Bool<(Length<_XV>::Value > 1) && (Length<_YV>::Value > 1)>, \
                _Is##NAME<DelFirst<_XV>, DelFirst<_YV>>, \
            Bool<(Length<_XV>::Value > 1)>, \
                _Is##NAME<DelFirst<_XV>, GetFirst<_YV>>, \
            \
                _Is##NAME<GetFirst<_XV>, DelFirst<_YV>> \
        >; \
    public: \
        using Type = Cat<_O0, _O1>; \
    }; \
    template <class U, U X, class T, T Y>       struct _Is##NAME<Vector<U,X>, Vector<T,Y>>      { using Type = ConditionalB<_##TEST(X,Y), True, False>; }; \
    template <class X, class T, T Y>            struct _Is##NAME<X, Vector<T,Y>>                { using Type = Cat<Is##NAME<GetFirst<X>,Vector<T,Y>>, Is##NAME<DelFirst<X>,Vector<T,Y>>>; }; \
    template <class T, class Y, T X>            struct _Is##NAME<Vector<T,X>, Y>                { using Type = Cat<Is##NAME<Vector<T,X>,GetFirst<Y>>, Is##NAME<Vector<T,X>,DelFirst<Y>>>; };


MAKE_MATHTESTBINARY(Greater,      GT)
MAKE_MATHTESTBINARY(Lower,        LT)
MAKE_MATHTESTBINARY(Equal,        EQ)
MAKE_MATHTESTBINARY(NotEqual,     NE)
MAKE_MATHTESTBINARY(GreaterEqual, GE)
MAKE_MATHTESTBINARY(LowerEqual,   LE)


// template <class T, T... X>       struct _IsPositiveValue                {};
// template <class T, T X>          using   IsPositiveValue                = typename _IsPositiveValue<T, X...>::Type;
// template <class T>               struct _IsPositiveValue<T>             { using Type = Bool<>; };
// template <class T, T X>          struct _IsPositiveValue<T, X>          { using Type = Conditional<(X > 0), True, False>; };
// template <class T, T X0, T... X> struct _IsPositiveValue<T, X0, X...>   { using Type = Cat<IsPositiveValue<T,X0>, IsPositiveValue<T,X...>>; };
// template <class X>               struct _IsPositive                     { using Type = IsPositive<AsVector<X>>; };
// template <class T, T... X>       struct _IsPositive<Vector<T, X...>>    { using Type = IsPositiveValue<T, X...>; };
// template <class X>               using   IsPositive                     = typename _IsPositive<X>::Type;

/// ---------------------------------------------------------------- ///
///     Reductions                                                   ///
/// ---------------------------------------------------------------- ///

#define MAKE_MATHREDUCE(NAME, OP, INIT) \
    template <class T, T... X>          struct __##NAME##V                {}; \
    template <class T, T... X>          struct _##NAME##V                 { using Type = Vector<T, __##NAME##V<T, X...>::Value>; }; \
    template <class T, T X0, T... X>    struct __##NAME##V<T, X0, X...>   { static constexpr T Value = _##OP(X0, (__##NAME##V<T, X...>::Value)); }; \
    template <class T, T X0>            struct __##NAME##V<T, X0>         { static constexpr T Value = X0; }; \
    template <class T>                  struct __##NAME##V<T>             { static constexpr T Value = static_cast<T>(INIT); }; \
    template <class A> \
    struct _##NAME \
    { \
    protected: \
        using _ItemType = ItemType<GetFirstValue<A>>; \
        using _AsVector = AsVector<A, _ItemType>; \
    public: \
        using Type = typename _##NAME<_AsVector>::Type; \
    }; \
    template <class T, T... X> \
    struct _##NAME<Vector<T, X...>> { \
        using Type = NAME##Values<T, X...>; \
    };

MAKE_MATHREDUCE(Prod, PROD, 1)
MAKE_MATHREDUCE(Sum,  SUM,  0)
MAKE_MATHREDUCE(Min,  MIN,  TypeInfo<T>::Max)
MAKE_MATHREDUCE(Max,  MAX,  TypeInfo<T>::Min)
MAKE_MATHREDUCE(Or,   OR,   false)

// /// ---------------------------------------------------------------- ///
// ///     Product                                                      ///
// /// ---------------------------------------------------------------- ///

// template <class T, T... X>          struct _ProdV                {};
// template <class T, T X0, T... X>    struct _ProdV<T, X0, X...>   { static constexpr T Value = X0 * _ProdV<T, X...>::Value; };
// template <class T, T X0>            struct _ProdV<T, X0>         { static constexpr T Value = X0; };
// template <class T>                  struct _ProdV<T>             { static constexpr T Value = static_cast<T>(1); };

// /// @brief      Compute the product of a series of values
// /// @details    ProdValues<T, N1, N2, ...>::Value == N1*N2*...
// /// @tparam T   Type of elements in the pack
// /// @tparam X   Elements
// template <class T, T... X>
// using ProdValues = Vector<T, _ProdV<T>::Value>;

// /// @brief      Alias for product of integers
// /// @details    ProdInt<N1, N2, ...>::Value == N1*N2*...
// template <int... X>
// using ProdInt = ProdValues<int, X...>;

// /// @brief      Alias for product of long integers
// /// @details    ProdLong<N1, N2, ...>::Value == N1*N2*...
// template <long... X>
// using ProdLong = ProdValues<long, X...>;

// template <class A>
// struct _Prod
// {
// protected:
//     using _ItemType = ItemType<GetFirstValue<A>>;
//     using _AsVector = AsVector<A, _ItemType>;
// public:
//     using Type = typename _Prod<_AsVector>::Type;
// };

// template <class T, T... X>
// struct _Prod<Vector<T, X...>> {
//     using Type = ProdValues<T, X...>;
// };

// /// @brief      Compute the product of a series of values
// /// @details    Prod<Vector<T, N1, N2, ...>>::Value == N1*N2*...
// /// @tparam A   Pack type
// template <class A>
// using Prod = typename _Prod<A>::Type;

// /// ---------------------------------------------------------------- ///
// ///     Sum                                                          ///
// /// ---------------------------------------------------------------- ///

// template <class T, T... X>          struct _SumV                 {};
// template <class T, T X0, T... X>    struct _SumV<T, X0, X...>    { static constexpr T Value = X0 + _SumV<T, X...>::Value; };
// template <class T, T X0>            struct _SumV<T, X0>          { static constexpr T Value = X0; };
// template <class T>                  struct _SumV<T>              { static constexpr T Value = static_cast<T>(0); };

// /// @brief      Compute the sum of a series of integers
// /// @details    Sum<T, N1, N2, ...>::Value == N1+N2+...
// /// @tparam T   Type of elements in the pack
// /// @tparam X   Elements
// template <class T, T... X>
// using SumValues = Vector<T, _SumV<T, X...>::Value>;

// /// @brief      Alias for sum of integers
// /// @details    SumInt<N1, N2, ...>::Value == N1*N2*...
// template <int... X>
// using SumInt = Sum<int, X...>;

// /// @brief      Alias for sum of long integers
// /// @details     SumLong<N1, N2, ...>::Value == N1*N2*...
// template <long... X>
// using SumLong = Sum<long, X...>;

// template <class A>
// struct _Sum
// {
// protected:
//     using _ItemType = ItemType<GetFirstValue<A>>;
//     using _AsVector = AsVector<A, _ItemType>;
// public:
//     using Type = typename _Sum<_AsVector>::Type;
// };

// template <class T, T... X>
// struct _Sum<Vector<T, X...>> {
//     using Type = SumValues<T, X...>;
// };

// /// @brief      Compute the sum of a series of integers
// /// @details    Sum<T, N1, N2, ...>::Value == N1+N2+...
// /// @tparam A   Pack type
// template <class A>
// using Sum = typename _Sum<A>::Type;

// /// ---------------------------------------------------------------- ///
// ///     Max                                                          ///
// /// ---------------------------------------------------------------- ///

// template <class T, T... X>              struct _MaxV                  {};
// template <class T, T X0, T N1, T... X>  struct _MaxV<T, X0, N1, X...> { static constexpr T Value = _MaxV<T, _MaxV<T, X0, N1>::Value, X...>::Value; };
// template <class T, T X0, T N1>          struct _MaxV<T, X0, N1>       { static constexpr T Value = X0 >= N1 ? X0 : N1; };
// template <class T, T X0>                struct _MaxV<T, X0>           { static constexpr T Value = X0; };
// template <class T>                      struct _MaxV<T>               { static constexpr T Value = TypeInfo<T>::Min; };

// /// @brief      Compute the max of a series of values
// /// @details    Max<T, N1, N2, ...>::Value == Nmax
// /// @tparam T   Type of elements in the pack
// /// @tparam X   Elements
// template <class T, T... X>
// using MaxValues = Vector<T, _MaxV::Value>;

// /// @brief      Alias for max of integers
// /// @details    MaxInt<N1, N2, ...>::Value == Nmax
// template <int... X>
// using MaxInt = Max<int, X...>;

// /// @brief      Alias for max of long integers
// /// @details    MaxLong<N1, N2, ...>::Value == Nmax
// template <long... X>
// using MaxLong = Max<long, X...>;

// template <class A>
// struct _Max
// {
// protected:
//     using _ItemType = ItemType<GetFirstValue<A>>;
//     using _AsVector = AsVector<A, _ItemType>;
// public:
//     using Type = typename _Max<_AsVector>::Type;
// };

// template <class T, T... X>
// struct _Max<Vector<T, X...>> {
//     using Type = MaxValues<T, X...>;
// };

// /// @brief      Compute the max of a series of values
// /// @details    Max<T, N1, N2, ...>::Value == Nmax
// /// @tparam A   Pack type
// template <class A>
// using Max = typename _Max<A>::Type;

// /// ---------------------------------------------------------------- ///
// ///     Min                                                          ///
// /// ---------------------------------------------------------------- ///

// template <class T, T... X>              struct _MinV                     {};
// template <class T, T X0, T N1, T... X>  struct _MinV<T, X0, N1, X...>    { static constexpr T Value = _MinV<T, _MinV<T, X0, N1>::Value, X...>::Value; };
// template <class T, T X0, T N1>          struct _MinV<T, X0, N1>          { static constexpr T Value = X0 >= N1 ? X0 : N1; };
// template <class T, T X0>                struct _MinV<T, X0>              { static constexpr T Value = X0; };
// template <class T>                      struct _MinV<T>                  { static constexpr T Value = TypeInfo<T>::Max; };

// /// @brief      Compute the min of a series of values
// /// @details    Min<T, N1, N2, ...>::Value == Nmin
// /// @tparam T   Type of elements in the pack
// /// @tparam X   Elements
// template <class T, T... X>
// using MinValues = Vector<T, _MinV::Value>;

// /// @brief      Alias for min of integers
// /// @details    MinInt<N1, N2, ...>::Value == Nmax
// template <int... X>
// using MinInt = Min<int, X...>;

// /// @brief      Alias for mminax of long integers
// /// @details    MinLong<N1, N2, ...>::Value == Nmax
// template <long... X>
// using MinLong = Min<long, X...>;

// template <class A>
// struct _Min
// {
// protected:
//     using _ItemType = ItemType<GetFirstValue<A>>;
//     using _AsVector = AsVector<A, _ItemType>;
// public:
//     using Type = typename _Min<_AsVector>::Type;
// };

// template <class T, T... X>
// struct _Min<Vector<T, X...>> {
//     using Type = MinValues<T, X...>;
// };

// /// @brief      Compute the min of a series of values
// /// @details    Min<T, N1, N2, ...>::Value == Nmin
// /// @tparam A   Pack type
// template <class A>
// using Min = typename _Min<A>::Type;

/// ---------------------------------------------------------------- ///
///     Count                                                        ///
/// ---------------------------------------------------------------- ///

template <class... T>               struct __CountT              {};
template <class T0, class... T>     struct __CountT<T0, T...>    { static constexpr size_t Value = 1 + __CountT<T...>::Value; };
template <class T0>                 struct __CountT<T0>          { static constexpr size_t Value = 1; };
template <>                         struct __CountT<>            { static constexpr size_t Value = 0; };
template <class... T>               struct  _CountT              { using Type = SizeT<__CountT<T...>::Value>; };

template <class T, T... X>          struct __CountV              {};
template <class T, T X0, T... X>    struct __CountV<T, X0, X...> { static constexpr size_t Value = 1 + __CountV<T, X...>::Value; };
template <class T, T X0>            struct __CountV<T, X0>       { static constexpr size_t Value = 1; };
template <class T>                  struct __CountV<T>           { static constexpr size_t Value = 0; };
template <class T, T... X>         struct   _CountV              { using Type = SizeT<__CountV<T, X...>::Value>; };

} // namespace meta
} // namespace miniten

#endif // MINITEN_META_MATH_IMPL_H
