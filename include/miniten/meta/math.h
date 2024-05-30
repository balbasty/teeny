/***********************************************************************
 * This file implements metaprogramming math
 *
 * IsPositive
 * IsNegative
 * IsZero
 * IsNonPositive
 * IsNonNegative
 * IsNonZero
 * Prod<class T, T...>
 * Sum<class T, T...>
 * Min<class T, T...>
 * Max<class T, T...>
 * Count<class T, T...>
 * CountTypes<class... T>
 ***********************************************************************/
#ifndef MINITEN_META_MATH_H
#define MINITEN_META_MATH_H

namespace miniten {

/// Forward declaration
namespace traits { template <typename T> struct type_info; }

namespace  meta {

/// ---------------------------------------------------------------- ///
///     Sign                                                         ///
/// ---------------------------------------------------------------- ///

template <typename T, T N>
struct IsPositive {
    static constexpr bool Value = N > 0;
};

template <typename T, T N>
struct IsNegative {
    static constexpr bool Value = N < 0;
};

template <typename T, T N>
struct IsZero {
    static constexpr bool Value = N == 0;
};

template <typename T, T N>
struct IsNonPositive {
    static constexpr bool Value = N <= 0;
};

template <typename T, T N>
struct IsNonNegative {
    static constexpr bool Value = N >= 0;
};

template <typename T, T N>
struct IsNonZero {
    static constexpr bool Value = N != 0;
};

/// ---------------------------------------------------------------- ///
///     Product                                                      ///
/// ---------------------------------------------------------------- ///

/// @brief      Compute the product of a series of values
/// @details    Prod<T, N1, N2, ...>::Value == N1*N2*...
/// @tparam T   Type of elements in the pack
/// @tparam N   Elements
template <typename T, T... N>
struct Prod {};

/// By default, multiply first two elements, then recurse
template <typename T, T N0, T... N>
struct Prod<T, N0, N...> {
    static constexpr T Value = N0 * Prod<T, N...>::Value;
};

// If only one element, return it
template <typename T, T N0>
struct Prod<T, N0> {
    static constexpr T Value = N0;
};

// If not elements, return 1
template <typename T>
struct Prod<T> {
    static constexpr T Value = static_cast<T>(1);
};

/// @brief      Alias for product of integers
/// @details    ProdInt<N1, N2, ...>::Value == N1*N2*...
template <int... N>
using ProdInt = Prod<int, N...>;

/// @brief      Alias for product of long integers
/// @details    ProdLong<N1, N2, ...>::Value == N1*N2*...
template <long... N>
using ProdLong = Prod<long, N...>;

/// ---------------------------------------------------------------- ///
///     Sum                                                          ///
/// ---------------------------------------------------------------- ///

/// @brief      Compute the sum of a series of integers
/// @details    Sum<T, N1, N2, ...>::Value == N1+N2+...
/// @tparam T   Type of elements in the pack
/// @tparam N   Elements
template <typename T, T... N>
struct Sum {};

/// By default, add first two elements, then recurse
template <typename T, T N0, T... N>
struct Sum<T, N0, N...> {
    static constexpr T Value = N0 + Sum<T, N...>::Value;
};

// If only one element, return it
template <typename T, T N0>
struct Sum<T, N0> {
    static constexpr T Value = N0;
};

// If not elements, return 0
template <typename T>
struct Sum<T> {
    static constexpr T Value = static_cast<T>(0);
};

/// @brief      Alias for sum of integers
/// @details    SumInt<N1, N2, ...>::Value == N1*N2*...
template <int... N>
using SumInt = Sum<int, N...>;

/// @brief      Alias for sum of long integers
/// @details     SumLong<N1, N2, ...>::Value == N1*N2*...
template <long... N>
using SumLong = Sum<long, N...>;

/// ---------------------------------------------------------------- ///
///     Count                                                        ///
/// ---------------------------------------------------------------- ///

/// @brief Count the number of elements in a parameter pack (of types)
template <typename... T>
struct CountTypes {};

/// By default, count 1 for the first element, then recurse
template <typename T0, typename... T>
struct CountTypes<T0, T...> {
    static constexpr long Value = 1 + CountTypes<T...>::Value;
};

/// I a single element, return 1
template <typename T0>
struct CountTypes<T0> {
    static constexpr long Value = 1;
};

/// I no elements, return 0
template <>
struct CountTypes<> {
    static constexpr long Value = 0;
};

/// @brief Count the number of elements in a parameter pack (of objects)
template <typename T, T... N>
struct Count {};

/// By default, count 1 for the first element, then recurse
template <typename T, T N0, T... N>
struct Count<T, N0, N...> {
    static constexpr long Value = 1 + Count<T, N...>::Value;
};

/// I a single element, return 1
template <typename T, T N0>
struct Count<T, N0> {
    static constexpr long Value = 1;
};

/// I no elements, return 0
template <typename T>
struct Count<T> {
    static constexpr long Value = 0;
};

/// @brief      Alias for integer count
template <int... N>
using CountInt = Count<int, N...>;

/// @brief      Alias for long integer count
template <long... N>
using CountLong = Count<long, N...>;

/// ---------------------------------------------------------------- ///
///     Max                                                          ///
/// ---------------------------------------------------------------- ///

/// @brief      Compute the max of a series of values
/// @details    Max<T, N1, N2, ...>::Value == Nmax
/// @tparam T   Type of elements in the pack
/// @tparam N   Elements
template <typename T, T... N>
struct Max {};

/// By default, max of first two elements, then recurse
template <typename T, T N0, T N1, T... N>
struct Max<T, N0, N1, N...> {
    static constexpr T Value = Max<T, Max<T, N0, N1>::Value, N...>::Value;
};

// If only two element, return the max
template <typename T, T N0, T N1>
struct Max<T, N0, N1> {
    static constexpr T Value = N0 >= N1 ? N0 : N1;
};

// If only one element, return it
template <typename T, T N0>
struct Max<T, N0> {
    static constexpr T Value = N0;
};

// If not elements, return Min
template <typename T>
struct Max<T> {
    static constexpr T Value = traits::type_info<T>::Min;
};

/// @brief      Alias for max of integers
/// @details    MaxInt<N1, N2, ...>::Value == Nmax
template <int... N>
using MaxInt = Max<int, N...>;

/// @brief      Alias for max of long integers
/// @details    MaxLong<N1, N2, ...>::Value == Nmax
template <long... N>
using MaxLong = Max<long, N...>;

/// ---------------------------------------------------------------- ///
///     Min                                                          ///
/// ---------------------------------------------------------------- ///

/// @brief      Compute the min of a series of values
/// @details    Min<T, N1, N2, ...>::Value == Nmin
/// @tparam T   Type of elements in the pack
/// @tparam N   Elements
template <typename T, T... N>
struct Min {};

/// By default, min of first two elements, then recurse
template <typename T, T N0, T N1, T... N>
struct Min<T, N0, N1, N...> {
    static constexpr T Value = Min<T, Min<T, N0, N1>::Value, N...>::Value;
};

// If only two element, return the min
template <typename T, T N0, T N1>
struct Min<T, N0, N1> {
    static constexpr T Value = N0 >= N1 ? N0 : N1;
};

// If only one element, return it
template <typename T, T N0>
struct Min<T, N0> {
    static constexpr T Value = N0;
};

// If not elements, return Max
template <typename T>
struct Min<T> {
    static constexpr T Value = traits::type_info<T>::Max;
};

/// @brief      Alias for min of integers
/// @details    MinInt<N1, N2, ...>::Value == Nmax
template <int... N>
using MinInt = Min<int, N...>;

/// @brief      Alias for mminax of long integers
/// @details    MinLong<N1, N2, ...>::Value == Nmax
template <long... N>
using MinLong = Min<long, N...>;

} // namespace meta
} // namespace miniten

#endif // MINITEN_META_MATH_H
