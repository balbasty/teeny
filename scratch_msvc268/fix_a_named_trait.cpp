// #268 -- FIX ATTEMPT A: same SFINAE-partitioned overload-set SHAPE as
// bare_repro.cpp, but the fold-expression moves OUT of enable_if's angle
// brackets into a named trait (is_all_index<Args...>) used by name. Tests
// whether MSVC specifically chokes on a fold-expression written directly
// inside enable_if_t<...>, vs. a pre-computed named trait.
#include <type_traits>

template <class A> struct is_index : std::is_integral<A> {};
template <class... Args> struct is_all_index
    : std::integral_constant<bool, (is_index<Args>::value && ...)> {};

template <class T>
struct minitensor {
    T val{};
    template <class... Args, std::enable_if_t<is_all_index<Args...>::value, int> = 0>
    T& operator()(Args...) noexcept { return val; }
    template <class... Args, std::enable_if_t<is_all_index<Args...>::value, int> = 0>
    const T& operator()(Args...) const noexcept { return val; }

    template <class... Args, std::enable_if_t<!is_all_index<Args...>::value, int> = 0>
    int operator()(Args...) noexcept { return 0; }
    template <class... Args, std::enable_if_t<!is_all_index<Args...>::value, int> = 0>
    int operator()(Args...) const noexcept { return 0; }
};

int main() {
    minitensor<double> a; a(0) = 1.0;
    minitensor<int> b;    b(0) = 1;
    return (a(0) == 1.0 && b(0) == 1) ? 0 : 1;
}
