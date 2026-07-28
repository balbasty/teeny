// #268 -- FIX ATTEMPT B: no SFINAE-partitioned overload set at all -- ONE
// operator() template per const-qualification, dispatching internally via
// `if constexpr`. This is the "Suggested next steps" remedy from the original
// issue. Tests whether ditching enable_if-partitioned overloads entirely
// sidesteps the MSVC defect.
#include <type_traits>

template <class A> struct is_index : std::is_integral<A> {};
template <class... Args> struct is_all_index
    : std::integral_constant<bool, (is_index<Args>::value && ...)> {};

template <class T>
struct minitensor {
    T val{};
    template <class... Args>
    decltype(auto) operator()(Args...) noexcept {
        if constexpr (is_all_index<Args...>::value) return (T&)val;
        else return 0;
    }
    template <class... Args>
    decltype(auto) operator()(Args...) const noexcept {
        if constexpr (is_all_index<Args...>::value) return (const T&)val;
        else return 0;
    }
};

int main() {
    minitensor<double> a; a(0) = 1.0;
    minitensor<int> b;    b(0) = 1;
    return (a(0) == 1.0 && b(0) == 1) ? 0 : 1;
}
