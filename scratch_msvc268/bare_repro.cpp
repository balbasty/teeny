// #268 -- bare-metal repro: does the enable_if_t-partitioned operator() overload-set
// SHAPE (no CCCL/mdspan involved at all, just <type_traits>) reproduce the MSVC
// "no matching call operator found" defect on its own? If it does, the bug is in
// this general SFINAE-partitioned-overload-set pattern, not anything specific to
// teeny's mdspan-based tensor implementation -- which would make the fix (likely
// restructuring to if constexpr/tag dispatch instead of enable_if partitioning,
// per the issue's original suggested next steps) much more targeted.
#include <type_traits>

template <class A> struct is_index : std::is_integral<A> {};

template <class T>
struct minitensor {
    T val{};
    // mirrors tensor.h's operator() overload set: 3 categories x {non-const,const},
    // partitioned via enable_if_t on a variadic Args pack -- same shape, no mdspan.
    template <class... Args, std::enable_if_t<(is_index<Args>::value && ...), int> = 0>
    T& operator()(Args...) noexcept { return val; }
    template <class... Args, std::enable_if_t<(is_index<Args>::value && ...), int> = 0>
    const T& operator()(Args...) const noexcept { return val; }

    template <class... Args, std::enable_if_t<!(is_index<Args>::value && ...), int> = 0>
    int operator()(Args...) noexcept { return 0; }
    template <class... Args, std::enable_if_t<!(is_index<Args>::value && ...), int> = 0>
    int operator()(Args...) const noexcept { return 0; }
};

int main() {
    minitensor<double> a; a(0) = 1.0;
    minitensor<int> b;    b(0) = 1;
    return (a(0) == 1.0 && b(0) == 1) ? 0 : 1;
}
