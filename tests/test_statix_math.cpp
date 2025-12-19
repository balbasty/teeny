#include <teeny/statix.h>

using namespace tny::statix;

int main()
{
    using tny::statix::div; // conflicts with std::div for some reason?

    static_assert( is_positive_value      <int,  1>(), "");
    static_assert(!is_positive_value      <int,  0>(), "");
    static_assert(!is_positive_value      <int, -1>(), "");
    static_assert(!is_negative_value      <int,  1>(), "");
    static_assert(!is_negative_value      <int,  0>(), "");
    static_assert( is_negative_value      <int, -1>(), "");
    static_assert(!is_nonpositive_value   <int,  1>(), "");
    static_assert( is_nonpositive_value   <int,  0>(), "");
    static_assert( is_nonpositive_value   <int, -1>(), "");
    static_assert( is_nonnegative_value   <int,  1>(), "");
    static_assert( is_nonnegative_value   <int,  0>(), "");
    static_assert(!is_nonnegative_value   <int, -1>(), "");
    static_assert(!is_zero_value          <int,  1>(), "");
    static_assert( is_zero_value          <int,  0>(), "");
    static_assert(!is_zero_value          <int, -1>(), "");
    static_assert( is_nonzero_value       <int,  1>(), "");
    static_assert(!is_nonzero_value       <int,  0>(), "");
    static_assert( is_nonzero_value       <int, -1>(), "");

    static_assert( is_positive       <cint< 1>>(), "");
    static_assert(!is_positive       <cint< 0>>(), "");
    static_assert(!is_positive       <cint<-1>>(), "");
    static_assert(!is_negative       <cint< 1>>(), "");
    static_assert(!is_negative       <cint< 0>>(), "");
    static_assert( is_negative       <cint<-1>>(), "");
    static_assert(!is_nonpositive    <cint< 1>>(), "");
    static_assert( is_nonpositive    <cint< 0>>(), "");
    static_assert( is_nonpositive    <cint<-1>>(), "");
    static_assert( is_nonnegative    <cint< 1>>(), "");
    static_assert( is_nonnegative    <cint< 0>>(), "");
    static_assert(!is_nonnegative    <cint<-1>>(), "");
    static_assert(!is_zero           <cint< 1>>(), "");
    static_assert( is_zero           <cint< 0>>(), "");
    static_assert(!is_zero           <cint<-1>>(), "");
    static_assert( is_nonzero        <cint< 1>>(), "");
    static_assert(!is_nonzero        <cint< 0>>(), "");
    static_assert( is_nonzero        <cint<-1>>(), "");

    static_assert(is_same<is_positive     <cint<1, 0, -1>>,  cbool<true,  false, false>>(), "");
    static_assert(is_same<is_negative     <cint<1, 0, -1>>,  cbool<false, false, true>>(),  "");
    static_assert(is_same<is_nonpositive  <cint<1, 0, -1>>,  cbool<false, true,  true>>(),  "");
    static_assert(is_same<is_nonnegative  <cint<1, 0, -1>>,  cbool<true,  true,  false>>(), "");
    static_assert(is_same<is_zero         <cint<1, 0, -1>>,  cbool<false, true,  false>>(), "");
    static_assert(is_same<is_nonzero      <cint<1, 0, -1>>,  cbool<true,  false, true>>(),  "");

    static_assert(is_same<is_equal         <cint<1, -1>, cint<1,  1>>,   cbool<true,  false>>(), "");
    static_assert(is_same<is_not_equal     <cint<1, -1>, cint<1,  1>>,   cbool<false, true>>(),  "");
    static_assert(is_same<is_greater       <cint<1,  1>, cint<1, -1>>,   cbool<false, true>>(),  "");
    static_assert(is_same<is_lower         <cint<1, -1>, cint<1,  1>>,   cbool<false, true>>(),  "");
    static_assert(is_same<is_greater_equal <cint<1,  1>, cint<1, -1>>,   cbool<true,  true>>(),  "");
    static_assert(is_same<is_lower_equal   <cint<1, -1>, cint<1,  1>>,   cbool<true,  true>>(),  "");

    static_assert(is_same<prod<cint<1, 2, 3>>, cint<6>>(), "");
    static_assert(is_same<sum <cint<1, 2, 3>>, cint<6>>(), "");
    static_assert(is_same<min <cint<1, 2, 3>>, cint<1>>(), "");
    static_assert(is_same<min <cint<2, 1, 3>>, cint<1>>(), "");
    static_assert(is_same<max <cint<1, 2, 3>>, cint<3>>(), "");
    static_assert(is_same<max <cint<1, 3, 2>>, cint<3>>(), "");
    static_assert( any<cbool<true,  false, true>>(),     "");
    static_assert(!any<cbool<false, false, false>>(),    "");
    static_assert( all<cbool<true,  true,  true>>(),     "");
    static_assert(!all<cbool<true,  false, true>>(),     "");

    static_assert(is_same<add<cint<1, 2, 3>, cint<1>>, cint<2, 3, 4>>(), "");
    static_assert(is_same<mul<cint<1, 2, 3>, cint<2>>, cint<2, 4, 6>>(), "");
    static_assert(is_same<sub<cint<1, 2, 3>, cint<1>>, cint<0, 1, 2>>(), "");
    static_assert(is_same<div<cint<1, 2, 3>, cint<2>>, cint<0, 1, 1>>(), "");

    static_assert(is_same<boolean_and<cbool<true, false>, ctrue>,  cbool<true,  false>>(), "");
    static_assert(is_same<boolean_and<cbool<true, false>, cfalse>, cbool<false, false>>(), "");
    static_assert(is_same<boolean_or <cbool<true, false>, ctrue>,  cbool<true,  true>>(),  "");
    static_assert(is_same<boolean_or <cbool<true, false>, cfalse>, cbool<true,  false>>(), "");
    static_assert(is_same<boolean_xor<cbool<true, false>, ctrue>,  cbool<false, true>>(),  "");
    static_assert(is_same<boolean_xor<cbool<true, false>, cfalse>, cbool<true,  false>>(), "");

    static_assert(is_same<add<cint<4, 5, 6>, cint<1, 2, 3>>, cint<5,  7,  9>>(), "");
    static_assert(is_same<mul<cint<4, 5, 6>, cint<1, 2, 3>>, cint<4, 10, 18>>(), "");
    static_assert(is_same<sub<cint<4, 5, 6>, cint<1, 2, 3>>, cint<3,  3,  3>>(), "");
    static_assert(is_same<div<cint<4, 5, 6>, cint<1, 2, 3>>, cint<4,  2,  2>>(), "");

    static_assert(is_same<cumprod<cint<1, 2, 3>>, cint<1, 2, 6>>(), "");
    static_assert(is_same<cumsum <cint<1, 2, 3>>, cint<1, 3, 6>>(), "");
    static_assert(is_same<cummin <cint<1, 2, 3>>, cint<1, 1, 1>>(), "");
    static_assert(is_same<cummin <cint<2, 1, 3>>, cint<2, 1, 1>>(), "");
    static_assert(is_same<cummax <cint<1, 2, 3>>, cint<1, 2, 3>>(), "");
    static_assert(is_same<cummax <cint<1, 3, 2>>, cint<1, 3, 3>>(), "");
    static_assert(is_same<cumany<cbool<true,  false, true>>,  cbool<true,   true, true>>(),     "");
    static_assert(is_same<cumany<cbool<false, true,  false>>, cbool<false,  true, true>>(),     "");
    static_assert(is_same<cumall<cbool<true,  true,  true>>,  cbool<true,   true, true>>(),     "");
    static_assert(is_same<cumall<cbool<true,  false, true>>,  cbool<true,  false, false>>(),    "");

    constexpr int MN = numeric_limits<int>::min();
    constexpr int MX = numeric_limits<int>::max();

    static_assert(is_same<shifted_cumprod<cint<1, 2, 3>>, cint<1, 1, 2>>(),  "");
    static_assert(is_same<shifted_cumsum <cint<1, 2, 3>>, cint<0, 1, 3>>(),  "");
    static_assert(is_same<shifted_cummin <cint<1, 2, 3>>, cint<MX, 1, 1>>(), "");
    static_assert(is_same<shifted_cummin <cint<2, 1, 3>>, cint<MX, 2, 1>>(), "");
    static_assert(is_same<shifted_cummax <cint<1, 2, 3>>, cint<MN, 1, 2>>(), "");
    static_assert(is_same<shifted_cummax <cint<1, 3, 2>>, cint<MN, 1, 3>>(), "");
    static_assert(is_same<shifted_cumany<cbool<true,  false, true>>, cbool<false, true,  true>>(),     "");
    static_assert(is_same<shifted_cumany<cbool<false, true, false>>, cbool<false, false, true>>(),     "");
    static_assert(is_same<shifted_cumall<cbool<true,  true,  true>>, cbool<true,  true,  true>>(),     "");
    static_assert(is_same<shifted_cumall<cbool<true,  false, true>>, cbool<true,  true,  false>>(),    "");
}
