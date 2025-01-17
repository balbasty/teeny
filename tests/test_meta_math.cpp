#include <miniten/meta.h>

using namespace miniten::meta;

int main()
{
    static_assert( IsPositiveValue      <int,  1>(), "");
    static_assert(!IsPositiveValue      <int,  0>(), "");
    static_assert(!IsPositiveValue      <int, -1>(), "");
    static_assert(!IsNegativeValue      <int,  1>(), "");
    static_assert(!IsNegativeValue      <int,  0>(), "");
    static_assert( IsNegativeValue      <int, -1>(), "");
    static_assert(!IsNonPositiveValue   <int,  1>(), "");
    static_assert( IsNonPositiveValue   <int,  0>(), "");
    static_assert( IsNonPositiveValue   <int, -1>(), "");
    static_assert( IsNonNegativeValue   <int,  1>(), "");
    static_assert( IsNonNegativeValue   <int,  0>(), "");
    static_assert(!IsNonNegativeValue   <int, -1>(), "");
    static_assert(!IsZeroValue          <int,  1>(), "");
    static_assert( IsZeroValue          <int,  0>(), "");
    static_assert(!IsZeroValue          <int, -1>(), "");
    static_assert( IsNonZeroValue       <int,  1>(), "");
    static_assert(!IsNonZeroValue       <int,  0>(), "");
    static_assert( IsNonZeroValue       <int, -1>(), "");

    static_assert( IsPositive       <Int< 1>>(), "");
    static_assert(!IsPositive       <Int< 0>>(), "");
    static_assert(!IsPositive       <Int<-1>>(), "");
    static_assert(!IsNegative       <Int< 1>>(), "");
    static_assert(!IsNegative       <Int< 0>>(), "");
    static_assert( IsNegative       <Int<-1>>(), "");
    static_assert(!IsNonPositive    <Int< 1>>(), "");
    static_assert( IsNonPositive    <Int< 0>>(), "");
    static_assert( IsNonPositive    <Int<-1>>(), "");
    static_assert( IsNonNegative    <Int< 1>>(), "");
    static_assert( IsNonNegative    <Int< 0>>(), "");
    static_assert(!IsNonNegative    <Int<-1>>(), "");
    static_assert(!IsZero           <Int< 1>>(), "");
    static_assert( IsZero           <Int< 0>>(), "");
    static_assert(!IsZero           <Int<-1>>(), "");
    static_assert( IsNonZero        <Int< 1>>(), "");
    static_assert(!IsNonZero        <Int< 0>>(), "");
    static_assert( IsNonZero        <Int<-1>>(), "");

    static_assert(IsSame<IsPositive     <Int<1, 0, -1>>,  Bool<true,  false, false>>(), "");
    static_assert(IsSame<IsNegative     <Int<1, 0, -1>>,  Bool<false, false, true>>(),  "");
    static_assert(IsSame<IsNonPositive  <Int<1, 0, -1>>,  Bool<false, true,  true>>(),  "");
    static_assert(IsSame<IsNonNegative  <Int<1, 0, -1>>,  Bool<true,  true,  false>>(), "");
    static_assert(IsSame<IsZero         <Int<1, 0, -1>>,  Bool<false, true,  false>>(), "");
    static_assert(IsSame<IsNonZero      <Int<1, 0, -1>>,  Bool<true,  false, true>>(),  "");

    static_assert(IsSame<IsEqual        <Int<1, -1>, Int<1,  1>>,   Bool<true,  false>>(), "");
    static_assert(IsSame<IsNotEqual     <Int<1, -1>, Int<1,  1>>,   Bool<false, true>>(),  "");
    static_assert(IsSame<IsGreater      <Int<1,  1>, Int<1, -1>>,   Bool<false, true>>(),  "");
    static_assert(IsSame<IsLower        <Int<1, -1>, Int<1,  1>>,   Bool<false, true>>(),  "");
    static_assert(IsSame<IsGreaterEqual <Int<1,  1>, Int<1, -1>>,   Bool<true,  true>>(),  "");
    static_assert(IsSame<IsLowerEqual   <Int<1, -1>, Int<1,  1>>,   Bool<true,  true>>(),  "");

    static_assert(IsSame<Prod<Int<1, 2, 3>>, Int<6>>(), "");
    static_assert(IsSame<Sum <Int<1, 2, 3>>, Int<6>>(), "");
    static_assert(IsSame<Min <Int<1, 2, 3>>, Int<1>>(), "");
    static_assert(IsSame<Min <Int<2, 1, 3>>, Int<1>>(), "");
    static_assert(IsSame<Max <Int<1, 2, 3>>, Int<3>>(), "");
    static_assert(IsSame<Max <Int<1, 3, 2>>, Int<3>>(), "");
    static_assert( Any<Bool<true,  false, true>>(),     "");
    static_assert(!Any<Bool<false, false, false>>(),    "");
    static_assert( All<Bool<true,  true,  true>>(),     "");
    static_assert(!All<Bool<true,  false, true>>(),     "");

    static_assert(IsSame<Add<Int<1, 2, 3>, Int<1>>, Int<2, 3, 4>>(), "");
    static_assert(IsSame<Mul<Int<1, 2, 3>, Int<2>>, Int<2, 4, 6>>(), "");
    static_assert(IsSame<Sub<Int<1, 2, 3>, Int<1>>, Int<0, 1, 2>>(), "");
    static_assert(IsSame<Div<Int<1, 2, 3>, Int<2>>, Int<0, 1, 1>>(), "");

    static_assert(IsSame<And<Bool<true, false>, True>,  Bool<true,  false>>(), "");
    static_assert(IsSame<And<Bool<true, false>, False>, Bool<false, false>>(), "");
    static_assert(IsSame<Or <Bool<true, false>, True>,  Bool<true,  true>>(),  "");
    static_assert(IsSame<Or <Bool<true, false>, False>, Bool<true,  false>>(), "");
    static_assert(IsSame<Xor<Bool<true, false>, True>,  Bool<false, true>>(),  "");
    static_assert(IsSame<Xor<Bool<true, false>, False>, Bool<true,  false>>(), "");

    static_assert(IsSame<Add<Int<4, 5, 6>, Int<1, 2, 3>>, Int<5,  7,  9>>(), "");
    static_assert(IsSame<Mul<Int<4, 5, 6>, Int<1, 2, 3>>, Int<4, 10, 18>>(), "");
    static_assert(IsSame<Sub<Int<4, 5, 6>, Int<1, 2, 3>>, Int<3,  3,  3>>(), "");
    static_assert(IsSame<Div<Int<4, 5, 6>, Int<1, 2, 3>>, Int<4,  2,  2>>(), "");

    static_assert(IsSame<CumProd<Int<1, 2, 3>>, Int<1, 2, 6>>(), "");
    static_assert(IsSame<CumSum <Int<1, 2, 3>>, Int<1, 3, 6>>(), "");
    static_assert(IsSame<CumMin <Int<1, 2, 3>>, Int<1, 1, 1>>(), "");
    static_assert(IsSame<CumMin <Int<2, 1, 3>>, Int<2, 1, 1>>(), "");
    static_assert(IsSame<CumMax <Int<1, 2, 3>>, Int<1, 2, 3>>(), "");
    static_assert(IsSame<CumMax <Int<1, 3, 2>>, Int<1, 3, 3>>(), "");
    static_assert(IsSame<CumAny<Bool<true,  false, true>>, Bool<true,   true, true>>(),     "");
    static_assert(IsSame<CumAny<Bool<false, true, false>>, Bool<false,  true, true>>(),     "");
    static_assert(IsSame<CumAll<Bool<true,  true,  true>>, Bool<true,   true, true>>(),     "");
    static_assert(IsSame<CumAll<Bool<true,  false, true>>, Bool<true,  false, false>>(),    "");

    constexpr int MN = miniten::TypeInfo<int>::Min;
    constexpr int MX = miniten::TypeInfo<int>::Max;

    static_assert(IsSame<ShiftedCumProd<Int<1, 2, 3>>, Int<1, 1, 2>>(),  "");
    static_assert(IsSame<ShiftedCumSum <Int<1, 2, 3>>, Int<0, 1, 3>>(),  "");
    static_assert(IsSame<ShiftedCumMin <Int<1, 2, 3>>, Int<MX, 1, 1>>(), "");
    static_assert(IsSame<ShiftedCumMin <Int<2, 1, 3>>, Int<MX, 2, 1>>(), "");
    static_assert(IsSame<ShiftedCumMax <Int<1, 2, 3>>, Int<MN, 1, 2>>(), "");
    static_assert(IsSame<ShiftedCumMax <Int<1, 3, 2>>, Int<MN, 1, 3>>(), "");
    static_assert(IsSame<ShiftedCumAny<Bool<true,  false, true>>, Bool<false, true,  true>>(),     "");
    static_assert(IsSame<ShiftedCumAny<Bool<false, true, false>>, Bool<false, false, true>>(),     "");
    static_assert(IsSame<ShiftedCumAll<Bool<true,  true,  true>>, Bool<true,  true,  true>>(),     "");
    static_assert(IsSame<ShiftedCumAll<Bool<true,  false, true>>, Bool<true,  true,  false>>(),    "");
}
