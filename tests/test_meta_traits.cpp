#include <miniten/meta.h>

using namespace miniten::meta;

void func() {}

int main()
{
    static_assert(IsSame<int,               int>(),     "");
    static_assert(!IsSame<int,              float>(),   "");
    static_assert(IsSame<Identity<int>,     int>(),     "");
    static_assert(IsSame<Void<int>,         void>(),    "");
    static_assert(IsSame<Void<int, float>,  void>(),    "");

    static_assert(IsSame<ForwardAs<int&,        float>,     float&>(),       "");
    static_assert(IsSame<ForwardAs<int&,        float&>,    float&>(),       "");
    static_assert(IsSame<ForwardAs<int&,        float&&>,   float&>(),       "");
    static_assert(IsSame<ForwardAs<int const&,  float>,     float const&>(), "");

    static_assert(IsSame<RemoveRef  <int&>,                 int>(), "");
    static_assert(IsSame<RemoveRef  <int>,                  int>(), "");
    static_assert(IsSame<RemovePtr  <int*>,                 int>(), "");
    static_assert(IsSame<RemovePtr  <int>,                  int>(), "");
    static_assert(IsSame<RemoveConst<const int>,            int>(), "");
    static_assert(IsSame<RemoveConst<int>,                  int>(), "");
    static_assert(IsSame<RemoveCV   <int>,                  int>(), "");
    static_assert(IsSame<RemoveCV   <const int>,            int>(), "");
    static_assert(IsSame<RemoveCV   <volatile int>,         int>(), "");
    static_assert(IsSame<RemoveCV   <const volatile int>,   int>(), "");

    static_assert(IsSame<AddRef         <int>,              int&>(),        "");
    static_assert(IsSame<AddRef         <int&>,             int&>(),        "");
    static_assert(IsSame<AddPtr         <int>,              int*>(),        "");
    static_assert(IsSame<AddPtr         <int*>,             int**>(),       "");
    static_assert(IsSame<AddConst       <int>,              const int>(),   "");
    static_assert(IsSame<AddConst       <const int>,        const int>(),   "");
    static_assert(IsSame<AddConstRef    <int>,              const int &>(), "");
    static_assert(IsSame<AddConstPtr    <int>,              const int *>(), "");

    static_assert(IsSame<Conditional<true,  int, float>,                int>(),   "");
    static_assert(IsSame<Conditional<false, int, float>,                float>(), "");
    static_assert(IsSame<IfElse     <True,  int, float>,                int>(),   "");
    static_assert(IsSame<IfElse     <False, int, float>,                float>(), "");
    static_assert(IsSame<IfElse     <True,  int, True,  float, bool>,   int>(),   "");
    static_assert(IsSame<IfElse     <True,  int, False, float, bool>,   int>(),   "");
    static_assert(IsSame<IfElse     <False, int, True,  float, bool>,   float>(), "");
    static_assert(IsSame<IfElse     <False, int, False, float, bool>,   bool>(),  "");

    auto lam = []() {};
    auto var = 1;
    static_assert(IsFunction<decltype(func)>(), "");
    static_assert(!IsFunction<decltype(lam)>(), "");
    static_assert(!IsFunction<decltype(var)>(), "");

    static_assert( HaveEq       <int, int>(),   "");
    static_assert( HaveEq       <int, float>(), "");
    static_assert(!HaveEq       <int, void>(),  "");
    static_assert( HaveLess     <int, int>(),   "");
    static_assert( HaveLess     <int, float>(), "");
    static_assert(!HaveLess     <int, void>(),  "");
    static_assert( IsComparable <int, int>(),   "");
    static_assert( IsComparable <int, float>(), "");
    static_assert(!IsComparable <int, void>(),  "");
}
