#include <teeny/statix.h>

using namespace tny::statix;

int main()
{
    using X = cint<1, 2, 3>;
    using Y = cint<4, 5>;
    using Z = cint<1, 2>;
    using P = pack<cint<1>, cint<2>>;

    static_assert( is_carray <X>(), "");
    static_assert(!is_pack   <X>(), "");
    static_assert(!is_tuple  <X>(), "");

    static_assert(is_same<size           <X>,                   csize<3>            >(), "");
    static_assert(is_same<empty_like     <X>,                   cint<>              >(), "");
    static_assert(is_same<reversed       <X>,                   cint<3,2,1>         >(), "");
    static_assert(is_same<like_from      <X,Y>,                 cint<4,5>           >(), "");
    static_assert(is_same<like_from      <X,P>,                 Z                   >(), "");
    static_assert(is_same<like           <X,clong<1>>,          cint<1>             >(), "");
    static_assert(is_same<as_carray      <P>,                   Z                   >(), "");
    static_assert(is_same<as_pack        <Z>,                   P                   >(), "");

    static_assert(is_same<get            <X,cint<1>>,           cint<2>             >(), "");
    static_assert(is_same<get_index      <X,1>,                 cint<2>             >(), "");
    static_assert(is_same<head           <X>,                   cint<1>             >(), "");
    static_assert(is_same<tail           <X>,                   cint<3>             >(), "");
    static_assert(is_same<head           <X,2>,                 cint<1,2>           >(), "");
    static_assert(is_same<tail           <X,2>,                 cint<2,3>           >(), "");

    static_assert(is_same<at             <X,cint<1>>,           cint<2>             >(), "");
    static_assert(is_same<at_index       <X,1>,                 cint<2>             >(), "");
    static_assert(is_same<front          <X>,                   cint<1>             >(), "");
    static_assert(is_same<back           <X>,                   cint<3>             >(), "");

    static_assert(is_same<erase          <X,cint<1>>,           cint<1,3>           >(), "");
    static_assert(is_same<erase_index    <X,1>,                 cint<1,3>           >(), "");
    static_assert(is_same<erase_head     <X>,                   cint<2,3>           >(), "");
    static_assert(is_same<erase_tail     <X>,                   cint<1,2>           >(), "");
    static_assert(is_same<erase_head     <X,2>,                 cint<3>             >(), "");
    static_assert(is_same<erase_tail     <X,2>,                 cint<1>             >(), "");
    static_assert(is_same<set_from       <X,cint<1, 2>,Y>,      cint<1,4,5>         >(), "");

    static_assert(is_same<set_head       <X,Y>,                 cint<4,5,3>         >(), "");
    static_assert(is_same<set_tail       <X,Y>,                 cint<1,4,5>         >(), "");

    static_assert(is_same<set            <X,cint<1>,cuz_0>,     cint<1,0,3>         >(), "");
    static_assert(is_same<set_index      <X,1,cuz_0>,           cint<1,0,3>         >(), "");
    static_assert(is_same<set_front      <X,cuz_0,cuz_0>,       cint<0,0,3>         >(), "");
    static_assert(is_same<set_back       <X,cuz_0,cuz_0>,       cint<1,0,0>         >(), "");

    static_assert(is_same<insert         <X,cint<1>,cint<0,0>>, cint<1,0,0,2,3>     >(), "");
    static_assert(is_same<insert_index   <X,1,cint<0,0>>,       cint<1,0,0,2,3>     >(), "");
    static_assert(is_same<prextend       <X,cint<0,0>>,         cint<0,0,1,2,3>     >(), "");
    static_assert(is_same<extend         <X,cint<0,0>>,         cint<1,2,3,0,0>     >(), "");

    static_assert(is_same<insert_values  <X,cuz_1,cuz_0,cuz_0>, cint<1,0,0,2,3>     >(), "");
    static_assert(is_same<prepend        <X,cuz_0,cuz_0>,       cint<0,0,1,2,3>     >(), "");
    static_assert(is_same<append         <X,cuz_0,cuz_0>,       cint<1,2,3,0,0>     >(), "");

    auto x = X();
    static_assert(x.at(cint<1>())     == 2, "");
    static_assert(x.front()           == 1, "");
    static_assert(x.back()            == 3, "");
}
