#include <teeny/statix.h>

using namespace tny::statix;

int main()
{
    using X = pack<int, float, bool>;
    using Y = pack<long, long>;
    using Z = pack<cint<1>, cint<2>>;
    using V = cint<1, 2>;

    static_assert( is_pack   <X>(), "");
    static_assert(!is_carray <X>(), "");
    static_assert(!is_tuple  <X>(), "");

    static_assert(is_same<size             <X>,               csize<3>                >(), "");
    static_assert(is_same<empty_like       <X>,               pack<>                  >(), "");
    static_assert(is_same<reversed         <X>,               pack<bool, float, int>  >(), "");
    static_assert(is_same<like_from        <X, Y>,            pack<long, long>        >(), "");
    static_assert(is_same<like_from        <X, V>,            pack<cint<1>, cint<2>>  >(), "");
    static_assert(is_same<like             <X, int>,          pack<int>               >(), "");
    static_assert(is_same<as_carray        <Z>,               cint<1, 2>              >(), "");
    static_assert(is_same<as_pack          <X>,               X                       >(), "");

    static_assert(is_same<get              <X,ciz_1>,         pack<float>             >(), "");
    static_assert(is_same<get_index        <X,1>,             pack<float>             >(), "");
    static_assert(is_same<head             <X>,               pack<int>               >(), "");
    static_assert(is_same<tail             <X>,               pack<bool>              >(), "");
    static_assert(is_same<head             <X,2>,             pack<int, float>        >(), "");
    static_assert(is_same<tail             <X,2>,             pack<float, bool>       >(), "");
    static_assert(is_same<at               <X,ciz_1>,         float                   >(), "");
    static_assert(is_same<at_index         <X,1>,             float                   >(), "");
    static_assert(is_same<front            <X>,               int                     >(), "");
    static_assert(is_same<back             <X>,               bool                    >(), "");

    static_assert(is_same<erase            <X,ciz_1>,         pack<int, bool>         >(), "");
    static_assert(is_same<erase_index      <X,1>,             pack<int, bool>         >(), "");
    static_assert(is_same<erase_head       <X>,               pack<float, bool>       >(), "");
    static_assert(is_same<erase_tail       <X>,               pack<int, float>        >(), "");
    static_assert(is_same<erase_head       <X,2>,             pack<bool>              >(), "");
    static_assert(is_same<erase_tail       <X,2>,             pack<int>               >(), "");

    static_assert(is_same<set_from        <X,cint<1, 2>,Y>,   pack<int,  long, long> >(), "");
    static_assert(is_same<set_head        <X,Y>,              pack<long, long, bool> >(), "");
    static_assert(is_same<set_tail        <X,Y>,              pack<int,  long, long> >(), "");

    static_assert(is_same<set             <X,ciz_1, long>,    pack<int,  long, bool> >(), "");
    static_assert(is_same<set_index       <X,1,long>,         pack<int,  long, bool> >(), "");
    static_assert(is_same<set_front       <X,long,long>,      pack<long, long, bool> >(), "");
    static_assert(is_same<set_back        <X,long,long>,      pack<int,  long, long> >(), "");

    static_assert(is_same<insert          <X,ciz_1,Y>,        pack<int,  long,  long,   float, bool> >(), "");
    static_assert(is_same<insert_index    <X,1,Y>,            pack<int,  long,  long,   float, bool> >(), "");
    static_assert(is_same<prextend        <X,Y>,              pack<long, long,  int,    float, bool> >(), "");
    static_assert(is_same<extend          <X,Y>,              pack<int,  float, bool,   long,  long> >(), "");

    static_assert(is_same<insert_values  <X,ciz_1,long,long>, pack<int,  long,  long,   float, bool> >(), "");
    static_assert(is_same<prepend        <X,long,long>,       pack<long, long,  int,    float, bool> >(), "");
    static_assert(is_same<append         <X,long,long>,       pack<int,  float, bool,   long,  long> >(), "");

    static_assert(is_same<apply_sizeof   <X>,                 csize<4, 4, 1> >(), "");
}
