#include <tuple>
#include <cuda/std/tuple>
#include <teeny/statix.h>
#include <iostream>

using namespace tny::statix;

int main()
{
    using X = tuple<int, float, bool>;
    using Y = tuple<long, long>;
    using Z = tuple<cint<1>, cint<2>>;
    using P = pack<cint<1>, cint<2>>;
    using V = cint<1, 2>;

    static_assert(!is_pack   <X>(), "");
    static_assert(!is_carray <X>(), "");
    static_assert( is_tuple  <X>(), "");

    static_assert(is_same<size            <X>,                csize<3>                >(), "");
    static_assert(is_same<empty_like      <X>,                tuple<>                 >(), "");
    static_assert(is_same<reversed        <X>,                tuple<bool, float, int> >(), "");
    static_assert(is_same<like_from       <X, Y>,             tuple<long, long>       >(), "");
    static_assert(is_same<like_from       <X, V>,             Z                       >(), "");
    static_assert(is_same<like            <X, int>,           tuple<int>              >(), "");
    static_assert(is_same<as_tuple        <Z>,                Z                       >(), "");
    static_assert(is_same<as_pack         <Z>,                P                       >(), "");
    static_assert(is_same<as_carray       <Z>,                V                       >(), "");

    static_assert(is_same<get             <X,cint<1>>,        tuple<float>            >(), "");
    static_assert(is_same<get_index       <X,1>,              tuple<float>            >(), "");
    static_assert(is_same<head            <X>,                tuple<int>              >(), "");
    static_assert(is_same<tail            <X>,                tuple<bool>             >(), "");
    static_assert(is_same<head            <X,2>,              tuple<int, float>       >(), "");
    static_assert(is_same<tail            <X,2>,              tuple<float, bool>      >(), "");

    static_assert(is_same<at              <X,cint<1>>,        float                   >(), "");
    static_assert(is_same<at_index        <X,1>,              float                   >(), "");
    static_assert(is_same<front           <X>,                int                     >(), "");
    static_assert(is_same<back            <X>,                bool                    >(), "");

    static_assert(is_same<erase           <X,cint<1>>,       tuple<int, bool>        >(), "");
    static_assert(is_same<erase_index     <X,1>,             tuple<int, bool>        >(), "");
    static_assert(is_same<erase_head      <X>,               tuple<float, bool>      >(), "");
    static_assert(is_same<erase_tail      <X>,               tuple<int, float>       >(), "");
    static_assert(is_same<erase_head      <X,2>,             tuple<bool>             >(), "");
    static_assert(is_same<erase_tail      <X,2>,             tuple<int>              >(), "");

    static_assert(is_same<set_from        <X,cint<1, 2>,Y>,  tuple<int,  long, long> >(), "");
    static_assert(is_same<set_head        <X,Y>,             tuple<long, long, bool> >(), "");
    static_assert(is_same<set_tail        <X,Y>,             tuple<int,  long, long> >(), "");

    static_assert(is_same<set             <X,cint<1>,long>,   tuple<int,  long, bool> >(), "");
    static_assert(is_same<set_index       <X,1,long>,         tuple<int,  long, bool> >(), "");
    static_assert(is_same<set_front       <X,long,long>,      tuple<long, long, bool> >(), "");
    static_assert(is_same<set_back        <X,long,long>,      tuple<int,  long, long> >(), "");

    static_assert(is_same<insert          <X,cint<1>,Y>,         tuple<int,  long,  long,   float, bool> >(), "");
    static_assert(is_same<insert_index    <X,1,Y>,               tuple<int,  long,  long,   float, bool> >(), "");
    static_assert(is_same<prextend        <X,Y>,                 tuple<long, long,  int,    float, bool> >(), "");
    static_assert(is_same<extend          <X,Y>,                 tuple<int,  float, bool,   long,  long> >(), "");

    static_assert(is_same<insert_values   <X,cint<1>,long,long>, tuple<int,  long,  long,   float, bool> >(), "");
    static_assert(is_same<prepend         <X,long,long>,         tuple<long, long,  int,    float, bool> >(), "");
    static_assert(is_same<append          <X,long,long>,         tuple<int,  float, bool,   long,  long> >(), "");

    static_assert(is_same<apply_sizeof    <X>,                   csize<4, 4, 1> >(), "");

    using Job = tuple<int, long, float>;
    Job job{1, 2L, 3.f};

}
