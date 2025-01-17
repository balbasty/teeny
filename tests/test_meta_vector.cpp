#include <miniten/meta.h>

using namespace miniten::meta;

int main()
{
    using X = Int<1, 2, 3>;
    using Y = Int<4, 5>;
    using Z = Int<1, 2>;
    using P = Pack<Int<1>, Int<2>>;

    static_assert( IsVector <X>(), "");
    static_assert(!IsPack   <X>(), "");
    static_assert(!IsTuple  <X>(), "");

    static_assert(IsSame<Length         <X>,                SizeT<3>                >(), "");
    static_assert(IsSame<EmptyLike      <X>,                Int<>                   >(), "");
    static_assert(IsSame<Reversed       <X>,                Int<3, 2, 1>            >(), "");
    static_assert(IsSame<LikeFrom       <X, Y>,             Int<4, 5>               >(), "");
    static_assert(IsSame<LikeFrom       <X, P>,             Int<1, 2>               >(), "");
    static_assert(IsSame<Like           <X, Long<1>>,       Int<1>                  >(), "");
    static_assert(IsSame<AsVector       <Z>,                Z                       >(), "");
    static_assert(IsSame<AsPack         <Z>,                P                       >(), "");

    static_assert(IsSame<Get            <X,Int<1>>,         Int<2>                  >(), "");
    static_assert(IsSame<GetIndex       <X,1>,              Int<2>                  >(), "");
    static_assert(IsSame<GetFirst       <X>,                Int<1>                  >(), "");
    static_assert(IsSame<GetLast        <X>,                Int<3>                  >(), "");
    static_assert(IsSame<GetFirst       <X,2>,              Int<1, 2>               >(), "");
    static_assert(IsSame<GetLast        <X,2>,              Int<2, 3>               >(), "");

    static_assert(IsSame<GetValue       <X,Int<1>>,         Int<2>                  >(), "");
    static_assert(IsSame<GetIndexValue  <X,1>,              Int<2>                  >(), "");
    static_assert(IsSame<GetFirstValue  <X>,                Int<1>                  >(), "");
    static_assert(IsSame<GetLastValue   <X>,                Int<3>                  >(), "");

    static_assert(IsSame<Del            <X,Int<1>>,         Int<1, 3>               >(), "");
    static_assert(IsSame<DelIndex       <X,1>,              Int<1, 3>               >(), "");
    static_assert(IsSame<DelFirst       <X>,                Int<2, 3>               >(), "");
    static_assert(IsSame<DelLast        <X>,                Int<1, 2>               >(), "");
    static_assert(IsSame<DelFirst       <X,2>,              Int<3>                  >(), "");
    static_assert(IsSame<DelLast        <X,2>,              Int<1>                  >(), "");

    static_assert(IsSame<SetFrom        <X,Int<1, 2>,Y>,    Int<1, 4, 5>            >(), "");
    static_assert(IsSame<SetFirstFrom   <X,Y>,              Int<4, 5, 3>            >(), "");
    static_assert(IsSame<SetLastFrom    <X,Y>,              Int<1, 4, 5>            >(), "");

    static_assert(IsSame<Set            <X,Int<1>,I0>,      Int<1, 0, 3>            >(), "");
    static_assert(IsSame<SetIndex       <X,1,I0>,           Int<1, 0, 3>            >(), "");
    static_assert(IsSame<SetFirst       <X,I0,I0>,          Int<0, 0, 3>            >(), "");
    static_assert(IsSame<SetLast        <X,I0,I0>,          Int<1, 0, 0>            >(), "");

    static_assert(IsSame<InsertFrom     <X,Int<1>,Int<0,0>>,Int<1, 0, 0, 2, 3>      >(), "");
    static_assert(IsSame<InsertIndexFrom<X,1,Int<0,0>>,     Int<1, 0, 0, 2, 3>      >(), "");
    static_assert(IsSame<PrependFrom    <X,Int<0,0>>,       Int<0, 0, 1, 2, 3>      >(), "");
    static_assert(IsSame<AppendFrom     <X,Int<0,0>>,       Int<1, 2, 3, 0, 0>      >(), "");

    static_assert(IsSame<Insert         <X,I1,I0,I0>,       Int<1, 0, 0, 2, 3>      >(), "");
    static_assert(IsSame<InsertIndex    <X,1,I0,I0>,        Int<1, 0, 0, 2, 3>      >(), "");
    static_assert(IsSame<Prepend        <X,I0,I0>,          Int<0, 0, 1, 2, 3>      >(), "");
    static_assert(IsSame<Append         <X,I0,I0>,          Int<1, 2, 3, 0, 0>      >(), "");
}
