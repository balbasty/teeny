#include <miniten/meta.h>

using namespace miniten::meta;

int main()
{
    using X = Pack<int, float, bool>;
    using Y = Pack<long, long>;
    using Z = Pack<Int<1>, Int<2>>;
    using V = Int<1, 2>;

    static_assert( IsPack   <X>(), "");
    static_assert(!IsVector <X>(), "");
    static_assert(!IsTuple  <X>(), "");

    static_assert(IsSame<Length         <X>,                SizeT<3>                >(), "");
    static_assert(IsSame<EmptyLike      <X>,                Pack<>                  >(), "");
    static_assert(IsSame<Reversed       <X>,                Pack<bool, float, int>  >(), "");
    static_assert(IsSame<LikeFrom       <X, Y>,             Pack<long, long>        >(), "");
    static_assert(IsSame<LikeFrom       <X, V>,             Z                       >(), "");
    static_assert(IsSame<Like           <X, int>,           Pack<int>               >(), "");
    static_assert(IsSame<AsVector       <Z>,                V                       >(), "");
    static_assert(IsSame<AsPack         <X>,                X                       >(), "");

    static_assert(IsSame<Get            <X,Int<1>>,         Pack<float>             >(), "");
    static_assert(IsSame<GetIndex       <X,1>,              Pack<float>             >(), "");
    static_assert(IsSame<GetFirst       <X>,                Pack<int>               >(), "");
    static_assert(IsSame<GetLast        <X>,                Pack<bool>              >(), "");
    static_assert(IsSame<GetFirst       <X,2>,              Pack<int, float>        >(), "");
    static_assert(IsSame<GetLast        <X,2>,              Pack<float, bool>       >(), "");

    static_assert(IsSame<GetValue       <X,Int<1>>,         float                   >(), "");
    static_assert(IsSame<GetIndexValue  <X,1>,              float                   >(), "");
    static_assert(IsSame<GetFirstValue  <X>,                int                     >(), "");
    static_assert(IsSame<GetLastValue   <X>,                bool                    >(), "");

    static_assert(IsSame<Del            <X,Int<1>>,         Pack<int, bool>         >(), "");
    static_assert(IsSame<DelIndex       <X,1>,              Pack<int, bool>         >(), "");
    static_assert(IsSame<DelFirst       <X>,                Pack<float, bool>       >(), "");
    static_assert(IsSame<DelLast        <X>,                Pack<int, float>        >(), "");
    static_assert(IsSame<DelFirst       <X,2>,              Pack<bool>              >(), "");
    static_assert(IsSame<DelLast        <X,2>,              Pack<int>               >(), "");

    static_assert(IsSame<SetFrom        <X,Int<1, 2>,Y>,    Pack<int,  long, long> >(), "");
    static_assert(IsSame<SetFirstFrom   <X,Y>,              Pack<long, long, bool> >(), "");
    static_assert(IsSame<SetLastFrom    <X,Y>,              Pack<int,  long, long> >(), "");

    static_assert(IsSame<Set            <X,Int<1>,long>,    Pack<int,  long, bool> >(), "");
    static_assert(IsSame<SetIndex       <X,1,long>,         Pack<int,  long, bool> >(), "");
    static_assert(IsSame<SetFirst       <X,long,long>,      Pack<long, long, bool> >(), "");
    static_assert(IsSame<SetLast        <X,long,long>,      Pack<int,  long, long> >(), "");

    static_assert(IsSame<InsertFrom     <X,Int<1>,Y>,       Pack<int,  long,  long,   float, bool> >(), "");
    static_assert(IsSame<InsertIndexFrom<X,1,Y>,            Pack<int,  long,  long,   float, bool> >(), "");
    static_assert(IsSame<PrependFrom    <X,Y>,              Pack<long, long,  int,    float, bool> >(), "");
    static_assert(IsSame<AppendFrom     <X,Y>,              Pack<int,  float, bool,   long,  long> >(), "");

    static_assert(IsSame<Insert         <X,I1,long,long>,   Pack<int,  long,  long,   float, bool> >(), "");
    static_assert(IsSame<InsertIndex    <X,1,long,long>,    Pack<int,  long,  long,   float, bool> >(), "");
    static_assert(IsSame<Prepend        <X,long,long>,      Pack<long, long,  int,    float, bool> >(), "");
    static_assert(IsSame<Append         <X,long,long>,      Pack<int,  float, bool,   long,  long> >(), "");

    static_assert(IsSame<ApplySizeOf    <X>,                SizeT<4, 4, 1> >(), "");
}
