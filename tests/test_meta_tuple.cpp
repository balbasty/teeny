#include <miniten/meta.h>

using namespace miniten::meta;

int main()
{
    using X = Tuple<int, float, bool>;
    using Y = Tuple<long, long>;
    using Z = Tuple<Int<1>, Int<2>>;
    using P = Pack<Int<1>, Int<2>>;
    using V = Int<1, 2>;

    static_assert(!IsPack   <X>(), "");
    static_assert(!IsVector <X>(), "");
    static_assert( IsTuple  <X>(), "");

    static_assert(IsSame<Length         <X>,                SizeT<3>                >(), "");
    static_assert(IsSame<EmptyLike      <X>,                Tuple<>                 >(), "");
    static_assert(IsSame<Reversed       <X>,                Tuple<bool, float, int> >(), "");
    static_assert(IsSame<LikeFrom       <X, Y>,             Tuple<long, long>       >(), "");
    static_assert(IsSame<LikeFrom       <X, V>,             Z                       >(), "");
    static_assert(IsSame<Like           <X, int>,           Tuple<int>              >(), "");
    static_assert(IsSame<AsTuple        <Z>,                Z                       >(), "");
    static_assert(IsSame<AsPack         <Z>,                P                       >(), "");
    static_assert(IsSame<AsVector       <Z>,                V                       >(), "");

    static_assert(IsSame<Get            <X,Int<1>>,         Tuple<float>            >(), "");
    static_assert(IsSame<GetIndex       <X,1>,              Tuple<float>            >(), "");
    static_assert(IsSame<GetFirst       <X>,                Tuple<int>              >(), "");
    static_assert(IsSame<GetLast        <X>,                Tuple<bool>             >(), "");
    static_assert(IsSame<GetFirst       <X,2>,              Tuple<int, float>       >(), "");
    static_assert(IsSame<GetLast        <X,2>,              Tuple<float, bool>      >(), "");

    static_assert(IsSame<GetValue       <X,Int<1>>,         float                   >(), "");
    static_assert(IsSame<GetIndexValue  <X,1>,              float                   >(), "");
    static_assert(IsSame<GetFirstValue  <X>,                int                     >(), "");
    static_assert(IsSame<GetLastValue   <X>,                bool                    >(), "");

    static_assert(IsSame<Del            <X,Int<1>>,         Tuple<int, bool>        >(), "");
    static_assert(IsSame<DelIndex       <X,1>,              Tuple<int, bool>        >(), "");
    static_assert(IsSame<DelFirst       <X>,                Tuple<float, bool>      >(), "");
    static_assert(IsSame<DelLast        <X>,                Tuple<int, float>       >(), "");
    static_assert(IsSame<DelFirst       <X,2>,              Tuple<bool>             >(), "");
    static_assert(IsSame<DelLast        <X,2>,              Tuple<int>              >(), "");

    static_assert(IsSame<SetFrom        <X,Int<1, 2>,Y>,    Tuple<int,  long, long> >(), "");
    static_assert(IsSame<SetFirstFrom   <X,Y>,              Tuple<long, long, bool> >(), "");
    static_assert(IsSame<SetLastFrom    <X,Y>,              Tuple<int,  long, long> >(), "");

    static_assert(IsSame<Set            <X,Int<1>,long>,    Tuple<int,  long, bool> >(), "");
    static_assert(IsSame<SetIndex       <X,1,long>,         Tuple<int,  long, bool> >(), "");
    static_assert(IsSame<SetFirst       <X,long,long>,      Tuple<long, long, bool> >(), "");
    static_assert(IsSame<SetLast        <X,long,long>,      Tuple<int,  long, long> >(), "");

    static_assert(IsSame<InsertFrom     <X,Int<1>,Y>,       Tuple<int,  long,  long,   float, bool> >(), "");
    static_assert(IsSame<InsertIndexFrom<X,1,Y>,            Tuple<int,  long,  long,   float, bool> >(), "");
    static_assert(IsSame<PrependFrom    <X,Y>,              Tuple<long, long,  int,    float, bool> >(), "");
    static_assert(IsSame<AppendFrom     <X,Y>,              Tuple<int,  float, bool,   long,  long> >(), "");

    static_assert(IsSame<Insert         <X,I1,long,long>,   Tuple<int,  long,  long,   float, bool> >(), "");
    static_assert(IsSame<InsertIndex    <X,1,long,long>,    Tuple<int,  long,  long,   float, bool> >(), "");
    static_assert(IsSame<Prepend        <X,long,long>,      Tuple<long, long,  int,    float, bool> >(), "");
    static_assert(IsSame<Append         <X,long,long>,      Tuple<int,  float, bool,   long,  long> >(), "");

    static_assert(IsSame<ApplySizeOf    <X>,                SizeT<4, 4, 1> >(), "");

    using Job = Tuple<int, long, float>;
    auto job = Job(1, 2, 3.);
}
