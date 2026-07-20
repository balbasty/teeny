#include <teeny/xarray.h>
#include <cuda/std/type_traits>

using namespace tny;
using namespace tny::statix;
using cuda::std::is_same;

using V = tuple<cnone, cvalue<long,3>, cvalue<long,4> >;   // [?, 3, 4]

// ---- type-level results ------------------------------------------------
static_assert(is_same<decltype(tny::select<2,0,1>(xarray<long,V>{}))::values_type,
                      tuple<cvalue<long,4>, cnone, cvalue<long,3> > >(), "select values");
static_assert(is_same<decltype(tny::erase<1>(xarray<long,V>{}))::values_type,
                      tuple<cnone, cvalue<long,4> > >(), "erase values");
static_assert(is_same<decltype(tny::reversed(xarray<long,V>{}))::values_type,
                      tuple<cvalue<long,4>, cvalue<long,3>, cnone > >(), "reversed values");

int main()
{
    xarray<long, V> a;            // [?, 3, 4]
    a[csize<0>()] = 9;            // -> [9, 3, 4]

    // ---- select / permute ----------------------------------------------
    auto p = tny::select<2,0,1>(a);   // -> [4, 9, 3]
    if (p[csize<0>()] != 4) return 1;   // was static slot 2
    if (p[csize<1>()] != 9) return 2;   // was dynamic slot 0 (copied)
    if (p[csize<2>()] != 3) return 3;   // was static slot 1
    static_assert(decltype(p)::num_dynamic() == 1, "permute keeps 1 dynamic");

    // gather with repeats / negative index
    auto g = tny::select<0,0,-1>(a);  // -> [9, 9, 4]
    if (g[csize<0>()] != 9 || g[csize<1>()] != 9 || g[csize<2>()] != 4) return 4;

    // ---- erase (squeeze) -----------------------------------------------
    auto e = tny::erase<1>(a);        // drop the static 3 -> [9, 4]
    if (e[csize<0>()] != 9) return 5;   // dynamic, copied
    if (e[csize<1>()] != 4) return 6;   // static
    static_assert(decltype(e)::size() == 2, "erase drops a dim");

    auto e0 = tny::erase<0>(a);       // drop the dynamic -> [3, 4] (fully static now)
    if (e0[csize<0>()] != 3 || e0[csize<1>()] != 4) return 7;
    static_assert(decltype(e0)::num_dynamic() == 0, "erasing the dynamic -> all static");

    // negative erase index
    auto en = tny::erase<-1>(a);      // drop last -> [9, 3]
    if (en[csize<0>()] != 9 || en[csize<1>()] != 3) return 8;

    // ---- reversed ------------------------------------------------------
    auto r = tny::reversed(a);        // -> [4, 3, 9]
    if (r[csize<0>()] != 4 || r[csize<1>()] != 3 || r[csize<2>()] != 9) return 9;

    // fully dynamic reversed
    dynarray<long,3> d; d[csize<0>()]=1; d[csize<1>()]=2; d[csize<2>()]=3;
    auto rd = tny::reversed(d);
    if (rd[csize<0>()] != 3 || rd[csize<1>()] != 2 || rd[csize<2>()] != 1) return 10;

    return 0;
}
