#include <teeny/xarray.h>
#include <cuda/std/type_traits>

using namespace tny;
using namespace tny::statix;
using cuda::std::is_same;

// Shapes.
using shape_dyn    = tuple<cnone, cnone, cnone>;                    // [?, ?, ?]
using shape_mixed  = tuple<cnone, cvalue<long,3>, cvalue<long,4> >; // [?, 3, 4]
using shape_static = tuple<cvalue<long,2>, cvalue<long,3>, cvalue<long,4> >;

// ---- fully-static folds are compile-time constants ---------------------

static_assert(tny::prod(xarray<long, shape_static>{}) == 24, "static prod value");
static_assert(tny::sum (xarray<long, shape_static>{}) == 9,  "static sum value");
static_assert(tny::max (xarray<long, shape_static>{}) == 4,  "static max value");
// ...and usable as a compile-time value (result carries ::value).
static_assert(decltype(tny::prod(xarray<long, shape_static>{}))::value == 24, "static prod ::value");
// empty product is the neutral 1.
static_assert(tny::prod(xarray<long, tuple<> >{}) == 1, "empty prod = 1");

// A fully-static fold returns a cvalue (empty), not a runtime long.
static_assert(sizeof(decltype(tny::prod(xarray<long, shape_static>{}))) == 1, "static fold -> cvalue");

// dot: static index . static stride folds fully.
static_assert(
    tny::dot(xarray<long, tuple<cvalue<long,1>, cvalue<long,2> > >{},   // index (1,2)
        xarray<long, tuple<cvalue<long,3>, cvalue<long,5> > >{})   // stride (3,5)
    == 1*3 + 2*5, "static dot");

int main()
{
    // ---- hybrid prod/sum/max (some dynamic) ----------------------------
    xarray<long, shape_mixed> shp;      // [?, 3, 4]
    shp[csize<0>()] = 5;                 // -> [5, 3, 4]
    if (tny::prod(shp) != 60) return 1;       // 5*3*4
    if (tny::sum(shp)  != 12) return 2;       // 5+3+4
    if (tny::max(shp)  != 5)  return 3;

    // fully dynamic
    dynarray<long, 3> d;
    d[csize<0>()] = 2; d[csize<1>()] = 7; d[csize<2>()] = 4;
    if (tny::prod(d) != 56) return 4;
    if (tny::max(d)  != 7)  return 5;
    if (tny::sum(d)  != 13) return 6;

    // ---- dot as an offset engine ---------------------------------------
    // index (i,j,k) . stride (12,4,1) for a [.,3,4] C-contiguous tensor.
    xarray<long, tuple<cnone,cnone,cnone> > idx;
    idx[csize<0>()] = 1; idx[csize<1>()] = 2; idx[csize<2>()] = 3;
    xarray<long, tuple<cvalue<long,12>, cvalue<long,4>, cvalue<long,1> > > str; // static strides
    if (tny::dot(idx, str) != 1*12 + 2*4 + 3*1) return 7;

    // ---- from_pointer: dynamic slots read, static slots kept -----------
    long raw[3] = {9, 999, 888};             // [9, (ignored), (ignored)]
    auto v = tny::from_pointer<shape_mixed>(raw); // static 3,4 are kept
    if (v[csize<0>()] != 9) return 8;         // dynamic slot took raw[0]
    if (v[csize<1>()] != 3) return 9;         // static
    if (v[csize<2>()] != 4) return 10;        // static
    if (tny::prod(v) != 108) return 11;            // 9*3*4

    long raw3[3] = {2, 7, 4};
    auto vd = tny::from_pointer<shape_dyn>(raw3);
    if (tny::prod(vd) != 56) return 12;

    // ---- for_each: unrolled, index tag + element -----------------------
    long acc = 0;
    tny::for_each(shp, [&](auto d_, long e){ (void)d_; acc += e; });
    if (acc != 12) return 13;
    // mutate dynamic slots through for_each (static slots are prvalues, skipped)
    dynarray<long,3> m; m[csize<0>()]=1; m[csize<1>()]=2; m[csize<2>()]=3;
    tny::for_each(m, [](auto, long& e){ e *= 10; });
    if (m[csize<0>()] != 10 || m[csize<2>()] != 30) return 14;

    return 0;
}
