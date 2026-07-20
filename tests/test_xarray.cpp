#include <teeny/xarray.h>
#include <cuda/std/type_traits>

using namespace tny;
using namespace tny::statix;

using cuda::std::is_same;

// A hybrid array: [ dyn, 3(static), dyn ] of longs.
using values_t = tuple<cnone, cvalue<long, 3>, cnone>;
using xa_t     = xarray<long, values_t>;

// ---- compile-time structure --------------------------------------------

static_assert(xa_t::size()        == 3, "size");       // static + dynamic
static_assert(xa_t::num_dynamic() == 2, "num_dynamic"); // only 2 are stored
static_assert(!xa_t::empty(), "not empty");

// Storage is a plain tuple of exactly the dynamic elements.
static_assert(is_same<xarray_tuple<long, values_t>, cuda::std::tuple<long, long> >(),
              "storage = dynamics only");

// num_dynamic works on pack- and carray-shaped `values` too.
static_assert(xarray_num_dynamic<pack<cnone, cvalue<long,3>, cnone> >::value == 2, "num_dynamic pack");
static_assert(xarray_num_dynamic<tuple<cvalue<long,1>, cvalue<long,2> > >::value == 0, "num_dynamic all-static");
static_assert(xarray_num_dynamic<tuple<cnone, cnone, cnone> >::value == 3, "num_dynamic all-dynamic");

// dynamic_ordinal: storage position of the dynamic slot at a logical index.
static_assert(dynamic_ordinal<values_t, 0>::value == 0, "ordinal slot 0");
static_assert(dynamic_ordinal<values_t, 2>::value == 1, "ordinal slot 2 -> stored at 1");

// ---- the core innovation: static elements cost no storage --------------
// Regression table (previously the [D,S,D,S]-style orderings over-allocated).

template <class V> using xl = xarray<long, V>;
static_assert(sizeof(xl<tuple<cnone,cvalue<long,3>,cnone> >)                        == 2*sizeof(long), "[D,S,D]");
static_assert(sizeof(xl<tuple<cnone,cvalue<long,2>,cnone,cvalue<long,4> > >)        == 2*sizeof(long), "[D,S,D,S]");
static_assert(sizeof(xl<tuple<cnone,cnone,cvalue<long,2>,cvalue<long,4> > >)        == 2*sizeof(long), "[D,D,S,S]");
static_assert(sizeof(xl<tuple<cvalue<long,2>,cnone,cvalue<long,4>,cnone> >)         == 2*sizeof(long), "[S,D,S,D]");
static_assert(sizeof(xl<tuple<cnone,cvalue<long,2>,cnone,cvalue<long,4>,cnone,cvalue<long,6> > >)
                                                                                    == 3*sizeof(long), "[D,S,D,S,D,S]");
// Fully static: no dynamic storage at all (empty-ish class).
static_assert(sizeof(xl<tuple<cvalue<long,2>,cvalue<long,4>,cvalue<long,6> > >)     == 1, "all static -> empty");
static_assert(xl<tuple<cvalue<long,2>,cvalue<long,4> > >::num_dynamic()             == 0, "all static num_dynamic");

// Must be trivially copyable so it can cross the __global__ boundary by value.
static_assert(cuda::std::is_trivially_copyable<xa_t>::value, "trivially copyable");

// No accidental implicit conversion to the element type (A3 footgun).
static_assert(!cuda::std::is_convertible<xa_t, long>::value, "no implicit conversion to long");

// ---- access return types -----------------------------------------------
// Dynamic slot -> reference; static slot -> value (prvalue).
static_assert(is_same<xarray_access_type<xa_t, csize<0> >, long &>(),             "dyn -> long&");
static_assert(is_same<xarray_access_type<xa_t, csize<1> >, long>(),               "static -> long");
static_assert(is_same<xarray_access_const_type<xa_t, csize<0> >, const long &>(), "dyn const -> const long&");
static_assert(is_same<xarray_access_const_type<xa_t, csize<1> >, long>(),         "static const -> long");

int main()
{
    // ---- runtime behaviour ---------------------------------------------

    xa_t a;                       // default: dynamics zero-initialised
    a[csize<0>()] = 10;           // write into dynamic slot 0 (returns long&)
    a[csize<2>()] = 20;           // write into dynamic slot 2

    if (a[csize<0>()] != 10) return 1;
    if (a[csize<1>()] != 3)  return 2;   // compile-time static value
    if (a[csize<2>()] != 20) return 3;
    if (a.at(csize<1>()) != 3) return 4;

    // front / back.
    if (a.front() != 10) return 5;
    if (a.back()  != 20) return 6;

    // Static value comes through a const view too.
    const xa_t & ca = a;
    if (ca[csize<1>()] != 3) return 7;
    if (ca.front() != 10)    return 8;

    // Negative (Python-style) static indices.
    if (a.at(cptrdiff<-1>()) != 20) return 9;
    if (a.at(cptrdiff<-2>()) != 3)  return 10;   // middle, static

    // Copy preserves the dynamic values (trivial copy).
    xa_t b = a;
    if (b[csize<0>()] != 10 || b[csize<2>()] != 20 || b[csize<1>()] != 3) return 11;
    b[csize<0>()] = 99;
    if (a[csize<0>()] != 10) return 12;          // independent storage

    // Fully-static xarray still reads its compile-time values.
    xarray<long, tuple<cvalue<long,2>, cvalue<long,4> > > s;
    if (s[csize<0>()] != 2) return 13;
    if (s[csize<1>()] != 4) return 14;

    // Fully-dynamic convenience alias.
    dynarray<long, 3> d;
    d[csize<0>()] = 7; d[csize<1>()] = 8; d[csize<2>()] = 9;
    if (d.front() != 7 || d.back() != 9 || d[csize<1>()] != 8) return 15;
    if (dynarray<long,3>::num_dynamic() != 3) return 16;

    return 0;
}
