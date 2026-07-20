#include <teeny/xarray.h>
#include <cuda/std/type_traits>

using namespace tny;
using namespace tny::statix;

using cuda::std::is_same;

// A hybrid array: [ dyn, 3(static), dyn ] of longs.
using values_t = tuple<cnone, cvalue<long, 3>, cnone>;
using xa_t     = xarray<long, values_t>;

// ---- compile-time structure --------------------------------------------

// Total number of elements (static + dynamic).
static_assert(xa_t::size() == 3, "size");
// Only two dynamic elements are stored.
static_assert(xa_t::num_dynamic() == 2, "num_dynamic");
static_assert(!xa_t::empty(), "not empty");

// Storage tuple: dynamic -> long, static -> carray<long,3>.
static_assert(is_same<
    xarray_tuple<long, values_t>,
    tuple<long, carray<long, 3>, long>
>(), "xarray_tuple layout");

// num_dynamic works on pack- and carray-shaped `values` too.
static_assert(xarray_num_dynamic<pack<cnone, cvalue<long,3>, cnone> >::value == 2, "num_dynamic pack");
static_assert(xarray_num_dynamic<tuple<cvalue<long,1>, cvalue<long,2> > >::value == 0, "num_dynamic all-static");
static_assert(xarray_num_dynamic<tuple<cnone, cnone, cnone> >::value == 3, "num_dynamic all-dynamic");

// ---- the core innovation: static elements cost no storage --------------

// EBO lock: only the two dynamic longs are stored.
static_assert(sizeof(xa_t) == 2 * sizeof(long), "only dynamics take space");

// Fully-static xarray: no *dynamic* storage at all. Each static element costs
// at most one empty-class byte -- never a full `T` -- so the whole array is
// <= size() bytes (vs. size()*sizeof(T) if it stored them).
using xa_static_t = xarray<long, tuple<cvalue<long,2>, cvalue<long,4> > >;
static_assert(xa_static_t::num_dynamic() == 0, "static num_dynamic");
static_assert(sizeof(xa_static_t) <= xa_static_t::size(), "no dynamic storage");
static_assert(sizeof(xa_static_t) < sizeof(long) * xa_static_t::size(),
              "static values are not stored as T");

// Must be trivially copyable so it can cross the __global__ boundary by value.
static_assert(cuda::std::is_trivially_copyable<xa_t>::value, "trivially copyable");

// ---- access return types -----------------------------------------------

// Dynamic element -> reference; static element -> value (prvalue).
static_assert(is_same<xarray_access_type<xa_t, csize<0> >, long &>(),       "dyn -> long&");
static_assert(is_same<xarray_access_type<xa_t, csize<1> >, long>(),         "static -> long");
static_assert(is_same<xarray_access_const_type<xa_t, csize<0> >, const long &>(), "dyn const -> const long&");
static_assert(is_same<xarray_access_const_type<xa_t, csize<1> >, long>(),   "static const -> long");

int main()
{
    // ---- runtime behaviour ---------------------------------------------

    xa_t a;                       // default: dynamics == 0, static slot == 3
    a[csize<0>()] = 10;           // write into dynamic slot 0 (returns long&)
    a[csize<2>()] = 20;           // write into dynamic slot 2

    // Read back.
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

    // Negative index (Python-style) via back()/cptrdiff.
    if (a.at(cptrdiff<-1>()) != 20) return 9;
    if (a.at(cptrdiff<-2>()) != 3)  return 10;   // middle, static

    // Fully-static xarray still reads its compile-time values.
    xa_static_t s;
    if (s[csize<0>()] != 2) return 11;
    if (s[csize<1>()] != 4) return 12;

    return 0;
}
