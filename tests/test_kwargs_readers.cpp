#include <teeny/teeny.h>

using namespace tny;

// ---- dtype_arg_t: explicit template arg > dtype<T>{} tag > library default ---
static_assert(cs::is_same<dtype_arg_t<void, float>, float>::value,
              "dtype_arg_t: no explicit, no tag -> default");
static_assert(cs::is_same<dtype_arg_t<double, float>, double>::value,
              "dtype_arg_t: explicit wins over default");
static_assert(cs::is_same<dtype_arg_t<void, float, dtype<int>>, int>::value,
              "dtype_arg_t: tag wins over default");
static_assert(cs::is_same<dtype_arg_t<void, float, storage_c<storage::gpu>, dtype<short>>, short>::value,
              "dtype_arg_t: finds the dtype tag among other tags, in any position");
static_assert(cs::is_same<dtype_arg_t<double, float, storage_c<storage::gpu>>, double>::value,
              "dtype_arg_t: explicit wins, unrelated tags present");

// ---- storage_arg: explicit template arg > storage_c<O>{} tag > library default -
static_assert(storage_arg<storage_deduce, storage_deduce>() == storage_deduce,
              "storage_arg: no explicit, no tag -> default");
static_assert(storage_arg<storage::gpu, storage_deduce>() == storage::gpu,
              "storage_arg: explicit wins over default");
static_assert((storage_arg<storage_deduce, storage_deduce, storage_c<storage::pinned>>()) == storage::pinned,
              "storage_arg: tag wins over default");
static_assert((storage_arg<storage_deduce, storage_deduce, dtype<int>, storage_c<storage::mapped>>()) == storage::mapped,
              "storage_arg: finds the storage tag among other tags");
static_assert((storage_arg<storage::gpu, storage_deduce, dtype<int>>()) == storage::gpu,
              "storage_arg: explicit wins, unrelated tags present");

// ---- layout_arg_t: explicit template arg > a bare layout tag > library default -
static_assert(cs::is_same<layout_arg_t<void, ccontiguous>, ccontiguous>::value,
              "layout_arg_t: no explicit, no tag -> default");
static_assert(cs::is_same<layout_arg_t<fcontiguous, ccontiguous>, fcontiguous>::value,
              "layout_arg_t: explicit wins over default");
static_assert(cs::is_same<layout_arg_t<void, ccontiguous, fcontiguous>, fcontiguous>::value,
              "layout_arg_t: tag wins over default");
static_assert(cs::is_same<layout_arg_t<void, ccontiguous, dtype<int>, fcontiguous>, fcontiguous>::value,
              "layout_arg_t: finds the layout tag among other tags");
static_assert(cs::is_same<layout_arg_t<ccontiguous, ccontiguous, dtype<int>>, ccontiguous>::value,
              "layout_arg_t: explicit wins, unrelated tag present");

// ---- is_keyword: open, diagnostics-only registry -- every value-carrier tag
//      registers itself next to its own definition ------------------------------
static_assert(_kw::is_keyword<dtype<double>>::value,          "dtype<T> is a keyword");
static_assert(_kw::is_keyword<storage_c<storage::gpu>>::value, "storage_c<O> is a keyword");
static_assert(_kw::is_keyword<ccontiguous>::value,             "ccontiguous is a keyword");
static_assert(_kw::is_keyword<fcontiguous>::value,             "fcontiguous is a keyword");
static_assert(_kw::is_keyword<axis<0,1>>::value,               "axis<...> is a keyword");
static_assert(_kw::is_keyword<keepdims_t>::value,              "keepdims_t is a keyword");
static_assert(_kw::is_keyword<into_t<int>>::value,             "into_t<D> is a keyword");
static_assert(!_kw::is_keyword<int>::value,                    "a plain int is not a keyword");
static_assert(!_kw::is_keyword<double>::value,                 "a plain double is not a keyword");

int main()
{
    return 0;
}
