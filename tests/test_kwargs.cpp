#include <teeny/kwargs.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

// ---- dummy keyword tags, independent of any real teeny tag --------------
struct tag_a  { int v; };
struct tag_b  { int v; };
struct tag_c  {};
struct not_a_tag {};

template <class> struct is_tag_a : cs::false_type {};
template <> struct is_tag_a<tag_a> : cs::true_type {};
template <class> struct is_tag_b : cs::false_type {};
template <> struct is_tag_b<tag_b> : cs::true_type {};
template <class> struct is_tag_c : cs::false_type {};
template <> struct is_tag_c<tag_c> : cs::true_type {};

// is_keyword is an OPEN registry: specialise it for the dummy tags here, same
// as a real keyword would specialise it next to its own definition.
namespace tny { namespace _kw {
template <> struct is_keyword<tag_a> : cs::true_type {};
template <> struct is_keyword<tag_b> : cs::true_type {};
template <> struct is_keyword<tag_c> : cs::true_type {};
} }

int main()
{
    using namespace tny::_kw;

    // ---- find_t: first type in the pack matching the predicate, else Fallback
    static_assert(cs::is_same<find_t<is_tag_a, unset, int, tag_a, tag_b>, tag_a>::value, "find_t hits tag_a");
    static_assert(cs::is_same<find_t<is_tag_b, unset, int, tag_a, tag_b>, tag_b>::value, "find_t hits tag_b");
    static_assert(cs::is_same<find_t<is_tag_c, unset, int, tag_a, tag_b>, unset>::value, "find_t misses -> Fallback");
    static_assert(cs::is_same<find_t<is_tag_a, unset>, unset>::value, "find_t on an empty pack -> Fallback");

    // ---- has / count: fold-expression queries --------------------------------
    static_assert(has<is_tag_a, int, tag_a, tag_b>(),  "has finds tag_a");
    static_assert(!has<is_tag_c, int, tag_a, tag_b>(), "has doesn't find tag_c");
    static_assert(count<is_tag_a, tag_a, tag_a, tag_b>() == 2, "count counts repeats");
    static_assert(count<is_tag_a, tag_b>() == 0,               "count of an absent tag is 0");

    // ---- get: value-level extraction (for state-carrying keywords) ----------
    if (get<is_tag_a>(-1, 1, 2, tag_a{42}, tag_b{7}).v != 42) return 1;
    if (get<is_tag_b>(-1, 1, 2, tag_a{42}, tag_b{7}).v != 7)  return 2;
    if (get<is_tag_a>(-1, 1, 2, tag_b{7}) != -1)              return 3;   // no match -> dflt
    static_assert(get<is_tag_a>(-1) == -1, "get on an empty pack -> dflt (constexpr)");

    // ---- accepts<Ps...>: known() (every supplied tag recognised) and
    //      unique() (no keyword supplied twice) -- independent checks ---------
    using ok = accepts<is_tag_a, is_tag_b>;
    static_assert(ok::known<tag_a, tag_b>(),      "known: every tag recognised");
    static_assert(ok::known<>(),                  "known: an empty pack is trivially known");
    static_assert(!ok::known<tag_a, not_a_tag>(), "known: an unrecognised tag is rejected");
    static_assert(ok::unique<tag_a, tag_b>(),      "unique: no repeats");
    static_assert(!ok::unique<tag_a, tag_a>(),     "unique: a repeat is rejected");
    static_assert(ok::unique<>(),                  "unique: an empty pack is trivially unique");
    // known() alone does not catch a duplicate of an otherwise-recognised tag --
    // that is unique()'s job, and a real call site must run both.
    static_assert(ok::known<tag_a, tag_a>(), "known() alone doesn't reject a duplicate");

    // ---- is_keyword: open, diagnostics-only registry -------------------------
    static_assert(is_keyword<tag_a>::value,      "tag_a is registered as a keyword");
    static_assert(!is_keyword<not_a_tag>::value, "not_a_tag is not a registered keyword");
    static_assert(!is_keyword<int>::value,       "a plain int is not a keyword");

    return 0;
}
