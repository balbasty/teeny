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

// ---- a payload-carrying keyword (the `dtype<T>` shape) + its unwrap step, to
//      exercise `_kw::resolve` -- the ONE copy of the precedence rule the real
//      per-keyword readers (dtype_arg_t / storage_arg / layout_arg_t) alias over.
template <class T> struct tag_t {};
template <class> struct is_tag_t : cs::false_type {};
template <class T> struct is_tag_t<tag_t<T>> : cs::true_type {};
template <class X, class D> struct unwrap_tag_t            { using type = D; };
template <class T, class D> struct unwrap_tag_t<tag_t<T>, D> { using type = T; };

// explicit template arg > matching value tag > library default (payload tag)
static_assert(cs::is_same<tny::_kw::resolve_t<unwrap_tag_t, void, void, is_tag_t, float>,
                          float>::value,  "resolve: no explicit, no tag -> default");
static_assert(cs::is_same<tny::_kw::resolve_t<unwrap_tag_t, double, void, is_tag_t, float>,
                          double>::value, "resolve: explicit wins over default");
static_assert(cs::is_same<tny::_kw::resolve_t<unwrap_tag_t, void, void, is_tag_t, float, tag_a, tag_t<int>>,
                          int>::value,    "resolve: tag wins over default, found among other tags");
static_assert(cs::is_same<tny::_kw::resolve_t<unwrap_tag_t, double, void, is_tag_t, float, tag_a>,
                          double>::value, "resolve: explicit wins, unrelated tags present");
// ...and the `keep_tag` unwrap, for a keyword whose tag IS the answer (layout,
// and storage through its storage_c<O> carrier)
static_assert(cs::is_same<tny::_kw::resolve_t<tny::_kw::keep_tag, void, void, is_tag_a, tag_b>,
                          tag_b>::value,  "resolve/keep_tag: no tag -> default");
static_assert(cs::is_same<tny::_kw::resolve_t<tny::_kw::keep_tag, void, void, is_tag_a, tag_b, tag_c, tag_a>,
                          tag_a>::value,  "resolve/keep_tag: the tag found IS the answer");

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

    // ---- check(): known() AND unique() in one question, for a caller that
    //      wants a single boolean (the two are still reported separately by
    //      the _TNY_KW_CHECK macro, which is what real call sites use) -------
    static_assert(ok::check<tag_a, tag_b>(),  "check: recognised and unique");
    static_assert(ok::check<>(),              "check: an empty pack passes");
    static_assert(!ok::check<tag_a, tag_a>(), "check: a duplicate fails it");
    static_assert(!ok::check<not_a_tag>(),    "check: an unrecognised tag fails it");

    // ---- the guard macro every call site opens with (it must accept a
    //      parenthesised predicate list and a pack, and compile in a body) ---
    _TNY_KW_CHECK("test_kwargs", "tag_a{} or tag_b{}", (is_tag_a, is_tag_b), tag_a, tag_b);

    // ---- is_keyword: open, diagnostics-only registry -------------------------
    static_assert(is_keyword<tag_a>::value,      "tag_a is registered as a keyword");
    static_assert(!is_keyword<not_a_tag>::value, "not_a_tag is not a registered keyword");
    static_assert(!is_keyword<int>::value,       "a plain int is not a keyword");

    return 0;
}
