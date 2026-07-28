#ifndef TNY_MD_KWARGS
#define TNY_MD_KWARGS
#include <cuda/std/type_traits>
#include <teeny/defines.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/**
 * @brief Generic keyword-argument mechanism for teeny's trailing value-carrier
 *        tags (`dtype<T>`, `storage_c<O>`, a layout tag, `axis<...>`, `into(dest)`,
 *        `keepdims`, ...).
 *
 * A call site that wants to accept K trailing keywords in any subset and any
 * order needs, today, one hand-written overload per arrangement (`_kw::accepts`
 * below documents why that stops scaling past 2-3 keywords). `_kw` instead lets
 * a call site take ONE variadic `Tags...` pack and ask type-level questions of
 * it: `find_t`/`get` extract the argument matching a predicate, `has`/`count`
 * query membership, `accepts<Ps...>` validates the whole pack against the
 * keywords a given call site recognises.
 *
 * Deliberately NOT here: a tuple, a `tag_pack` object, runtime storage, type
 * erasure. The pack stays the pack; `_kw` only asks questions about it. Every
 * function is `constexpr`/`_TNY_API`, lambda-free, no `std::` runtime facility
 * — safe to use from device-callable call sites.
 *
 * Positional-vs-keyword contract (see the design note on #277): required
 * positional arguments come first, in fixed order; keywords are trailing and
 * order-free among themselves. `_kw` does not enforce that shape itself — it
 * just answers questions about whatever pack it's given — but every reader
 * built on it (`dtype_arg_t`, `storage_arg`, `layout_arg_t`, ...) assumes it.
 */
namespace _kw {

/** @brief Sentinel: "this keyword was not supplied". */
struct unset {};

/* --- type level: the FIRST type in Args... matching predicate P, else Fallback.
 *     `conditional_t` picks a metafunction (not its `::type` directly), so the
 *     `find<P, Fallback, R...>` tail is only INSTANTIATED when `P<A0>::value`
 *     is false — required for this to work through an arbitrarily long pack
 *     without instantiating every suffix eagerly. */
template <template <class> class P, class Fallback, class... A>
struct find { using type = Fallback; };
template <template <class> class P, class Fallback, class A0, class... R>
struct find<P, Fallback, A0, R...> {
    using type = typename cs::conditional_t<P<A0>::value,
                     cs::enable_if<true, A0>, find<P, Fallback, R...>>::type;
};
template <template <class> class P, class Fallback, class... A>
using find_t = typename find<P, Fallback, A...>::type;

/* --- fold-expression queries: does ANY / how MANY of Args... match P --- */
template <template <class> class P, class... A>
_TNY_API constexpr bool has() { return (false || ... || P<A>::value); }
template <template <class> class P, class... A>
_TNY_API constexpr unsigned count() { return (0u + ... + (P<A>::value ? 1u : 0u)); }

/** @brief Value level: the first argument in `(a0, rest...)` matching `P`, else
 *         `dflt`. For keywords that carry STATE (e.g. `into_t<D>`'s destination
 *         reference, a future `stream_t<S>`'s handle) — `find_t`/`has`/`count`
 *         answer questions about the PACK's types; `get` extracts the actual
 *         matching argument. */
template <template <class> class P, class D>
_TNY_API constexpr D get(D dflt) { return dflt; }
template <template <class> class P, class D, class A0, class... R>
_TNY_API constexpr auto get(D dflt, A0 a0, R... rest) {
    if constexpr (P<A0>::value) return a0;
    else                        return get<P>(dflt, rest...);
}

/**
 * @brief The set of keywords ONE call site accepts, built from its predicates
 *        (`Ps...`, e.g. `_is_dtype`, `_is_storage_tag`, `_is_layout_tag`).
 *        `known<Tags...>()` — every supplied tag matches some `Ps`;
 *        `unique<Tags...>()` — no keyword was supplied more than once.
 *        Put both in a `static_assert` ahead of any other logic in a call site's
 *        body, so a bad call fails on a clean, single message instead of
 *        overload resolution dumping every rejected candidate.
 */
template <template <class> class... Ps>
struct accepts {
    template <class A>    static constexpr bool one()    { return (false || ... || Ps<A>::value); }
    template <class... A> static constexpr bool known()  { return (true && ... && one<A>()); }
    template <class... A> static constexpr bool unique() { return (true && ... && (count<Ps, A...>() <= 1)); }
};

/**
 * @brief OPEN registry, DIAGNOSTICS ONLY: is `X` recognised as a keyword tag AT
 *        ALL (by any call site anywhere), regardless of whether the site being
 *        checked happens to accept it? Specialised next to each keyword's own
 *        definition (`dtype<T>`, `storage_c<O>`, a layout tag, `into_t<D>`, ...)
 *        — never here. Lets a call site with an unconstrained positional
 *        parameter (`full(Shape, V value, Tags...)`) catch a keyword landing in
 *        that slot by mistake, with a message naming the actual mistake instead
 *        of a wrong numeric result or an unrelated compile error deeper in.
 */
template <class X> struct is_keyword : cs::false_type {};

} // namespace _kw
_TNY_NAMESPACE_END(tny)
#endif // TNY_MD_KWARGS
