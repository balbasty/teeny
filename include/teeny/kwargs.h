#ifndef TNY_KWARGS_H
#define TNY_KWARGS_H
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
 * Two of those are what a call site actually reaches for, and each exists ONCE
 * so it is never transcribed by hand (#376): the `_TNY_KW_CHECK(...)` macro at
 * the bottom of this file — the guard every keyword-accepting site opens with —
 * and `resolve` — the single copy of "an explicit template argument beats a
 * matching value tag beats the library default", which each per-keyword reader
 * (`dtype_arg_t`, `storage_arg`, `layout_arg_t`) is a one-line alias over.
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
 *        `unique<Tags...>()` — no keyword was supplied more than once;
 *        `check<Tags...>()` — both at once, for a caller that wants ONE boolean
 *        (a SFINAE constraint, say) rather than the two separately-worded
 *        diagnostics.
 *
 *        A call site does not spell any of these by hand: `_TNY_KW_CHECK` below
 *        is the single entry point every one of them opens with, so "did this
 *        site remember `unique()`?" cannot be answered by omission.
 */
template <template <class> class... Ps>
struct accepts {
    template <class A>    _TNY_API static constexpr bool one()    { return (false || ... || Ps<A>::value); }
    template <class... A> _TNY_API static constexpr bool known()  { return (true && ... && one<A>()); }
    template <class... A> _TNY_API static constexpr bool unique() { return (true && ... && (count<Ps, A...>() <= 1)); }
    template <class... A> _TNY_API static constexpr bool check()  { return known<A...>() && unique<A...>(); }
};

/**
 * @brief The ONE precedence rule every per-keyword reader implements: an
 *        **explicit template argument** beats a **matching value tag** beats the
 *        **library default**, and supplying both an explicit argument and a tag
 *        for the same keyword is a `static_assert`.
 *
 *        - `Expl`      the explicit template argument, as a TYPE (a keyword whose
 *                      explicit form is a *value* — `storage_arg`'s `storage` —
 *                      passes its own carrier, e.g. `storage_c<O>`);
 *        - `NotGiven`  the sentinel `Expl` equals when no explicit argument was
 *                      given (`void`, `storage_c<storage_deduce>`, ...);
 *        - `P`         the keyword's predicate (`_is_dtype`, `_is_layout_tag`, ...);
 *        - `Dflt`      the library default, in the same currency as the answer;
 *        - `Unwrap<Found, Dflt>::type`  the keyword's own "tag -> answer" step, the
 *                      only part that differs between keywords: `dtype<T>` unwraps
 *                      to `T`, a layout tag / `storage_c<O>` IS the answer already
 *                      (`keep_tag`). It is also what maps the `unset` no-tag-found
 *                      result to `Dflt`.
 *
 *        Each header then declares its reader as a one-line alias over this — see
 *        `dtype_arg_t` (`alias.h`), `storage_arg` (`storage.h`), `layout_arg_t`
 *        (`layout.h`) — so the precedence rule and its diagnostic exist once.
 */
template <template <class, class> class Unwrap, class Expl, class NotGiven,
          template <class> class P, class Dflt, class... Tags>
struct resolve {
    static_assert(cs::is_same<Expl, NotGiven>::value || !has<P, Tags...>(),
        "this keyword was given BOTH as an explicit template argument and as a value "
        "tag -- pick one (which keyword: the reader named in the instantiation trace "
        "-- dtype_arg_t / storage_arg / layout_arg_t)");
    /** The matching tag as supplied, or `unset` if the caller gave none. */
    using found = find_t<P, unset, Tags...>;
    using type  = cs::conditional_t<!cs::is_same<Expl, NotGiven>::value, Expl,
                                    typename Unwrap<found, Dflt>::type>;
};
template <template <class, class> class Unwrap, class Expl, class NotGiven,
          template <class> class P, class Dflt, class... Tags>
using resolve_t = typename resolve<Unwrap, Expl, NotGiven, P, Dflt, Tags...>::type;

/** @brief `resolve`'s unwrap step for a keyword whose tag IS the answer (a layout
 *         tag; `storage_c<O>`, read through its own `::value` afterwards): keep the
 *         tag found, or fall back to the default when there was none. */
template <class Found, class Dflt> struct keep_tag              { using type = Found; };
template <class Dflt>              struct keep_tag<unset, Dflt> { using type = Dflt;  };

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

/**
 * @brief The guard EVERY keyword-accepting call site opens with, in one line.
 *
 *     _TNY_KW_CHECK("empty()",
 *                   "dtype<T>{}, storage_c<...>{} or a layout tag (ccontiguous{}/fcontiguous{})",
 *                   (_is_dtype, _is_storage_tag, _is_layout_tag), Tags...);
 *
 *  - `SITE`     the call site's own name, as a string literal (`"empty()"`, or a
 *               `#NAME` stringized macro parameter in `math.h`'s reduction macros);
 *  - `EXPECTED` a string literal listing the keywords this site takes — it is what
 *               the "unrecognised argument" message tells the caller to use instead;
 *  - `PREDS`    the matching predicate list, **in parentheses** (so its commas
 *               survive as one macro argument): `(_is_into_tag, _is_keepdims_tag)`;
 *  - the rest   the trailing pack to check — `Tags...`, or `Tag0, Tags...` for a
 *               site whose first keyword is a separate parameter.
 *
 *  Expands to the two `static_assert`s (unrecognised keyword / duplicated keyword)
 *  the sites used to hand-copy, so both diagnostics keep their own wording and a
 *  site cannot silently forget one of them (#376). Place it ahead of any other
 *  logic in the body, so a bad call fails on one clean named message instead of
 *  overload resolution dumping every rejected candidate.
 */
// `_TNY_KW_PREDS PREDS` drops the parentheses that kept the predicate list one
// macro argument; `_TNY_KW_EXPAND` is the identity, forcing the extra rescan pass
// MSVC's traditional (non-conforming) preprocessor wants before it will expand a
// macro invocation formed that way — a no-op everywhere else.
#define _TNY_KW_PREDS(...)  __VA_ARGS__
#define _TNY_KW_EXPAND(...) __VA_ARGS__
#define _TNY_KW_CHECK(SITE, EXPECTED, PREDS, ...)                                                             \
    static_assert(_TNY_KW_EXPAND(::tny::_kw::accepts<_TNY_KW_PREDS PREDS>::template known<__VA_ARGS__>()),    \
                  SITE ": unrecognised trailing argument — expected " EXPECTED);                              \
    static_assert(_TNY_KW_EXPAND(::tny::_kw::accepts<_TNY_KW_PREDS PREDS>::template unique<__VA_ARGS__>()),   \
                  SITE ": the same keyword was given twice")

#endif // TNY_KWARGS_H
