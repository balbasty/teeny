/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file implements a compile-time "pack of types"                     **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef TNY__META__PACK_IMPL
#define TNY__META__PACK_IMPL
#include <teeny/core.h>
#include <teeny/_statix/_pack/decl.h>
#include <teeny/_statix/_packapi/decl.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

/* ------------------------------------------------------------------ *
 *     Pack Specialization                                            *
 * ------------------------------------------------------------------ */

struct pack_base {};

template <class... X>
struct pack: public pack_base {
    using this_type = pack<X...>;
};

template <class X>
struct pack<X>: public pack_base {
    using type      = X;
    using this_type = pack<X>;
};

/* ------------------------------------------------------------------ *
 *     Repeat                                                         *
 * ------------------------------------------------------------------ */

template <long N, class T>
struct _npack {
    using type = append<npack<N-1, T>, T>;
};

template <class T>
struct _npack<0, T> {
    using type = pack<>;
};

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif /// TNY__META__PACK_IMPL
