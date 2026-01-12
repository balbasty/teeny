#ifndef TNY__STATIX__TUPLE_DECL
#define TNY__STATIX__TUPLE_DECL
#include <teeny/core.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

template <class... T>
struct tuple;

template <size_t N, class T>
struct _ntuple;

/**
 * @brief Helper to create a tuple with N elements of type T
 *
 * ntuple<N, T> = tuple<T, T, ..., T> (N times)
 *
 * @tparam N Number of elements
 * @tparam T Type of each element
 */
template <size_t N, class T>
using ntuple = typename _ntuple<N, T>::type;

template <class T, class I = csize<0>>
struct tuple_iterator;

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)


#endif // TNY__STATIX__TUPLE_DECL
