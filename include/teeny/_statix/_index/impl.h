/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file implements indexing utility for template parameter packs      **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef TNY__STATIX__INDEX_IMPL
#define TNY__STATIX__INDEX_IMPL
#include <teeny/core.h>
#include <teeny/_statix/_index/decl.h>
#include <teeny/_statix/_carray/decl.h>     // carray, cptrdiff, cbool, cuz_0
#include <teeny/_statix/_packapi/decl.h>    // get
#include <teeny/_statix/_math/decl.h>       // is_negative, nonzero, boolean_not, isin

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(statix)

/* ================================================================== *
 *      Index complement                                              *
 * ================================================================== */

template <class N, class I>
struct _index_complement {
private:
    using del_indices = wrap_index<N, I>;
    using all_indices = wrap_index<N, slice<cuz_0, N>>;
    using mask        = boolean_not<isin<all_indices, del_indices>>;
    using get_indices = get<all_indices, mask>;
public:
    using type = get_indices;
};

template <class N, bool... I>
struct _index_complement<N, cbool<I...>> {
private:
    using mask        = boolean_not<cbool<I...>>;
    using all_indices = wrap_index<N, slice<cuz_0, N>>;
    using get_indices = get<all_indices, mask>;
public:
    using type = get_indices;
};

/* ================================================================== *
 *      Python-like negative indexing                                 *
 * ================================================================== */

/* ---------------------------------------- *
 * Simple version that uses concrete values *
 * ---------------------------------------- */

// Helper that takes a single (concrete) index and wraps it
template <size_t length, ptrdiff_t index,
          bool from_end = (index < 0)>
struct __wrap_index_value
{
    static constexpr ptrdiff_t value = index;
};

template <size_t length, ptrdiff_t index>
struct __wrap_index_value<length, index, true>
{
    static constexpr ptrdiff_t value = length + index;
};

// Wrapper that hides `from_end` -- used by public alias
template <size_t length, ptrdiff_t... index>
struct _wrap_index_value {};

template <ptrdiff_t index>
struct _wrap_index_value<0, index> {
    using type = cptrdiff<(index < 0) ? 0 : index>;
};

template <size_t length>
struct _wrap_index_value<length> {
    using type = cptrdiff<>;
};

template <size_t length, ptrdiff_t index>
struct _wrap_index_value<length, index> {
    using type = cptrdiff<__wrap_index_value<length, index>::value>;
};

template <size_t length, ptrdiff_t index, ptrdiff_t... indices>
struct _wrap_index_value<length, index, indices...> {
    using left_vector  = wrap_index_value<length, index>;
    using right_vector = wrap_index_value<length, indices...>;
public:
    using type = cat<left_vector, right_vector>;
};

/* ------------------------------------- *
 * Generic version that uses meta-values *
 * ------------------------------------- */

template <class length, class index>
struct _wrap_index {
private:
    using _length = as_carray<length, size_t>;
    using _index  = as_carray<index, ptrdiff_t>;
public:
    using type = wrap_index<_length, _index>;
};

template <size_t length, ptrdiff_t... indices>
struct _wrap_index<csize<length>, cptrdiff<indices...>> {
    using type = wrap_index_value<length, indices...>;
};

// Specialization for slices
template <class length, class start, class stop, class step>
struct _wrap_index<length, slice<start,stop,step>> {
private:
    using this_slice     = slice<start,stop,step>;
    using slice_as_index = as_index_carray<this_slice, length::value>;
public:
    using type = wrap_index<length, slice_as_index>;
};

// Specialization for simple slices
template <class length, ptrdiff_t start, ptrdiff_t stop, ptrdiff_t step>
struct _wrap_index<length, simple_slice<start,stop,step>> {
private:
    using this_slice     = simple_slice<start,stop,step>;
    using slice_as_index = as_index_carray<this_slice, length::value>;
public:
    using type = wrap_index<length, slice_as_index>;
};

// Specialization for boolean masks
template <class length, bool... mask>
struct _wrap_index<length, cbool<mask...>> {
    using type = wrap_index<length, nonzero<cbool<mask...>>>;
};

/* ================================================================== *
 *      Convert Scalar (or None) to integer                           *
 * ================================================================== */

template <class output_type, class scalar_type, output_type default_value>
struct _set_default {
    using type = carray<output_type, static_cast<output_type>(scalar_type::value)>;
};

template <class output_type, output_type default_value>
struct _set_default<output_type, cnone, default_value> {
    using type = carray<output_type, default_value>;
};

/* ================================================================== *
 *      Python-like slice                                             *
 * ================================================================== */

template <class _start, class _stop, class _step>
struct slice {
public:
    using start = _start;
    using stop  = _stop;
    using step  = _step;

private:
    /// Step as a ptrdiff
    static constexpr ptrdiff_t eval_step = set_default<ptrdiff_t, step, 1>::value;
    /// Whether the step is negative (= moving backward)
    static constexpr bool has_negative_step = eval_step < 0;

    /// Whether the step is +/- 1 (= contiguous section)
    static constexpr bool is_contiguous = (eval_step == 1) || (eval_step == -1);

    /// Convert slice to vector of indices, given the tuple length
    template <size_t length, bool reached_end = (length <= 0) >
    struct _slice_as_carray {
    private:
        static constexpr ptrdiff_t abs_step = (
            eval_step > 0
            ? eval_step
            : -eval_step
        );
        static constexpr ptrdiff_t default_start = (
            has_negative_step
            ? static_cast<ptrdiff_t>(length - 1)
            : static_cast<ptrdiff_t>(0)
        );
        static constexpr ptrdiff_t eval_start = wrap_index_value<
            length,
            set_default<ptrdiff_t, start, default_start>::value
        >::value;
        static constexpr ptrdiff_t eval_stop = (
            has_negative_step && is_cnone<stop>::value
            ? static_cast<ptrdiff_t>(-1)
            : wrap_index_value<
                length, set_default<ptrdiff_t, start, length>::value
            >::value
        );
    public:
        using type = cat<
            cptrdiff<eval_start>,
            as_index_carray<
                simple_slice<eval_start+eval_step, eval_stop, eval_step>,
                length-abs_step
            >
        >;
    };

    /// Negative length means we've reached the end of the slice
    template <size_t length>
    struct _slice_as_carray<length, true>
    {
        using type = cptrdiff<>;
    };

public:
    /// Convert slice to vector of indices, given the tuple length
    template <size_t length>
    using as_carray = typename _slice_as_carray<length>::type;
};

/* ================================================================== *
 *      Simple slice                                                  *
 * ================================================================== */

template <ptrdiff_t _start, ptrdiff_t _stop, ptrdiff_t _step>
struct simple_slice {
public:
    static constexpr ptrdiff_t start = _start;
    static constexpr ptrdiff_t stop  = _stop;
    static constexpr ptrdiff_t step  = (_step == 0 ? static_cast<ptrdiff_t>(1) : _step);

private:
    /// Whether the step is negative (= moving backward)
    static constexpr bool has_negative_step = (step < 0);

    /// Whether the step is 1 (= contiguous section)
    static constexpr bool is_contiguous = (step == 1 || step == -1);

    /// Convert slice to vector of indices, given the tuple length
    template <
        size_t length,
        bool reached_end = (length <= 0) || (
            step > 0
            ? wrap_index_value<length, start>::value >= wrap_index_value<length, stop>::value
            : wrap_index_value<length, start>::value <= wrap_index_value<length, stop>::value
        )
    >
    struct _slice_as_carray {
    private:
        static constexpr ptrdiff_t wrap_start = wrap_index_value<length, start>::value;
    public:
        using type = cat<
            cptrdiff<wrap_start>,
            as_index_carray<
                simple_slice<wrap_start+step, stop, step>,
                length-step
            >
        >;
    };

    /// Negative length means we've reached the end of the slice
    template <size_t length>
    struct _slice_as_carray<length, true>
    {
        using type = cptrdiff<>;
    };

public:
    /// Convert slice to vector of indices, given the tuple length
    template <size_t length>
    using as_carray = typename _slice_as_carray<length>::type;
};

/* ================================================================== *
 *      Conversion to vector                                          *
 * ================================================================== */

template <class slice, size_t length>
struct _as_index_carray {
    using type = as_carray<slice, ptrdiff_t>;
};

template <ptrdiff_t start, ptrdiff_t stop, ptrdiff_t step, size_t length>
struct _as_index_carray<simple_slice<start,stop,step>, length> {
    using type = typename simple_slice<start,stop,step>::template as_carray<length>;
};

template <class start, class stop, class step, size_t length>
struct _as_index_carray<slice<start,stop,step>, length> {
    using type = typename slice<start,stop,step>::template as_carray<length>;
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                           *
 *                     Register compile-time indices                         *
 *                                                                           *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

template <class T, T... X>
struct _is_static_index<carray<T, X...>>
{
    using type = ctrue;
};

template <ptrdiff_t start, ptrdiff_t stop, ptrdiff_t step>
struct _is_static_index<simple_slice<start, stop, step>>
{
    using type = ctrue;
};

template <class start, class stop, class step>
struct _is_static_index<slice<start, stop, step>>
{
    using type = ctrue;
};

template <>
struct _is_static_index<pack<>>
{
    using type = ctrue;
};

template <class T0, class... T>
struct _is_static_index<pack<T0, T...>>
{
    using type = boolean_and<is_static_index<T0>, is_static_index<pack<T...>>>;
};

_TNY_NAMESPACE_END(statix)
_TNY_NAMESPACE_END(tny)

#endif // TNY__STATIX__INDEX_IMPL
