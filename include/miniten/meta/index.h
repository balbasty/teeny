/***********************************************************************
 * This file implements indexing utility for template parameter packs
 *
 * WrapIndex        : Makes python-like negative indices positive
 * WrapIndexVector  : WrapIndex applied to all elements in a meta::Vector
 * SmartSlice       : The metatemplating equivalent of python's `slice
 * Slice            : A simpler slice that only accepts concrete arguments
 ***********************************************************************/
#ifndef MINITEN_META_INDEX_H
#define MINITEN_META_INDEX_H
#include "../defines.h"
#include "../types.h"
#include "../show.h"

namespace miniten {
namespace meta {

/// ---------------------------------------------------------------- ///
///     Forward declarations                                         ///
/// ---------------------------------------------------------------- ///

struct None;
struct Error;

template <class T, T... N>                     struct Vector;
template <long  Start, long  Stop, long  Step> struct Slice;
template <class Start, class Stop, class Step> struct SmartSlice;

template <long... N> using Long = Vector<long, N...>;

namespace _vector { template <class A, class B> struct Cat; }

/// ---------------------------------------------------------------- ///
///     Python-like negative indexing                                ///
/// ---------------------------------------------------------------- ///

/// Transform negative indices into positive indices (Python convention).
///
/// WrapIndices<Length, Index>
///  ::Value = Index
///
/// @tparam Length  Number of elements in the tuple/vector.
/// @tparam Index   Index to (maybe) wrap. If negative, counts from the end.
/// @tparam FromEnd Whether Index < 0 (auto). It true, counts from end.
template <long Length, long Index, bool FromEnd = IsNegative<long, Index>::Value>
struct WrapIndex
{
    static constexpr long Value = Index;
};

/// Specialization for negative indices
template <long Length, long Index>
struct WrapIndex<Length, Index, true>
{
    static constexpr long Value = Length + Index;
};

/// Transform negative indices into positive indices (Python convention).
///
/// WrapIndices<Length, I0, I1, ...>
///  ::Type = Long<I0, I1, ...>
///
/// @tparam Length  Number of elements in the tuple/vector.
/// @tparam Indices Indices to (maybe) wrap. If negative, counts from the end.
template <long Length, long... Indices>
struct WrapIndices
{};

template <long Length, long Index, long... Indices>
struct WrapIndices<Length, Index, Indices...>
{
private:
    using LeftVector  = Long< WrapIndex<Length, Index>::Value >;
    using RightVector = typename WrapIndices<Length, Indices...>::Type;
public:
    using Type = typename _vector::Cat<LeftVector, RightVector>::Type;
};

template <long Length, long Index>
struct WrapIndices<Length, Index>
{
    using Type = Long<WrapIndex<Length, Index>::Value>;
};

template <long Length>
struct WrapIndices<Length>
{
    using Type = Long<>;
};


/// Transform negative indices into positive indices (Python convention).
///
/// WrapIndices<Length, Long<I0, I1, ...> >
///  ::Type = Long<I0, I1, ...>
///
/// WrapIndices<Length, Slice<Start, Stop, Step> >
///  ::Type = Long<I0, I1, ...>
///
/// WrapIndices<Length, SmartSlice<Start, Stop, Step> >
///  ::Type = Long<I0, I1, ...>
///
/// @tparam Length  Number of elements in the tuple/vector.
/// @tparam Indices Vector of indices to (maybe) wrap. If negative, counts from the end.
template <long Length, class Indices>
struct WrapIndexVector
{};

template <long Length, long Index, long... Indices>
struct WrapIndexVector<Length, Long<Index, Indices...> >
{
private:
    using LeftVector  = Long<WrapIndex<Length, Index>::Value>;
    using RightVector = typename WrapIndices<Length, Indices...>::Type;
public:
    using Type = typename _vector::Cat<LeftVector, RightVector>::Type;
};

template <long Length, long Index>
struct WrapIndexVector<Length, Long<Index> >
{
    using Type = Long< WrapIndex<Length, Index>::Value >;
};

template <long Length>
struct WrapIndexVector<Length, Long<> >
{
    using Type = Long<>;
};

template <long Length, class Start, class Stop, class Step>
struct WrapIndexVector<Length, SmartSlice<Start, Stop, Step> >
{
    using Type = typename SmartSlice<Start, Stop, Step>::template AsVector<Length>;
};

template <long Length, long Start, long Stop, long Step>
struct WrapIndexVector<Length, Slice<Start, Stop, Step> >
{
    using Type = typename Slice<Start, Stop, Step>::template AsVector<Length>;
};

/// ---------------------------------------------------------------- ///
///     Convert Scalar (or None) to integer                          ///
/// ---------------------------------------------------------------- ///

/// Convert a type that can be `Scalar<OutputType, Value>` or `None`
/// into `Value` or `Default`.
///
/// @tparam OutputType  The scalar data type (e.g., `long`)
/// @tparam ScalarType  The meta scalar type (e.g., `Long<1>`)
/// @tparam Default     Default value to use instead of `None`
template <typename OutputType, typename ScalarType,
          OutputType Default=static_cast<OutputType>(0)>
struct OptionalScalarToNumber {
    static constexpr OutputType Value = ScalarType::Value;
};

/// Specialization when ScalarType is None
template <typename OutputType, OutputType Default>
struct OptionalScalarToNumber<OutputType, None, Default> {
    static constexpr OutputType Value = Default;
};

/// ---------------------------------------------------------------- ///
///     Python-like slice                                            ///
/// ---------------------------------------------------------------- ///

/// The metatemplating equivalent of python's `slice`.
///
/// Start/Stop/Step must be typename if we want to support the value `None`.
/// For values that are not None, users should wrap them in a `Scalar`
/// (i.e., a length-1 Vector) to transform them into types.
///
/// For example, to select odd indices: SmartSlice< Long<1>, None, Long<2> >
///
/// @tparam Start           First (inclusive) index: Scalar or None
/// @tparam Stop            Last  (exclusive) index: Scalar or None
/// @tparam Step            Step between indices:    Scalar or None
template <typename START = None, typename STOP = None, typename STEP = None>
struct SmartSlice {

    using Start = START;
    using Stop  = STOP;
    using Step  = STEP;

    /// Whether the step is negative (= moving backward)
    static constexpr bool NegativeStep = (OptionalScalarToNumber<long, Step>::Value < 0);

    /// Whether the step is +/- 1 (= contiguous section)
    static constexpr bool Contiguous = (
        OptionalScalarToNumber<long, Step, 1L>::Value ==  1 ||
        OptionalScalarToNumber<long, Step, 1L>::Value == -1
    );

    /// Convert Slice to vector of indices, given the tuple length
    template <long Length, bool ReachedEnd = (Length <= 0) >
    struct _AsVector {
    private:
        static constexpr long EvalStep  = OptionalScalarToNumber<long, Step, 1L>::Value;
        static constexpr long AbsStep   = EvalStep > 0 ? EvalStep : -EvalStep;
        static constexpr long EvalStart = WrapIndex<
            Length,
            OptionalScalarToNumber<long, Start, NegativeStep ? Length - 1 : 0L>::Value
        >::Value;
        static constexpr long EvalStop  = (
            NegativeStep && IsNone<Stop>::Value
            ? -1L
            : WrapIndex<Length, OptionalScalarToNumber<long, Start, Length>::Value>::Value
        );
    public:
        using Type = typename Vector<long, EvalStart>::template Append<
            Slice<EvalStart+EvalStep, EvalStop, EvalStep>::template AsVector<Length-AbsStep>
        >;
    };

    /// Negative length means we've reached the end of the slice
    template <long Length>
    struct _AsVector<Length, true>
    {
        using Type = Vector<long>;
    };

    /// Convert Slice to vector of indices, given the tuple length
    template <long Length>
    using AsVector = typename _AsVector<Length>::Type;
};

/// ---------------------------------------------------------------- ///
///     Simple slice                                                 ///
/// ---------------------------------------------------------------- ///

/// A simpler slice that only accepts concrete arguments (not None).
///
/// For example, to select odd indices: Slice<1, Length, 1>
///
/// @tparam Start
/// @tparam Stop
/// @tparam Step
/// @tparam NegativeStep
template <long START, long STOP, long STEP = 1>
struct Slice {
    static constexpr long Start = START;
    static constexpr long Stop  = STOP;
    static constexpr long Step  = (STEP == 0 ? static_cast<long>(1) : STEP);

    /// Whether the step is negative (= moving backward)
    static constexpr bool NegativeStep = (Step < 0);

    /// Whether the step is 1 (= contiguous section)
    static constexpr bool Contiguous = (Step == 1 || Step == -1);

    /// Convert Slice to vector of indices, given the tuple length
    template <long Length,
              bool ReachedEnd = (Length <= 0) ||
                (Step > 0
                ? WrapIndex<Length, Start>::Value >= WrapIndex<Length, Stop>::Value
                : WrapIndex<Length, Start>::Value <= WrapIndex<Length, Stop>::Value
                )>
    struct _AsVector {
    private:
        static constexpr long WrapStart = WrapIndex<Length, Start>::Value;
    public:
        using Type = typename _vector::Cat<
            Long<WrapStart>,
            typename Slice<WrapStart+Step, Stop, Step>
                ::template _AsVector<Length-Step>::Type
        >::Type;
    };

    /// Negative length means we've reached the end of the slice
    template <long Length>
    struct _AsVector<Length, true>
    {
        using Type = Long<>;
    };

    /// Convert Slice to vector of indices, given the tuple length
    template <long Length>
    using AsVector = typename _AsVector<Length>::Type;
};

} // namespace meta
} // namespace miniten

#endif // MINITEN_META_INDEX_H
