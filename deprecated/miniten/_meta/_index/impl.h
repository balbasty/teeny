/** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
 **                                                                         **
 ** This file implements indexing utility for template parameter packs      **
 **                                                                         **
 ** WrapIndex        : Makes python-like negative indices positive          **
 ** WrapIndexVector  : WrapIndex applied to all elements in a meta::Vector  **
 ** SmartSlice       : The metatemplating equivalent of python's `slice     **
 ** Slice            : A simpler slice that only accepts concrete arguments **
 **                                                                         **
 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **/
#ifndef MINITEN__META__INDEX_IMPL
#define MINITEN__META__INDEX_IMPL
#include <miniten/_core/defines.h>
#include <miniten/_core/types.h>
#include <miniten/_meta/_index/decl.h>
#include <miniten/_meta/_vector/decl.h>
#include <miniten/_meta/_packapi/decl.h>
#include <miniten/_meta/_math/decl.h>

NAMESPACE_BEGIN(miniten)
NAMESPACE_BEGIN(meta)

/* ================================================================== *
 *      Index complement                                              *
 * ================================================================== */

template <class N, class I>
struct _IndexComplement {
private:
    using DelIndices = WrapIndex<N, I>;
    using AllIndices = WrapIndex<N, Slice<UZ_0, N>>;
    using Mask       = Not<IsIn<AllIndices, DelIndices>>;
    using GetIndices = Get<AllIndices, Mask>;
public:
    using Type = GetIndices;
};

template <class N, bool... I>
struct _IndexComplement<N, Bool<I...>> {
private:
    using Mask = Not<Bool<I...>>;
    using AllIndices = WrapIndex<N, Slice<UZ_0, N>>;
    using GetIndices = Get<AllIndices, Mask>;
public:
    using Type = GetIndices;
};

template <class N, class I>
using IndexComplement = typename _IndexComplement<N, I>::Type;

/* ================================================================== *
 *      Python-like negative indexing                                 *
 * ================================================================== */

/* ---------------------------------------- *
 * Simple version that uses concrete values *
 * ---------------------------------------- */

// Helper that takes a single (concrete) index and wraps it
template <size_t Length, ptrdiff_t Index,
          bool FromEnd = IsNegative<PtrDiff<Index>>::Value>
struct __WrapIndexValue
{
    static constexpr ptrdiff_t Value = Index;
};

template <size_t Length, ptrdiff_t Index>
struct __WrapIndexValue<Length, Index, true>
{
    static constexpr ptrdiff_t Value = Length + Index;
};

// Wrapper that hides `FromEnd` -- used by public alias
template <size_t Length, ptrdiff_t... Index>
struct _WrapIndexValue {};

template <size_t Length>
struct _WrapIndexValue<Length> {
    using Type = PtrDiff<>;
};

template <size_t Length, ptrdiff_t Index>
struct _WrapIndexValue<Length, Index> {
    using Type = PtrDiff<__WrapIndexValue<Length, Index>::Value>;
};

template <size_t Length, ptrdiff_t Index, ptrdiff_t... Indices>
struct _WrapIndexValue<Length, Index, Indices...> {
    using LeftVector  = WrapIndexValue<Length, Index>;
    using RightVector = WrapIndexValue<Length, Indices...>;
public:
    using Type = Cat<LeftVector, RightVector>;
};

/* ------------------------------------- *
 * Generic version that uses meta-values *
 * ------------------------------------- */

template <class Length, class Index>
struct _WrapIndex {
private:
    using _Length = AsVector<Length, size_t>;
    using _Index  = AsVector<Index, ptrdiff_t>;
public:
    using Type = WrapIndex<_Length, _Index>;
};

template <class Length, ptrdiff_t... Indices>
struct _WrapIndex<Length, PtrDiff<Indices...>> {
    using Type = WrapIndexValue<Length::Value, Indices...>;
};

// Specialization for slices
template <class Length, class START, class STOP, class STEP>
struct _WrapIndex<Length, Slice<START,STOP,STEP>> {
protected:
    using ThisSlice    = Slice<START,STOP,STEP>;
    using SliceAsIndex = AsIndexVector<ThisSlice, Length::Value>;
public:
    using Type = WrapIndex<Length, SliceAsIndex>;
};

// Specialization for simple slices
template <class Length, ptrdiff_t START, ptrdiff_t STOP, ptrdiff_t STEP>
struct _WrapIndex<Length, SimpleSlice<START,STOP,STEP>> {
protected:
    using ThisSlice    = SimpleSlice<START,STOP,STEP>;
    using SliceAsIndex = AsIndexVector<ThisSlice, Length::Value>;
public:
    using Type = WrapIndex<Length, SliceAsIndex>;
};

// Specialization for boolean masks
template <class Length, bool... Mask>
struct _WrapIndex<Length, Bool<Mask...>> {
    using Type = WrapIndex<Length, NonZero<Bool<Mask...>>>;
};

/* ================================================================== *
 *      Convert Scalar (or None) to integer                           *
 * ================================================================== */

template <class OutputType, class ScalarType,
          OutputType Default = static_cast<OutputType>(0)>
struct _SetDefault {
    using Type = Vector<OutputType, static_cast<OutputType>(ScalarType::Value)>;
};

template <class OutputType, OutputType Default>
struct _SetDefault<OutputType, None, Default> {
    using Type = Vector<OutputType, Default>;
};

/**
 * @brief Convert a type that can be `Scalar<OutputType, Value>` or `None`
 *        into `Value` or `Default`.
 *
 * @tparam OutputType  The scalar data type (e.g., `long`)
 * @tparam ScalarType  The meta scalar type (e.g., `Long<1>`)
 * @tparam Default     Default value to use instead of `None`
 */
template <class OutputType, class ScalarType,
          OutputType Default = static_cast<OutputType>(0)>
using SetDefault = typename _SetDefault<OutputType, ScalarType, Default>::Type;


/* ================================================================== *
 *      Python-like slice                                             *
 * ================================================================== */

/**
 * @brief The meta-templating equivalent of python's `slice`.
 *
 * Start/Stop/Step must be typename if we want to support the value `None`.
 * For values that are not None, users should wrap them in a `Scalar`
 * (i.e., a length-1 Vector) to transform them into types.
 *
 * For example, to select odd indices: SmartSlice< Long<1>, None, Long<2> >
 *
 * @tparam Start           First (inclusive) index: Scalar or None
 * @tparam Stop            Last  (exclusive) index: Scalar or None
 * @tparam Step            Step between indices:    Scalar or None
 */
template <class START, class STOP, class STEP>
struct Slice {
public:
    using Start = START;
    using Stop  = STOP;
    using Step  = STEP;

private:
    /// Step as a ptrdiff
    static constexpr ptrdiff_t EvalStep = SetDefault<ptrdiff_t, STEP, 1>::Value;

    /// Whether the step is negative (= moving backward)
    static constexpr bool HasNegativeStep = EvalStep < 0;

    /// Whether the step is +/- 1 (= contiguous section)
    static constexpr bool IsContiguous = (EvalStep == 1) || (EvalStep == -1);

    /// Convert Slice to vector of indices, given the tuple length
    template <size_t Length, bool ReachedEnd = (Length <= 0) >
    struct _AsVector {
    private:
        static constexpr ptrdiff_t AbsStep   = EvalStep > 0 ? EvalStep : -EvalStep;
        static constexpr ptrdiff_t DefaultStart = (
            HasNegativeStep
            ? static_cast<ptrdiff_t>(Length - 1)
            : static_cast<ptrdiff_t>(0)
        );
        static constexpr ptrdiff_t EvalStart = WrapIndexValue<
            Length,
            SetDefault<ptrdiff_t, Start, DefaultStart>::Value
        >::Value;
        static constexpr ptrdiff_t EvalStop  = (
            HasNegativeStep && IsNone<Stop>::Value
            ? static_cast<ptrdiff_t>(-1)
            : WrapIndexValue<
                Length, SetDefault<ptrdiff_t, Start, Length>::Value
            >::Value
        );
    public:
        using Type = Cat<
            PtrDiff<EvalStart>,
            AsIndexVector<
                SimpleSlice<EvalStart+EvalStep, EvalStop, EvalStep>,
                Length-AbsStep
            >
        >;
    };

    /// Negative length means we've reached the end of the slice
    template <size_t Length>
    struct _AsVector<Length, true>
    {
        using Type = Vector<size_t>;
    };

public:
    /// Convert Slice to vector of indices, given the tuple length
    template <size_t Length>
    using AsVector = typename _AsVector<Length>::Type;
};

/* ================================================================== *
 *      Simple slice                                                  *
 * ================================================================== */

/**
 * @brief A simpler slice that only accepts concrete arguments (not None).
 *
 * For example, to select odd indices: Slice<1, Length, 1>
 *
 * @tparam Start
 * @tparam Stop
 * @tparam Step
 * @tparam NegativeStep
 */
template <ptrdiff_t START, ptrdiff_t STOP, ptrdiff_t STEP>
struct SimpleSlice {
public:
    static constexpr ptrdiff_t Start = START;
    static constexpr ptrdiff_t Stop  = STOP;
    static constexpr ptrdiff_t Step  = (STEP == 0 ? static_cast<ptrdiff_t>(1) : STEP);

private:
    /// Whether the step is negative (= moving backward)
    static constexpr bool HasNegativeStep = (Step < 0);

    /// Whether the step is 1 (= contiguous section)
    static constexpr bool IsContiguous = (Step == 1 || Step == -1);

    /// Convert Slice to vector of indices, given the tuple length
    template <size_t Length,
              bool ReachedEnd = (Length <= 0) ||
                (Step > 0
                ? WrapIndexValue<Length, Start>::Value >= WrapIndexValue<Length, Stop>::Value
                : WrapIndexValue<Length, Start>::Value <= WrapIndexValue<Length, Stop>::Value
                )>
    struct _AsVector {
    private:
        static constexpr ptrdiff_t WrapStart = WrapIndexValue<Length, Start>::Value;
    public:
        using Type = Cat<
            PtrDiff<WrapStart>,
            AsIndexVector<
                SimpleSlice<WrapStart+Step, Stop, Step>,
                Length-Step
            >
        >;
    };

    /// Negative length means we've reached the end of the slice
    template <size_t Length>
    struct _AsVector<Length, true>
    {
        using Type = PtrDiff<>;
    };

public:
    /// Convert Slice to vector of indices, given the tuple length
    template <size_t Length>
    using AsVector = typename _AsVector<Length>::Type;
};

/* ================================================================== *
 *      Conversion to vector                                          *
 * ================================================================== */

template <class SLICE, size_t LENGTH>
struct _AsIndexVector {
    using Type = AsVector<SLICE, ptrdiff_t>;
};

template <ptrdiff_t START, ptrdiff_t STOP, ptrdiff_t STEP, size_t LENGTH>
struct _AsIndexVector<SimpleSlice<START,STOP,STEP>, LENGTH> {
    using Type = typename SimpleSlice<START,STOP,STEP>::template AsVector<LENGTH>;
};

template <class START, class STOP, class STEP, size_t LENGTH>
struct _AsIndexVector<Slice<START,STOP,STEP>, LENGTH> {
    using Type = typename Slice<START,STOP,STEP>::template AsVector<LENGTH>;
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                           *
 *                     Register compile-time indices                         *
 *                                                                           *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

template <class T, T... X>
struct _IsStaticIndex<Vector<T, X...>> { using Type = True; };

template <ptrdiff_t Start, ptrdiff_t Stop, ptrdiff_t Step>
struct _IsStaticIndex<SimpleSlice<Start, Stop, Step>> { using Type = True; };

template <class Start, class Stop, class Step>
struct _IsStaticIndex<Slice<Start, Stop, Step>> { using Type = True; };

template <>
struct _IsStaticIndex<Pack<>> { using Type = True; };

template <class T0, class... T>
struct _IsStaticIndex<Pack<T0, T...>> { using Type = And<IsStaticIndex<T0>, IsStaticIndex<Pack<T...>>>; };

NAMESPACE_END(meta)
NAMESPACE_END(miniten)


#endif // MINITEN__META__INDEX_IMPL
