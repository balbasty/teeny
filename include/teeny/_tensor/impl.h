#ifndef TNY__TENSOR_IMPL
#define TNY__TENSOR_IMPL
#include <cuda/std/utility>       // index_sequence, make_index_sequence
#include <cuda/std/type_traits>
#include <teeny/core.h>
#include <teeny/statix.h>
#include <teeny/xarray.h>

_TNY_NAMESPACE_BEGIN(tny)

/**
 * @brief Non-owning N-dimensional strided tensor view.
 *
 * Holds a raw data pointer plus a `shape` and a `stride`, each a hybrid
 * `xarray` so that any extent or stride may be known at compile time or at
 * run time, per dimension. Dimensions known statically fold their index
 * arithmetic to immediates; dynamic ones are carried as ordinary values.
 *
 * The view owns nothing -- lifetime of the pointed-to memory is the caller's.
 * It is an aggregate with no user-declared special members, hence trivially
 * copyable, so it can be passed into a `__global__` kernel by value.
 *
 * @tparam T       Element type (e.g. `float`).
 * @tparam Offset  Index / stride integer type (e.g. `long`).
 * @tparam Shape   `values` pack for the shape xarray (cvalue / cnone slots).
 * @tparam Stride  `values` pack for the stride xarray; defaults to fully
 *                 dynamic with the same rank as `Shape`.
 */
template <class T, class Offset, class Shape,
          class Stride = dynamic_values<
              statix::size<statix::as_tuple<Shape> >::value> >
struct tensor {
    using value_type  = T;
    using offset_type = Offset;
    using shape_type  = xarray<Offset, Shape>;
    using stride_type = xarray<Offset, Stride>;

    static constexpr size_t ndim = statix::size<statix::as_tuple<Shape> >::value;
    static_assert(statix::size<statix::as_tuple<Stride> >::value == ndim,
                  "tensor: shape and stride must have the same rank");

    /* --- data (public: this is an aggregate) ---------------------- */
    T *         data;
    shape_type  shape;
    stride_type stride;

    /* --- geometry ------------------------------------------------- */

    /** @brief Number of dimensions. */
    _TNYDEF(H,D,S,CX) size_t dim() noexcept { return ndim; }

    /** @brief Total element count (folds to a constant if shape is static). */
    _TNYDEF(H,D,I,CX) auto numel() const noexcept { return tny::prod(shape); }

    /** @brief Extent of dimension `d` (a static index). */
    template <class Dim>
    _TNYDEF(H,D,I,CX) auto size(Dim d) const noexcept { return shape[d]; }

    /** @brief Stride of dimension `d` (a static index). */
    template <class Dim>
    _TNYDEF(H,D,I,CX) auto stride_at(Dim d) const noexcept { return stride[d]; }

    /* --- element access ------------------------------------------- */

    /**
     * @brief Access the element at a multi-index.
     *
     * Exactly `ndim` indices, each either a runtime integer or a static index
     * (`csize<K>` / `cptrdiff<K>`). Static index * static stride folds away.
     */
    template <class... Ix>
    _TNYDEF(H,D,I,CX) T & operator()(Ix... ix) const noexcept {
        static_assert(sizeof...(Ix) == ndim, "tensor: wrong number of indices");
        return data[offset(cuda::std::make_index_sequence<ndim>{}, ix...)];
    }

    /** @brief Raw access by an already-computed memory offset. */
    _TNYDEF(H,D,I,CX) T & operator[](Offset raw) const noexcept { return data[raw]; }

    /* --- structural views ----------------------------------------- */

    /**
     * @brief Bind dimension `D` to index `i`, returning an (ndim-1) view.
     *
     * The data pointer is advanced by `i * stride[D]` and dimension `D` is
     * dropped from both shape and stride. `i` may be a runtime integer or a
     * static index.
     */
    template <size_t D, class Ix>
    _TNYDEF(H,D,I) auto sub(Ix i) const noexcept
        -> tensor<T, Offset,
                  statix::erase_index<statix::as_tuple<Shape>,  (ptrdiff_t)D>,
                  statix::erase_index<statix::as_tuple<Stride>, (ptrdiff_t)D> >
    {
        T * p = data + as_offset(i) * static_cast<Offset>(stride[statix::csize<D>()]);
        return { p, tny::erase<(ptrdiff_t)D>(shape), tny::erase<(ptrdiff_t)D>(stride) };
    }

    /**
     * @brief Reorder the dimensions (a permutation of `0..ndim-1`).
     */
    template <ptrdiff_t... I>
    _TNYDEF(H,D,I) auto permute() const noexcept
        -> tensor<T, Offset,
                  statix::get_index<statix::as_tuple<Shape>,  I...>,
                  statix::get_index<statix::as_tuple<Stride>, I...> >
    {
        static_assert(sizeof...(I) == ndim, "permute: need exactly ndim indices");
        return { data, tny::select<I...>(shape), tny::select<I...>(stride) };
    }

    /**
     * @brief Memory offset of a C-contiguous (row-major) linear index.
     *
     * Decodes `linear` against `shape`, weighting by `stride` -- the
     * `index2offset` operation, with static extents/strides folded.
     */
    _TNYDEF(H,D,I,CX) Offset offset_at(Offset linear) const noexcept {
        Offset off = 0;
        decode(off, linear, cuda::std::make_index_sequence<ndim>{});
        return off;
    }

    /**
     * @brief Memory offset of a Fortran-contiguous (column-major) linear index.
     *
     * Decodes `linear` with dimension 0 varying fastest --
     * `linear = i0 + size0*(i1 + size1*(i2 + ...))` -- weighting by `stride`.
     * This matches the batch-linearisation convention used by CUDA-launch
     * loops (jitfields `index2offset`), and static extents/strides fold away.
     */
    _TNYDEF(H,D,I,CX) Offset foffset(Offset linear) const noexcept {
        Offset off = 0, cur = 1;
        fdecode(off, cur, linear, cuda::std::make_index_sequence<ndim>{});
        return off;
    }

private:
    /* term of a multi-index: value of a static index, or the runtime int. */
    template <class Ix>
    _TNYDEF(H,D,S,I,CX) Offset as_offset(Ix ix) noexcept {
        if constexpr (statix::is_static_index<Ix>::value) { (void)ix; return static_cast<Offset>(Ix::value); }
        else                                              { return static_cast<Offset>(ix); }
    }

    template <size_t... Dm, class... Ix>
    _TNYDEF(H,D,I,CX) Offset offset(cuda::std::index_sequence<Dm...>, Ix... ix) const noexcept {
        Offset off = 0;
        ((off += as_offset(ix) * static_cast<Offset>(stride[statix::csize<Dm>()])), ...);
        return off;
    }

    template <size_t... Dm>
    _TNYDEF(H,D,I,CX) void decode(Offset & off, Offset & linear,
                                  cuda::std::index_sequence<Dm...>) const noexcept {
        // Process dimensions last-to-first (row-major: last dim is fastest).
        (decode_one(off, linear, statix::csize<ndim - 1 - Dm>()), ...);
    }

    template <class Dim>
    _TNYDEF(H,D,I,CX) void decode_one(Offset & off, Offset & linear, Dim d) const noexcept {
        const Offset sz = static_cast<Offset>(shape[d]);
        off    += (linear % sz) * static_cast<Offset>(stride[d]);
        linear /= sz;
    }

    template <size_t... Dm>
    _TNYDEF(H,D,I,CX) void fdecode(Offset & off, Offset & cur, Offset linear,
                                   cuda::std::index_sequence<Dm...>) const noexcept {
        // Process dimensions first-to-last (column-major: dim 0 is fastest).
        (fdecode_one(off, cur, linear, statix::csize<Dm>()), ...);
    }

    template <class Dim>
    _TNYDEF(H,D,I,CX) void fdecode_one(Offset & off, Offset & cur, Offset linear, Dim d) const noexcept {
        const Offset sz  = static_cast<Offset>(shape[d]);
        const Offset nxt = cur * sz;
        off += ((linear % nxt) / cur) * static_cast<Offset>(stride[d]);
        cur  = nxt;
    }
};

/* ------------------------------------------------------------------ *
 *     make_tensor  (kernel-boundary construction)                    *
 * ------------------------------------------------------------------ */

/**
 * @brief Build a tensor view from a data pointer and raw size/stride arrays.
 *
 * `sizes` and `strides` each hold `ndim` values; static slots in `Shape` /
 * `Stride` keep their compile-time value and ignore the corresponding entry.
 *
 * @tparam Shape   Shape `values` pack (required, names the rank & static dims).
 * @tparam Stride  Stride `values` pack (defaults to fully dynamic).
 */
template <class Shape,
          class Stride = dynamic_values<statix::size<statix::as_tuple<Shape> >::value>,
          class T, class Offset>
_TNYDEF(H,D,I) tensor<T, Offset, Shape, Stride>
make_tensor(T * data, const Offset * sizes, const Offset * strides) {
    return tensor<T, Offset, Shape, Stride>{
        data,
        from_pointer<Shape>(sizes),
        from_pointer<Stride>(strides)
    };
}

_TNY_NAMESPACE_END(tny)

#endif // TNY__TENSOR_IMPL
