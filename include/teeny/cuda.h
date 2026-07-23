#ifndef TNY_MD_CUDA
#define TNY_MD_CUDA
// CUDA memory-space storage for tensor (gpu / pinned / mapped owning
// modes). Safe to include ALWAYS — the body activates ONLY when the CUDA runtime
// is reachable: compiling with nvcc, or <cuda_runtime.h> is on the include path
// (auto-detected via __has_include). Otherwise this header is empty, so teeny.h
// pulls it in unconditionally and users never have to think about it. Define
// TNY_NO_CUDA to force it off even when the runtime is available.
#if !defined(TNY_NO_CUDA) && (defined(__CUDACC__) || (defined(__has_include) && __has_include(<cuda_runtime.h>)))
#  define TNY_HAS_CUDA 1
#endif

#ifdef TNY_HAS_CUDA
#include <cuda_runtime.h>
#include <cuda/std/cstddef>
#include <teeny/defines.h>
#include <teeny/storage.h>
#include <teeny/tensor.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/* ------------------------------------------------------------------ *
 *     CUDA allocator policies                                        *
 * ------------------------------------------------------------------ */

/** @brief Device (GPU) memory (`cudaMalloc`). Not host-dereferenceable. */
struct cuda_gpu_alloc {
    template <class T> _TNY_HOST static T * allocate(cs::size_t n) {
        void * p = nullptr; cudaMalloc(&p, n * sizeof(T)); _TNY_CHECK(p, "cudaMalloc failed"); return static_cast<T *>(p);
    }
    template <class T> _TNY_HOST static void deallocate(T * p) { cudaFree(p); }
};

/** @brief Page-locked ("pinned") host memory (`cudaMallocHost`). */
struct cuda_pinned_alloc {
    template <class T> _TNY_HOST static T * allocate(cs::size_t n) {
        void * p = nullptr; cudaMallocHost(&p, n * sizeof(T)); _TNY_CHECK(p, "cudaMallocHost failed"); return static_cast<T *>(p);
    }
    template <class T> _TNY_HOST static void deallocate(T * p) { cudaFreeHost(p); }
};

/** @brief Page-locked + device-mapped (zero-copy) host memory (`cudaHostAlloc`). */
struct cuda_mapped_alloc {
    template <class T> _TNY_HOST static T * allocate(cs::size_t n) {
        void * p = nullptr; cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped); _TNY_CHECK(p, "cudaHostAlloc failed"); return static_cast<T *>(p);
    }
    template <class T> _TNY_HOST static void deallocate(T * p) { cudaFreeHost(p); }
};

/* ------------------------------------------------------------------ *
 *     Storage specializations for the CUDA `own` modes               *
 * ------------------------------------------------------------------ */

template <class T, cs::size_t N>
struct storage<T, own::gpu, N> : owning_storage<T, cuda_gpu_alloc> {
    using owning_storage<T, cuda_gpu_alloc>::owning_storage;
};
template <class T, cs::size_t N>
struct storage<T, own::pinned, N> : owning_storage<T, cuda_pinned_alloc> {
    using owning_storage<T, cuda_pinned_alloc>::owning_storage;
};
template <class T, cs::size_t N>
struct storage<T, own::mapped, N> : owning_storage<T, cuda_mapped_alloc> {
    using owning_storage<T, cuda_mapped_alloc>::owning_storage;
};

/* ------------------------------------------------------------------ *
 *     Factories                                                      *
 * ------------------------------------------------------------------ */

/** @brief Owning tensor in device (GPU) memory (move-only). `gpu<T,E>(extents)`. */
template <class T, class Shape, class Layout = cs::layout_right>
using gpu = tensor<T, Shape, Layout, own::gpu>;
/** @brief Owning tensor in page-locked ("pinned") host memory (move-only).
 *         `pinned<T,E>(extents)` — pytorch's `pin_memory`. */
template <class T, class Shape, class Layout = cs::layout_right>
using pinned = tensor<T, Shape, Layout, own::pinned>;
/** @brief Owning tensor in mapped (zero-copy) host memory (move-only). `mapped<T,E>(extents)`. */
template <class T, class Shape, class Layout = cs::layout_right>
using mapped = tensor<T, Shape, Layout, own::mapped>;

/* --- functional factories (deduce the Shape type from the argument; `T` defaults
 *     to `float`, like the host factories). Thin spellings of the unified
 *     `empty<T, own::gpu/pinned/mapped>` factory (tensor.h). --- */
template <class T = float, class Layout = cs::layout_right, class Shape>
_TNY_HOST auto make_gpu(Shape e)    { return empty<T, own::gpu,    Layout>(e); }
template <class T = float, class Layout = cs::layout_right, class Shape>
_TNY_HOST auto make_pinned(Shape e) { return empty<T, own::pinned, Layout>(e); }
template <class T = float, class Layout = cs::layout_right, class Shape>
_TNY_HOST auto make_mapped(Shape e) { return empty<T, own::mapped, Layout>(e); }

/* ------------------------------------------------------------------ *
 *     Memory-backend `to` — the CUDA half of pytorch's `.to`         *
 *                                                                    *
 *  The member `x.to<T2>()` (tensor.h) does the dtype half on the      *
 *  host; these free functions add the MEMORY-SPACE move (a member     *
 *  can't reach cuda.h, which is included after tensor.h).            *
 * ------------------------------------------------------------------ */
namespace _detail {
// A dense, row-major HOST copy of `x` with element type `E2`. A host-accessible
// source (view/stack/heap/pinned/mapped) is read + converted directly, gathering
// only the viewed extent. A DEVICE source (`gpu` owning OR `gpu_view` — a
// slice/permute of a gpu tensor) is downloaded via one `cudaMemcpy` of its device
// SPAN (lowest..highest addressed element), re-imposing `x`'s mapping (so any layout
// — C, F, or strided — is preserved, not transposed), then densified on the host.
// `Force=true` on the inner `.to` guarantees an OWNING dense buffer, never a borrow
// of the local.
//
// The span is computed over SIGNED strides: a reversed/flipped device view has a
// negative stride, so `x.data()` points INTO the region (at the axis's last element),
// not at its start. We walk each axis to find the lowest (`lo <= 0`) and highest
// (`hi >= 0`) byte offset from `x.data()`, copy `[data()+lo .. data()+hi]`, and place
// the logical origin at `raw.data() - lo` — so a flip downloads correctly instead of
// reading past the end. (Can't lean on `required_span_size()`: it assumes
// non-negative strides — see layout.h.)
//
// NOTE the span == numel for a CONTIGUOUS device view (so it copies exactly the
// viewed extent — optimal), but span > numel for a STRIDED device view (e.g. a
// column of a big matrix): the download then over-copies the underlying volume.
// Correct, but wasteful for pathologically strided device views; a run-wise /
// contiguous-then-copy gather is the follow-up (#50).
template <class E2, class T, class Shape, class Layout, own O>
_TNY_HOST auto dense_host(const tensor<T, Shape, Layout, O> & x) {
    using Ts  = cs::remove_cv_t<T>;
    using Idx = typename tensor<T, Shape, Layout, O>::index_type;
    if constexpr (own_is_device(O)) {
        // Signed extent of the addressed region relative to x.data(): [lo, hi].
        // Guard rank-0 (a single element, span 1): CCCL constrains the runtime
        // stride(r) to rank > 0, so the loop body must not instantiate for rank 0.
        Idx lo = 0, hi = 0;
        bool empty = false;
        if constexpr (Shape::rank() > 0) {
            for (cs::size_t r = 0; r < x.rank(); ++r) {
                const Idx e = static_cast<Idx>(x.extent(r));
                if (e == 0) { empty = true; break; }     // an empty axis -> nothing to copy
                const Idx reach = static_cast<Idx>(x.stride(r)) * (e - 1);   // signed
                if (reach < 0) lo += reach; else hi += reach;
            }
        }
        const cs::size_t span = empty ? 0 : static_cast<cs::size_t>(hi - lo + 1);
        auto raw = make_heap<Ts>(cs::dextents<cs::int64_t, 1>{ static_cast<cs::int64_t>(span) });   // 1-D host span buffer
        if (span) cudaMemcpy(raw.data(), x.data() + lo, span * sizeof(Ts), cudaMemcpyDeviceToHost);
        tensor<Ts, Shape, Layout, own::view> hv(raw.data() - lo, x.mapping());   // re-impose x's layout; origin at -lo
        return hv.template to<E2, true>();               // densify to row-major (owns its buffer; raw can die)
    } else {
        return x.template to<E2, true>();                // host-accessible: read + convert into a dense OWNING copy
    }
}
} // namespace _detail

/**
 * @brief Move `x` to memory space `Space` (`own::gpu`/`pinned`/`mapped`/`heap`),
 *        optionally converting the element type to `ET` — the memory-backend half
 *        of pytorch's `.to`. `ET` defaults to the source type.
 *
 *            auto d = to<own::gpu>(h);           // upload host -> device
 *            auto e = to<own::gpu, half>(h);     // convert to half AND upload
 *            auto c = to<own::heap>(d);          // download device -> host
 *
 * **No-op when already there:** if the source is already an owning tensor of
 * element type `ET` in space `Space`, and `Force` is false, this returns a *view*
 * of it (no copy, borrows the source) — like any teeny view op. Pass `Force =
 * true` to always materialise a fresh owning copy (a force-clone into a space):
 *
 *            auto v = to<own::gpu>(g);           // g is already gpu<T> -> a view, no copy
 *            auto k = to<own::gpu, void, true>(g);  // forced: a fresh gpu copy
 *
 * Any **device** source — an owning `gpu` OR a `gpu_view` (a slice/permute/peel
 * of a gpu tensor) — is downloaded via `cudaMemcpy` (any layout, C/F/strided, is
 * preserved and densified on the host); a host-accessible source is read directly,
 * gathering only the viewed extent. `Space == stack` needs a static shape. Since
 * #15, a device view is correctly *typed* (`gpu_view`), so it takes the download
 * path instead of being host-dereferenced — the hazard the earlier version warned
 * about is closed. NB a **contiguous** device view downloads exactly its `numel`
 * elements; a **strided** device view currently downloads its full span (over-copies
 * — a run-wise gather is a tracked follow-up, #50).
 *
 * @note The no-copy branch returns a **borrow** of `x` (a `gpu_view` when `x` is a
 * gpu tensor, else a host `view`), so it must outlive the result — same lifetime
 * rule as `view()`/`permute()`/slicing. Calling it on a temporary lvalue would
 * dangle; the rvalue overload below forces a copy instead.
 */
template <own Space, class ET = void, bool Force = false, class T, class Shape, class Layout, own O>
_TNY_HOST auto to(const tensor<T, Shape, Layout, O> & x) {
    static_assert(!own_is_view(Space), "to<Space>: Space must be an owning space or stack, not a view kind");
    using Tb = cs::remove_cv_t<T>;
    using E2 = cs::conditional_t<cs::is_same<ET, void>::value, Tb, ET>;
    if constexpr (!Force && cs::is_same<E2, Tb>::value && O == Space) {
        return tensor<const Tb, Shape, Layout, own_view_of(O)>(x.data(), x.mapping());  // already there -> borrow (gpu_view if device)
    } else if constexpr (Space == own::stack) {
        auto host = _detail::dense_host<E2>(x);
        tensor<E2, Shape, cs::layout_right, own::stack> dst{};   // static shape
        cudaMemcpy(dst.data(), host.data(), static_cast<cs::size_t>(dst.numel()) * sizeof(E2), cudaMemcpyHostToHost);
        return dst;
    } else {
        auto host = _detail::dense_host<E2>(x);
        tensor<E2, Shape, cs::layout_right, Space> dst(x.extents());
        const cudaMemcpyKind kind = (Space == own::gpu) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
        cudaMemcpy(dst.data(), host.data(), static_cast<cs::size_t>(dst.numel()) * sizeof(E2), kind);
        return dst;
    }
}

/** @brief Rvalue overload of `to<Space>`: a temporary source cannot be borrowed
 *  (the no-copy branch would dangle — and for a `gpu` temporary would point at
 *  freed device memory), so this always **forces a fresh owning copy**. */
template <own Space, class ET = void, bool Force = false, class T, class Shape, class Layout, own O>
_TNY_HOST auto to(tensor<T, Shape, Layout, O> && x) {
    return to<Space, ET, /*Force=*/true>(x);   // x is a named lvalue here -> the const& overload, copy branch
}

_TNY_NAMESPACE_END(tny)

#endif // TNY_HAS_CUDA

#endif // TNY_MD_CUDA
