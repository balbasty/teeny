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
#include <teeny/alias.h>
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
 *     Storage specializations for the CUDA `storage` modes               *
 * ------------------------------------------------------------------ */

template <class T, cs::size_t N>
struct storage_policy<T, storage::gpu, N> : owning_storage<T, cuda_gpu_alloc> {
    using owning_storage<T, cuda_gpu_alloc>::owning_storage;
};
template <class T, cs::size_t N>
struct storage_policy<T, storage::pinned, N> : owning_storage<T, cuda_pinned_alloc> {
    using owning_storage<T, cuda_pinned_alloc>::owning_storage;
};
template <class T, cs::size_t N>
struct storage_policy<T, storage::mapped, N> : owning_storage<T, cuda_mapped_alloc> {
    using owning_storage<T, cuda_mapped_alloc>::owning_storage;
};

/* ------------------------------------------------------------------ *
 *     Factories                                                      *
 * ------------------------------------------------------------------ */

/** @brief Owning tensor in device (GPU) memory (move-only). `gpu<T,E>(extents)`. */
template <class T, class Shape, class Layout = ccontiguous>
using gpu = tensor<T, Shape, Layout, storage::gpu>;
/** @brief Owning tensor in page-locked ("pinned") host memory (move-only).
 *         `pinned<T,E>(extents)` — pytorch's `pin_memory`. */
template <class T, class Shape, class Layout = ccontiguous>
using pinned = tensor<T, Shape, Layout, storage::pinned>;
/** @brief Owning tensor in mapped (zero-copy) host memory (move-only). `mapped<T,E>(extents)`. */
template <class T, class Shape, class Layout = ccontiguous>
using mapped = tensor<T, Shape, Layout, storage::mapped>;

/* --- functional factories (deduce the Shape type from the argument; `T` defaults
 *     to `float`, like the host factories). Thin spellings of the unified
 *     `empty<T, storage::gpu/pinned/mapped>` factory (tensor.h). --- */
template <class T = float, class Layout = ccontiguous, class Shape>
_TNY_HOST auto make_gpu(Shape e)    { return empty<T, storage::gpu,    Layout>(e); }
template <class T = float, class Layout = ccontiguous, class Shape>
_TNY_HOST auto make_pinned(Shape e) { return empty<T, storage::pinned, Layout>(e); }
template <class T = float, class Layout = ccontiguous, class Shape>
_TNY_HOST auto make_mapped(Shape e) { return empty<T, storage::mapped, Layout>(e); }

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
// viewed extent — optimal). For a STRIDED device view span > numel; when the
// innermost axis is unit-stride (a padded sub-block) the run-wise path below
// gathers only the runs (#50). A view with NO unit-stride axis (e.g. a single
// strided column) still over-copies its span — a general device gather kernel is
// the remaining follow-up.
template <class E2, class T, class Shape, class Layout, storage O>
_TNY_HOST auto dense_host(const tensor<T, Shape, Layout, O> & x) {
    using Ts  = cs::remove_cv_t<T>;
    using Idx = typename tensor<T, Shape, Layout, O>::index_type;
    if constexpr (storage_is_device(O)) {
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
        // #50 run-wise gather: when the INNERMOST axis is unit-stride and the span
        // has GAPS (span > numel — e.g. a padded sub-block: rows are contiguous but
        // separated by the parent's row pitch), copy the contiguous runs straight
        // into a dense row-major buffer instead of dragging the whole span (gaps and
        // all) across the bus. One D2H memcpy per run (`numel` elements total) vs one
        // of `span`. A stride-1 inner axis with span == numel is already fully
        // contiguous (the single-span copy below is optimal); other layouts (no
        // unit-stride axis, e.g. a strided column) still fall back — a general device
        // gather kernel is the remaining follow-up.
        if constexpr (Shape::rank() > 0) {
            const Idx inner = static_cast<Idx>(x.extent(x.rank() - 1));
            if (!empty && span > static_cast<cs::size_t>(x.numel())
                       && static_cast<Idx>(x.stride(x.rank() - 1)) == Idx(1) && inner > Idx(1)) {
                auto dense = make_heap<Ts>(x.extents());                 // dense row-major, x's extents
                const Idx nouter = static_cast<Idx>(x.numel()) / inner;   // one run per outer index tuple
                for (Idx o = 0; o < nouter; ++o) {
                    Idx rem = o, src = 0;                                  // decode o over the outer axes (row-major)
                    for (int d = static_cast<int>(x.rank()) - 2; d >= 0; --d) {
                        const Idx e = static_cast<Idx>(x.extent(static_cast<cs::size_t>(d)));
                        const Idx k = rem % e; rem /= e;
                        src += k * static_cast<Idx>(x.stride(static_cast<cs::size_t>(d)));   // signed
                    }
                    cudaMemcpy(dense.data() + o * inner, x.data() + src,
                               static_cast<cs::size_t>(inner) * sizeof(Ts), cudaMemcpyDeviceToHost);
                }
                return dense.template to<E2, true>();     // already dense row-major; convert to E2 (owning)
            }
        }
        auto raw = make_heap<Ts>(cs::dextents<cs::int64_t, 1>{ static_cast<cs::int64_t>(span) });   // 1-D host span buffer
        if (span) cudaMemcpy(raw.data(), x.data() + lo, span * sizeof(Ts), cudaMemcpyDeviceToHost);
        tensor<Ts, Shape, Layout, storage::view> hv(raw.data() - lo, x.mapping());   // re-impose x's layout; origin at -lo
        return hv.template to<E2, true>();               // densify to row-major (owns its buffer; raw can die)
    } else {
        return x.template to<E2, true>();                // host-accessible: read + convert into a dense OWNING copy
    }
}
} // namespace _detail

/**
 * @brief Move `x` to memory space `Space` (`storage::gpu`/`pinned`/`mapped`/`heap`),
 *        optionally converting the element type to `ET` — the memory-backend half
 *        of pytorch's `.to`. `ET` defaults to the source type.
 *
 *            auto d = to<storage::gpu>(h);           // upload host -> device
 *            auto e = to<storage::gpu, half>(h);     // convert to half AND upload
 *            auto c = to<storage::heap>(d);          // download device -> host
 *
 * **Stays put when already there (#58):** with `Force` false and no dtype change,
 * a source already in a **compatible** space borrows instead of copying — the exact
 * same space (`heap`->`heap`, `gpu`->`gpu`), OR a device source moving to a device
 * space (a `gpu` OR a `gpu_view` slice -> `gpu`). So the common "send data that's
 * already on the device to the device" returns a `gpu_view`, NOT a host round-trip.
 * Pass `Force = true` for a fresh owning copy:
 *
 *            auto v = to<storage::gpu>(g);              // g is gpu (or a gpu_view slice) -> a view, no copy
 *            auto k = to<storage::gpu, void, true>(g);  // forced: a fresh gpu copy
 *
 * When a copy IS made: a **device -> device** copy (Force, same dtype) of a dense
 * row-major source is a single **device-to-device** `cudaMemcpy` — no host hop; a
 * strided device source falls back to the host densify (a device gather kernel is
 * the #50 follow-up). A **device -> host** copy downloads via `cudaMemcpy` (any
 * layout C/F/strided preserved, densified on the host). A **host -> device** copy
 * reads the source directly (gathering only the viewed extent) and uploads.
 * `Space == stack` needs a static shape.
 *
 * @note The no-copy branch returns a **borrow** of `x` (a `gpu_view` for a device
 * source, else a host `view`), so it must outlive the result — same lifetime rule
 * as `view()`/`permute()`/slicing. On a temporary the rvalue overload below instead
 * **moves** a same-space dense owning source (steals its buffer) or forces a copy,
 * so nothing dangles. NB a **contiguous** device download copies exactly `numel`;
 * a **strided** device download still copies its full span (over-copies — #50).
 */
template <storage Space, class ET = void, bool Force = false, class T, class Shape, class Layout, storage O>
_TNY_HOST auto to(const tensor<T, Shape, Layout, O> & x) {
    static_assert(!storage_is_view(Space), "to<Space>: Space must be an owning space or stack, not a view kind");
    using Tb = cs::remove_cv_t<T>;
    using E2 = cs::conditional_t<cs::is_same<ET, void>::value, Tb, ET>;
    if constexpr (!Force && cs::is_same<E2, Tb>::value
                  && (O == Space || (storage_is_device(O) && storage_is_device(Space)))) {
        // Already in a compatible space (exact same space, or both device) and
        // same dtype -> borrow, no copy. This covers the common "move to the
        // device data that's ALREADY on the device" (a gpu OR a gpu_view slice):
        // it returns a gpu_view instead of round-tripping through the host.
        return tensor<const Tb, Shape, Layout, storage_view_of(O)>(x.data(), x.mapping());
    } else if constexpr (storage_is_device(O) && storage_is_device(Space) && cs::is_same<E2, Tb>::value) {
        // Device -> device, same dtype, but a fresh owning copy is wanted (Force).
        // A DENSE row-major source densifies with a single device-to-device memcpy
        // — no host round-trip. A strided/permuted device source still needs a
        // reorder we can't do on-device without a kernel, so it falls back to the
        // host densify (a device gather kernel is the #50 follow-up).
        tensor<E2, Shape, ccontiguous, Space> dst(x.extents());
        if (x.template is_contiguous<ccontiguous>())
            cudaMemcpy(dst.data(), x.data(), static_cast<cs::size_t>(dst.numel()) * sizeof(E2), cudaMemcpyDeviceToDevice);
        else {
            auto host = _detail::dense_host<E2>(x);
            cudaMemcpy(dst.data(), host.data(), static_cast<cs::size_t>(dst.numel()) * sizeof(E2), cudaMemcpyHostToDevice);
        }
        return dst;
    } else if constexpr (Space == storage::stack) {
        auto host = _detail::dense_host<E2>(x);
        tensor<E2, Shape, ccontiguous, storage::stack> dst{};   // static shape
        cudaMemcpy(dst.data(), host.data(), static_cast<cs::size_t>(dst.numel()) * sizeof(E2), cudaMemcpyHostToHost);
        return dst;
    } else {
        auto host = _detail::dense_host<E2>(x);
        tensor<E2, Shape, ccontiguous, Space> dst(x.extents());
        const cudaMemcpyKind kind = (Space == storage::gpu) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
        cudaMemcpy(dst.data(), host.data(), static_cast<cs::size_t>(dst.numel()) * sizeof(E2), kind);
        return dst;
    }
}

/** @brief Rvalue overload of `to<Space>`: a temporary source cannot be borrowed
 *  (the no-copy branch would dangle — and for a device temporary would point at
 *  freed device memory). A same-space, same-dtype, dense OWNING temporary is
 *  **moved** (its buffer stolen — no copy, no round-trip); otherwise this forces a
 *  fresh owning copy (which, for a device->device contiguous source, is the
 *  device-to-device path above, not a host round-trip). */
template <storage Space, class ET = void, bool Force = false, class T, class Shape, class Layout, storage O>
_TNY_HOST auto to(tensor<T, Shape, Layout, O> && x) {
    using Tb = cs::remove_cv_t<T>;
    using E2 = cs::conditional_t<cs::is_same<ET, void>::value, Tb, ET>;
    if constexpr (!Force && storage_is_owning(O) && O == Space && cs::is_same<E2, Tb>::value
                  && cs::is_same<T, Tb>::value   // non-const element: a const-T owning rvalue has no move ctor
                  && cs::is_same<Layout, ccontiguous>::value) {
        return tensor<Tb, Shape, Layout, O>(cs::move(x));   // steal the buffer (already dense in-place)
    } else {
        return to<Space, ET, /*Force=*/true>(x);   // x is a named lvalue here -> the const& copy path
    }
}

_TNY_NAMESPACE_END(tny)

#endif // TNY_HAS_CUDA

#endif // TNY_MD_CUDA
