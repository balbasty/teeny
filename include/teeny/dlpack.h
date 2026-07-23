#ifndef TNY_MD_DLPACK
#define TNY_MD_DLPACK
// teeny <-> DLPack (https://dmlc.github.io/dlpack) zero-copy interchange with
// numpy / torch / cupy / jax. DLPack exchanges a *pointer + metadata*, never the
// data itself, so import/export are pure host-side struct work — no CUDA needed
// (the device field just LABELS where the pointer lives).
//
// We VENDOR the DLPack C structs (below) rather than depend on the user's include
// path. They sit behind DLPack's own include guard (`DLPACK_DLPACK_H_`), so if the
// user also includes a framework's <dlpack/dlpack.h> only one copy is used — and
// the core `DLManagedTensor` layout is ABI-stable, so a torch `DLManagedTensor*`
// matches this one field-for-field either way. We target the classic
// `DLManagedTensor` (what torch's `__dlpack__` emits); the newer
// `DLManagedTensorVersioned` is a later add.
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <teeny/defines.h>
#include <teeny/tensor.h>
#include <teeny/dynamic.h>
#include <teeny/half.h>

/* ============================ vendored DLPack ============================ */
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    kDLCPU = 1, kDLCUDA = 2, kDLCUDAHost = 3, kDLOpenCL = 4, kDLVulkan = 7,
    kDLMetal = 8, kDLVPI = 9, kDLROCM = 10, kDLROCMHost = 11, kDLExtDev = 12,
    kDLCUDAManaged = 13, kDLOneAPI = 14, kDLWebGPU = 15, kDLHexagon = 16,
} DLDeviceType;
typedef struct { DLDeviceType device_type; int32_t device_id; } DLDevice;
typedef enum { kDLInt = 0U, kDLUInt = 1U, kDLFloat = 2U, kDLOpaqueHandle = 3U,
               kDLBfloat = 4U, kDLComplex = 5U, kDLBool = 6U } DLDataTypeCode;
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;
typedef struct {
    void * data; DLDevice device; int32_t ndim; DLDataType dtype;
    int64_t * shape; int64_t * strides; uint64_t byte_offset;
} DLTensor;
typedef struct DLManagedTensor {
    DLTensor dl_tensor; void * manager_ctx;
    void (*deleter)(struct DLManagedTensor * self);
} DLManagedTensor;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // DLPACK_DLPACK_H_

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

namespace _dl {

// teeny element type <-> DLDataType (lanes always 1). half/bfloat16 are teeny's.
template <class T> _TNY_HOST DLDataType dtype_of() {
    static_assert(cs::is_arithmetic<T>::value || cs::is_same<T, half>::value || cs::is_same<T, bfloat16>::value,
                  "DLPack: unsupported element type (arithmetic / half / bfloat16 only)");
    DLDataType d; d.lanes = 1;
    if      constexpr (cs::is_same<T, half>::value)     { d.code = kDLFloat;  d.bits = 16; }
    else if constexpr (cs::is_same<T, bfloat16>::value) { d.code = kDLBfloat; d.bits = 16; }
    else if constexpr (cs::is_same<T, bool>::value)     { d.code = kDLBool;   d.bits = 8;  }
    else if constexpr (cs::is_floating_point<T>::value) { d.code = kDLFloat;  d.bits = sizeof(T) * 8; }
    else if constexpr (cs::is_unsigned<T>::value)       { d.code = kDLUInt;   d.bits = sizeof(T) * 8; }
    else                                                { d.code = kDLInt;    d.bits = sizeof(T) * 8; }
    return d;
}
template <class T> _TNY_HOST bool dtype_matches(const DLDataType & d) {
    const DLDataType e = dtype_of<T>();
    return d.code == e.code && d.bits == e.bits && d.lanes == e.lanes;
}

// teeny ownership -> DLDevice type. A device tensor/view (gpu/gpu_view) is CUDA;
// pinned/mapped and their views (pinned_view/mapped_view) are kDLCUDAHost; a plain
// host view/stack/heap is kDLCPU (override any of these with the explicit-device
// to_dlpack overload).
template <own O> _TNY_HOST constexpr DLDeviceType device_of() {
    return own_is_device(O) ? kDLCUDA          // gpu OR gpu_view
         : (O == own::pinned || O == own::pinned_view) ? kDLCUDAHost
         : (O == own::mapped || O == own::mapped_view) ? kDLCUDAHost
                                            // cudaHostAllocMapped = page-locked HOST memory (zero-copy),
                                            //   NOT managed/UVM — kDLCUDAHost is the honest label; a view
                                            //   of pinned/mapped keeps that label (pinned_view/mapped_view)
                            : kDLCPU;   // host view / stack / heap
}

// Whether a DLPack device's memory is dereferenceable from the HOST: plain CPU,
// page-locked CUDA host memory, and CUDA managed/UVM all are; a plain `kDLCUDA`
// pointer is NOT. Used at import to reject tagging a device pointer as a host view.
_TNY_HOST constexpr bool device_is_host_accessible(DLDeviceType dt) {
    return dt == kDLCPU || dt == kDLCUDAHost || dt == kDLCUDAManaged;
}

// The heap block that backs an exported DLManagedTensor: the managed struct, the
// shape/stride arrays it points at, and (for an owning export) the moved-in
// tensor whose buffer must outlive the capsule. `deleter` frees exactly this.
template <class Owner>
struct holder {
    DLManagedTensor mt;
    cs::int64_t *   shape;
    cs::int64_t *   stride;
    Owner           owner;   // an owning tensor (move-in) or an empty tag for a view
    static void deleter(DLManagedTensor * self) {
        holder * h = static_cast<holder *>(self->manager_ctx);
        delete[] h->shape; delete[] h->stride; delete h;   // ~owner frees the buffer if owning
    }
};
struct no_owner {};   // view export: nothing to free beyond the metadata arrays

template <class Owner, class T, class Shape, class Layout, own O>
_TNY_HOST DLManagedTensor * make_managed(const tensor<T, Shape, Layout, O> & t, DLDevice dev, Owner && owner) {
    const int nd = static_cast<int>(t.rank());
    auto * h  = new holder<Owner>{};
    h->shape  = new cs::int64_t[nd ? nd : 1];
    h->stride = new cs::int64_t[nd ? nd : 1];
    // Read ALL of `t` (data, extents, strides) BEFORE moving `owner` in — for an
    // owning export `owner` aliases `t`, and the move would leave `t` empty. The
    // `if constexpr` is needed for a rank-0 (scalar) tensor: `t.stride(i)` with a
    // runtime index would still instantiate `layout::mapping::stride`, which CCCL
    // constrains to rank > 0.
    if constexpr (Shape::rank() > 0) {
        for (int i = 0; i < nd; ++i) {
            h->shape[i]  = static_cast<cs::int64_t>(t.extent(i));
            h->stride[i] = static_cast<cs::int64_t>(t.stride(i));   // DLPack strides are in ELEMENTS
        }
    }
    DLTensor & dt = h->mt.dl_tensor;
    dt.data = const_cast<void *>(static_cast<const void *>(t.data()));
    dt.device = dev; dt.ndim = nd; dt.dtype = dtype_of<T>();
    dt.shape = h->shape; dt.strides = h->stride; dt.byte_offset = 0;
    h->mt.manager_ctx = h;
    h->mt.deleter     = &holder<Owner>::deleter;
    h->owner = static_cast<Owner &&>(owner);   // take ownership LAST (empties `t` if owning)
    return &h->mt;
}

} // namespace _dl

/* ============================ export (teeny -> DLPack) ============================ */

/** @brief Export a **view** (`view` / `gpu_view` / `pinned_view` / `mapped_view`)
 *         to a `DLManagedTensor` (borrows the data — the caller must keep the
 *         underlying memory alive; only the metadata is owned by the capsule).
 *         The device defaults to the tensor's memory space (`kDLCPU` for a host
 *         view, `kDLCUDA` for a `gpu_view`, `kDLCUDAHost` for a `pinned_view`/
 *         `mapped_view`; pass `dev` to override). The consumer owns the returned
 *         pointer and MUST call `m->deleter(m)` exactly once. */
template <class T, class Shape, class Layout, own O,
          cs::enable_if_t<own_is_view(O), int> = 0>
_TNY_HOST DLManagedTensor * to_dlpack(const tensor<T, Shape, Layout, O> & t,
                                      DLDevice dev = { _dl::device_of<O>(), 0 }) {
    return _dl::make_managed(t, dev, _dl::no_owner{});
}

/** @brief Export an **owning** tensor, TRANSFERRING ownership of the buffer into
 *         the capsule (the tensor is moved-from; the capsule's `deleter` frees the
 *         buffer). Device is taken from the tensor's memory space. */
template <class T, class Shape, class Layout, own O,
          cs::enable_if_t<own_is_owning(O), int> = 0>
_TNY_HOST DLManagedTensor * to_dlpack(tensor<T, Shape, Layout, O> && t) {
    using Owner = tensor<T, Shape, Layout, O>;
    DLDevice dev{ _dl::device_of<O>(), 0 };
    return _dl::make_managed(static_cast<const Owner &>(t), dev, static_cast<Owner &&>(t));
}

/* ============================ import (DLPack -> teeny) ============================ */

/** @brief Import a `DLManagedTensor` of known element type `T` as an `anyrank`
 *         (runtime rank). The shape/stride METADATA is copied into the carrier
 *         (so it is self-contained), while the DATA is BORROWED — the caller keeps
 *         `m` alive while the view is used, then calls `m->deleter(m)`. A null
 *         `strides` (DLPack's C-contiguous shorthand) is expanded to row-major.
 *         `byte_offset` is folded into the data pointer.
 *
 *         `Space` is the memory space to tag the carrier with (default `own::view`
 *         = host); every view peeled off it inherits it. It is **checked against
 *         `m->dl_tensor.device`**: importing a `kDLCUDA` capsule as the default
 *         host `Space` trips `_TNY_CHECK` — spell `from_dlpack<T, own::gpu_view>(m)`
 *         so `fixed()`/`peel_front` yield device-tagged views (no host deref of a
 *         device pointer). (Closes the #38 hole where the device field was ignored
 *         and a device capsule silently became a host view.) */
template <class T, own Space = own::view>
_TNY_HOST anyrank<T, cs::int64_t, _meta_store<cs::int64_t, TNY_MAX_RANK>, Space>
from_dlpack(const DLManagedTensor * m) {
    const DLTensor & dt = m->dl_tensor;
    _TNY_CHECK(_dl::dtype_matches<T>(dt.dtype), "from_dlpack: DLPack dtype does not match T");
    _TNY_CHECK(own_is_host_accessible(Space) == _dl::device_is_host_accessible(dt.device.device_type),
        "from_dlpack: Space host/device does not match the capsule's device — import a kDLCUDA capsule as from_dlpack<T, own::gpu_view>(m)");
    const int nd = dt.ndim;
    // Trust boundary: `ndim` comes straight from the producer (torch allows 64
    // dims). CLAMP the local fills to TNY_MAX_RANK UNCONDITIONALLY — this must
    // hold even under -DNDEBUG, where _TNY_CHECK is compiled out. An oversized
    // ndim then simply never matches dispatch_rank / fixed<R>.
    _TNY_CHECK(nd <= static_cast<int>(TNY_MAX_RANK), "from_dlpack: ndim exceeds TNY_MAX_RANK (raise -DTNY_MAX_RANK)");
    const int n = nd < static_cast<int>(TNY_MAX_RANK) ? nd : static_cast<int>(TNY_MAX_RANK);
    T * data = reinterpret_cast<T *>(reinterpret_cast<char *>(dt.data) + dt.byte_offset);
    cs::int64_t st[TNY_MAX_RANK];
    if (dt.strides) { for (int i = 0; i < n; ++i) st[i] = dt.strides[i]; }
    else { cs::int64_t s = 1; for (int i = n - 1; i >= 0; --i) { st[i] = s; s *= dt.shape[i]; } }  // C-contiguous
    return as_anyrank<TNY_MAX_RANK, Space>(data, dt.shape, st, nd, copy_meta);
}

/** @brief Import as a **fixed-rank** view (requires `m->dl_tensor.ndim == R`).
 *         Returns a `layout_stride` tensor view borrowing the data; the caller
 *         owns `m`'s lifetime. `Space` (default host `own::view`) tags the view and
 *         is checked against the capsule's device, as in the `anyrank` overload —
 *         `from_dlpack<T, R, own::gpu_view>(m)` for a device capsule. */
template <class T, cs::size_t R, own Space = own::view>
_TNY_HOST dyn_tensor<T, cs::int64_t, R, own_view_of(Space)> from_dlpack(const DLManagedTensor * m) {
    const DLTensor & dt = m->dl_tensor;
    _TNY_CHECK(_dl::dtype_matches<T>(dt.dtype), "from_dlpack<T,R>: DLPack dtype does not match T");
    _TNY_CHECK(dt.ndim == static_cast<int>(R),  "from_dlpack<T,R>: ndim != R");
    _TNY_CHECK(own_is_host_accessible(Space) == _dl::device_is_host_accessible(dt.device.device_type),
        "from_dlpack<T,R>: Space host/device does not match the capsule's device — use from_dlpack<T, R, own::gpu_view>(m)");
    T * data = reinterpret_cast<T *>(reinterpret_cast<char *>(dt.data) + dt.byte_offset);
    // Read only min(R, ndim) from the producer's arrays so a wrong-rank call can
    // never read out of bounds (the check above is debug-only under NDEBUG).
    const cs::size_t n = (dt.ndim >= 0 && static_cast<cs::size_t>(dt.ndim) < R) ? static_cast<cs::size_t>(dt.ndim) : R;
    cs::array<cs::int64_t, R> ext{}, st{};
    for (cs::size_t i = 0; i < n; ++i) ext[i] = dt.shape[i];
    if (dt.strides) { for (cs::size_t i = 0; i < n; ++i) st[i] = dt.strides[i]; }
    else { cs::int64_t s = 1; for (int i = int(n) - 1; i >= 0; --i) { st[i] = s; s *= dt.shape[i]; } }
    using E = cs::dextents<cs::int64_t, R>;
    cs::layout_stride::mapping<E> mp(E(ext), st);
    return dyn_tensor<T, cs::int64_t, R, own_view_of(Space)>(data, mp);
}

/** @brief Import + dispatch: read the dtype/rank from the `DLManagedTensor` and
 *         call `f` with a fixed-rank typed view (one instantiation per (dtype,
 *         rank)). Returns false if the dtype/rank is outside the supported set.
 *         Data borrowed; caller owns `m`. `Space` (default host `own::view`) tags
 *         the views and is checked against the capsule's device — dispatch a device
 *         capsule with `dispatch_dlpack<own::gpu_view>(m, f)`. */
template <own Space = own::view, class F>
_TNY_HOST bool dispatch_dlpack(const DLManagedTensor * m, F && f) {
    const DLDataType d = m->dl_tensor.dtype;
    auto by_rank = [&](auto tag) -> bool {
        using T = decltype(tag);
        if (!_dl::dtype_matches<T>(d)) return false;
        return dispatch_rank(from_dlpack<T, Space>(m), f);
    };
    if (d.lanes != 1) return false;
    if (d.code == kDLBool && d.bits == 8) return by_rank(bool{});
    if (d.code == kDLFloat && d.bits == 32) return by_rank(float{});
    if (d.code == kDLFloat && d.bits == 64) return by_rank(double{});
    if (d.code == kDLFloat && d.bits == 16) return by_rank(half{});
    if (d.code == kDLBfloat && d.bits == 16) return by_rank(bfloat16{});
    if (d.code == kDLInt) {
        if (d.bits == 8)  return by_rank((cs::int8_t)0);
        if (d.bits == 16) return by_rank((cs::int16_t)0);
        if (d.bits == 32) return by_rank((cs::int32_t)0);
        if (d.bits == 64) return by_rank((cs::int64_t)0);
    }
    if (d.code == kDLUInt) {
        if (d.bits == 8)  return by_rank((cs::uint8_t)0);
        if (d.bits == 16) return by_rank((cs::uint16_t)0);
        if (d.bits == 32) return by_rank((cs::uint32_t)0);
        if (d.bits == 64) return by_rank((cs::uint64_t)0);
    }
    return false;
}

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_DLPACK
