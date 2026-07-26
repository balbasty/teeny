#ifndef TNY_MD_DLPACK
#define TNY_MD_DLPACK
// teeny <-> DLPack (https://dmlc.github.io/dlpack) zero-copy interchange with
// numpy / torch / cupy / jax. DLPack exchanges a *pointer + metadata*, never the
// data itself, so import/export are pure host-side struct work — no CUDA needed
// (the device field just LABELS where the pointer lives).
//
// DLPack's C structs come from the OFFICIAL header, so teeny never ships an
// incomplete shadow of the ABI. Precedence:
//   1. the app already included <dlpack/dlpack.h>  -> use it (guard is set);
//   2. a system/framework <dlpack/dlpack.h> exists -> include it;
//   3. neither                                     -> the complete upstream copy
//      teeny vendors under external/dlpack/ (pinned; same as CCCL).
// Whichever wins, `DLManagedTensor` / `DLTensor` / `DLManagedTensorVersioned` and
// the version macros are all defined field-for-field — so teeny composes with any
// framework that imports the real header, in any include order.
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <teeny/defines.h>
#include <teeny/tensor.h>
#include <teeny/dynamic.h>
#include <teeny/half.h>

/* =========================== official DLPack ============================ */
#if defined(DLPACK_DLPACK_H_)
    // Already included by the application — use its definitions.
#elif defined(__has_include) && __has_include(<dlpack/dlpack.h>)
#   include <dlpack/dlpack.h>          // a system / framework header is available
#else
#   include "../../external/dlpack/dlpack/dlpack.h"   // vendored complete upstream copy
#endif

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
template <storage O> _TNY_HOST constexpr DLDeviceType device_of() {
    return storage_is_device(O) ? kDLCUDA          // gpu OR gpu_view
         : (O == storage::pinned || O == storage::pinned_view) ? kDLCUDAHost
         : (O == storage::mapped || O == storage::mapped_view) ? kDLCUDAHost
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

template <class Owner, class T, class Shape, class Layout, storage O>
_TNY_HOST DLManagedTensor * make_managed(const tensor<T, Shape, Layout, O> & t, DLDevice dev, Owner && owner) {
    const int nd = static_cast<int>(t.rank());
    auto * h  = new holder<Owner>{};
    h->shape  = new cs::int64_t[nd ? nd : 1];
    h->stride = new cs::int64_t[nd ? nd : 1];
    // Read ALL of `t` (data, shape, strides) BEFORE moving `owner` in — for an
    // owning export `owner` aliases `t`, and the move would leave `t` empty. The
    // `if constexpr` is needed for a rank-0 (scalar) tensor: `t.stride(i)` with a
    // runtime index would still instantiate `layout::mapping::stride`, which CCCL
    // constrains to rank > 0.
    if constexpr (Shape::rank() > 0) {
        for (int i = 0; i < nd; ++i) {
            h->shape[i]  = static_cast<cs::int64_t>(t.shape(i));
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

// Dispatch on the DLPack **dtype** only: pick the teeny element type from
// `(code, bits)` and call `g(T{})` (a generic callable) -> bool. Shared by the
// rank-collapsing `dispatch_dlpack` and the rank-preserving `dispatch_dlpack_dtype`
// so the supported-dtype table lives in exactly one place. Returns false for an
// unsupported dtype (lanes != 1, or a code/bits teeny has no element type for).
template <class G>
_TNY_HOST bool dispatch_dtype(const DLDataType & d, G && g) {
    if (d.lanes != 1) return false;
    if (d.code == kDLBool  && d.bits ==  8) return g(bool{});
    if (d.code == kDLFloat && d.bits == 32) return g(float{});
    if (d.code == kDLFloat && d.bits == 64) return g(double{});
    if (d.code == kDLFloat && d.bits == 16) return g(half{});
    if (d.code == kDLBfloat && d.bits == 16) return g(bfloat16{});
    if (d.code == kDLInt) {
        if (d.bits ==  8) return g((cs::int8_t)0);
        if (d.bits == 16) return g((cs::int16_t)0);
        if (d.bits == 32) return g((cs::int32_t)0);
        if (d.bits == 64) return g((cs::int64_t)0);
    }
    if (d.code == kDLUInt) {
        if (d.bits ==  8) return g((cs::uint8_t)0);
        if (d.bits == 16) return g((cs::uint16_t)0);
        if (d.bits == 32) return g((cs::uint32_t)0);
        if (d.bits == 64) return g((cs::uint64_t)0);
    }
    return false;
}

// The `DLTensor` payload of any DLPack carrier — the classic managed tensor, a
// bare (unmanaged) `DLTensor`, or the versioned managed tensor. Import/dispatch
// read only this payload; the carrier differs only in who frees it (nobody, for a
// bare DLTensor; `m->deleter` for either managed form — the CALLER's job).
_TNY_HOST inline const DLTensor & as_dltensor(const DLManagedTensor & m)          { return m.dl_tensor; }
_TNY_HOST inline const DLTensor & as_dltensor(const DLTensor & d)                 { return d; }
_TNY_HOST inline const DLTensor & as_dltensor(const DLManagedTensorVersioned & m) { return m.dl_tensor; }

// Core import (shared by every carrier): borrow the DATA, copy the shape/stride
// METADATA into a self-contained carrier. A null `strides` (DLPack's C-contiguous
// shorthand) expands to row-major; `byte_offset` folds into the pointer.
template <class T, storage Space>
_TNY_HOST anyrank<T, cs::int64_t, _meta_store<cs::int64_t, TNY_MAX_RANK>, Space>
import_anyrank(const DLTensor & dt) {
    _TNY_CHECK(dtype_matches<T>(dt.dtype), "from_dlpack: DLPack dtype does not match T");
    _TNY_CHECK(storage_is_host_accessible(Space) == device_is_host_accessible(dt.device.device_type),
        "from_dlpack: Space host/device does not match the tensor's device — import a kDLCUDA tensor as from_dlpack<T, storage::gpu_view>(...)");
    const int nd = dt.ndim;
    // Trust boundary: `ndim` is the producer's (torch allows 64). CLAMP the local
    // fills to TNY_MAX_RANK UNCONDITIONALLY (must hold under -DNDEBUG too); an
    // oversized ndim then simply never matches dispatch_rank / fixed<R>.
    _TNY_CHECK(nd <= static_cast<int>(TNY_MAX_RANK), "from_dlpack: ndim exceeds TNY_MAX_RANK (raise -DTNY_MAX_RANK)");
    const int n = nd < static_cast<int>(TNY_MAX_RANK) ? nd : static_cast<int>(TNY_MAX_RANK);
    T * data = reinterpret_cast<T *>(reinterpret_cast<char *>(dt.data) + dt.byte_offset);
    cs::int64_t st[TNY_MAX_RANK];
    if (dt.strides) { for (int i = 0; i < n; ++i) st[i] = dt.strides[i]; }
    else { cs::int64_t s = 1; for (int i = n - 1; i >= 0; --i) { st[i] = s; s *= dt.shape[i]; } }  // C-contiguous
    return as_anyrank<TNY_MAX_RANK, Space>(data, dt.shape, st, nd, copy_meta);
}
template <class T, cs::size_t R, storage Space>
_TNY_HOST dyn_tensor<T, cs::int64_t, R, storage_view_of(Space)>
import_fixed(const DLTensor & dt) {
    _TNY_CHECK(dtype_matches<T>(dt.dtype), "from_dlpack<T,R>: DLPack dtype does not match T");
    _TNY_CHECK(dt.ndim == static_cast<int>(R),  "from_dlpack<T,R>: ndim != R");
    _TNY_CHECK(storage_is_host_accessible(Space) == device_is_host_accessible(dt.device.device_type),
        "from_dlpack<T,R>: Space host/device does not match the tensor's device — use from_dlpack<T, R, storage::gpu_view>(...)");
    T * data = reinterpret_cast<T *>(reinterpret_cast<char *>(dt.data) + dt.byte_offset);
    // Read only min(R, ndim) so a wrong-rank call can never read out of bounds
    // (the check above is debug-only under NDEBUG).
    const cs::size_t n = (dt.ndim >= 0 && static_cast<cs::size_t>(dt.ndim) < R) ? static_cast<cs::size_t>(dt.ndim) : R;
    cs::array<cs::int64_t, R> ext{}, st{};
    for (cs::size_t i = 0; i < n; ++i) ext[i] = dt.shape[i];
    if (dt.strides) { for (cs::size_t i = 0; i < n; ++i) st[i] = dt.strides[i]; }
    else { cs::int64_t s = 1; for (int i = int(n) - 1; i >= 0; --i) { st[i] = s; s *= dt.shape[i]; } }
    using E = cs::dextents<cs::int64_t, R>;
    cs::layout_stride::mapping<E> mp(E(ext), st);
    return dyn_tensor<T, cs::int64_t, R, storage_view_of(Space)>(data, mp);
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
template <class T, class Shape, class Layout, storage O,
          cs::enable_if_t<storage_is_view(O), int> = 0>
_TNY_HOST DLManagedTensor * to_dlpack(const tensor<T, Shape, Layout, O> & t,
                                      DLDevice dev = { _dl::device_of<O>(), 0 }) {
    return _dl::make_managed(t, dev, _dl::no_owner{});
}

/** @brief Export an **owning** tensor, TRANSFERRING ownership of the buffer into
 *         the capsule (the tensor is moved-from; the capsule's `deleter` frees the
 *         buffer). Device is taken from the tensor's memory space. */
template <class T, class Shape, class Layout, storage O,
          cs::enable_if_t<storage_is_owning(O), int> = 0>
_TNY_HOST DLManagedTensor * to_dlpack(tensor<T, Shape, Layout, O> && t) {
    using Owner = tensor<T, Shape, Layout, O>;
    DLDevice dev{ _dl::device_of<O>(), 0 };
    return _dl::make_managed(static_cast<const Owner &>(t), dev, static_cast<Owner &&>(t));
}

/** @brief Export to a **bare `DLTensor`** (unmanaged — no capsule, no deleter, no
 *         allocation). Borrows both the data AND the shape/stride arrays: the caller
 *         supplies `shape_out`/`strides_out` (each ≥ `t.rank()` `int64_t`s), which
 *         this fills, and the returned `DLTensor` points at them + `t.data()`. The
 *         caller must keep the tensor's memory *and* those two buffers alive for as
 *         long as the `DLTensor` is used. Use for a consumer that takes a plain
 *         `DLTensor` rather than a managed capsule. Device defaults to the tensor's
 *         memory space (override with `dev`). Works for any storage (a pure borrow). */
template <class T, class Shape, class Layout, storage O>
_TNY_HOST DLTensor to_dltensor(const tensor<T, Shape, Layout, O> & t,
                               cs::int64_t * shape_out, cs::int64_t * strides_out,
                               DLDevice dev = { _dl::device_of<O>(), 0 }) {
    const int nd = static_cast<int>(t.rank());
    if constexpr (Shape::rank() > 0) {   // rank-0: skip the runtime-index stride() (CCCL constrains it to rank>0)
        for (int i = 0; i < nd; ++i) {
            shape_out[i]   = static_cast<cs::int64_t>(t.shape(i));
            strides_out[i] = static_cast<cs::int64_t>(t.stride(i));   // DLPack strides are in ELEMENTS
        }
    }
    DLTensor dt;
    dt.data = const_cast<void *>(static_cast<const void *>(t.data()));
    dt.device = dev; dt.ndim = nd; dt.dtype = _dl::dtype_of<T>();
    dt.shape = shape_out; dt.strides = strides_out; dt.byte_offset = 0;
    return dt;
}

/* ============================ import (DLPack -> teeny) ============================ */

/** @brief Import a `DLManagedTensor` of known element type `T` as an `anyrank`
 *         (runtime rank). The shape/stride METADATA is copied into the carrier
 *         (so it is self-contained), while the DATA is BORROWED — the caller keeps
 *         `m` alive while the view is used, then calls `m->deleter(m)`. A null
 *         `strides` (DLPack's C-contiguous shorthand) is expanded to row-major.
 *         `byte_offset` is folded into the data pointer.
 *
 *         `Space` is the memory space to tag the carrier with (default `storage::view`
 *         = host); every view peeled off it inherits it. It is **checked against
 *         `m->dl_tensor.device`**: importing a `kDLCUDA` capsule as the default
 *         host `Space` trips `_TNY_CHECK` — spell `from_dlpack<T, storage::gpu_view>(m)`
 *         so `fixed()`/`peel_front` yield device-tagged views (no host deref of a
 *         device pointer). (Closes the #38 hole where the device field was ignored
 *         and a device capsule silently became a host view.) */
template <class T, storage Space = storage::view>
_TNY_HOST auto from_dlpack(const DLManagedTensor * m)          { return _dl::import_anyrank<T, Space>(_dl::as_dltensor(*m)); }
/** @brief Import a **bare `DLTensor`** (unmanaged — no deleter). Borrows the data,
 *         copies the metadata; the CALLER owns the whole lifetime (there is nothing
 *         to free). Use when a producer hands you a plain `DLTensor*` rather than a
 *         capsule. Same `Space`/device check as the managed overload. */
template <class T, storage Space = storage::view>
_TNY_HOST auto from_dlpack(const DLTensor * dt)               { return _dl::import_anyrank<T, Space>(_dl::as_dltensor(*dt)); }
/** @brief Import a **versioned** managed tensor (DLPack 1.0+, what a modern
 *         `__dlpack__(max_version=…)` emits). Reads its `dl_tensor` payload; as with
 *         the classic capsule the caller keeps `m` alive and calls `m->deleter(m)`. */
template <class T, storage Space = storage::view>
_TNY_HOST auto from_dlpack(const DLManagedTensorVersioned * m) { return _dl::import_anyrank<T, Space>(_dl::as_dltensor(*m)); }

/** @brief Import as a **fixed-rank** view (requires the payload's `ndim == R`).
 *         Returns a `layout_stride` tensor view borrowing the data. `Space` (default
 *         host `storage::view`) tags the view and is checked against the device, as in
 *         the `anyrank` overloads — `from_dlpack<T, R, storage::gpu_view>(m)` for a
 *         device tensor. Accepts all three carriers (managed / bare / versioned). */
template <class T, cs::size_t R, storage Space = storage::view>
_TNY_HOST auto from_dlpack(const DLManagedTensor * m)          { return _dl::import_fixed<T, R, Space>(_dl::as_dltensor(*m)); }
template <class T, cs::size_t R, storage Space = storage::view>
_TNY_HOST auto from_dlpack(const DLTensor * dt)               { return _dl::import_fixed<T, R, Space>(_dl::as_dltensor(*dt)); }
template <class T, cs::size_t R, storage Space = storage::view>
_TNY_HOST auto from_dlpack(const DLManagedTensorVersioned * m) { return _dl::import_fixed<T, R, Space>(_dl::as_dltensor(*m)); }

/** @brief Import + dispatch: read the dtype/rank from the `DLManagedTensor` and
 *         call `f` with a fixed-rank typed view (one instantiation per (dtype,
 *         rank)). Returns false if the dtype/rank is outside the supported set.
 *         Data borrowed; caller owns `m`. `Space` (default host `storage::view`) tags
 *         the views and is checked against the capsule's device — dispatch a device
 *         capsule with `dispatch_dlpack<storage::gpu_view>(m, f)`. */
template <storage Space = storage::view, class Carrier, class F>
_TNY_HOST bool dispatch_dlpack(const Carrier * m, F && f) {
    return _dl::dispatch_dtype(_dl::as_dltensor(*m).dtype, [&](auto tag) -> bool {
        using T = decltype(tag);
        return dispatch_rank(from_dlpack<T, Space>(m), f);   // dtype x total rank -> static view
    });
}

/** @brief Import + **dtype-only** dispatch that PRESERVES the rank: read the dtype
 *         from the capsule and call `f` with the **typed `anyrank`** (rank still
 *         dynamic), instead of collapsing to a fixed rank like `dispatch_dlpack`.
 *         The caller then peels its own axes — the `(*batch, *spatial, C)` batch
 *         idiom `for (auto cell : at.peel_front<-Sr>()) …`, which instantiates the
 *         kernel **once per `Sr`**, not once per total rank. Returns false for an
 *         unsupported dtype. Data borrowed; caller owns `m`. `Space` tags the carrier
 *         and is checked against the capsule's device (see `from_dlpack`). */
template <storage Space = storage::view, class Carrier, class F>
_TNY_HOST bool dispatch_dlpack_dtype(const Carrier * m, F && f) {
    return _dl::dispatch_dtype(_dl::as_dltensor(*m).dtype, [&](auto tag) -> bool {
        using T = decltype(tag);
        f(from_dlpack<T, Space>(m));   // typed anyrank; caller does peel_front<-Sr>
        return true;
    });
}

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_DLPACK
