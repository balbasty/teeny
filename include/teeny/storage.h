#ifndef TNY_MD_STORAGE
#define TNY_MD_STORAGE
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <teeny/defines.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/**
 * @brief Ownership / memory-space of a tensor's storage.
 *
 * `view` / `stack` need no allocator. The owning modes differ only in where the
 * memory lives and how it is (de)allocated:
 *   - `heap`   : ordinary C++ `new[]` / `delete[]` (host memory).
 *   - `gpu`    : `cudaMalloc`     (device memory; not host-dereferenceable).
 *   - `pinned` : `cudaMallocHost` (page-locked host memory — pytorch's "pinned").
 *   - `mapped` : `cudaHostAlloc`  (page-locked + device-mapped / zero-copy).
 * The `gpu`/`pinned`/`mapped` storage is defined in the opt-in `teeny/cuda.h`
 * (which needs the CUDA runtime); using them without it is a compile error.
 */
//  - `gpu_view` : a non-owning VIEW of device memory (the memory-space-carrying
//                 counterpart of `view`). Slicing / permuting / peeling a `gpu`
//                 tensor yields a `gpu_view`, not a `view`, so device pointers
//                 stay distinguishable from host ones in the type.
//  - `pinned_view` / `mapped_view` : the same idea for `pinned` / `mapped` — a
//                 view of page-locked host memory keeps its space (both are
//                 host-dereferenceable, so they behave like `view` everywhere
//                 except the DLPack device label, which stays `kDLCUDAHost`).
// NB when adding an `own` kind, update EVERY classifier so it isn't silently
// misclassified as a plain host view: `own_is_owning`, `own_is_view`,
// `own_is_device`, `own_view_of` (below), and `_dl::device_of` (dlpack.h) — plus a
// `storage<T, own::NEW, N>` specialization (its absence is at least a hard error).
enum class own { view, stack, heap, gpu, pinned, mapped, gpu_view, pinned_view, mapped_view };

/** @brief Whether the mode owns (and therefore allocates) its storage. */
_TNY_API constexpr bool own_is_owning(own o) noexcept {
    return o == own::heap || o == own::gpu || o == own::pinned || o == own::mapped;
}
/** @brief Whether the mode is a non-owning view (`view`/`gpu_view`/`pinned_view`/
 *         `mapped_view`) — the pointer-wrapping modes (vs `stack`'s inline array). */
_TNY_API constexpr bool own_is_view(own o) noexcept {
    return o == own::view || o == own::gpu_view
        || o == own::pinned_view || o == own::mapped_view;
}
/** @brief Whether the storage lives in device (GPU) memory (owning or view). */
_TNY_API constexpr bool own_is_device(own o) noexcept {
    return o == own::gpu || o == own::gpu_view;
}
/** @brief Whether the storage is dereferenceable from the host. */
_TNY_API constexpr bool own_is_host_accessible(own o) noexcept {
    return !own_is_device(o);
}
/** @brief The non-owning VIEW kind that preserves a source's memory space: a
 *         device source (`gpu`/`gpu_view`) -> `gpu_view`, a `pinned`/`mapped`
 *         source -> `pinned_view`/`mapped_view`, anything else -> `view`. Every
 *         view-producing op (slice / permute / peel / reshape / at) tags its
 *         result with this so a view never loses (or misreports) its space. */
_TNY_API constexpr own own_view_of(own o) noexcept {
    return own_is_device(o)                            ? own::gpu_view
         : (o == own::pinned || o == own::pinned_view) ? own::pinned_view
         : (o == own::mapped || o == own::mapped_view) ? own::mapped_view
                                                       : own::view;
}

/** @brief Factory sentinel meaning "deduce the ownership from the shape" — a fully
 *         static shape -> `stack` (host+device), any dynamic extent -> `heap`
 *         (host). It is the default backend of `empty` (and the creation
 *         factories), out of the enum's normal range so it never names storage. */
inline constexpr own own_deduce = static_cast<own>(-1);
/** @brief Value-tag carrier for an ownership mode, for the factories' value-tag
 *         backend form, e.g. `empty<T>(shape, own_c<own::gpu>{})`. */
template <own O> using own_c = cs::integral_constant<own, O>;
/** @brief Resolve a factory's ownership: an explicitly named mode passes through,
 *         `own_deduce` becomes `stack` for a static shape / `heap` for a dynamic one. */
_TNY_API constexpr own own_resolve(own o, bool static_shape) noexcept {
    return o != own_deduce ? o : (static_shape ? own::stack : own::heap);
}

/* ------------------------------------------------------------------ *
 *     Allocator policies                                             *
 * ------------------------------------------------------------------ */

/** @brief Host allocator using C++ `new[]` / `delete[]`. */
struct cpp_alloc {
    template <class T> _TNY_HOST static T *  allocate(cs::size_t n) { return n ? new T[n]() : nullptr; }
    template <class T> _TNY_HOST static void deallocate(T * p)      { delete[] p; }
};

/**
 * @brief Generic owning storage (move-only, no ref-counting), parameterised by
 *        an allocator policy. Shared by all owning `own` modes.
 */
template <class T, class Alloc>
struct owning_storage {
    T * p = nullptr;
    owning_storage() = default;
    _TNY_HOST explicit owning_storage(cs::size_t n) : p(Alloc::template allocate<T>(n)) {}
    owning_storage(const owning_storage &)             = delete;
    owning_storage & operator=(const owning_storage &) = delete;
    _TNY_HOST owning_storage(owning_storage && o) noexcept : p(o.p) { o.p = nullptr; }
    _TNY_HOST owning_storage & operator=(owning_storage && o) noexcept {
        if (this != &o) { Alloc::deallocate(p); p = o.p; o.p = nullptr; }
        return *this;
    }
    _TNY_HOST ~owning_storage() { Alloc::deallocate(p); }
    _TNY_API T *       data()       noexcept { return p; }
    _TNY_API const T * data() const noexcept { return p; }
};

/* ------------------------------------------------------------------ *
 *     Storage policy per `own` mode                                  *
 * ------------------------------------------------------------------ */

template <class T, own O, cs::size_t N>
struct storage;

/* --- the four non-owning VIEW kinds all wrap a bare pointer; they differ only in
 *     the `own` tag they carry (view = host; gpu_view = device memory; pinned_view
 *     / mapped_view = page-locked host memory, so DLPack labels them kDLCUDAHost).
 *     Share one storage so a tweak can't land in three copies and miss the fourth.
 *     Trivially copyable (single pointer, defaulted specials) -> kernel-passable. */
template <class T>
struct _ptr_storage {
    T * p = nullptr;
    _ptr_storage() = default;
    _TNY_API constexpr _ptr_storage(T * q) noexcept : p(q) {}
    _TNY_API constexpr T * data() const noexcept { return p; }
};
template <class T, cs::size_t N> struct storage<T, own::view,        N> : _ptr_storage<T> { using _ptr_storage<T>::_ptr_storage; };
template <class T, cs::size_t N> struct storage<T, own::gpu_view,    N> : _ptr_storage<T> { using _ptr_storage<T>::_ptr_storage; };
template <class T, cs::size_t N> struct storage<T, own::pinned_view, N> : _ptr_storage<T> { using _ptr_storage<T>::_ptr_storage; };
template <class T, cs::size_t N> struct storage<T, own::mapped_view, N> : _ptr_storage<T> { using _ptr_storage<T>::_ptr_storage; };

/* --- stack: inline array (fully-static shape) --------------------- */
template <class T, cs::size_t N>
struct storage<T, own::stack, N> {
    cs::array<T, N> a{};
    _TNY_API constexpr T *       data()       noexcept { return a.data(); }
    _TNY_API constexpr const T * data() const noexcept { return a.data(); }
};

/* --- heap: C++ new/delete (host) ---------------------------------- */
template <class T, cs::size_t N>
struct storage<T, own::heap, N> : owning_storage<T, cpp_alloc> {
    using owning_storage<T, cpp_alloc>::owning_storage;
};

/** @brief Storage element count for a stack tensor (0 for view/owning). */
template <class Mapping, bool Stack>
struct storage_size { static constexpr cs::size_t value = 0; };
template <class Mapping>
struct storage_size<Mapping, true> {
    static constexpr cs::size_t value =
        static_cast<cs::size_t>(Mapping().required_span_size());
};

_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_STORAGE
