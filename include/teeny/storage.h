#ifndef TNY_MD_STORAGE
#define TNY_MD_STORAGE
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <teeny/defines.h>
#include <teeny/kwargs.h>

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
// NB when adding a `storage` kind, update EVERY classifier so it isn't silently
// misclassified as a plain host view: `storage_is_owning`, `storage_is_view`,
// `storage_is_device`, `storage_view_of` (below), and `_dl::device_of` (dlpack.h) — plus a
// `storage_policy<T, storage::NEW, N>` specialization (its absence is at least a hard error).
enum class storage { view, stack, heap, gpu, pinned, mapped, gpu_view, pinned_view, mapped_view };

/** @brief Whether the mode owns (and therefore allocates) its storage. */
_TNY_API constexpr bool storage_is_owning(storage o) noexcept {
    return o == storage::heap || o == storage::gpu || o == storage::pinned || o == storage::mapped;
}
/** @brief Whether the mode is a non-owning view (`view`/`gpu_view`/`pinned_view`/
 *         `mapped_view`) — the pointer-wrapping modes (vs `stack`'s inline array). */
_TNY_API constexpr bool storage_is_view(storage o) noexcept {
    return o == storage::view || o == storage::gpu_view
        || o == storage::pinned_view || o == storage::mapped_view;
}
/** @brief Whether the storage lives in device (GPU) memory (owning or view). */
_TNY_API constexpr bool storage_is_device(storage o) noexcept {
    return o == storage::gpu || o == storage::gpu_view;
}
/** @brief Whether the storage is dereferenceable from the host. */
_TNY_API constexpr bool storage_is_host_accessible(storage o) noexcept {
    return !storage_is_device(o);
}
/** @brief The non-owning VIEW kind that preserves a source's memory space: a
 *         device source (`gpu`/`gpu_view`) -> `gpu_view`, a `pinned`/`mapped`
 *         source -> `pinned_view`/`mapped_view`, anything else -> `view`. Every
 *         view-producing op (slice / permute / peel / reshape / at) tags its
 *         result with this so a view never loses (or misreports) its space. */
_TNY_API constexpr storage storage_view_of(storage o) noexcept {
    return storage_is_device(o)                            ? storage::gpu_view
         : (o == storage::pinned || o == storage::pinned_view) ? storage::pinned_view
         : (o == storage::mapped || o == storage::mapped_view) ? storage::mapped_view
                                                       : storage::view;
}

/** @brief Factory sentinel meaning "deduce the ownership from the shape" — a fully
 *         static shape -> `stack` (host+device), any dynamic extent -> `heap`
 *         (host). It is the default backend of `empty` (and the creation
 *         factories), out of the enum's normal range so it never names storage. */
inline constexpr storage storage_deduce = static_cast<storage>(-1);
/** @brief Value-tag carrier for an ownership mode, for the factories' value-tag
 *         backend form, e.g. `empty<T>(shape, storage_c<storage::gpu>{})`. */
template <storage O> using storage_c = cs::integral_constant<storage, O>;
/** @brief A ready-made `storage_c<O>` VALUE — the no-braces spelling of the value tag:
 *         `wrap(p, e, storage_v<storage::gpu>)` instead of `storage_c<storage::gpu>{}`. */
template <storage O> inline constexpr storage_c<O> storage_v{};
template <class> struct _is_storage_tag : cs::false_type {};
template <storage O> struct _is_storage_tag<storage_c<O>> : cs::true_type {};
namespace _kw { template <storage O> struct is_keyword<storage_c<O>> : cs::true_type {}; }

/** @brief storage_arg<Oexpl, Dflt, Tags...>(): the backend a call site should use --
 *         an explicit template argument (Oexpl != storage_deduce) wins, else a
 *         storage_c<O>{} tag found in Tags..., else the library default Dflt
 *         (typically storage_deduce itself, resolved later from the shape by
 *         storage_resolve). static_assert if BOTH an explicit Oexpl and a tag
 *         were supplied for the same keyword. */
template <storage Oexpl, storage Dflt, class... Tags>
_TNY_API constexpr storage storage_arg() {
    static_assert(Oexpl == storage_deduce || !_kw::has<_is_storage_tag, Tags...>(),
        "storage given both as an explicit template argument and as a storage_c<...>{} tag -- pick one");
    return Oexpl != storage_deduce ? Oexpl : _kw::find_t<_is_storage_tag, storage_c<Dflt>, Tags...>::value;
}
/** @brief Resolve a factory's ownership: an explicitly named mode passes through,
 *         `storage_deduce` becomes `stack` for a static shape / `heap` for a dynamic one. */
_TNY_API constexpr storage storage_resolve(storage o, bool static_shape) noexcept {
    return o != storage_deduce ? o : (static_shape ? storage::stack : storage::heap);
}

/* ------------------------------------------------------------------ *
 *     Allocator policies                                             *
 * ------------------------------------------------------------------ */

/** @brief Host allocator using C++ `new[]` / `delete[]`. */
struct cpp_alloc {
    template <class T> _TNY_HOST static T *  allocate(cs::size_t n)        { return n ? new T[n]() : nullptr; }
    template <class T> _TNY_HOST static T *  allocate_uninit(cs::size_t n) { return n ? new T[n]   : nullptr; }
    template <class T> _TNY_HOST static void deallocate(T * p)             { delete[] p; }
};

// Internal tag: construct owning/stack storage WITHOUT the value-initialisation, for
// `empty()` (numpy `np.empty` semantics — the caller fills it). Not public API; the
// public spelling is `empty<T>(...)` (uninitialised) vs `zeros<T>(...)` (zeroed).
struct _uninit_t { explicit _uninit_t() = default; };
inline constexpr _uninit_t _uninit{};

/**
 * @brief Generic owning storage (move-only, no ref-counting), parameterised by
 *        an allocator policy. Shared by all owning `storage` modes.
 */
template <class T, class Alloc>
struct owning_storage {
    T * p = nullptr;
    owning_storage() = default;
    _TNY_HOST explicit owning_storage(cs::size_t n) : p(Alloc::template allocate<T>(n)) {}
    _TNY_HOST owning_storage(cs::size_t n, _uninit_t) : p(Alloc::template allocate_uninit<T>(n)) {}   // no value-init
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
 *     Storage policy per `storage` mode                                  *
 * ------------------------------------------------------------------ */

template <class T, storage O, cs::size_t N>
struct storage_policy;

/* --- the four non-owning VIEW kinds all wrap a bare pointer; they differ only in
 *     the `storage` tag they carry (view = host; gpu_view = device memory; pinned_view
 *     / mapped_view = page-locked host memory, so DLPack labels them kDLCUDAHost).
 *     Share one storage so a tweak can't land in three copies and miss the fourth.
 *     Trivially copyable (single pointer, defaulted specials) -> kernel-passable. */
template <class T>
struct ptr_storage {
    T * p = nullptr;
    ptr_storage() = default;
    _TNY_API constexpr ptr_storage(T * q) noexcept : p(q) {}
    _TNY_API constexpr T * data() const noexcept { return p; }
};
template <class T, cs::size_t N> struct storage_policy<T, storage::view,        N> : ptr_storage<T> { using ptr_storage<T>::ptr_storage; };
template <class T, cs::size_t N> struct storage_policy<T, storage::gpu_view,    N> : ptr_storage<T> { using ptr_storage<T>::ptr_storage; };
template <class T, cs::size_t N> struct storage_policy<T, storage::pinned_view, N> : ptr_storage<T> { using ptr_storage<T>::ptr_storage; };
template <class T, cs::size_t N> struct storage_policy<T, storage::mapped_view, N> : ptr_storage<T> { using ptr_storage<T>::ptr_storage; };

/* --- stack: inline array (fully-static shape) --------------------- */
template <class T, cs::size_t N>
struct storage_policy<T, storage::stack, N> {
    cs::array<T, N> a;
    // Default construction VALUE-INITIALISES (zeros) — so `local<...>{}` / `zeros(...)`
    // keep their zero-fill. `_uninit` leaves `a` indeterminate for `empty()` (the array
    // has no NSDMI, so the uninit ctor's empty init-list really skips it). Copy/move/dtor
    // stay implicit+trivial, so a stack tensor is still trivially copyable.
    _TNY_API constexpr storage_policy() noexcept : a{} {}
    _TNY_API storage_policy(_uninit_t) noexcept {}
    _TNY_API constexpr T *       data()       noexcept { return a.data(); }
    _TNY_API constexpr const T * data() const noexcept { return a.data(); }
};

/* --- heap: C++ new/delete (host) ---------------------------------- */
template <class T, cs::size_t N>
struct storage_policy<T, storage::heap, N> : owning_storage<T, cpp_alloc> {
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
