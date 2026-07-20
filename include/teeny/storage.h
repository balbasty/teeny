#ifndef TNY_MD_STORAGE
#define TNY_MD_STORAGE
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <teeny/_core/defines.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

/**
 * @brief Ownership / memory-space of a tensor's storage.
 *
 * `view` / `stack` need no allocator. The owning modes differ only in where the
 * memory lives and how it is (de)allocated:
 *   - `heap`   : ordinary C++ `new[]` / `delete[]` (host memory).
 *   - `device` : `cudaMalloc`     (device memory; not host-dereferenceable).
 *   - `host`   : `cudaMallocHost` (page-locked host memory).
 *   - `pinned` : `cudaHostAlloc`  (pinned/mapped host memory).
 * The `device`/`host`/`pinned` storage is defined in the opt-in `teeny/cuda.h`
 * (which needs the CUDA runtime); using them without it is a compile error.
 */
enum class own { view, stack, heap, device, host, pinned };

/** @brief Whether the mode owns (and therefore allocates) its storage. */
_TNY_API constexpr bool own_is_owning(own o) noexcept {
    return o == own::heap || o == own::device || o == own::host || o == own::pinned;
}
/** @brief Whether the storage is dereferenceable from the host. */
_TNY_API constexpr bool own_is_host_accessible(own o) noexcept {
    return o != own::device;
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

/* --- view: non-owning pointer ------------------------------------- */
template <class T, cs::size_t N>
struct storage<T, own::view, N> {
    T * p = nullptr;
    storage() = default;
    _TNY_API constexpr storage(T * q) noexcept : p(q) {}
    _TNY_API constexpr T * data() const noexcept { return p; }
};

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
