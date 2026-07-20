#ifndef TNY_MD_STORAGE
#define TNY_MD_STORAGE
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <teeny/_core/defines.h>

_TNY_NAMESPACE_BEGIN(tny)
_TNY_NAMESPACE_BEGIN(md)

namespace cs = cuda::std;

/** @brief Ownership kind of a tensor's storage. */
enum class own {
    view,    ///< non-owning pointer (trivially copyable, kernel-passable)
    stack,   ///< inline array (fully-static shape only; value semantics)
    heap     ///< host-only owning pointer (move-only, no ref-counting)
};

/**
 * @brief Storage policy. Selected by `own`; its special members are what give
 *        the enclosing tensor its copy/move semantics.
 *
 * @tparam T  Element type.
 * @tparam O  Ownership kind.
 * @tparam N  Number of elements (used by the stack policy only).
 */
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

/* --- heap: host-only owning pointer, move-only -------------------- */
template <class T, cs::size_t N>
struct storage<T, own::heap, N> {
    T * p = nullptr;
    storage() = default;
    _TNY_HOST explicit storage(cs::size_t n) : p(n ? new T[n]() : nullptr) {}
    storage(const storage &)             = delete;
    storage & operator=(const storage &) = delete;
    _TNY_HOST storage(storage && o) noexcept : p(o.p) { o.p = nullptr; }
    _TNY_HOST storage & operator=(storage && o) noexcept {
        if (this != &o) { delete[] p; p = o.p; o.p = nullptr; }
        return *this;
    }
    _TNY_HOST ~storage() { delete[] p; }
    _TNY_API T *       data()       noexcept { return p; }
    _TNY_API const T * data() const noexcept { return p; }
};

/** @brief Storage element count for a stack tensor (0 for view/heap). */
template <class Mapping, bool Stack>
struct storage_size { static constexpr cs::size_t value = 0; };
template <class Mapping>
struct storage_size<Mapping, true> {
    static constexpr cs::size_t value =
        static_cast<cs::size_t>(Mapping().required_span_size());
};

_TNY_NAMESPACE_END(md)
_TNY_NAMESPACE_END(tny)

#endif // TNY_MD_STORAGE
