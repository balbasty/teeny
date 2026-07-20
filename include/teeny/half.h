#ifndef TNY_HALF
#define TNY_HALF
// Portable half-precision element types: `half` (IEEE binary16) and `bfloat16`
// (the two PyTorch uses). Host+device, no CUDA dependency: storage is a single
// uint16 and all arithmetic is done in `float` then rounded back, so these work
// as tensor element types in every teeny engine. On a CUDA build you may prefer
// to map these to `__half` / `__nv_bfloat16` for native device math; the bit
// layout is identical, so a reinterpret is valid.
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <teeny/_core/defines.h>

_TNY_NAMESPACE_BEGIN(tny)

namespace cs = cuda::std;

namespace _detail {
union _f32 { float f; cs::uint32_t u; };
_TNY_API inline cs::uint32_t f2u(float f) { _f32 t; t.f = f; return t.u; }
_TNY_API inline float u2f(cs::uint32_t u) { _f32 t; t.u = u; return t.f; }

// float32 -> IEEE binary16 (round to nearest even)
_TNY_API inline cs::uint16_t f32_to_f16(float f) {
    cs::uint32_t x = f2u(f);
    cs::uint16_t sign = static_cast<cs::uint16_t>((x >> 16) & 0x8000u);
    cs::uint32_t exp  = (x >> 23) & 0xffu;
    cs::uint32_t mant = x & 0x7fffffu;
    if (exp == 0xff)                                    // inf / nan
        return static_cast<cs::uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
    int e = static_cast<int>(exp) - 127 + 15;
    if (e >= 0x1f) return static_cast<cs::uint16_t>(sign | 0x7c00u);   // overflow -> inf
    if (e <= 0) {                                       // subnormal / zero
        if (e < -10) return sign;
        mant |= 0x800000u;
        cs::uint32_t shift = static_cast<cs::uint32_t>(14 - e);
        cs::uint16_t h = static_cast<cs::uint16_t>(mant >> shift);
        cs::uint32_t rem = mant & ((1u << shift) - 1u);
        cs::uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return static_cast<cs::uint16_t>(sign | h);
    }
    cs::uint16_t h = static_cast<cs::uint16_t>((static_cast<cs::uint32_t>(e) << 10) | (mant >> 13));
    cs::uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;   // round to nearest even
    return static_cast<cs::uint16_t>(sign | h);
}
// IEEE binary16 -> float32
_TNY_API inline float f16_to_f32(cs::uint16_t h) {
    cs::uint32_t sign = static_cast<cs::uint32_t>(h & 0x8000u) << 16;
    cs::uint32_t exp  = (h >> 10) & 0x1fu;
    cs::uint32_t mant = h & 0x3ffu;
    cs::uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {                                          // subnormal
            int e = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; --e; }
            mant &= 0x3ffu;
            out = sign | (static_cast<cs::uint32_t>(e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    return u2f(out);
}
// float32 -> bfloat16 (round to nearest even) and back
_TNY_API inline cs::uint16_t f32_to_bf16(float f) {
    cs::uint32_t x = f2u(f);
    if (((x >> 23) & 0xffu) == 0xffu && (x & 0x7fffffu))   // nan -> quiet nan
        return static_cast<cs::uint16_t>((x >> 16) | 0x40u);
    cs::uint32_t r = x + 0x7fffu + ((x >> 16) & 1u);
    return static_cast<cs::uint16_t>(r >> 16);
}
_TNY_API inline float bf16_to_f32(cs::uint16_t h) { return u2f(static_cast<cs::uint32_t>(h) << 16); }
} // namespace _detail

// a half type is a uint16 store + convert-through-float arithmetic. The CRTP-ish
// macro shares the operator soup between `half` and `bfloat16`.
#define _TNY_HALF_TYPE(NAME, TO_F, FROM_F)                                                         \
struct NAME {                                                                                       \
    cs::uint16_t bits{};                                                                            \
    NAME() = default;                                                                               \
    template <class U, cs::enable_if_t<cs::is_arithmetic<U>::value, int> = 0>                       \
    _TNY_API NAME(U v) : bits(FROM_F(static_cast<float>(v))) {}                                     \
    _TNY_API operator float() const { return TO_F(bits); }                                          \
    _TNY_API NAME operator-() const { return NAME(-TO_F(bits)); }                                   \
    _TNY_API NAME & operator+=(NAME o) { *this = NAME(TO_F(bits) + TO_F(o.bits)); return *this; }   \
    _TNY_API NAME & operator-=(NAME o) { *this = NAME(TO_F(bits) - TO_F(o.bits)); return *this; }   \
    _TNY_API NAME & operator*=(NAME o) { *this = NAME(TO_F(bits) * TO_F(o.bits)); return *this; }   \
    _TNY_API NAME & operator/=(NAME o) { *this = NAME(TO_F(bits) / TO_F(o.bits)); return *this; }   \
};                                                                                                 \
_TNY_API inline NAME operator+(NAME a, NAME b) { return NAME(TO_F(a.bits) + TO_F(b.bits)); }        \
_TNY_API inline NAME operator-(NAME a, NAME b) { return NAME(TO_F(a.bits) - TO_F(b.bits)); }        \
_TNY_API inline NAME operator*(NAME a, NAME b) { return NAME(TO_F(a.bits) * TO_F(b.bits)); }        \
_TNY_API inline NAME operator/(NAME a, NAME b) { return NAME(TO_F(a.bits) / TO_F(b.bits)); }        \
_TNY_API inline bool operator==(NAME a, NAME b) { return TO_F(a.bits) == TO_F(b.bits); }            \
_TNY_API inline bool operator!=(NAME a, NAME b) { return TO_F(a.bits) != TO_F(b.bits); }            \
_TNY_API inline bool operator<(NAME a, NAME b)  { return TO_F(a.bits) <  TO_F(b.bits); }            \
_TNY_API inline bool operator>(NAME a, NAME b)  { return TO_F(a.bits) >  TO_F(b.bits); }

_TNY_HALF_TYPE(half,     _detail::f16_to_f32, _detail::f32_to_f16)
_TNY_HALF_TYPE(bfloat16, _detail::bf16_to_f32, _detail::f32_to_bf16)
#undef _TNY_HALF_TYPE

static_assert(sizeof(half) == 2 && sizeof(bfloat16) == 2, "half types are 16-bit");
static_assert(cs::is_trivially_copyable<half>::value, "half must be kernel-passable");

/**
 * @brief The type math should ACCUMULATE in for element type `T`.
 *
 * Half types accumulate in `float` (reductions / repeated adds in 16-bit lose
 * precision fast — jitfields' `reduce_t` pattern). Everything else accumulates
 * in itself.
 */
template <class T> struct compute_type          { using type = T; };
template <>        struct compute_type<half>     { using type = float; };
template <>        struct compute_type<bfloat16> { using type = float; };
template <class T> using compute_type_t = typename compute_type<T>::type;

_TNY_NAMESPACE_END(tny)

#endif // TNY_HALF
