// Cross-width broadcasting (#167 / #115 open-Q3): the broadcast RESULT inherits the
// WIDER of the two operands' offset index types. This is lossless (the engine already
// casts both operands to the result index type) and prevents a wide operand's strides
// from being truncated to a narrow result's width. Same-width mixes are unchanged.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

template <class T> using idx_of = typename T::index_type;

int main() {
    double buf[12]; for (int i = 0; i < 12; ++i) buf[i] = i;

    // (1) int32 (lhs) + int64 (rhs) -> result index type broadens to int64.
    auto a32 = wrap(buf, shape32<2,3>{});          // int32-indexed view
    auto b64 = wrap(buf, shape<2,3>{});            // int64-indexed view
    auto c1  = a32 + b64;
    static_assert(cs::is_same<idx_of<decltype(c1)>, cs::int64_t>::value, "int32+int64 -> int64 result");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j)
        if (c1(i,j) != a32(i,j) + b64(i,j)) return 1;

    // (2) order-independent: int64 (lhs) + int32 (rhs) -> int64 too.
    auto c2 = b64 + a32;
    static_assert(cs::is_same<idx_of<decltype(c2)>, cs::int64_t>::value, "int64+int32 -> int64 result");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j)
        if (c2(i,j) != b64(i,j) + a32(i,j)) return 2;

    // (3) same-width unchanged: int32+int32 -> int32, int64+int64 -> int64.
    auto s3 = wrap(buf, shape32<2,3>{}) + wrap(buf, shape32<2,3>{});
    static_assert(cs::is_same<idx_of<decltype(s3)>, cs::int32_t>::value, "int32+int32 stays int32");
    auto s4 = wrap(buf, shape<2,3>{}) + wrap(buf, shape<2,3>{});
    static_assert(cs::is_same<idx_of<decltype(s4)>, cs::int64_t>::value, "int64+int64 stays int64");

    // (4) mixed-width WITH broadcast (stretch on the narrow operand): (int32 2x1) +
    //     (int64 2x3) -> (2,3) int64. Element identity vs the int64 reference proves the
    //     wide operand's strides are NOT truncated through a narrow result.
    auto col32 = wrap(buf, shape32<2,1>{});        // int32, stretches across axis 1
    auto M64   = wrap(buf, shape<2,3>{});
    auto c4    = col32 + M64;
    static_assert(cs::is_same<idx_of<decltype(c4)>, cs::int64_t>::value, "broadcast mix -> int64");
    static_assert(decltype(c4)::extents_type::static_extent(0) == 2 &&
                  decltype(c4)::extents_type::static_extent(1) == 3, "result 2x3");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j)
        if (c4(i,j) != col32(i,0) + M64(i,j)) return 3;

    // (5) comparisons share the same result extents -> the bool result also broadens.
    auto m = a32 < b64;
    static_assert(cs::is_same<idx_of<decltype(m)>, cs::int64_t>::value, "compare result broadens too");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j)
        if (m(i,j) != (a32(i,j) < b64(i,j))) return 4;

    // (6) dynamic-shape mixed width -> heap result, still int64-indexed.
    auto da = wrap(buf, shape_as<cs::int32_t,-1,-1>{2,3});   // int32 dynamic
    auto db = wrap(buf, shape<-1,-1>{2,3});                  // int64 dynamic
    auto c6 = da + db;
    static_assert(cs::is_same<idx_of<decltype(c6)>, cs::int64_t>::value, "dynamic mix -> int64");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j)
        if (c6(i,j) != da(i,j) + db(i,j)) return 5;

    return 0;
}
