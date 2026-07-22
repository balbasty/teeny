#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main()
{
    // ---- fixed-width value forms + numpy short spellings (bytes) ------------
    static_assert(cs::is_same<I1<1>, Int8<1>>::value,   "I1 == Int8");
    static_assert(cs::is_same<I2<1>, Int16<1>>::value,  "I2 == Int16");
    static_assert(cs::is_same<I4<3>, Int32<3>>::value,  "I4 == Int32");
    static_assert(cs::is_same<I8<1>, Int64<1>>::value,  "I8 == Int64");
    static_assert(cs::is_same<U1<1>, UInt8<1>>::value,  "U1 == UInt8");
    static_assert(cs::is_same<U4<1>, UInt32<1>>::value, "U4 == UInt32");
    static_assert(I4<7>::value == 7, "value carried");
    static_assert(UInt8<200>::value == 200, "unsigned value");
    // still convert to a runtime integer (usable as an index)
    long a[5] = {0,1,2,3,4};
    auto v = wrap(a, shape<5>{});
    if (v(I4<2>()) != 2) return 1;

    // ---- numpy dtype type aliases (bytes) ----------------------------------
    static_assert(cs::is_same<i1, cs::int8_t>::value,   "i1");
    static_assert(cs::is_same<i4, cs::int32_t>::value,  "i4");
    static_assert(cs::is_same<i8, cs::int64_t>::value,  "i8");
    static_assert(cs::is_same<u1, cs::uint8_t>::value,  "u1");
    static_assert(cs::is_same<u8, cs::uint64_t>::value, "u8");
    static_assert(cs::is_same<f4, float>::value,        "f4 == float");
    static_assert(cs::is_same<f8, double>::value,       "f8 == double");
    static_assert(cs::is_same<f2, half>::value,         "f2 == half");
    static_assert(cs::is_same<bf16, bfloat16>::value,   "bf16 == bfloat16");

    // ---- numpy-style layout spellings --------------------------------------
    static_assert(cs::is_same<corder, layout_right>::value, "corder == layout_right");
    static_assert(cs::is_same<forder, layout_left>::value,  "forder == layout_left");
    double lb[6] = {0,1,2,3,4,5};
    auto cw = wrap<corder>(lb, shape<2,3>{});   // row-major: strides (3,1)
    auto fw = wrap<forder>(lb, shape<2,3>{});   // col-major: strides (1,2)
    if (cw(0,1) != 1 || fw(1,0) != 1) return 2;

    // ---- rank<N>: fully-dynamic shape of a given rank ----------------------
    static_assert(rank<3>::rank() == 3, "rank<3> is rank 3");
    static_assert(cs::is_same<rank<3>, shape<-1,-1,-1>>::value, "rank<3> == shape<-1,-1,-1>");
    static_assert(rank<3>::rank_dynamic() == 3, "rank<3> is all-dynamic");
    static_assert(rank<0>::rank() == 0, "rank<0> is the scalar shape");
    double rb[6] = {0,1,2,3,4,5};
    auto rv = wrap(rb, rank<2>{2,3});           // runtime 2x3 view spelled with rank<2>
    if (rv.extent(0) != 2 || rv.extent(1) != 3 || rv(1,2) != 5) return 3;

    // usable as an element type in a factory
    auto z = zeros<i4>(shape<2,2>{});
    static_assert(cs::is_same<decltype(z)::element_type, cs::int32_t>::value, "zeros<i4>");
    auto d = make_local<f8>(shape<3>{});
    static_assert(cs::is_same<decltype(d)::element_type, double>::value, "make_local<f8>");

    return 0;
}
