// strides<...> layout: per-dim static OR dynamic strides (the stride analogue
// of extents), plus the make_* functional factories.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main() {
    double pad[12]; for (int i=0;i<12;++i) pad[i]=i;

    // --- fully static strides: ptr-only ctor, folds, EBO (view == a pointer) ---
    tensor<double, shape<3,3>, strides<4,1>> A(pad);        // padded row stride 4
    static_assert(sizeof(A) == sizeof(double*), "static-strides view is just a pointer (EBO)");
    static_assert(cs::is_same<decltype(A.stride(Int<0>())), cs::integral_constant<long,4>>::value, "stride 0 folds to 4");
    static_assert(cs::is_same<decltype(A.stride(Int<1>())), cs::integral_constant<long,1>>::value, "stride 1 folds to 1");
    if (A(2,1) != pad[2*4+1]) return 1;

    // via view_strided (back-compat: layout_static_stride == strides)
    auto A2 = view_strided<4,1>(pad, shape<3,3>{});
    if (A2(2,1) != A(2,1)) return 2;

    // --- MIXED: outer stride dynamic (-1), inner static 1 ---
    using MixL = strides<-1, 1>;                            // -1 == dynamic_stride
    MixL::mapping<shape<3,3>> m(shape<3,3>{}, cs::array<long,1>{4});
    tensor<double, shape<3,3>, MixL> B(pad, m);
    static_assert(cs::is_same<decltype(B.stride(Int<1>())), cs::integral_constant<long,1>>::value, "mixed: inner folds");
    static_assert(cs::is_same<decltype(B.stride(Int<0>())), long>::value, "mixed: outer is runtime");
    if (B.stride(0) != 4 || B(2,1) != pad[2*4+1]) return 3;

    // dynamic_stride spelling works the same as -1
    static_assert(cs::is_same<strides<dynamic_stride,1>, strides<-1,1>>::value, "dynamic_stride == -1");

    // --- make_* factories (deduce the extents type) ---
    auto v = make_view(pad, shape<3,4>{});
    static_assert(decltype(v)::ownership == own::view, "make_view -> view");
    if (v(1,1) != pad[1*4+1]) return 4;

    auto s = make_local<double>(shape<2,2>{});             // stack, static
    static_assert(decltype(s)::ownership == own::stack, "make_local -> stack");
    s.fill_(7.0); if (s(1,1) != 7.0) return 5;

    using DynE = shape<-1,-1>;
    auto h = make_heap<float>(DynE{2,3});                   // heap, runtime shape deduced
    static_assert(decltype(h)::ownership == own::heap, "make_heap -> heap");
    if (h.extent(0) != 2 || h.extent(1) != 3) return 6;

    return 0;
}
