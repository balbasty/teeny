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

    // via wrap(..., strides<S...>{}) — compile-time strides fold into the type
    auto A2 = wrap(pad, shape<3,3>{}, strides<4,1>{});
    if (A2(2,1) != A(2,1)) return 2;

    // --- MIXED: outer stride dynamic, inner static 1 ---
    // The clean spelling (analogue of shape<-1,3>{2}): static pattern in the
    // template, runtime strides for the dynamic_stride slots in braces.
    auto B = wrap<dynamic_stride, 1>(pad, shape<3,3>{}, {4});   // outer runtime 4, inner static 1
    static_assert(cs::is_same<decltype(B.stride(Int<1>())), cs::integral_constant<long,1>>::value, "mixed: inner folds");
    static_assert(cs::is_same<decltype(B.stride(Int<0>())), long>::value, "mixed: outer is runtime");
    if (B.stride(0) != 4 || B(2,1) != pad[2*4+1]) return 3;
    // it matches the explicit strides<dynamic_stride,1> mapping construction
    using MixL = strides<dynamic_stride, 1>;
    MixL::mapping<shape<3,3>> m(shape<3,3>{}, cs::array<long,1>{4});
    tensor<double, shape<3,3>, MixL> B2(pad, m);
    if (B2(2,1) != B(2,1)) return 30;

    // --- NEGATIVE static stride: -1 means stride -1, NOT dynamic ---
    static_assert(strides<-4,1>::S_[0] == -4, "strides<-4,1> is a real -4 stride");
    static_assert(!cs::is_same<strides<-1,1>, strides<dynamic_stride,1>>::value, "-1 stride != dynamic");
    // a reversed-row view: row r starts at pad[8 - 4r], columns forward
    tensor<double, shape<3,3>, strides<-4,1>> R(pad + 8);
    static_assert(cs::is_same<decltype(R.stride(Int<0>())), cs::integral_constant<long,-4>>::value, "negative stride folds");
    if (R(0,1) != pad[8+1] || R(1,0) != pad[4] || R(2,0) != pad[0]) return 7;

    // --- make_* factories (deduce the extents type) ---
    auto v = make_view(pad, shape<3,4>{});
    static_assert(decltype(v)::ownership == storage::view, "make_view -> view");
    if (v(1,1) != pad[1*4+1]) return 4;

    auto s = make_local<double>(shape<2,2>{});             // stack, static
    static_assert(decltype(s)::ownership == storage::stack, "make_local -> stack");
    s.fill_(7.0); if (s(1,1) != 7.0) return 5;

    using DynE = shape<-1,-1>;
    auto h = make_heap<float>(DynE{2,3});                   // heap, runtime shape deduced
    static_assert(decltype(h)::ownership == storage::heap, "make_heap -> heap");
    if (h.extent(0) != 2 || h.extent(1) != 3) return 6;

    // --- wrap(ptr, shape, runtime strides) -> a layout_stride view ---
    auto rs = wrap(pad, shape<3,4>{}, {4,1});               // row-major, runtime strides
    static_assert(cs::is_same<decltype(rs)::layout_type, cs::layout_stride>::value, "wrap+strides -> layout_stride");
    if (rs(1,1) != pad[4+1]) return 8;
    auto cs_ = wrap(pad, shape<3,4>{}, {1,3});              // column-major
    if (cs_(1,0) != pad[1] || cs_(0,1) != pad[3]) return 9;
    auto dyn = wrap(pad, shape<-1,4>{3}, {4,1});            // dynamic outer + strides
    if (dyn(2,3) != pad[2*4+3]) return 10;
    auto neg = wrap(pad + 8, shape<3,4>{}, {-4,1});         // reversed axis 0 (negative stride)
    if (neg(0,0) != pad[8] || neg(2,0) != pad[0]) return 11;

    // --- every way of passing the same geometry AGREES on element access -------
    // static tag folds to a strides<4,1> layout; the runtime {4,1} is layout_stride;
    // plain wrap is C-order. Different TYPES, identical addressing.
    auto c_ord  = wrap(pad, shape<3,4>{});                  // contiguous C-order
    auto st_tag = wrap(pad, shape<3,4>{}, strides<4,1>{});  // compile-time strides (fold)
    auto rt_arr = wrap(pad, shape<3,4>{}, {4,1});           // runtime strides (layout_stride)
    static_assert(decltype(st_tag.stride(Int<0>()))::value == 4, "static tag folds the stride");
    static_assert(cs::is_same<decltype(rt_arr.stride(Int<0>())), long>::value, "runtime array stays runtime");
    for (long i = 0; i < 3; ++i)
        for (long j = 0; j < 4; ++j)
            if (c_ord(i,j) != st_tag(i,j) || st_tag(i,j) != rt_arr(i,j)) return 12;

    // negative stride agrees between the static tag and the runtime array too
    auto neg_tag = wrap(pad + 8, shape<3,4>{}, strides<-4,1>{});
    for (long i = 0; i < 3; ++i)
        for (long j = 0; j < 4; ++j)
            if (neg_tag(i,j) != neg(i,j)) return 13;

    return 0;
}
