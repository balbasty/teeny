// strides<...> layout: per-dim static OR dynamic strides (the stride analogue
// of extents), plus the make_* functional factories.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

// --- #389 element-identity probe ------------------------------------------
// The static stride pack used to live in a `static constexpr cs::int64_t S_[N]`
// array data member. nvcc never places such a member in device memory, so
// indexing it with a RUNTIME rank -- which `mapping::stride(r)` and the gather
// both do -- made every `strides<...>`-layout view uncompilable for the device.
// It is now read through pack folds instead. This probe pins the replacement to
// the retired member's exact semantics: for a battery of packs, every accessor
// must agree with a literal reference array, at every rank, with a RUNTIME index
// (`volatile` defeats constant folding, so it exercises the path that broke).
template <class L, int N>
static int probe_pack(const cs::int64_t (&ref)[N]) {
    if (L::N != static_cast<cs::size_t>(N)) return 1;
    cs::size_t ndyn_ref = 0;
    for (int r = 0; r < N; ++r) if (ref[r] == dynamic_stride) ++ndyn_ref;
    if (L::ndyn() != ndyn_ref)                 return 2;
    if (L::all_static() != (ndyn_ref == 0))    return 3;
    for (int r = 0; r < N; ++r) {
        volatile int rv = r;                                   // a RUNTIME rank
        if (L::static_stride(static_cast<cs::size_t>(rv)) != ref[r]) return 4;
        cs::size_t slot_ref = 0;
        for (int i = 0; i < r; ++i) if (ref[i] == dynamic_stride) ++slot_ref;
        if (L::slot(static_cast<cs::size_t>(rv)) != slot_ref)        return 5;
    }
    return 0;
}

int main() {
    double pad[12]; for (int i=0;i<12;++i) pad[i]=i;

    // --- fully static strides: ptr-only ctor, folds, EBO (view == a pointer) ---
    tensor<double, shape<3,3>, strides<4,1>> A(pad);        // padded row stride 4
    static_assert(sizeof(A) == sizeof(double*), "static-strides view is just a pointer (EBO)");
    static_assert(cs::is_same<decltype(A.stride(Int<0>())), cs::integral_constant<cs::int64_t,4>>::value, "stride 0 folds to 4");
    static_assert(cs::is_same<decltype(A.stride(Int<1>())), cs::integral_constant<cs::int64_t,1>>::value, "stride 1 folds to 1");
    if (A(2,1) != pad[2*4+1]) return 1;

    // via wrap(..., strides<S...>{}) — compile-time strides fold into the type
    auto A2 = wrap(pad, shape<3,3>{}, strides<4,1>{});
    if (A2(2,1) != A(2,1)) return 2;

    // --- MIXED: outer stride dynamic, inner static 1 ---
    // The clean spelling (analogue of shape<-1,3>{2}): static pattern in the
    // template, runtime strides for the dynamic_stride slots in braces.
    auto B = wrap<dynamic_stride, 1>(pad, shape<3,3>{}, {4});   // outer runtime 4, inner static 1
    static_assert(cs::is_same<decltype(B.stride(Int<1>())), cs::integral_constant<cs::int64_t,1>>::value, "mixed: inner folds");
    static_assert(cs::is_same<decltype(B.stride(Int<0>())), cs::int64_t>::value, "mixed: outer is runtime");
    if (B.stride(0) != 4 || B(2,1) != pad[2*4+1]) return 3;
    // it matches the explicit strides<dynamic_stride,1> mapping construction
    using MixL = strides<dynamic_stride, 1>;
    MixL::mapping<shape<3,3>> m(shape<3,3>{}, cs::array<cs::int64_t,1>{4});
    tensor<double, shape<3,3>, MixL> B2(pad, m);
    if (B2(2,1) != B(2,1)) return 30;

    // --- NEGATIVE static stride: -1 means stride -1, NOT dynamic ---
    static_assert(strides<-4,1>::static_stride(0) == -4, "strides<-4,1> is a real -4 stride");
    static_assert(!cs::is_same<strides<-1,1>, strides<dynamic_stride,1>>::value, "-1 stride != dynamic");
    // a reversed-row view: row r starts at pad[8 - 4r], columns forward
    tensor<double, shape<3,3>, strides<-4,1>> R(pad + 8);
    static_assert(cs::is_same<decltype(R.stride(Int<0>())), cs::integral_constant<cs::int64_t,-4>>::value, "negative stride folds");
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
    static_assert(cs::is_same<decltype(rt_arr.stride(Int<0>())), cs::int64_t>::value, "runtime array stays runtime");
    for (long i = 0; i < 3; ++i)
        for (long j = 0; j < 4; ++j)
            if (c_ord(i,j) != st_tag(i,j) || st_tag(i,j) != rt_arr(i,j)) return 12;

    // negative stride agrees between the static tag and the runtime array too
    auto neg_tag = wrap(pad + 8, shape<3,4>{}, strides<-4,1>{});
    for (long i = 0; i < 3; ++i)
        for (long j = 0; j < 4; ++j)
            if (neg_tag(i,j) != neg(i,j)) return 13;

    // --- rank-0 element access on a strides<...> mapping (squeeze to scalar) ---
    // Regression for a zero-length-array bug (same class as #313): this
    // mapping's operator() decoded I... into a bare `T id[]`, which is a
    // zero-size array when I... is empty (rank-0) -- a GCC/Clang extension
    // MSVC rejects.
    auto v0 = local<double, shape<1>>();
    v0(0) = 42;
    auto s0 = v0.squeeze<0>();
    static_assert(decltype(s0)::rank() == 0, "squeeze<0> on shape<1> -> rank 0");
    if (s0() != 42) return 14;

    // --- #389: the stride pack, read at COMPILE time and at RUNTIME rank ------
    // Compile-time reads still fold to immediates (these are the folds every
    // slice/peel output layout is built from).
    static_assert(strides<12,dynamic_stride,1>::static_stride(0) == 12, "static slot folds");
    static_assert(strides<12,dynamic_stride,1>::static_stride(1) == dynamic_stride, "dynamic slot reads as the sentinel");
    static_assert(strides<12,dynamic_stride,1>::ndyn() == 1, "one runtime stride");
    static_assert(strides<12,dynamic_stride,1>::slot(2) == 1, "dim 2 sits past one dynamic slot");
    static_assert(strides<0,1>::static_stride(0) == 0, "a stride-0 (broadcast) axis is a real 0, not dynamic");
    static_assert(strides<>::ndyn() == 0 && strides<>::all_static(), "rank-0 pack");

    // ...and a runtime rank agrees with the literal reference, every pack shape.
    {
        const cs::int64_t p1[] = {4,1};                                    if (int e = probe_pack<strides<4,1>>(p1))                                    return 40 + e;
        const cs::int64_t p2[] = {-4,1};                                   if (int e = probe_pack<strides<-4,1>>(p2))                                   return 50 + e;
        const cs::int64_t p3[] = {0,1};                                    if (int e = probe_pack<strides<0,1>>(p3))                                    return 60 + e;
        const cs::int64_t p4[] = {dynamic_stride,1};                       if (int e = probe_pack<strides<dynamic_stride,1>>(p4))                       return 70 + e;
        const cs::int64_t p5[] = {dynamic_stride,dynamic_stride,dynamic_stride};
                                                                           if (int e = probe_pack<strides<dynamic_stride,dynamic_stride,dynamic_stride>>(p5)) return 80 + e;
        const cs::int64_t p6[] = {20,dynamic_stride,5,1};                  if (int e = probe_pack<strides<20,dynamic_stride,5,1>>(p6))                  return 90 + e;
    }

    // The hot-path entry itself: mapping::stride(RUNTIME r) over a mixed pack.
    {
        using ML = strides<12, dynamic_stride, 1>;
        ML::mapping<shape<2,3,4>> mm(shape<2,3,4>{}, cs::array<cs::int64_t,1>{4});
        const cs::int64_t want[] = {12,4,1};
        for (int r = 0; r < 3; ++r) { volatile int rv = r; if (mm.stride(static_cast<cs::size_t>(rv)) != want[r]) return 15; }
        // ...and the offset it feeds agrees with the hand-computed one
        if (mm(1,2,3) != 12*1 + 4*2 + 1*3) return 16;
    }

    // EBO invariant: an all-static pack leaves the mapping (and so the tensor) empty.
    static_assert(sizeof(strides<4,1>::mapping<shape<3,3>>) == 1, "fully-static strides mapping is EMPTY (EBO)");
    static_assert(sizeof(tensor<double, shape<3,3>, strides<4,1>>) == sizeof(double*), "...so the view is just a pointer");

    return 0;
}
