#include <teeny/tensor.h>
#include <cuda/std/type_traits>

using namespace tny;
using namespace tny::statix;
using cuda::std::is_same;

// Descriptors for a rank-3 [2,3,4] tensor.
using shape_static = tuple<cvalue<long,2>, cvalue<long,3>, cvalue<long,4> >;
using stride_c     = tuple<cvalue<long,12>, cvalue<long,4>, cvalue<long,1> >;   // row-major
using shape_mixed  = tuple<cnone, cvalue<long,3>, cvalue<long,4> >;             // [?,3,4]

using tensor_static = tensor<long, long, shape_static, stride_c>;

// ---- compile-time properties -------------------------------------------

static_assert(tensor_static::ndim == 3, "ndim");
static_assert(cuda::std::is_trivially_copyable<tensor_static>::value,
              "trivially copyable (kernel-by-value)");
// A fully-static tensor view carries no runtime shape/stride storage: it is
// the data pointer plus a small constant, far smaller than a dynamic view
// (which would store ndim extents + ndim strides).
static_assert(sizeof(tensor_static) < sizeof(long*) + tensor_static::ndim * 2 * sizeof(long),
              "static view stores no shape/stride data");
static_assert(sizeof(tensor_static)
              < sizeof(tensor<long, long, dynamic_values<3> >),
              "static view is smaller than a dynamic one");

int main()
{
    long buf[24];
    for (long i = 0; i < 24; ++i) buf[i] = i;
    long sizes[3]   = {2, 3, 4};
    long strides[3] = {12, 4, 1};

    // ---- fully-static tensor -------------------------------------------
    auto t = make_tensor<shape_static, stride_c>(buf, sizes, strides);

    // numel folds to a compile-time constant.
    static_assert(decltype(t.numel())::value == 24, "static numel folds");
    if (t.numel() != 24) return 1;

    // operator() matches manual row-major offset.
    for (long i = 0; i < 2; ++i)
    for (long j = 0; j < 3; ++j)
    for (long k = 0; k < 4; ++k)
        if (t(i, j, k) != i*12 + j*4 + k) return 2;

    // Mixed static / runtime indices.
    if (t(csize<1>(), 2L, csize<3>()) != 1*12 + 2*4 + 3) return 3;

    // Static extents / strides read back.
    if (t.size(csize<0>())      != 2)  return 4;
    if (t.stride_at(csize<2>()) != 1)  return 5;

    // offset_at decodes a row-major linear index (contiguous -> identity).
    for (long lin = 0; lin < 24; ++lin)
        if (t.offset_at(lin) != lin) return 6;

    // Writes go through the view into the backing buffer.
    t(1, 1, 1) = 777;
    if (buf[12 + 4 + 1] != 777) return 7;

    // ---- mixed static/dynamic shape, dynamic strides -------------------
    auto m = make_tensor<shape_mixed>(buf, sizes, strides);   // stride all-dynamic
    if (m.numel() != 24) return 8;                            // 2*3*4 (dim0 runtime)
    if (m.size(csize<1>()) != 3) return 9;                    // static
    if (m.size(csize<0>()) != 2) return 10;                   // dynamic, read from sizes[]
    buf[12 + 4 + 1] = 55;
    if (m(1, 1, 1) != 55) return 11;

    // The dynamic-shape view stores exactly the dynamic extents (dim0) +
    // all three dynamic strides.
    static_assert(decltype(m)::shape_type::num_dynamic()  == 1, "mixed shape stores 1 extent");
    static_assert(decltype(m)::stride_type::num_dynamic() == 3, "dynamic stride stores 3");

    // ---- fully dynamic view --------------------------------------------
    auto d = make_tensor<dynamic_values<3> >(buf, sizes, strides);
    if (d.numel() != 24) return 12;
    if (d(0, 2, 3) != 2*4 + 3) return 13;

    return 0;
}
