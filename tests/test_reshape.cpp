// Sub-task 2 of smart reshape (#129): reshape()/flatten() are now numpy
// view-when-stride-compatible, not merely "C-contiguous or bust". A reshape is a
// VIEW whenever the layout can be regrouped without a copy (split a contiguous axis,
// merge a contiguous run) — including non-C-contiguous and strided/permuted sources.
// The output is a folded strides<...> view: compile-time strides when the source is
// fully static, runtime otherwise. This test checks element identity, the folded
// output strides, the inferred (-1) dim folding, and the dynamic runtime path.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    double buf[48]; for (int i = 0; i < 48; ++i) buf[i] = i;

    // ---- static C-contiguous source: folds to compile-time strides<...> --------
    {
        auto t = local<double, shape<6,4>>{};             // (6,4), C-contiguous, strides (4,1)
        for (long i = 0; i < 6; ++i) for (long j = 0; j < 4; ++j) t(i,j) = i*4 + j;
        auto r = t.reshape<2,3,4>();                      // (6,4) -> (2,3,4)
        static_assert(decltype(r)::rank() == 3, "rank");
        static_assert(decltype(r)::extents_type::static_extent(0) == 2 &&
                      decltype(r)::extents_type::static_extent(1) == 3 &&
                      decltype(r)::extents_type::static_extent(2) == 4, "static extents");
        // result layout is a folded strides<...> with the C-contiguous strides (12,4,1)
        static_assert(_is_strides<decltype(r)::layout_type>::value, "reshape -> strides<...> view");
        static_assert(decltype(r)::layout_type::static_stride(0) == 12 &&
                      decltype(r)::layout_type::static_stride(1) == 4  &&
                      decltype(r)::layout_type::static_stride(2) == 1, "folded C strides");
        if (!r.is_contiguous()) return 1;                 // still C-order
        for (long a = 0; a < 2; ++a) for (long b = 0; b < 3; ++b) for (long c = 0; c < 4; ++c)
            if (r(a,b,c) != t(a*3 + b, c)) return 2;
        r(1,2,3) = 999.0;                                 // it's a view
        if (t(5,3) != 999.0) return 3;
    }

    // ---- static reshape<6,-1>: the inferred dim folds to a compile-time extent ---
    {
        auto t = local<double, shape<6,4>>{};
        auto r = t.reshape<6,-1>();                       // infer 4
        static_assert(decltype(r)::extents_type::static_extent(1) == 4, "inferred dim folds static");
        static_assert(_is_strides<decltype(r)::layout_type>::value, "strides<...> view");
        if (r.extent(0) != 6 || r.extent(1) != 4) return 4;
    }

    // ---- static NON-C-contiguous source that is still viewable (split inner) -----
    // (3,8) with a row GAP: strides (16,1) over a 48-wide buffer. Not C-contiguous,
    // but the inner contiguous run of 8 splits into (2,4) without a copy.
    {
        auto t = tensor<double, shape<3,8>, strides<16,1>>(buf);   // static, gapped rows
        if (t.is_contiguous()) return 5;                  // sanity: the row gap makes it non-C
        auto r = t.reshape<3,2,4>();                      // 8 -> (2,4): a contiguous split
        static_assert(_is_strides<decltype(r)::layout_type>::value, "strides<...> view");
        // derived strides: outer 16 (unchanged), inner run (4,1)
        static_assert(decltype(r)::layout_type::static_stride(0) == 16 &&
                      decltype(r)::layout_type::static_stride(1) == 4  &&
                      decltype(r)::layout_type::static_stride(2) == 1, "folded split strides");
        for (long i = 0; i < 3; ++i) for (long j = 0; j < 2; ++j) for (long k = 0; k < 4; ++k)
            if (r(i,j,k) != t(i, j*4 + k)) return 13;
    }

    // ---- dynamic source: runtime path, all-dynamic strides<...>, static extents --
    {
        auto t = wrap(buf, shape<-1,4>{6,4});             // dynamic outer, C-contiguous
        auto r = t.reshape<2,3,4>();
        static_assert(decltype(r)::extents_type::static_extent(0) == 2, "literal extents static");
        static_assert(_is_strides<decltype(r)::layout_type>::value, "strides<...> view");
        static_assert(!decltype(r)::layout_type::all_static(), "dynamic source -> runtime strides");
        if (r.extent(0) != 2 || r.extent(1) != 3 || r.extent(2) != 4) return 6;
        for (long a = 0; a < 2; ++a) for (long b = 0; b < 3; ++b) for (long c = 0; c < 4; ++c)
            if (r(a,b,c) != buf[(a*3 + b)*4 + c]) return 7;
        // flatten of a dynamic source -> dynamic 1-D
        auto f = t.flatten();
        static_assert(decltype(f)::rank() == 1, "flatten rank 1");
        if (f.extent(0) != 24 || f(23) != 23.0) return 8;
    }

    // ---- dynamic NON-contiguous but viewable (runtime split) --------------------
    {
        auto t = wrap(buf, shape<3,8>{}, {16,1});         // dynamic_strides, gapped
        if (t.is_contiguous()) return 9;
        auto r = t.reshape<3,2,4>();                      // viewable split
        static_assert(_is_strides<decltype(r)::layout_type>::value, "strides<...> view");
        for (long i = 0; i < 3; ++i) for (long j = 0; j < 2; ++j) for (long k = 0; k < 4; ++k)
            if (r(i,j,k) != buf[i*16 + j*4 + k]) return 10;
    }

    // ---- a permuted (transposed) view reshapes without a copy (identity/append) --
    {
        auto p = wrap(buf, shape<3,16>{}).permute<1,0>(); // (16,3) strides (1,16)
        auto r = p.reshape<16,3,1>();                     // append a size-1 axis: viewable
        static_assert(decltype(r)::rank() == 3, "rank 3");
        if (r.extent(0) != 16 || r.extent(1) != 3 || r.extent(2) != 1) return 11;
        for (long i = 0; i < 16; ++i) for (long j = 0; j < 3; ++j)
            if (r(i,j,0) != p(i,j)) return 12;
    }

    // ---- reshape preserves ownership KIND (view stays a view) -------------------
    {
        auto t = local<double, shape<2,6>>{};
        static_assert(storage_is_view(decltype(t.reshape<3,4>())::ownership), "reshape of stack -> a view");
    }

    return 0;
}
