// Robustness harness for nd-peel + recast (the (*batch, *spatial) idiom):
//   (a) STATIC INFERENCE — peeling the batch axes off shape<-1,-1,-1,M,N> yields a
//       cell whose trailing extents AND strides stay STATIC <M,N> (staticity is not
//       lost just because the batch dims are dynamic).
//   (b) NO SPURIOUS RE-ADDRESSING — writing into a peeled (and recast) cell at (i,j)
//       hits exactly the same element as writing t(*batch, i, j) on the source,
//       for contiguous AND permuted (non-row-major) sources.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

template <class Tn, long Ax, long V> constexpr bool ext_is = (decltype(cs::declval<Tn>().extent(Int<Ax>()))::value == V);
template <class Tn, long Ax, long V> constexpr bool str_is = (decltype(cs::declval<Tn>().stride(Int<Ax>()))::value == V);

int main() {
    // ================= (a) static inference through peel =================

    // fully static source -> static <5,6> cell, strides fold (6,1)
    auto s = local<double, shape<2,3,4,5,6>>{};
    using CellS = decltype(peel_front_at<3>(s, 0));
    static_assert(CellS::rank() == 2, "static src: cell rank 2");
    static_assert(ext_is<CellS,0,5> && ext_is<CellS,1,6>, "static src: cell extents <5,6>");
    static_assert(str_is<CellS,0,6> && str_is<CellS,1,1>, "static src: cell strides fold (6,1)");

    // DYNAMIC batch, STATIC trailing: shape<-1,-1,-1,5,6> -> cell STILL static <5,6>
    double buf[2*3*4*5*6]; for (int i = 0; i < 2*3*4*5*6; ++i) buf[i] = i;
    auto d = wrap(buf, shape<-1,-1,-1,5,6>{2,3,4});
    using CellD = decltype(peel_front_at<3>(d, 0));
    static_assert(CellD::rank() == 2, "dyn-batch: cell rank 2");
    static_assert(ext_is<CellD,0,5> && ext_is<CellD,1,6>, "dyn-batch: cell extents STILL static <5,6>");
    static_assert(str_is<CellD,0,6> && str_is<CellD,1,1>, "dyn-batch: cell strides STILL fold (6,1)");

    // the subset peel<axes...> form preserves it too (peel the 3 leading axes)
    using CellP = decltype(peel_at<0,1,2>(d, 0));
    static_assert(ext_is<CellP,0,5> && ext_is<CellP,1,6>, "peel<subset>: extents static <5,6>");
    static_assert(str_is<CellP,1,1>, "peel<subset>: unit stride folds");

    // the range object peel_front<N>(t) yields the same static cell type
    using CellR = decltype(*peel_front<3>(d).begin());
    static_assert(ext_is<CellR,0,5> && ext_is<CellR,1,6>, "peel_front range: extents static <5,6>");

    // ================= (b) no spurious re-addressing =====================
    // a rank-4 source (2 batch dims + M,N), so peel_front<2> -> rank-2 <M,N> cells
    const long B0 = 2, B1 = 3, M = 5, N = 6;
    double buf2[B0*B1*M*N];
    auto d2 = wrap(buf2, shape<-1,-1,5,6>{B0,B1});

    // ---- contiguous source: peel_front write == full-index write ----------
    for (int i = 0; i < B0*B1*M*N; ++i) buf2[i] = -1;
    long k = 0;
    for (auto c : peel_front<2>(d2)) {                // c is (M,N), one per (b0,b1)
        for (long i = 0; i < M; ++i) for (long j = 0; j < N; ++j) c(i,j) = 1000*k + 10*i + j;
        ++k;
    }
    k = 0;
    for (long b0 = 0; b0 < B0; ++b0) for (long b1 = 0; b1 < B1; ++b1) {
        for (long i = 0; i < M; ++i) for (long j = 0; j < N; ++j)
            if (d2(b0,b1,i,j) != 1000*k + 10*i + j) return 1;      // cell(i,j) must be t(*batch,i,j)
        ++k;
    }

    // ---- peel_front_at + RECAST to static <M,N> then write ----------------
    for (int i = 0; i < B0*B1*M*N; ++i) buf2[i] = -1;
    for (long q = 0; q < B0*B1; ++q) {
        auto c = peel_front_at<2>(d2, q).recast<shape<5,6>>();    // force a fully static cell
        static_assert(decltype(c)::rank() == 2 && ext_is<decltype(c),0,5>, "recast cell static");
        for (long i = 0; i < M; ++i) for (long j = 0; j < N; ++j) c(i,j) = 2000 + 100*q + 10*i + j;
    }
    for (long b0 = 0, q = 0; b0 < B0; ++b0) for (long b1 = 0; b1 < B1; ++b1, ++q)
        for (long i = 0; i < M; ++i) for (long j = 0; j < N; ++j)
            if (d2(b0,b1,i,j) != 2000 + 100*q + 10*i + j) return 2;

    // ---- PERMUTED (non-row-major) source: still addresses correctly -------
    // swap the two batch axes: strides are no longer descending, so a naive
    // row-major recompute would mis-address — this pins that it does not.
    for (int i = 0; i < B0*B1*M*N; ++i) buf2[i] = -1;
    auto dp = d2.permute<1,0,2,3>();                  // (B1,B0,M,N)
    for (long q = 0; q < B1*B0; ++q) {
        auto c = peel_front_at<2>(dp, q).recast<shape<5,6>>();
        for (long i = 0; i < M; ++i) for (long j = 0; j < N; ++j) c(i,j) = 3000 + 100*q + 10*i + j;
    }
    for (long a0 = 0, q = 0; a0 < B1; ++a0) for (long a1 = 0; a1 < B0; ++a1, ++q)
        for (long i = 0; i < M; ++i) for (long j = 0; j < N; ++j)
            if (dp(a0,a1,i,j) != 3000 + 100*q + 10*i + j) return 3;   // matches the permuted view's own index
    // and cross-check against the ORIGINAL layout: dp(a0,a1,..) == d2(a1,a0,..)
    for (long a0 = 0; a0 < B1; ++a0) for (long a1 = 0; a1 < B0; ++a1)
        for (long i = 0; i < M; ++i) for (long j = 0; j < N; ++j)
            if (dp(a0,a1,i,j) != d2(a1,a0,i,j)) return 4;

    return 0;
}
