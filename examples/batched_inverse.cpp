// batched_inverse — the flagship "efficient kernel" idiom: a runtime tensor whose
// INNER dims are static (a C×C matrix) and OUTER dims are dynamic (the batch),
// one matrix per thread, on CPU threads AND CUDA.
//
// Why it's efficient: the matrix is `shape<dynamic_extent, C, C>` — the batch
// extent is a runtime value, but C is baked into the type, so the C×C inversion
// (register work arrays, unrolled loops, folded strides) compiles to straight-
// line code with no shape/stride loads. `dispatch_value<2,3,4>` turns a runtime
// C into that static type once, at the launch boundary.
//
// Proof it folds: for a static `shape<3,3>` view, `A(0,0)+A(1,1)+A(2,2)` compiles
// (clang -O2) to exactly:
//     movsd (%rdi), %xmm0 ; addsd 32(%rdi), %xmm0 ; addsd 64(%rdi), %xmm0 ; ret
// — constant byte offsets 0/32/64, no stride loads. The abstraction is free.
#include <teeny/teeny.h>
#include <thread>
#include <vector>
#include <cmath>
#include <cstdio>

using namespace tny;
namespace cs = cuda::std;

// ---- the kernel core: invert one static C×C matrix (Gauss-Jordan) ----------
// A and out are (C,C) views of ANY stride; C is a compile-time constant, so the
// whole thing unrolls and the strides fold to immediates.
template <class MatA, class MatO>
_TNY_API void invert(const MatA & A, MatO & out) {
    constexpr int C = static_cast<int>(decltype(A.extent(Int<0>()))::value);   // static extent, friendly
    double m[C][C], inv[C][C];
    for (int i=0;i<C;++i) for (int j=0;j<C;++j) { m[i][j]=A(i,j); inv[i][j]=(i==j)?1.0:0.0; }
    for (int col=0; col<C; ++col) {
        double p = 1.0 / m[col][col];                       // (demo: assume nonsingular)
        for (int j=0;j<C;++j) { m[col][j]*=p; inv[col][j]*=p; }
        for (int row=0; row<C; ++row) if (row!=col) {
            double f = m[row][col];
            for (int j=0;j<C;++j) { m[row][j]-=f*m[col][j]; inv[row][j]-=f*inv[col][j]; }
        }
    }
    for (int i=0;i<C;++i) for (int j=0;j<C;++j) out(i,j)=inv[i][j];
}

// the i-th batch element: peel the (single) leading batch axis -> a (C,C) view.
// `_TNY_API` so it is callable from a CUDA kernel unchanged.
template <class In, class Out>
_TNY_API void invert_at(const In & in, Out & out, long i) {
    auto A  = peel_front_at<1>(in,  i);                    // (C,C) view into the batch
    auto Oi = peel_front_at<1>(out, i);
    invert(A, Oi);
}

// ---- CPU driver: split the batch across std::threads -----------------------
template <class In, class Out>
static void invert_batch_cpu(const In & in, Out & out) {
    const long n = in.extent(0);
    const unsigned nt = 4;
    std::vector<std::thread> pool;
    for (unsigned t=0; t<nt; ++t)
        pool.emplace_back([&,t]{ for (long i=t; i<n; i+=nt) invert_at(in, out, i); });
    for (auto & th : pool) th.join();
}

// ---- CUDA driver: one thread per matrix, grid-stride (compiled by nvcc) -----
#ifdef __CUDACC__
template <class In, class Out>
__global__ void invert_batch_kernel(In in, Out out) {          // views are trivially copyable
    for (long i = blockIdx.x*blockDim.x + threadIdx.x; i < in.extent(0); i += gridDim.x*blockDim.x)
        invert_at(in, out, i);
}
template <class In, class Out>
static void invert_batch_cuda(const In & in, Out & out) {
    int block = 256, grid = (int)((in.extent(0)+block-1)/block);
    invert_batch_kernel<<<grid, block>>>(in, out);             // in/out are device views
}
#endif

// ---- runtime spatial size C -> a static type, at the launch boundary -------
// `data` is (n, C, C) row-major with C known only at run time; dispatch to the
// specialised kernel for C in {2,3,4}.
static void invert_batch(double * data, double * out, long n, int C) {
    dispatch_value<2,3,4>(C, [&](auto CC) {
        constexpr long c = CC.value;
        auto in  = view(data, shape<dynamic_extent, c, c>{n});   // static inner, dynamic outer
        auto ov  = view(out,  shape<dynamic_extent, c, c>{n});
        invert_batch_cpu(in, ov);                                // (or invert_batch_cuda on device)
    });
}

// ---- validate: A · A⁻¹ == I for every batch element ------------------------
int main() {
    const long n = 500;
    const int  C = 3;                                            // known only at run time here
    std::vector<double> A(n*C*C), Ainv(n*C*C);
    unsigned s = 12345;
    auto rnd = [&]{ s = s*1103515245u + 12345u; return ((s>>9)&0xffff)/65535.0; };
    for (long b=0;b<n;++b)                                       // diagonally-dominant => invertible
        for (int i=0;i<C;++i) for (int j=0;j<C;++j)
            A[(b*C+i)*C+j] = (i==j) ? (C + rnd()) : (rnd()-0.5);

    invert_batch(A.data(), Ainv.data(), n, C);

    for (long b=0;b<n;++b)
        for (int i=0;i<C;++i) for (int j=0;j<C;++j) {
            double acc=0; for (int k=0;k<C;++k) acc += A[(b*C+i)*C+k]*Ainv[(b*C+k)*C+j];
            if (std::fabs(acc - (i==j?1.0:0.0)) > 1e-9) { std::printf("batch %ld bad\n", b); return 1; }
        }
    std::printf("batched_inverse: OK (%ld matrices %dx%d, CPU threads; CUDA path under nvcc)\n", n, C, C);
    return 0;
}
