// A posdef-style kernel on the md line: Cholesky factorisation + solve of a
// small SPD system, run on a matrix stored with PER-DIMENSION COMPILE-TIME
// STRIDES (a padded row stride -- jitfields' posdef Pointer<T,S> pattern) via
// layout_static_stride, with stack-owned tensors for the factor and solution.
#include <teeny/teeny.h>
#include <cmath>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;

// L = chol(A) (lower), A read-only. A and L may have different layouts.
template <class MatA, class MatL>
static void cholesky(const MatA & A, MatL & L) {
    const long n = A.extent(0);
    for (long j = 0; j < n; ++j) {
        double s = A(j,j);
        for (long k = 0; k < j; ++k) s -= L(j,k) * L(j,k);
        L(j,j) = std::sqrt(s);
        for (long i = j+1; i < n; ++i) {
            double t = A(i,j);
            for (long k = 0; k < j; ++k) t -= L(i,k) * L(j,k);
            L(i,j) = t / L(j,j);
        }
    }
}

// solve L L^T x = b   (forward then backward substitution)
template <class MatL, class VecB, class VecX>
static void solve(const MatL & L, const VecB & b, VecX & x) {
    const long n = L.extent(0);
    for (long i = 0; i < n; ++i)   { double t = b(i); for (long k = 0;   k < i; ++k) t -= L(i,k) * x(k); x(i) = t / L(i,i); }
    for (long i = n-1; i >= 0; --i){ double t = x(i); for (long k = i+1; k < n; ++k) t -= L(k,i) * x(k); x(i) = t / L(i,i); }
}

static bool close(double a, double b) { return std::fabs(a - b) < 1e-9; }

int main()
{
    // SPD A = [[4,1,1],[1,3,0],[1,0,2]], stored with a PADDED row stride of 4
    // (so row i starts at 4*i, not 3*i) -> static strides (4,1), non-contiguous.
    double pad[12] = {0};
    double Avals[3][3] = {{4,1,1},{1,3,0},{1,0,2}};
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) pad[i*4 + j] = Avals[i][j];

    auto A = wrap(pad, shape<3,3>{}, strides<4,1>{});   // compile-time strides (4,1)
    static_assert(decltype(A)::rank() == 2, "matrix");

    // factor + solve, with stack-owned L and x (fully static shape)
    auto L = local<double, shape<3,3>>();            // 9 doubles, no padding
    static_assert(sizeof(L) == 9*sizeof(double), "stack matrix is exactly its data");
    cholesky(A, L);

    auto b = local<double, shape<3>>();
    auto x = local<double, shape<3>>();
    b(0)=6; b(1)=4; b(2)=3;
    solve(L, b, x);

    // verify A x == b
    for (int i=0;i<3;++i) {
        double axi = 0;
        for (int j=0;j<3;++j) axi += Avals[i][j] * x(j);
        if (!close(axi, b(i))) return 1;
    }

    // sanity: L is lower-triangular and L L^T == A
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        double llt = 0;
        for (int k=0;k<3;++k) llt += L(i,k) * L(j,k);
        if (!close(llt, Avals[i][j])) return 2;
        if (j > i && !close(L(i,j), 0.0)) return 3;         // upper part is zero
    }

    return 0;
}
