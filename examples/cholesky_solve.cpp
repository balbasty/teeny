// cholesky_solve — factor a small SPD matrix and solve A x = b, on a matrix
// stored with PER-DIMENSION COMPILE-TIME STRIDES (a padded row stride, the
// posdef Pointer<T,S> pattern). Work tensors are stack-owned and exactly
// sizeof their data.
//
// Shows: wrap(ptr, shape, strides<S...>{}) (folded strides), local<T,E> stack
// tensors, and layout-agnostic kernels (A and L can have different layouts).
#include <teeny/teeny.h>
#include <cmath>
#include <cstdio>

using namespace tny;

// L = chol(A), lower-triangular. A read-only; A and L may differ in layout.
template <class MatA, class MatL>
static void cholesky(const MatA & A, MatL & L) {
    const long n = A.shape(0);
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

// solve L L^T x = b (forward then backward substitution).
template <class MatL, class VecB, class VecX>
static void solve(const MatL & L, const VecB & b, VecX & x) {
    const long n = L.shape(0);
    for (long i = 0; i < n; ++i)    { double t = b(i); for (long k=0;   k<i; ++k) t -= L(i,k)*x(k); x(i) = t / L(i,i); }
    for (long i = n-1; i >= 0; --i) { double t = x(i); for (long k=i+1; k<n; ++k) t -= L(k,i)*x(k); x(i) = t / L(i,i); }
}

static bool close(double a, double b) { return std::fabs(a-b) < 1e-9; }

int main() {
    // SPD A = [[4,1,1],[1,3,0],[1,0,2]] stored with a PADDED row stride of 4
    // -> compile-time strides (4,1), non-contiguous.
    double pad[12] = {0};
    double Av[3][3] = {{4,1,1},{1,3,0},{1,0,2}};
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) pad[i*4 + j] = Av[i][j];

    auto A = wrap(pad, shape<3,3>{}, strides<4,1>{});
    auto L = local<double, shape<3,3>>();
    static_assert(sizeof(L) == 9*sizeof(double), "stack matrix is exactly its data");
    cholesky(A, L);

    auto b = local<double, shape<3>>();
    auto x = local<double, shape<3>>();
    b(0)=6; b(1)=4; b(2)=3;
    solve(L, b, x);

    for (int i=0;i<3;++i) {
        double axi = 0; for (int j=0;j<3;++j) axi += Av[i][j] * x(j);
        if (!close(axi, b(i))) { std::printf("A x != b @%d\n", i); return 1; }
    }
    std::printf("cholesky_solve: OK  (x = %.4f %.4f %.4f)\n", x(0), x(1), x(2));
    return 0;
}
