#include <teeny/teeny.h>
#include <cuda/std/type_traits>
#include <cmath>
using namespace tny;
namespace cs = cuda::std;

static bool close(double a, double b){ return std::fabs(a-b) < 1e-9; }

int main() {
    auto A = local<double, extents<long,2,2>>();
    auto B = local<double, extents<long,2,2>>();
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    B(0,0)=10;B(0,1)=20;B(1,0)=30;B(1,1)=40;

    // (4) out-of-place methods, tensor rhs
    auto C = A.add(B);
    if (C(0,0)!=11 || C(1,1)!=44) return 1;
    if (A.sub(B)(0,0) != -9) return 2;
    if (A.mul(B)(1,1) != 160) return 3;

    // (4) out-of-place methods, scalar rhs
    auto S = A.mul(10.0);
    if (S(0,0)!=10 || S(1,1)!=40) return 4;
    if (A.add(1.0)(0,0) != 2) return 5;

    // (4) tensor (+) scalar and scalar (+) tensor operators
    if ((A + 100.0)(0,0) != 101) return 6;
    if ((2.0 * A)(1,1) != 8) return 7;
    if ((A - 1.0)(0,1) != 1) return 8;

    // (6) pow (method + in-place)
    auto P = A.pow(2.0);
    if (P(1,1) != 16) return 9;                 // 4^2
    auto Q = A; Q.pow_(2.0);
    if (Q(1,0) != 9) return 10;                 // 3^2

    // (6) unary out-of-place + in-place
    auto E = local<double, extents<long,3>>(); E(0)=0; E(1)=1; E(2)=2;
    if (!close(exp(E)(1), std::exp(1.0))) return 11;
    if (!close(sqrt(A)(1,1), 2.0)) return 12;   // sqrt(4)
    auto N = A; N.neg_();
    if (N(0,0) != -1) return 13;
    auto Ab = local<double, extents<long,2>>(); Ab(0)=-3; Ab(1)=5; Ab.abs_();
    if (Ab(0) != 3 || Ab(1) != 5) return 14;
    if (!close(sin(E)(2), std::sin(2.0))) return 15;

    // dynamic out-of-place (host, heap)
    using DynE = extents<long, dynamic_extent>;
    auto Da = owned<double, DynE>(DynE{3}); Da(0)=1; Da(1)=2; Da(2)=3;
    auto Dc = Da.mul(2.0);
    static_assert(decltype(Dc)::ownership == storage::heap, "dynamic -> heap");
    if (Dc(2) != 6) return 16;

    return 0;
}
