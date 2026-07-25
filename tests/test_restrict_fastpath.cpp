// #161: out-of-place elementwise ops take a contiguous linear fast path (a
// __restrict__ destination + a flat i-indexed loop) when every operand has the
// same rank/extents as the fresh result and is C-contiguous; otherwise they fall
// back to the mixed-radix broadcast decode. This test pins BOTH paths to
// hand-written references, and checks the in-place path (which must NOT take the
// fast path) still behaves.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
#include <cmath>
using namespace tny;
namespace cs = cuda::std;

static bool close(double x, double y) { double d = x - y; if (d < 0) d = -d; return d <= 1e-12 * (1.0 + (y < 0 ? -y : y)); }

int main() {
    // -------- (a) FAST PATH: contiguous, same-shape operands --------
    auto A = local<double, shape<3,4>>();
    auto B = local<double, shape<3,4>>();
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) { A(i,j) = i*4+j+1; B(i,j) = (i+j)*0.5 - 2.0; }

    // a + b (tensor+tensor, w_set store, fast path)
    auto S = A + B;
    static_assert(decltype(S)::is_static && decltype(S)::rank()==2, "static same-shape add");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (S(i,j) != A(i,j) + B(i,j)) return 1;

    // a * b
    auto P = A * B;
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (P(i,j) != A(i,j) * B(i,j)) return 2;

    // unary exp(a) (out-of-place unaryo, fast path). engine computes in compute
    // type = double, so it matches std::exp exactly.
    auto E = exp(A);
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (!close(E(i,j), std::exp(A(i,j)))) return 3;

    // scalar a * 2 (out-of-place scalo, fast path)
    auto T2 = A * 2.0;
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (T2(i,j) != A(i,j) * 2.0) return 4;

    // comparison a < b -> bool tensor (bcmp, fast path)
    auto LT = A < B;
    static_assert(cs::is_same<decltype(LT)::element_type, bool>::value, "compare -> bool");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (LT(i,j) != (A(i,j) < B(i,j))) return 5;

    // a + a : sources alias each other (a and b same pointer). The fast path does
    // NOT restrict the sources, only the (fresh) destination, so this is correct.
    auto AA = A + A;
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (AA(i,j) != 2.0 * A(i,j)) return 6;

    // DYNAMIC (heap) contiguous a+b -> heap result also takes the fast path.
    using D2 = shape<-1,-1>;
    auto Da = owned<double,D2>(D2{3,4}), Db = owned<double,D2>(D2{3,4});
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) { Da(i,j)=i+j; Db(i,j)=i*j+1; }
    auto Ds = Da + Db;
    static_assert(decltype(Ds)::ownership==storage::heap, "dynamic -> heap");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (Ds(i,j) != Da(i,j) + Db(i,j)) return 7;

    // -------- (b) FALLBACK PATH: broadcast and strided/permuted operands --------
    // broadcast col + M : different ranks-after-stretch (extent-1 axis) -> fallback.
    auto col = local<double, shape<3,1>>();
    for (long i=0;i<3;++i) col(i,0) = (i+1)*100;
    auto RB = col + A;                       // (3,1) + (3,4) -> (3,4)
    static_assert(decltype(RB)::rank()==2, "broadcast result rank 2");
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (RB(i,j) != col(i,0) + A(i,j)) return 8;

    // strided/permuted operand: At = A^T (4x3), not C-contiguous -> fallback decode.
    auto At = A.permute<1,0>();
    auto Bt = B.permute<1,0>();
    if (At.is_contiguous()) return 9;        // sanity: the permuted view is NOT contiguous
    auto RT = At + Bt;                        // both permuted, equal extents -> fallback
    for (long i=0;i<4;++i) for (long j=0;j<3;++j) if (RT(i,j) != At(i,j) + Bt(i,j)) return 10;

    // one contiguous, one permuted (b not contiguous) -> still fallback, correct.
    auto C = local<double, shape<4,3>>();
    for (long i=0;i<4;++i) for (long j=0;j<3;++j) C(i,j) = i - j;
    auto RM = C + At;                         // C contiguous, At permuted
    for (long i=0;i<4;++i) for (long j=0;j<3;++j) if (RM(i,j) != C(i,j) + At(i,j)) return 11;

    // scalar op on a permuted operand -> scalo fallback
    auto RS = At * 3.0;
    for (long i=0;i<4;++i) for (long j=0;j<3;++j) if (RS(i,j) != At(i,j) * 3.0) return 12;

    // unary on a permuted operand -> unaryo fallback
    auto RN = -At;
    for (long i=0;i<4;++i) for (long j=0;j<3;++j) if (RN(i,j) != -At(i,j)) return 13;

    // comparison on permuted operands -> bcmp fallback
    auto RC = At < Bt;
    for (long i=0;i<4;++i) for (long j=0;j<3;++j) if (RC(i,j) != (At(i,j) < Bt(i,j))) return 14;

    // -------- (c) IN-PLACE unchanged (must not take the restrict fast path) --------
    auto M = A;                               // stack copy
    M.add_(B);
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (M(i,j) != A(i,j) + B(i,j)) return 15;

    // in-place where rhs aliases lhs (M += M): the in-place path never restricts,
    // so this read-modify-write is well defined.
    auto M2 = A; M2.add_(M2);
    for (long i=0;i<3;++i) for (long j=0;j<4;++j) if (M2(i,j) != 2.0 * A(i,j)) return 16;

    // integer bitwise out-of-place (w_set, contiguous) also hits the fast path.
    auto Ia = local<int, shape<2,3>>(), Ib = local<int, shape<2,3>>();
    for (long i=0;i<2;++i) for (long j=0;j<3;++j) { Ia(i,j)=i*3+j; Ib(i,j)=(i+j)&3; }
    auto Ix = Ia ^ Ib;
    for (long i=0;i<2;++i) for (long j=0;j<3;++j) if (Ix(i,j) != (Ia(i,j) ^ Ib(i,j))) return 17;

    return 0;
}
