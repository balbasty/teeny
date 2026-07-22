// Operator ergonomics: compound assignment (+=,-=,*=,/=), scalar-on-the-left
// (s-a, s/a), unary minus, rank-0 <-> scalar interop (at/item/convert), and the
// Atomic flag on add_/sub_ (host path == plain accumulate).
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;

int main() {
    // ---- compound assignment: scalar rhs ------------------------------
    auto a = local<double, extents<long,2,3>>(); a.fill_(1.0);
    a += 2.0;                       // 3
    a *= 2.0;                       // 6
    a -= 1.0;                       // 5
    a /= 5.0;                       // 1
    for (long i=0;i<2;++i) for (long j=0;j<3;++j) if (a(i,j) != 1.0) return 1;

    // ---- compound assignment: tensor rhs (broadcasts) -----------------
    auto col = local<double, extents<long,2,1>>(); col(0,0)=10; col(1,0)=20;
    a += col;                       // row 0 -> 11, row 1 -> 21
    for (long j=0;j<3;++j) if (a(0,j)!=11.0 || a(1,j)!=21.0) return 2;

    // ---- scalar on the left: + * commute, - / reversed ----------------
    auto b = local<double, extents<long,3>>(); b(0)=1; b(1)=2; b(2)=4;
    auto sub = 10.0 - b;            // 9,8,6
    if (sub(0)!=9.0 || sub(1)!=8.0 || sub(2)!=6.0) return 3;
    auto dvd = 8.0 / b;             // 8,4,2
    if (dvd(0)!=8.0 || dvd(1)!=4.0 || dvd(2)!=2.0) return 4;
    auto neg = -b;                  // -1,-2,-4
    if (neg(0)!=-1.0 || neg(2)!=-4.0) return 5;

    // ---- rank-0 <-> scalar interop ------------------------------------
    auto m = local<double, extents<long,2,3>>(); m.iota_(0.0, 1.0);   // 0..5
    double v = m.at(1, 2);          // implicit rank-0 -> double
    if (v != 5.0) return 6;
    m.at(0, 0) = 42.0;              // assign through the rank-0 view
    if (m(0,0) != 42.0) return 7;
    if (m.at(1,1).item() != 4.0) return 8;

    // ---- Atomic flag on the host == plain accumulate ------------------
    auto acc = local<double, extents<long,4>>(); acc.zero_();
    acc.at(2).add_<true>(1.5);
    acc.at(2).add_<true>(2.5);      // 4.0
    acc.add_at(9.0, 3);             // add_at == at().add_<true>()
    if (acc(2) != 4.0 || acc(3) != 9.0) return 9;

    // atomic add_ over a whole view (region scatter), host path
    auto reg = local<double, extents<long,3>>(); reg.fill_(1.0);
    auto one = local<double, extents<long,3>>(); one.fill_(2.0);
    reg.add_<true>(one);            // 3,3,3
    if (reg(0)!=3.0 || reg(2)!=3.0) return 10;

    return 0;
}
