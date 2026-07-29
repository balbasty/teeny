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
    auto a = local<double, shape<2,3>>(); a.fill_(1.0);
    a += 2.0;                       // 3
    a *= 2.0;                       // 6
    a -= 1.0;                       // 5
    a /= 5.0;                       // 1
    for (long i=0;i<2;++i) for (long j=0;j<3;++j) if (a(i,j) != 1.0) return 1;

    // ---- compound assignment: tensor rhs (broadcasts) -----------------
    auto col = local<double, shape<2,1>>(); col(0,0)=10; col(1,0)=20;
    a += col;                       // row 0 -> 11, row 1 -> 21
    for (long j=0;j<3;++j) if (a(0,j)!=11.0 || a(1,j)!=21.0) return 2;

    // ---- scalar on the left: + * commute, - / reversed ----------------
    auto b = local<double, shape<3>>(); b(0)=1; b(1)=2; b(2)=4;
    auto sub = 10.0 - b;            // 9,8,6
    if (sub(0)!=9.0 || sub(1)!=8.0 || sub(2)!=6.0) return 3;
    auto dvd = 8.0 / b;             // 8,4,2
    if (dvd(0)!=8.0 || dvd(1)!=4.0 || dvd(2)!=2.0) return 4;
    auto neg = -b;                  // -1,-2,-4
    if (neg(0)!=-1.0 || neg(2)!=-4.0) return 5;

    // ---- rank-0 <-> scalar interop ------------------------------------
    auto m = local<double, shape<2,3>>(); m.iota_(0.0, 1.0);   // 0..5
    double v = m.at(1, 2);          // implicit rank-0 -> double
    if (v != 5.0) return 6;
    m.at(0, 0) = 42.0;              // assign through the rank-0 view
    if (m(0,0) != 42.0) return 7;
    if (m.at(1,1).item() != 4.0) return 8;

    // ---- Atomic flag: single-threaded value check (real host atomicity is ----
    // proven under actual contention by test_atomic_fetch_add.cpp, #257) --------
    auto acc = local<double, shape<4>>(); acc.zero_();
    acc.at(2).add_<true>(1.5);
    acc.at(2).add_<true>(2.5);      // 4.0
    acc.at(3).add_<true>(9.0);      // scatter into one cell (atomic, host and device)
    if (acc(2) != 4.0 || acc(3) != 9.0) return 9;

    // atomic add_ over a whole view (region scatter), host path
    auto reg = local<double, shape<3>>(); reg.fill_(1.0);
    auto one = local<double, shape<3>>(); one.fill_(2.0);
    reg.add_<true>(one);            // 3,3,3
    if (reg(0)!=3.0 || reg(2)!=3.0) return 10;

    // ---- rounding / sign / clamp (in-place and free) ------------------
    auto g = local<double, shape<4>>(); g(0)=-1.7; g(1)=2.3; g(2)=0.0; g(3)=3.5;
    auto fl = floor(g); if (fl(0)!=-2.0 || fl(1)!=2.0 || fl(3)!=3.0) return 11;
    auto sg = sign(g);  if (sg(0)!=-1.0 || sg(2)!=0.0 || sg(3)!=1.0) return 12;
    auto gr = g.clone(); gr.round_(); if (gr(0)!=-2.0 || gr(1)!=2.0 || gr(3)!=4.0) return 13;
    auto gc = g.clone(); gc.clamp_(0.0, 3.0); if (gc(0)!=0.0 || gc(1)!=2.3 || gc(3)!=3.0) return 14;

    // ---- minimum / maximum (broadcast) + mean -------------------------
    auto x = local<double, shape<3>>(); x(0)=1; x(1)=5; x(2)=3;
    auto y = local<double, shape<3>>(); y(0)=4; y(1)=2; y(2)=3;
    auto mn = minimum(x,y); if (mn(0)!=1 || mn(1)!=2 || mn(2)!=3) return 15;
    auto mx = maximum(x,2.5); if (mx(0)!=2.5 || mx(1)!=5 || mx(2)!=3) return 16;
    if (mean(x) != 3.0) return 17;

    // ---- minimum_ / maximum_ (in place, #325): running min/max update ------
    auto rmin = x.clone(); rmin.minimum_(y);   // min(1,4)=1, min(5,2)=2, min(3,3)=3
    if (rmin(0)!=1 || rmin(1)!=2 || rmin(2)!=3) return 18;
    auto rmax = x.clone(); rmax.maximum_(y);   // max(1,4)=4, max(5,2)=5, max(3,3)=3
    if (rmax(0)!=4 || rmax(1)!=5 || rmax(2)!=3) return 19;
    auto rmins = x.clone(); rmins.minimum_(2.5);   // min(1,2.5)=1, min(5,2.5)=2.5, min(3,2.5)=2.5
    if (rmins(0)!=1 || rmins(1)!=2.5 || rmins(2)!=2.5) return 20;
    auto rmaxs = x.clone(); rmaxs.maximum_(2.5);   // max(1,2.5)=2.5, max(5,2.5)=5, max(3,2.5)=3
    if (rmaxs(0)!=2.5 || rmaxs(1)!=5 || rmaxs(2)!=3) return 21;
    // running-min idiom: seed a "best" cell then fold in candidates one at a time
    auto best = local<double, shape<>>(); best.fill_(1e30);
    best.minimum_(5.0); best.minimum_(2.0); best.minimum_(9.0);
    if (best.item() != 2.0) return 22;

    // ---- ++ / -- (prefix in place; postfix static -> stack copy) ------
    auto p = local<double, shape<2>>(); p.fill_(5.0);
    ++p; if (p(0)!=6.0) return 23;
    --p; --p; if (p(0)!=4.0) return 24;
    auto pre = p++;                 // postfix returns pre-value (4), p becomes 5
    if (pre(0)!=4.0 || p(0)!=5.0) return 25;

    // ---- bitwise (integer element types) ------------------------------
    auto bi = local<int, shape<2>>(); bi(0)=0xC; bi(1)=0x6;   // 1100, 0110
    auto bj = local<int, shape<2>>(); bj(0)=0xA; bj(1)=0xF;   // 1010, 1111
    auto ba = bi & bj; if (ba(0)!=0x8 || ba(1)!=0x6) return 26;
    auto bo = bi | bj; if (bo(0)!=0xE) return 27;
    auto bx = bi ^ bj; if (bx(0)!=0x6) return 28;
    auto bn = ~bi;     if (bn(0)!=~0xC) return 29;
    bi &= bj;          if (bi(0)!=0x8) return 30;
    bi |= 0x1;         if (bi(0)!=0x9) return 31;

    return 0;
}
