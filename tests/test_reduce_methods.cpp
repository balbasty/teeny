// Reductions as methods (#247): m.sum()/m.mean()/m.dot(b)/m.norm()/... — thin
// forwarders to the free forms, same overload shapes (full / axis / accumulator /
// axis<...> value form / into(dest)). Each check compares the method against the
// free function it mirrors.
#include <teeny/teeny.h>
#include <cmath>
using namespace tny;

static bool close(double a, double b) { return std::fabs(a - b) < 1e-9; }

int main()
{
    auto m = local<double, shape<2,3>>();
    m(0,0)=1; m(0,1)=2; m(0,2)=3; m(1,0)=4; m(1,1)=5; m(1,2)=6;

    // ---- full reductions: method == free ------------------------------
    if (!close(m.sum(),  sum(m)))   return 1;   // 21
    if (!close(m.prod(), prod(m)))  return 2;   // 720
    if (!close(m.max(),  max(m)))   return 3;   // 6
    if (!close(m.min(),  min(m)))   return 4;   // 1
    if (!close(m.mean(), mean(m)))  return 5;   // 3.5
    if (!close(m.sum(), 21.0))      return 6;

    // leading TYPE = accumulator + result
    if (!close(m.sum<double>(), sum<double>(m))) return 7;

    // integer mean -> double (numpy rule), via the method
    auto mi = local<int, shape<4>>(); mi(0)=1; mi(1)=2; mi(2)=2; mi(3)=3;
    if (!close(mi.mean(), 2.0))     return 8;   // 8/4

    // ---- axis reductions: template form + value form ------------------
    auto col  = m.sum<0>();               // reduce axis 0 -> length 3
    if (col(0)!=5 || col(1)!=7 || col(2)!=9) return 9;
    auto colv = m.sum(axis<0>{});         // value form == m.sum<0>()
    if (colv(0)!=5 || colv(2)!=9)         return 10;
    auto row  = m.mean<1>();              // mean over axis 1 -> length 2
    if (!close(row(0), 2.0) || !close(row(1), 5.0)) return 11;
    auto colf = m.sum<float,0>();         // accumulator + axis
    if (!close(colf(1), 7.0f))            return 12;
    static_assert(decltype(col)::rank() == 1, "axis method drops one axis");

    // ---- into(dest): full -> rank-0, axis -> lower-rank ---------------
    auto cell = local<double, shape<>>();
    auto & rc = m.sum(into(cell));
    if (!close(cell.item(), 21.0))        return 13;
    if (&rc != &cell)                     return 14;   // returns dest by ref
    auto cbuf = local<double, shape<3>>();
    m.sum<0>(into(cbuf));
    if (cbuf(0)!=5 || cbuf(2)!=9)         return 15;
    cbuf.zero_(); m.sum(axis<0>{}, into(cbuf));         // value form + into
    if (cbuf(1)!=7)                       return 16;

    // ---- vector algebra methods: sqnorm / norm ------------------------
    auto v = local<double, shape<3>>(); v(0)=3; v(1)=0; v(2)=4;
    if (!close(v.sqnorm(), 25.0))         return 17;   // 9+0+16
    if (!close(v.norm(),    5.0))         return 18;   // sqrt(25)
    if (!close(v.norm(), norm(v)))        return 19;
    auto nr = m.norm<1>();                // per-row L2 norm -> length 2
    if (!close(nr(0), std::sqrt(1.0+4.0+9.0)) || !close(nr(1), std::sqrt(16.0+25.0+36.0))) return 20;

    // ---- dot as a method ---------------------------------------------
    auto w = local<double, shape<3>>(); w.fill_(1.0);
    if (!close(v.dot(w), 7.0))            return 21;   // 3+0+4
    if (!close(v.dot(w), dot(v,w)))       return 22;
    if (!close(v.dot<double>(w), 7.0))    return 23;
    v.dot(w, into(cell));                              // dot into rank-0
    if (!close(cell.item(), 7.0))         return 24;

    return 0;
}
