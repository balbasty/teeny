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

    // ---- DYNAMIC-shape receiver: the axis methods must resolve to the ------
    // HOST (heap-allocating) free overload, not the host+device one. The method
    // declarations mirror the free functions' static/dynamic split, so a
    // _TNY_API (__host__ __device__) forwarder never calls a __host__-only
    // allocator. NB reducing axis 0 of shape<-1,3> leaves a STATIC shape<3>
    // (the _TNY_API arm); reducing axis 1 leaves a DYNAMIC shape<-1> (the
    // _TNY_HOST arm) — both arms are exercised here.
    auto md = zeros<double>(shape<-1,3>{2}); md.copy_(m);
    auto dcol = md.sum<0>();                             // -> static shape<3> (device-safe arm)
    static_assert(decltype(dcol)::extents_type::rank_dynamic() == 0, "sum<0> of shape<-1,3> is static");
    if (dcol(0)!=5 || dcol(1)!=7 || dcol(2)!=9)          return 25;
    auto drow = md.sum<1>();                             // -> dynamic shape<-1> (host-only arm)
    static_assert(decltype(drow)::extents_type::rank_dynamic() == 1, "sum<1> of shape<-1,3> is dynamic");
    static_assert(decltype(drow)::ownership == storage::heap, "dynamic axis result is heap-owned");
    if (drow(0)!=6 || drow(1)!=15)                       return 26;
    auto drowv = md.sum(axis<1>{});                      // value form, dynamic result
    if (drowv(0)!=6 || drowv(1)!=15)                     return 27;
    auto drowa = md.sum<double,1>();                     // accumulator + axis, dynamic result
    if (!close(drowa(0), 6.0) || !close(drowa(1), 15.0)) return 28;
    auto dmean = md.mean<1>();
    if (!close(dmean(0), 2.0) || !close(dmean(1), 5.0))  return 29;
    auto dnorm = md.norm<1>();
    if (!close(dnorm(0), std::sqrt(14.0)) || !close(dnorm(1), std::sqrt(77.0))) return 30;
    // ...and the into() twins of the dynamic axis forms
    auto dbuf = zeros<double>(shape<-1>{2});
    md.sum<1>(into(dbuf));            if (dbuf(0)!=6 || dbuf(1)!=15)  return 31;
    dbuf.zero_();
    md.sum(axis<1>{}, into(dbuf));    if (dbuf(0)!=6 || dbuf(1)!=15)  return 32;
    dbuf.zero_();
    md.sum<double,1>(into(dbuf));     if (dbuf(0)!=6 || dbuf(1)!=15)  return 33;
    md.mean(axis<1>{}, into(dbuf));
    if (!close(dbuf(0), 2.0) || !close(dbuf(1), 5.0))                 return 34;
    md.sum(into(cell));               if (!close(cell.item(), 21.0))  return 35;   // full form still fine

    return 0;
}
