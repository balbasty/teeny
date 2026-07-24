// Axis reductions: sum/prod/max/min/mean<Axes...> -> a lower-rank tensor.
// Static shape -> stack result; dynamic -> heap. Reduced axes are removed.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main() {
    // (2,3) matrix, row-major values 0..5
    auto m = local<double, shape<2,3>>(); m.iota_(0.0, 1.0);   // [[0,1,2],[3,4,5]]

    // sum over axis 0 -> shape (3): column sums [3,5,7]
    auto c = sum<0>(m);
    static_assert(decltype(c)::rank() == 1, "axis 0 removed");
    static_assert(decltype(c)::ownership == storage::stack, "static -> stack");
    static_assert(decltype(c.extent(Int<0>()))::value == 3, "kept extent static");
    if (c(0)!=3.0 || c(1)!=5.0 || c(2)!=7.0) return 1;

    // sum over axis 1 -> shape (2): row sums [3,12]
    auto r = sum<1>(m);
    if (r(0)!=3.0 || r(1)!=12.0) return 2;

    // sum over a negative axis (== last)
    auto rn = sum<-1>(m);
    if (rn(0)!=3.0 || rn(1)!=12.0) return 3;

    // mean over axis 0 -> [1.5, 2.5, 3.5]
    auto mn = mean<0>(m);
    if (mn(0)!=1.5 || mn(1)!=2.5 || mn(2)!=3.5) return 4;

    // max / min over axis 1 -> [2,5] / [0,3]
    auto mx = max<1>(m); if (mx(0)!=2.0 || mx(1)!=5.0) return 5;
    auto mi = min<1>(m); if (mi(0)!=0.0 || mi(1)!=3.0) return 6;

    // prod over axis 1 -> [0*1*2, 3*4*5] = [0,60]
    auto pr = prod<1>(m); if (pr(0)!=0.0 || pr(1)!=60.0) return 7;

    // reduce multiple axes of a 3-D tensor -> scalar-shaped rank-1... here keep one
    auto t = local<double, shape<2,2,2>>(); t.iota_(1.0, 1.0);   // 1..8
    auto s02 = sum<0,2>(t);    // reduce axes 0 and 2 -> shape (2)
    static_assert(decltype(s02)::rank() == 1, "two axes removed");
    // axis1==0: t(:, 0, :) = 1,2,5,6 -> 14 ; axis1==1: 3,4,7,8 -> 22
    if (s02(0)!=14.0 || s02(1)!=22.0) return 8;

    // reducing the DYNAMIC axis leaves a fully-static result -> stack
    auto d = owned<double, shape<-1,3>>(shape<-1,3>{2,3}); d.iota_(0.0,1.0);   // (2,3), 0..5
    auto dc = sum<0>(d);                       // axis0 (dynamic) removed -> shape<3> static
    static_assert(decltype(dc)::ownership == storage::stack, "reduced the dynamic axis -> static");
    if (dc(0)!=3.0 || dc(2)!=7.0) return 9;

    // reducing a STATIC axis and keeping a dynamic one -> heap result
    auto dr = sum<1>(d);                        // axis1 (static 3) removed -> shape<-1> dynamic
    static_assert(decltype(dr)::ownership == storage::heap, "dynamic result -> heap");
    if (dr(0)!=3.0 || dr(1)!=12.0) return 10;

    return 0;
}
