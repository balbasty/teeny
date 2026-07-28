// wrap(mdspan): a spelling of as_tensor(mdspan) under the one factory name
// users already reach for. (#270)
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    double buf[6] = {1,2,3,4,5,6};   // row-major 2x3: [[1,2,3],[4,5,6]]
    auto vc = wrap(buf, shape<2,3>{});

    auto md = vc.mdspan();
    auto w1 = wrap(md);
    static_assert(cs::is_same<decltype(w1)::element_type, double>::value, "");
    if (w1(1,2)!=6)                              return 1;
    auto w2 = as_tensor(md);                     // as_tensor stays available, identical result
    if (w1(0,1)!=w2(0,1))                        return 2;
    auto ws = wrap(vc.permute(axis<1,0>{}).mdspan());   // works on a transposed submdspan too
    if (ws(2,1)!=6)                              return 3;

    return 0;
}
