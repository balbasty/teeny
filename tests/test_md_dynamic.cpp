#include <teeny/md.h>
#include <cuda/std/type_traits>

using namespace tny::md;
namespace cs = cuda::std;

int main()
{
    double buf[24];
    for (long i = 0; i < 24; ++i) buf[i] = i;

    // any_tensor is a trivially-copyable rank-erased carrier.
    static_assert(cs::is_trivially_copyable<any_tensor<double,long>>::value, "trivially copyable");
    static_assert(any_tensor<double,long>::max_rank == 8, "default MaxRank");

    long shape[3]  = {2,3,4};
    long stride[3] = {12,4,1};
    auto at = any(buf, shape, stride, 3);        // runtime rank 3
    if (at.ndim != 3) return 1;

    // fixed<R>() -> a concrete rank-R md::tensor view.
    auto v = at.fixed<3>();
    static_assert(decltype(v)::rank() == 3, "fixed rank");
    if (v(1,2,3) != buf[1*12+2*4+3]) return 2;
    if (v.numel() != 24) return 3;

    // dispatch_rank: pick the fixed rank by runtime ndim, run a generic kernel.
    double total = 0;
    bool ok = dispatch_rank(at, [&](auto view){ total = sum(view); });
    if (!ok) return 4;
    double expect = 0; for (int i=0;i<24;++i) expect += buf[i];
    if (total != expect) return 5;

    // works for other ranks through the SAME call site.
    long sh2[2] = {4,6}, st2[2] = {6,1};
    auto at2 = any(buf, sh2, st2, 2);
    double t2 = 0;
    dispatch_rank(at2, [&](auto view){ t2 = sum(view); });
    if (t2 != expect) return 6;                  // same 24 elements, contiguous

    // an out-of-range ndim (here 0) is reported, not dispatched.
    auto zero = any(buf, shape, stride, 0);      // ndim 0 -> no fixed rank matches
    if (dispatch_rank(zero, [](auto){}) != false) return 7;

    return 0;
}
