#include <teeny/teeny.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;
using cs::extents;

// atomic_add_/atomic_sub_ are the readable aliases of add_<true>/sub_<true>.
// On the host the atomic write is a plain += (see fetch_add), so the alias must
// produce exactly the same result as the underlying <true> form, element by
// element — for a broadcasting tensor rhs, a scalar rhs, and a rank-0 at() cell.

int main()
{
    // ---- scalar rhs: atomic_add_(s) == add_<true>(s) ------------------
    {
        double b1[4] = {1, 2, 3, 4};
        double b2[4] = {1, 2, 3, 4};
        auto a1 = wrap(b1, shape<2,2>{});
        auto a2 = wrap(b2, shape<2,2>{});
        a1.atomic_add_(10.0);
        a2.add_<true>(10.0);
        for (int i = 0; i < 4; ++i) if (b1[i] != b2[i]) return 1;
        // and it actually accumulated the delta
        if (b1[0] != 11 || b1[3] != 14) return 2;

        a1.atomic_sub_(5.0);
        a2.sub_<true>(5.0);
        for (int i = 0; i < 4; ++i) if (b1[i] != b2[i]) return 3;
        if (b1[0] != 6 || b1[3] != 9) return 4;
    }

    // ---- tensor rhs (broadcasts): atomic_add_(b) == add_<true>(b) -----
    {
        double b1[6] = {0, 1, 2, 3, 4, 5};
        double b2[6] = {0, 1, 2, 3, 4, 5};
        auto a1 = wrap(b1, shape<2,3>{});
        auto a2 = wrap(b2, shape<2,3>{});
        // rhs of shape (3,) broadcasts over the leading axis
        double rb[3] = {10, 20, 30};
        auto r = wrap(rb, shape<3>{});
        a1.atomic_add_(r);
        a2.add_<true>(r);
        for (int i = 0; i < 6; ++i) if (b1[i] != b2[i]) return 5;
        if (b1[0] != 10 || b1[1] != 21 || b1[5] != 35) return 6;

        a1.atomic_sub_(r);
        a2.sub_<true>(r);
        for (int i = 0; i < 6; ++i) if (b1[i] != b2[i]) return 7;
        // back to the original values
        if (b1[0] != 0 || b1[1] != 1 || b1[5] != 5) return 8;
    }

    // ---- rank-0 at() cell: scatter-accumulate spelling ---------------
    {
        double b1[4] = {0, 0, 0, 0};
        double b2[4] = {0, 0, 0, 0};
        auto a1 = wrap(b1, shape<2,2>{});
        auto a2 = wrap(b2, shape<2,2>{});
        // several scatters into the same cell (would race on device; atomic there)
        a1.at(0, 1).atomic_add_(3.0);
        a1.at(0, 1).atomic_add_(4.0);
        a2.at(0, 1).add_<true>(3.0);
        a2.at(0, 1).add_<true>(4.0);
        for (int i = 0; i < 4; ++i) if (b1[i] != b2[i]) return 9;
        if (b1[1] != 7) return 10;

        a1.at(1, 1).atomic_sub_(2.0);
        a2.at(1, 1).sub_<true>(2.0);
        for (int i = 0; i < 4; ++i) if (b1[i] != b2[i]) return 11;
        if (b1[3] != -2) return 12;
    }

    // ---- the aliases return tensor& (chainable, like the in-place ops) ---
    {
        double b1[2] = {1, 1};
        auto a1 = wrap(b1, shape<2>{});
        static_assert(
            cs::is_same<decltype(a1.atomic_add_(1.0)), decltype(a1)&>::value,
            "atomic_add_ returns tensor&");
        static_assert(
            cs::is_same<decltype(a1.atomic_sub_(1.0)), decltype(a1)&>::value,
            "atomic_sub_ returns tensor&");
        a1.atomic_add_(1.0).atomic_add_(1.0);   // chain
        if (b1[0] != 3) return 13;
    }

    return 0;
}
