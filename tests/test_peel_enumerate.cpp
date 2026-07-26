// #213: the peel / peel_front range-for can ALSO yield the peeled multi-index —
// `it.index(d)` / `it.index()` on the iterator, and `enumerate()` for the ergonomic
// `for (auto [m, cell] : peel(...).enumerate())`. The default `for (auto cell : ...)`
// keeps the cell LEAN (no coordinate words); enumerate is opt-in.
#include <teeny/teeny.h>
#include <cuda/std/type_traits>
using namespace tny;
namespace cs = cuda::std;

int main() {
    auto t = zeros<double>(shape<3,4>{}); t.iota_(0, 1);      // row-major 0..11

    // ---- enumerate over BOTH axes: rank-0 cells, m = (i,j), value == flat(m) ----
    long seen = 0;
    for (auto [m, cell] : peel(t, axis<0,1>{}).enumerate()) {
        if (double(cell) != double(m[0]*4 + m[1])) return 1;
        ++seen;
    }
    if (seen != 12) return 2;

    // ---- enumerate over axis 0: rank-1 row cells, m = (i) ----
    long rows = 0;
    for (auto [m, row] : peel(t, axis<0>{}).enumerate()) {
        if (row(0) != double(m[0]*4)) return 3;               // first element of row m[0]
        if (row(3) != double(m[0]*4 + 3)) return 4;
        ++rows;
    }
    if (rows != 3) return 5;

    // ---- iterator.index(d) / index() on the bare range (explicit loop) ----
    auto rng = peel(t, axis<0,1>{});
    auto it = rng.begin();
    if (it.index(0) != 0 || it.index(1) != 0) return 6;
    ++it;                                                     // row-major: last peeled axis fastest
    if (it.index(0) != 0 || it.index(1) != 1) return 7;
    auto m0 = it.index();                                     // whole tuple
    if (m0[0] != 0 || m0[1] != 1) return 8;

    // ---- enumerate composes with subrange (chunked/threaded sweep) ----
    long cnt = 0, acc = 0;
    for (auto [m, cell] : peel(t, axis<0,1>{}).enumerate().subrange(4, 8)) {
        if (double(cell) != double(m[0]*4 + m[1])) return 9;
        ++cnt; acc += m[0]*4 + m[1];
    }
    if (cnt != 4 || acc != 4+5+6+7) return 10;

    // ---- peel_front<N> enumerate (batch multi-index) ----
    auto b = zeros<double>(shape<2,3,4>{}); b.iota_(0, 1);    // (2,3) batch, len-4 rows
    long bc = 0;
    for (auto [m, row] : peel_front<2>(b).enumerate()) {      // peel the first 2 (batch) dims
        // row is the length-4 innermost; its first element is at flat (m[0]*3 + m[1])*4
        if (row(0) != double((m[0]*3 + m[1]) * 4)) return 11;
        ++bc;
    }
    if (bc != 6) return 12;

    // ---- the DEFAULT cell is unchanged (lean): enumerate adds nothing to `*it` ----
    static_assert(cs::is_same<decltype(*peel(t, axis<0>{}).begin()),
                              decltype(peel(t, axis<0>{}))::Cell>::value,
                  "default *it still yields the bare lean cell");

    return 0;
}
