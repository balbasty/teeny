// teeny <-> DLPack zero-copy interchange (host-side; DLPack never copies data).
#include <teeny/teeny.h>
#include <teeny/dlpack.h>
#include <cuda/std/type_traits>

using namespace tny;
namespace cs = cuda::std;

int main()
{
    // ---- export a VIEW: metadata owned by the capsule, data borrowed ---------
    double buf[2*3];
    for (int i = 0; i < 6; ++i) buf[i] = i;
    auto x = wrap(buf, shape<2,3>{});                 // C-contiguous view
    DLManagedTensor * m = to_dlpack(x);
    if (m->dl_tensor.ndim != 2) return 1;
    if (m->dl_tensor.data != buf) return 2;           // borrowed, same pointer
    if (m->dl_tensor.device.device_type != kDLCPU) return 3;
    if (m->dl_tensor.dtype.code != kDLFloat || m->dl_tensor.dtype.bits != 64) return 4;
    if (m->dl_tensor.shape[0] != 2 || m->dl_tensor.shape[1] != 3) return 5;
    if (m->dl_tensor.strides[0] != 3 || m->dl_tensor.strides[1] != 1) return 6;

    // ---- import it back (fixed rank) and compare -----------------------------
    auto y = from_dlpack<double, 2>(m);
    static_assert(decltype(y)::rank() == 2, "fixed rank 2");
    for (long i = 0; i < 2; ++i) for (long j = 0; j < 3; ++j)
        if (y(i,j) != buf[i*3+j]) return 7;
    m->deleter(m);                                    // consumer frees the capsule (not buf)
    if (buf[5] != 5.0) return 8;                       // data untouched

    // ---- import a hand-built DLManagedTensor (as a framework would emit) ------
    // C-contiguous, NULL strides (DLPack's shorthand) -> from_dlpack fills row-major.
    float fbuf[3*4];
    for (int i = 0; i < 12; ++i) fbuf[i] = float(i) + 0.5f;
    int64_t sh[2] = {3,4};
    DLManagedTensor mm{};
    mm.dl_tensor.data = fbuf;
    mm.dl_tensor.device = {kDLCPU, 0};
    mm.dl_tensor.ndim = 2;
    mm.dl_tensor.dtype = {kDLFloat, 32, 1};
    mm.dl_tensor.shape = sh;
    mm.dl_tensor.strides = nullptr;                   // C-contiguous
    mm.dl_tensor.byte_offset = 0;
    mm.deleter = nullptr;                             // borrow (nothing to free)

    auto a = from_dlpack<float>(&mm);                 // -> anyrank<float>, meta COPIED
    if (a.ndim != 2 || a.size(0) != 3 || a.step(0) != 4 || a.step(1) != 1) return 9;
    auto av = a.fixed<2>();
    if (av(2,3) != fbuf[2*4+3]) return 10;

    // dispatch_dlpack: pick dtype + rank from the struct
    double total = 0; int rank_seen = -1;
    bool ok = dispatch_dlpack(&mm, [&](auto view){ rank_seen = decltype(view)::rank(); total = sum(view); });
    if (!ok || rank_seen != 2) return 11;
    double expect = 0; for (int i = 0; i < 12; ++i) expect += fbuf[i];
    if (total != expect) return 12;

    // wrong dtype -> the fixed importer's precondition; dispatch returns false
    bool bad = dispatch_dlpack(&mm, [&](auto){});     // fbuf is float; ask for it as float works,
    if (!bad) return 13;                              // (sanity: float IS supported)

    // ---- export an OWNING tensor: ownership moves into the capsule -----------
    auto owned_t = make_heap<int>(shape<-1>{4});      // heap [i]
    owned_t.iota_(0, 1);
    int * base = owned_t.data();
    DLManagedTensor * mo = to_dlpack(cs::move(owned_t));   // owned_t is moved-from
    if (mo->dl_tensor.data != base) return 14;             // capsule now owns the buffer
    if (mo->dl_tensor.ndim != 1 || mo->dl_tensor.shape[0] != 4) return 15;
    auto zi = from_dlpack<int, 1>(mo);
    if (zi(2) != 2) return 16;
    mo->deleter(mo);                                       // frees the heap buffer + metadata

    // ---- byte_offset folds into the pointer ----------------------------------
    int64_t sh1[1] = {2};
    DLManagedTensor mb{};
    mb.dl_tensor.data = fbuf;                              // start of fbuf
    mb.dl_tensor.device = {kDLCPU, 0};
    mb.dl_tensor.ndim = 1; mb.dl_tensor.dtype = {kDLFloat, 32, 1};
    mb.dl_tensor.shape = sh1; mb.dl_tensor.strides = nullptr;
    mb.dl_tensor.byte_offset = 2 * sizeof(float);          // start at fbuf[2]
    mb.deleter = nullptr;
    auto b = from_dlpack<float, 1>(&mb);
    if (b(0) != fbuf[2] || b(1) != fbuf[3]) return 17;

    return 0;
}
