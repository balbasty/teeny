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

    // ---- rank-0 (scalar) export compiles + round-trips -----------------------
    double s = 7.0;
    auto sc = wrap(&s, shape<>{});                    // a rank-0 view
    DLManagedTensor * ms = to_dlpack(sc);
    if (ms->dl_tensor.ndim != 0 || ms->dl_tensor.data != &s) return 18;
    auto s0 = from_dlpack<double, 0>(ms);
    if (double(s0) != 7.0) return 19;                 // rank-0 <-> scalar
    ms->deleter(ms);

    // ---- #38: memory space at the rank-erased boundary -----------------------
    // A host capsule imports as host views (default), a kDLCUDA capsule imports
    // as gpu_view-tagged views when asked for storage::gpu_view. We only inspect
    // METADATA (shape/stride live in the carrier / on the host) — never deref the
    // views here, so `dbuf` needn't be real device memory for a type/label test.
    {
        // default host import -> storage::view everywhere.
        static_assert(decltype(from_dlpack<float>(&mm))::space == storage::view, "host anyrank space");
        static_assert(decltype(from_dlpack<float, 2>(&mm))::ownership == storage::view, "host fixed -> view");

        // a device-tagged capsule (kDLCUDA); data pointer is host memory here but
        // we never dereference it — only the tag/rank is under test.
        long dshape[3] = {2, 3, 4}, dstride[3] = {12, 4, 1};
        DLManagedTensor md{};
        md.dl_tensor.data = buf;                       // stand-in pointer (not dereferenced)
        md.dl_tensor.device = {kDLCUDA, 0};
        md.dl_tensor.ndim = 3;
        md.dl_tensor.dtype = {kDLFloat, 32, 1};
        md.dl_tensor.shape = dshape; md.dl_tensor.strides = dstride;

        auto g = from_dlpack<float, storage::gpu_view>(&md);          // device anyrank
        static_assert(decltype(g)::space == storage::gpu_view, "device anyrank tagged gpu_view");
        static_assert(decltype(g)::is_device, "device anyrank is_device");
        if (g.ndim != 3 || g.size(1) != 3) return 20;             // metadata read (host), no deref
        static_assert(decltype(g.fixed<3>())::ownership == storage::gpu_view, "device fixed -> gpu_view");
        static_assert(decltype(g.peel_front_at<-2>(0))::ownership == storage::gpu_view, "device peel -> gpu_view");
        if (g.size_front<-2>() != 2) return 21;                    // batch count from host metadata

        auto gf = from_dlpack<float, 3, storage::gpu_view>(&md);      // device fixed-rank
        static_assert(decltype(gf)::ownership == storage::gpu_view, "device from_dlpack<T,R> -> gpu_view");
        if (gf.extent(0) != 2 || gf.extent(2) != 4) return 22;    // extents only (no deref)
    }

    // ---- #181: dispatch_dlpack_dtype PRESERVES the rank (typed anyrank) --------
    // dispatch_dlpack collapses to a fixed rank (f per dtype x total rank); this
    // sibling hands f the TYPED anyrank (rank still dynamic) so the caller drives
    // the (*batch, *spatial, C) batch idiom with peel_front<-Sr>.
    {
        float vol[2*3*4];
        for (int i = 0; i < 24; ++i) vol[i] = float(i);
        int64_t vsh[3] = {2,3,4};                          // (batch=2, spatial=3, C=4)
        DLManagedTensor mv{};
        mv.dl_tensor.data = vol; mv.dl_tensor.device = {kDLCPU, 0};
        mv.dl_tensor.ndim = 3; mv.dl_tensor.dtype = {kDLFloat, 32, 1};
        mv.dl_tensor.shape = vsh; mv.dl_tensor.strides = nullptr; mv.dl_tensor.byte_offset = 0;
        mv.deleter = nullptr;

        // NB: dtype dispatch instantiates f for EVERY supported type (only the
        // matching one runs), so f must be compile-valid for all — no static_assert
        // on a specific element type here. `space`/`rank()` hold for every type.
        int ndim_seen = -1; double s = 0; int ncells = 0;
        bool ok = dispatch_dlpack_dtype(&mv, [&](auto at) {
            static_assert(decltype(at)::space == storage::view, "typed anyrank, host-tagged (rank not collapsed)");
            ndim_seen = int(at.ndim);                       // RUNTIME rank -> still dynamic (3)
            for (auto cell : at.template peel_front<-2>()) {   // keep trailing (spatial,C); peel the 1 batch dim
                static_assert(decltype(cell)::rank() == 2, "cell keeps the trailing 2 dims");
                ++ncells; s += static_cast<double>(sum(cell));
            }
        });
        if (!ok || ndim_seen != 3 || ncells != 2) return 23;   // float ran: 2 batch cells
        double expect = 0; for (int i = 0; i < 24; ++i) expect += vol[i];
        if (s != expect) return 24;

        // the dtype really dispatches: a float64 capsule lands in the double arm.
        double dvol[6]; for (int i = 0; i < 6; ++i) dvol[i] = i + 0.5;
        int64_t dsh[2] = {2,3};
        DLManagedTensor mdd{};
        mdd.dl_tensor.data = dvol; mdd.dl_tensor.device = {kDLCPU, 0};
        mdd.dl_tensor.ndim = 2; mdd.dl_tensor.dtype = {kDLFloat, 64, 1};
        mdd.dl_tensor.shape = dsh; mdd.dl_tensor.strides = nullptr; mdd.deleter = nullptr;
        bool got_double = false;
        dispatch_dlpack_dtype(&mdd, [&](auto at) {
            got_double = cs::is_same<typename decltype(at.template fixed<2>())::element_type, double>::value;
        });
        if (!got_double) return 25;

        // unsupported dtype (lanes != 1) -> false, f never called.
        DLManagedTensor mu = mv; mu.dl_tensor.dtype = {kDLFloat, 32, 2};
        bool called = false;
        bool ok2 = dispatch_dlpack_dtype(&mu, [&](auto){ called = true; });
        if (ok2 || called) return 26;
    }

    return 0;
}
