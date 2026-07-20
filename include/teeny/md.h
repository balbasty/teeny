#ifndef TNY_MD
#define TNY_MD

// An mdspan-based tensor library: one `tny::md::tensor<T, Extents, Layout, own>`
// with view / stack / heap ownership, a valarray-like math layer, a custom
// per-dimension static-stride layout, and small CUDA-kernel helpers. mdspan
// (from CCCL) does the extents, layouts, and offset mapping.

#include <teeny/md/storage.h>
#include <teeny/md/layout.h>
#include <teeny/md/tensor.h>
#include <teeny/md/math.h>
#include <teeny/md/helpers.h>

#endif // TNY_MD
