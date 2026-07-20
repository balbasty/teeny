#ifndef TNY_TEENY
#define TNY_TEENY

// teeny: an mdspan-based tensor library: one `tny::tensor<T, Extents, Layout, own>`
// with view / stack / heap ownership, a valarray-like math layer, a custom
// per-dimension static-stride layout, and small CUDA-kernel helpers. mdspan
// (from CCCL) does the extents, layouts, and offset mapping.

#include <teeny/storage.h>
#include <teeny/layout.h>
#include <teeny/tensor.h>
#include <teeny/math.h>
#include <teeny/helpers.h>
#include <teeny/iterate.h>
#include <teeny/dynamic.h>

#endif // TNY_TEENY
