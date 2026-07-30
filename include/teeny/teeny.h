#ifndef TNY_TEENY_H
#define TNY_TEENY_H

// teeny: an mdspan-based tensor library: one `tny::tensor<T, Extents, Layout, storage>`
// with view / stack / heap ownership, a valarray-like math layer, a custom
// per-dimension static-stride layout, and small CUDA-kernel helpers. mdspan
// (from CCCL) does the extents, layouts, and offset mapping.

#include <teeny/alias.h>
#include <teeny/half.h>
#include <teeny/storage.h>
#include <teeny/layout.h>
#include <teeny/tensor.h>
#include <teeny/math.h>
#include <teeny/iterate.h>
#include <teeny/dynamic.h>
#include <teeny/cuda.h>   // self-guarded: no-op unless the CUDA runtime is reachable

#endif // TNY_TEENY_H
