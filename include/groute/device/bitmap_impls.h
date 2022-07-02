// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef SEP_GRAPH_BITMAP_IMPLS_H
#define SEP_GRAPH_BITMAP_IMPLS_H

#include <groute/device/compressed_bitmap.cuh>
#include <groute/device/array_bitmap.h>

#ifdef ARRAY_BITMAP
    typedef sepgraph::ArrayBitmap Bitmap;
    typedef sepgraph::dev::ArrayBitmap BitmapDeviceObject;
 #else
     typedef sepgraph::CompressedBitmap Bitmap;
     typedef sepgraph::dev::CompressedBitmap BitmapDeviceObject;
 #endif

#endif //SEP_GRAPH_BITMAP_IMPLS_H
