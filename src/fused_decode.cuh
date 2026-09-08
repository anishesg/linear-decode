#pragma once

#include "config.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Launch the fused single-step decode kernel.
// One thread block per head; dynamic shared memory holds state + staging.
// state: [num_heads, d_k, d_v] float32, updated in-place.
// out:   [num_heads, d_v] float16, written.
void fused_decode_step(
    float*              state,
    const __half*       q,
    const __half*       k,
    const __half*       v,
    const float*        gate,
    __half*             out,
    const RecurrentConfig& cfg,
    cudaStream_t        stream = 0);

// Returns required dynamic smem in bytes for the given config.
size_t fused_decode_smem_bytes(const RecurrentConfig& cfg);
