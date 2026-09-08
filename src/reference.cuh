#pragma once

#include "config.cuh"
#include <cuda_runtime.h>

// Launch the sequential reference decode step.
// One kernel launch covers all num_heads heads (one block per head).
// state: [num_heads, d_k, d_v] float32, updated in-place.
// out:   [num_heads, d_v] float16, written.
void reference_decode_step(
    float*              state,
    const __half*       q,
    const __half*       k,
    const __half*       v,
    const float*        gate,
    __half*             out,
    const RecurrentConfig& cfg,
    cudaStream_t        stream = 0);
