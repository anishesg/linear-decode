#pragma once

#include "config.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Launch the multi-head persistent decode step.
// Computes heads_per_group = floor(smem_limit / (d_k * d_v * 4)) at runtime.
// Grid = ceil(num_heads / heads_per_group); each block processes one group.
// state: [num_heads, d_k, d_v] float32, updated in-place.
// out:   [num_heads, d_v] float16, written.
void multi_head_decode_step(
    float*              state,
    const __half*       q,
    const __half*       k,
    const __half*       v,
    const float*        gate,
    __half*             out,
    const RecurrentConfig& cfg,
    cudaStream_t        stream = 0);

// Returns heads_per_group for the given config and device.
int compute_heads_per_group(const RecurrentConfig& cfg, int device = 0);
