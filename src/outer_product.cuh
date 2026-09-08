#pragma once

#include "config.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Gated rank-1 outer product update on a shared-memory-resident state matrix.
//
// S[i][j] = decay(i,j) * S[i][j] + k[i] * v[j]
//
// smem_state: pointer to d_k * d_v float32 elements in shared memory (row-major)
// smem_k:     pointer to d_k float32 elements in shared memory
// smem_v:     pointer to d_v float32 elements in shared memory
// smem_gate:  pointer to d_k * d_v float32 gate logits (GLA only; unused otherwise)
// decay_val:  scalar decay for DecayType::fixed / ::retnet
//
// Called from within a __global__ kernel; blockDim.x must be a multiple of 32.
// Warps are distributed across rows: warp w handles rows w, w+num_warps, ...
// __syncthreads() is called at the end to ensure all writes are visible.

template <DecayType DT>
__device__ __forceinline__ void outer_product_update(
    float* __restrict__ smem_state,
    const float* __restrict__ smem_k,
    const float* __restrict__ smem_v,
    const float* __restrict__ smem_gate,
    int d_k, int d_v,
    float decay_val)
{
    const int lane      = threadIdx.x & 31;
    const int warp_id   = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    // Each warp iterates over rows assigned to it.
    for (int row = warp_id; row < d_k; row += num_warps) {
        // All lanes in the warp load the same k[row] value.
        float k_val = smem_k[row];
        // Broadcast via shfl (already uniform but keeps register pressure low).
        k_val = __shfl_sync(0xffffffff, k_val, 0);

        // Each lane covers d_v / 32 columns in a strided pattern.
        for (int col = lane; col < d_v; col += 32) {
            float v_val = smem_v[col];
            float s     = smem_state[row * d_v + col];

            float decay;
            if constexpr (DT == DecayType::none) {
                decay = 1.0f;
            } else if constexpr (DT == DecayType::fixed || DT == DecayType::retnet) {
                decay = decay_val;
            } else { // gla
                float logit = smem_gate[row * d_v + col];
                decay = 1.0f / (1.0f + expf(-logit));
            }

            smem_state[row * d_v + col] = decay * s + k_val * v_val;
        }
    }

    __syncthreads();
}
