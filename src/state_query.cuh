#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Warp-cooperative state-query contraction: o = S^T @ q
//
// Computes o[j] = sum_i S[i][j] * q[i] for each output column j.
//
// smem_state:  d_k * d_v float32, row-major, in shared memory (post-update)
// smem_q:      d_k float32 in shared memory (converted from float16 inputs)
// smem_out:    d_v float32 output buffer in shared memory (written by this function)
// d_k, d_v:    dimensions
//
// Warp assignment: warp w handles output columns w, w+num_warps, ...
// Within a warp, lane l accumulates the partial sum for a single output column
// by iterating over all d_k rows. No intra-warp reduction is needed because
// each warp owns its output columns exclusively: each lane accumulates
// independently and writes to smem_out[col].
//
// __syncthreads() is NOT called here; the caller must synchronize before
// reading smem_out if other warps need those values.

__device__ __forceinline__ void state_query_contract(
    const float* __restrict__ smem_state,
    const float* __restrict__ smem_q,
    float*       __restrict__ smem_out,
    int d_k, int d_v)
{
    const int lane      = threadIdx.x & 31;
    const int warp_id   = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    // Each warp handles a strided subset of output columns.
    // Within the warp, each lane independently accumulates dot(S[:,j], q).
    for (int col = warp_id * 32 + lane; col < d_v; col += num_warps * 32) {
        float acc = 0.0f;
        for (int row = 0; row < d_k; row++) {
            acc += smem_state[row * d_v + col] * smem_q[row];
        }
        smem_out[col] = acc;
    }
}

// Variant where q is provided as float16 and converted inline.
// smem_q_f16: d_k float16 elements in shared memory; converted to float32 on the fly.
__device__ __forceinline__ void state_query_contract_f16(
    const float*  __restrict__ smem_state,
    const __half* __restrict__ smem_q_f16,
    float*        __restrict__ smem_out,
    int d_k, int d_v)
{
    const int lane      = threadIdx.x & 31;
    const int warp_id   = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    for (int col = warp_id * 32 + lane; col < d_v; col += num_warps * 32) {
        float acc = 0.0f;
        for (int row = 0; row < d_k; row++) {
            acc += smem_state[row * d_v + col] * __half2float(smem_q_f16[row]);
        }
        smem_out[col] = acc;
    }
}
