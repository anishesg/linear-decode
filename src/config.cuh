#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// Decay mode for the gated rank-1 update S_t = decay * S_{t-1} + k_t outer v_t
enum class DecayType : int {
    none    = 0,  // no decay, pure accumulation
    fixed   = 1,  // fixed scalar exp(-gamma) broadcast to all elements
    gla     = 2,  // data-dependent per-element sigmoid gate from gate vector
    retnet  = 3,  // per-head fixed decay, same as fixed but named for clarity
};

struct RecurrentConfig {
    int d_k;               // key/query dimension
    int d_v;               // value/output dimension
    int num_heads;
    DecayType decay_type;
    float decay_val;       // scalar decay value when decay_type is fixed/retnet
};

// Device accessor: returns pointer to head head_idx's d_k x d_v state block.
// State buffer is laid out as [num_heads, d_k, d_v] contiguous float32.
// Each head slice is 128-byte aligned when d_k * d_v * 4 is a multiple of 128,
// which holds for d_k*d_v in {32, 64, 128, ...} * 32 element multiples.
__device__ __host__ __forceinline__
float* get_head_state(float* base_ptr, int head_idx, int d_k, int d_v) {
    return base_ptr + static_cast<ptrdiff_t>(head_idx) * d_k * d_v;
}

__device__ __host__ __forceinline__
const float* get_head_state(const float* base_ptr, int head_idx, int d_k, int d_v) {
    return base_ptr + static_cast<ptrdiff_t>(head_idx) * d_k * d_v;
}

// Per-step inputs for one decode step across all heads.
// All device pointers; dimensions: q/k/v are [num_heads, dim].
struct StepInput {
    const __half* q;          // [num_heads, d_k] float16 query vectors
    const __half* k;          // [num_heads, d_k] float16 key vectors
    const __half* v;          // [num_heads, d_v] float16 value vectors
    const float*  gate;       // [num_heads, d_k * d_v] or [num_heads] depending on decay_type
                              //   fixed/retnet: NULL (use config.decay_val)
                              //   gla: [num_heads, d_k * d_v] per-element gate logits (pre-sigmoid)
};

// Per-step outputs: updated state written back in-place via state pointer,
// output vectors written to out.
struct StepOutput {
    float*  state;    // [num_heads, d_k, d_v] float32, updated in-place
    __half* out;      // [num_heads, d_v] float16 output vectors
};
