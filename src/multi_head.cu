#include "multi_head.cuh"
#include "outer_product.cuh"
#include "state_query.cuh"
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define CHECK_CUDA(expr) do {                                         \
    cudaError_t _e = (expr);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e));                              \
        exit(1);                                                      \
    }                                                                 \
} while (0)

// Shared memory layout per block (for heads_per_group heads):
//   states:   heads_per_group * d_k * d_v floats
//   k_stage:  d_k floats  (single head, reused)
//   v_stage:  d_v floats
//   q_stage:  d_k floats
//   out_stage: d_v floats
//   gate_stage: d_k*d_v floats (GLA only)
//
// The block iterates heads in [group_start, group_start + heads_per_group).
// All states are loaded first, then each head is processed in sequence.

template <DecayType DT>
__global__ void multi_head_kernel(
    float*        state_global,  // [H, d_k, d_v]
    const __half* q_global,      // [H, d_k]
    const __half* k_global,      // [H, d_k]
    const __half* v_global,      // [H, d_v]
    const float*  gate_global,   // [H, d_k*d_v] GLA only
    __half*       out_global,    // [H, d_v]
    int d_k, int d_v,
    int num_heads,
    int heads_per_group,
    float decay_val)
{
    extern __shared__ float smem[];

    const int tid         = threadIdx.x;
    const int group_start = blockIdx.x * heads_per_group;
    const int group_end   = min(group_start + heads_per_group, num_heads);
    const int group_size  = group_end - group_start;

    int state_elems = d_k * d_v;

    // Shared memory partitioning
    float* smem_states = smem;                                      // group_size * d_k * d_v
    float* smem_k      = smem_states + heads_per_group * state_elems; // d_k
    float* smem_v      = smem_k + d_k;                             // d_v
    float* smem_q      = smem_v + d_v;                             // d_k
    float* smem_out    = smem_q + d_k;                             // d_v
    float* smem_gate   = smem_out + d_v;                           // d_k*d_v for GLA

    // Load all states in the group from global memory
    for (int h = 0; h < group_size; h++) {
        int head = group_start + h;
        float* src  = state_global + (ptrdiff_t)head * state_elems;
        float* dst  = smem_states  + (ptrdiff_t)h    * state_elems;
        for (int i = tid; i < state_elems; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
    __syncthreads();

    // Process each head in the group sequentially
    for (int h = 0; h < group_size; h++) {
        int head = group_start + h;
        float*        smem_state = smem_states + (ptrdiff_t)h * state_elems;
        const __half* q_g = q_global + (ptrdiff_t)head * d_k;
        const __half* k_g = k_global + (ptrdiff_t)head * d_k;
        const __half* v_g = v_global + (ptrdiff_t)head * d_v;

        // Load k, v, q for this head
        for (int i = tid; i < d_k; i += blockDim.x) {
            smem_k[i] = __half2float(k_g[i]);
            smem_q[i] = __half2float(q_g[i]);
        }
        for (int i = tid; i < d_v; i += blockDim.x) {
            smem_v[i] = __half2float(v_g[i]);
        }

        if constexpr (DT == DecayType::gla) {
            const float* gate_g = gate_global + (ptrdiff_t)head * state_elems;
            for (int i = tid; i < state_elems; i += blockDim.x) {
                smem_gate[i] = gate_g[i];
            }
        }

        __syncthreads();

        outer_product_update<DT>(smem_state, smem_k, smem_v, smem_gate, d_k, d_v, decay_val);
        // outer_product_update calls __syncthreads()

        state_query_contract(smem_state, smem_q, smem_out, d_k, d_v);

        __syncthreads();

        // Write output for this head
        __half* o_g = out_global + (ptrdiff_t)head * d_v;
        for (int i = tid; i < d_v; i += blockDim.x) {
            o_g[i] = __float2half(smem_out[i]);
        }

        __syncthreads();
    }

    // Store all updated states back to global memory
    for (int h = 0; h < group_size; h++) {
        int head = group_start + h;
        float* src = smem_states  + (ptrdiff_t)h    * state_elems;
        float* dst = state_global + (ptrdiff_t)head * state_elems;
        for (int i = tid; i < state_elems; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

int compute_heads_per_group(const RecurrentConfig& cfg, int device) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    size_t max_smem = prop.sharedMemPerBlockOptin;

    // Reserve staging area: k + v + q + out + optional gate
    size_t staging = ((size_t)cfg.d_k + cfg.d_v + cfg.d_k + cfg.d_v) * sizeof(float);
    if (cfg.decay_type == DecayType::gla) {
        staging += (size_t)cfg.d_k * cfg.d_v * sizeof(float);
    }

    size_t per_head = (size_t)cfg.d_k * cfg.d_v * sizeof(float);
    if (per_head + staging > max_smem) return 1; // can't even fit one head

    int heads_per_group = (int)((max_smem - staging) / per_head);
    return std::max(1, std::min(heads_per_group, cfg.num_heads));
}

void multi_head_decode_step(
    float*              state,
    const __half*       q,
    const __half*       k,
    const __half*       v,
    const float*        gate,
    __half*             out,
    const RecurrentConfig& cfg,
    cudaStream_t        stream)
{
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    int hpg = compute_heads_per_group(cfg, device);

    size_t state_smem   = (size_t)hpg * cfg.d_k * cfg.d_v * sizeof(float);
    size_t staging_smem = ((size_t)cfg.d_k + cfg.d_v + cfg.d_k + cfg.d_v) * sizeof(float);
    if (cfg.decay_type == DecayType::gla) {
        staging_smem += (size_t)cfg.d_k * cfg.d_v * sizeof(float);
    }
    size_t smem = state_smem + staging_smem;

    int grid = (cfg.num_heads + hpg - 1) / hpg;
    int threads = 128;

    // Set max dynamic smem
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    size_t max_smem = prop.sharedMemPerBlockOptin;

    auto set_max = [&](void* kptr) {
        CHECK_CUDA(cudaFuncSetAttribute(
            kptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)max_smem));
    };

    switch (cfg.decay_type) {
        case DecayType::none:
            set_max((void*)multi_head_kernel<DecayType::none>);
            multi_head_kernel<DecayType::none>
                <<<grid, threads, smem, stream>>>(
                    state, q, k, v, gate, out,
                    cfg.d_k, cfg.d_v, cfg.num_heads, hpg, cfg.decay_val);
            break;
        case DecayType::fixed:
        case DecayType::retnet:
            set_max((void*)multi_head_kernel<DecayType::fixed>);
            multi_head_kernel<DecayType::fixed>
                <<<grid, threads, smem, stream>>>(
                    state, q, k, v, gate, out,
                    cfg.d_k, cfg.d_v, cfg.num_heads, hpg, cfg.decay_val);
            break;
        case DecayType::gla:
            set_max((void*)multi_head_kernel<DecayType::gla>);
            multi_head_kernel<DecayType::gla>
                <<<grid, threads, smem, stream>>>(
                    state, q, k, v, gate, out,
                    cfg.d_k, cfg.d_v, cfg.num_heads, hpg, cfg.decay_val);
            break;
    }

    CHECK_CUDA(cudaGetLastError());
}
