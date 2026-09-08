#include "fused_decode.cuh"
#include "outer_product.cuh"
#include "state_query.cuh"
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(expr) do {                                         \
    cudaError_t _e = (expr);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e));                              \
        exit(1);                                                      \
    }                                                                 \
} while (0)

// Shared memory layout (all float32):
//   [0                  .. d_k*d_v)        state matrix
//   [d_k*d_v            .. d_k*d_v+d_k)    k vector
//   [d_k*d_v+d_k        .. d_k*d_v+d_k+dv) v vector
//   [d_k*d_v+d_k+d_v    .. +d_k)           q vector (float32 converted)
//   [d_k*d_v+d_k+d_v+dk .. +d_v)           output staging
//
// For GLA, gate logits are loaded directly to the state region temporarily
// before being overwritten by state values -- actually kept in a separate
// gate_offset region appended after output staging when DT==gla.

template <DecayType DT>
__global__ void fused_decode_kernel(
    float*        state_global,  // [H, d_k, d_v]
    const __half* q_global,      // [H, d_k]
    const __half* k_global,      // [H, d_k]
    const __half* v_global,      // [H, d_v]
    const float*  gate_global,   // [H, d_k*d_v] for GLA; NULL otherwise
    __half*       out_global,    // [H, d_v]
    int d_k, int d_v,
    float decay_val)
{
    extern __shared__ float smem[];

    const int head = blockIdx.x;
    const int tid  = threadIdx.x;

    // Partition shared memory
    float* smem_state = smem;                              // d_k * d_v floats
    float* smem_k     = smem_state + d_k * d_v;           // d_k floats
    float* smem_v     = smem_k + d_k;                     // d_v floats
    float* smem_q     = smem_v + d_v;                     // d_k floats
    float* smem_out   = smem_q + d_k;                     // d_v floats
    float* smem_gate  = smem_out + d_v;                   // d_k*d_v for GLA; unused otherwise

    // Global pointers for this head
    float*        S_g  = state_global + (ptrdiff_t)head * d_k * d_v;
    const __half* q_g  = q_global     + (ptrdiff_t)head * d_k;
    const __half* k_g  = k_global     + (ptrdiff_t)head * d_k;
    const __half* v_g  = v_global     + (ptrdiff_t)head * d_v;
    __half*       o_g  = out_global   + (ptrdiff_t)head * d_v;

    // Prologue: coalesced warp-stride load of state from global to shared memory
    for (int i = tid; i < d_k * d_v; i += blockDim.x) {
        smem_state[i] = S_g[i];
    }

    // Load k, v, q (float16 -> float32 conversion)
    for (int i = tid; i < d_k; i += blockDim.x) {
        smem_k[i] = __half2float(k_g[i]);
        smem_q[i] = __half2float(q_g[i]);
    }
    for (int i = tid; i < d_v; i += blockDim.x) {
        smem_v[i] = __half2float(v_g[i]);
    }

    // Load GLA gate logits if needed
    if constexpr (DT == DecayType::gla) {
        const float* gate_g = gate_global + (ptrdiff_t)head * d_k * d_v;
        for (int i = tid; i < d_k * d_v; i += blockDim.x) {
            smem_gate[i] = gate_g[i];
        }
    }

    __syncthreads();

    // Gated rank-1 update (writes smem_state in-place)
    outer_product_update<DT>(smem_state, smem_k, smem_v, smem_gate, d_k, d_v, decay_val);
    // outer_product_update calls __syncthreads() at the end

    // Query contraction: o = S^T @ q
    state_query_contract(smem_state, smem_q, smem_out, d_k, d_v);

    __syncthreads();

    // Epilogue: store updated state to global memory
    for (int i = tid; i < d_k * d_v; i += blockDim.x) {
        S_g[i] = smem_state[i];
    }

    // Write output vector
    for (int i = tid; i < d_v; i += blockDim.x) {
        o_g[i] = __float2half(smem_out[i]);
    }
}

size_t fused_decode_smem_bytes(const RecurrentConfig& cfg) {
    size_t state_bytes = (size_t)cfg.d_k * cfg.d_v * sizeof(float);
    size_t kv_bytes    = ((size_t)cfg.d_k + cfg.d_v) * sizeof(float);
    size_t q_bytes     = (size_t)cfg.d_k * sizeof(float);
    size_t out_bytes   = (size_t)cfg.d_v * sizeof(float);
    size_t gate_bytes  = (cfg.decay_type == DecayType::gla)
                         ? (size_t)cfg.d_k * cfg.d_v * sizeof(float)
                         : 0;
    return state_bytes + kv_bytes + q_bytes + out_bytes + gate_bytes;
}

void fused_decode_step(
    float*              state,
    const __half*       q,
    const __half*       k,
    const __half*       v,
    const float*        gate,
    __half*             out,
    const RecurrentConfig& cfg,
    cudaStream_t        stream)
{
    size_t smem = fused_decode_smem_bytes(cfg);

    // Validate against device limit
    {
        int device;
        CHECK_CUDA(cudaGetDevice(&device));
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
        size_t max_smem = prop.sharedMemPerBlockOptin;
        if (smem > max_smem) {
            fprintf(stderr,
                "fused_decode: smem required %zu bytes exceeds device limit %zu\n",
                smem, max_smem);
            exit(1);
        }
        // Request maximum shared memory for this kernel
        void* kptr = nullptr;
        switch (cfg.decay_type) {
            case DecayType::none:
                kptr = (void*)fused_decode_kernel<DecayType::none>;   break;
            case DecayType::fixed:
            case DecayType::retnet:
                kptr = (void*)fused_decode_kernel<DecayType::fixed>;  break;
            case DecayType::gla:
                kptr = (void*)fused_decode_kernel<DecayType::gla>;    break;
        }
        CHECK_CUDA(cudaFuncSetAttribute(
            kptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)max_smem));
    }

    int threads = 128; // 4 warps; enough for d_v=128 with warp-stride pattern
    dim3 grid(cfg.num_heads);
    dim3 block(threads);

    switch (cfg.decay_type) {
        case DecayType::none:
            fused_decode_kernel<DecayType::none>
                <<<grid, block, smem, stream>>>(
                    state, q, k, v, gate, out, cfg.d_k, cfg.d_v, cfg.decay_val);
            break;
        case DecayType::fixed:
        case DecayType::retnet:
            fused_decode_kernel<DecayType::fixed>
                <<<grid, block, smem, stream>>>(
                    state, q, k, v, gate, out, cfg.d_k, cfg.d_v, cfg.decay_val);
            break;
        case DecayType::gla:
            fused_decode_kernel<DecayType::gla>
                <<<grid, block, smem, stream>>>(
                    state, q, k, v, gate, out, cfg.d_k, cfg.d_v, cfg.decay_val);
            break;
    }

    CHECK_CUDA(cudaGetLastError());
}
