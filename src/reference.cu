#include "reference.cuh"
#include <cuda_fp16.h>
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

// One thread block per head.
// Threads are laid out 1D with blockDim.x = d_v (capped at 256).
// Each thread handles one or more columns of the d_k x d_v state.
template <DecayType DT>
__global__ void reference_step_kernel(
    float*        state,       // [H, d_k, d_v]
    const __half* q,           // [H, d_k]
    const __half* k,           // [H, d_k]
    const __half* v,           // [H, d_v]
    const float*  gate,        // [H, d_k*d_v] for GLA; [H] ignored for fixed; NULL for none
    __half*       out,         // [H, d_v]
    int d_k, int d_v,
    float         decay_val)
{
    const int head = blockIdx.x;
    const int tid  = threadIdx.x;

    float* S = state + (ptrdiff_t)head * d_k * d_v;
    const __half* qh = q + (ptrdiff_t)head * d_k;
    const __half* kh = k + (ptrdiff_t)head * d_k;
    const __half* vh = v + (ptrdiff_t)head * d_v;
    __half*       oh = out + (ptrdiff_t)head * d_v;

    // Gated rank-1 update: for each row i, for each col j owned by this thread
    for (int i = 0; i < d_k; i++) {
        float ki = __half2float(kh[i]);
        for (int j = tid; j < d_v; j += blockDim.x) {
            float vj = __half2float(vh[j]);
            float s  = S[i * d_v + j];

            float decay;
            if constexpr (DT == DecayType::none) {
                decay = 1.0f;
            } else if constexpr (DT == DecayType::fixed || DT == DecayType::retnet) {
                decay = decay_val;
            } else { // gla
                const float* gh = gate + (ptrdiff_t)head * d_k * d_v;
                decay = 1.0f / (1.0f + expf(-gh[i * d_v + j]));
            }

            S[i * d_v + j] = decay * s + ki * vj;
        }
    }

    __syncthreads();

    // Query contraction: o[j] = sum_i S[i][j] * q[i]
    for (int j = tid; j < d_v; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < d_k; i++) {
            acc += S[i * d_v + j] * __half2float(qh[i]);
        }
        oh[j] = __float2half(acc);
    }
}

void reference_decode_step(
    float*              state,
    const __half*       q,
    const __half*       k,
    const __half*       v,
    const float*        gate,
    __half*             out,
    const RecurrentConfig& cfg,
    cudaStream_t        stream)
{
    int threads = min(cfg.d_v, 256);
    dim3 grid(cfg.num_heads);
    dim3 block(threads);

    auto launch = [&](auto decay_type_tag) {
        reference_step_kernel<decltype(decay_type_tag)::value>
            <<<grid, block, 0, stream>>>(
                state, q, k, v, gate, out,
                cfg.d_k, cfg.d_v, cfg.decay_val);
    };

    switch (cfg.decay_type) {
        case DecayType::none:
            reference_step_kernel<DecayType::none>
                <<<grid, block, 0, stream>>>(
                    state, q, k, v, gate, out, cfg.d_k, cfg.d_v, cfg.decay_val);
            break;
        case DecayType::fixed:
        case DecayType::retnet:
            reference_step_kernel<DecayType::fixed>
                <<<grid, block, 0, stream>>>(
                    state, q, k, v, gate, out, cfg.d_k, cfg.d_v, cfg.decay_val);
            break;
        case DecayType::gla:
            reference_step_kernel<DecayType::gla>
                <<<grid, block, 0, stream>>>(
                    state, q, k, v, gate, out, cfg.d_k, cfg.d_v, cfg.decay_val);
            break;
    }

    CHECK_CUDA(cudaGetLastError());
}
