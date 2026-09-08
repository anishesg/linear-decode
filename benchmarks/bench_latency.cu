#include "reference.cuh"
#include "fused_decode.cuh"
#include "multi_head.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(expr) do {                                         \
    cudaError_t _e = (expr);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e));                              \
        exit(1);                                                      \
    }                                                                 \
} while (0)

static uint64_t rng_state = 0xcafe12345678ULL;
static float next_rand() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)(rng_state & 0xFFFFFF) / (float)0xFFFFFF * 2.0f - 1.0f;
}

// Simulated two-kernel baseline: separate update and contract kernels
// with global memory intermediate between them.

// Update-only kernel: S_new = decay * S + k outer v, writes to global mem
__global__ void update_kernel(
    float* __restrict__ state,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    int d_k, int d_v, float decay)
{
    int head = blockIdx.x;
    int tid  = threadIdx.x;
    float* S       = state + (ptrdiff_t)head * d_k * d_v;
    const __half* kh = k + (ptrdiff_t)head * d_k;
    const __half* vh = v + (ptrdiff_t)head * d_v;
    for (int i = 0; i < d_k; i++) {
        float ki = __half2float(kh[i]);
        for (int j = tid; j < d_v; j += blockDim.x) {
            S[i * d_v + j] = decay * S[i * d_v + j] + ki * __half2float(vh[j]);
        }
    }
}

// Contraction-only kernel: o = S @ q, reads from global mem
__global__ void contract_kernel(
    const float* __restrict__ state,
    const __half* __restrict__ q,
    __half* __restrict__ out,
    int d_k, int d_v)
{
    int head = blockIdx.x;
    int tid  = threadIdx.x;
    const float* S  = state + (ptrdiff_t)head * d_k * d_v;
    const __half* qh = q + (ptrdiff_t)head * d_k;
    __half* oh       = out + (ptrdiff_t)head * d_v;
    for (int j = tid; j < d_v; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < d_k; i++) acc += S[i * d_v + j] * __half2float(qh[i]);
        oh[j] = __float2half(acc);
    }
}

struct BenchResult {
    float ref_us, fused_us, mh_us, twokernel_us;
    size_t state_bytes;
    float ref_bw_gbps, fused_bw_gbps, mh_bw_gbps;
};

static BenchResult run_bench(int d_k, int d_v, int num_heads, int warmup = 10, int iters = 100) {
    RecurrentConfig cfg;
    cfg.d_k = d_k; cfg.d_v = d_v; cfg.num_heads = num_heads;
    cfg.decay_type = DecayType::fixed; cfg.decay_val = 0.99f;

    size_t state_elems = (size_t)num_heads * d_k * d_v;
    size_t q_elems     = (size_t)num_heads * d_k;
    size_t v_elems     = (size_t)num_heads * d_v;

    float *d_state_ref, *d_state_fused, *d_state_mh, *d_state_2k;
    __half *d_q, *d_k_buf, *d_v_buf, *d_out;

    CHECK_CUDA(cudaMalloc(&d_state_ref,   state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_state_fused, state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_state_mh,    state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_state_2k,    state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q,    q_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_k_buf, q_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_v_buf, v_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_out,   v_elems * sizeof(__half)));

    // Fill with random data
    std::vector<float> init(state_elems);
    for (auto& x : init) x = next_rand();
    std::vector<__half> hq(q_elems), hk(q_elems), hv(v_elems);
    for (auto& x : hq) x = __float2half(next_rand());
    for (auto& x : hk) x = __float2half(next_rand());
    for (auto& x : hv) x = __float2half(next_rand());

    CHECK_CUDA(cudaMemcpy(d_state_ref,   init.data(), state_elems*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_state_fused, init.data(), state_elems*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_state_mh,    init.data(), state_elems*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_state_2k,    init.data(), state_elems*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q,    hq.data(), q_elems*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k_buf, hk.data(), q_elems*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v_buf, hv.data(), v_elems*sizeof(__half), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    auto time_fn = [&](auto fn) -> float {
        for (int i = 0; i < warmup; i++) fn();
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) fn();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        return ms * 1000.0f / iters; // microseconds per iter
    };

    float ref_us = time_fn([&]() {
        reference_decode_step(d_state_ref, d_q, d_k_buf, d_v_buf, nullptr, d_out, cfg);
    });

    float fused_us = time_fn([&]() {
        fused_decode_step(d_state_fused, d_q, d_k_buf, d_v_buf, nullptr, d_out, cfg);
    });

    float mh_us = time_fn([&]() {
        multi_head_decode_step(d_state_mh, d_q, d_k_buf, d_v_buf, nullptr, d_out, cfg);
    });

    int threads2k = std::min(d_v, 256);
    float twokernel_us = time_fn([&]() {
        update_kernel<<<num_heads, threads2k>>>(d_state_2k, d_k_buf, d_v_buf, d_k, d_v, 0.99f);
        contract_kernel<<<num_heads, threads2k>>>(d_state_2k, d_q, d_out, d_k, d_v);
    });

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    size_t state_bytes = state_elems * sizeof(float);
    // Bandwidth: state read + write (2x) for reference/two-kernel
    float ref_bw  = (2.0f * state_bytes) / (ref_us * 1e-6f) / 1e9f;
    float fused_bw = (2.0f * state_bytes) / (fused_us * 1e-6f) / 1e9f;
    float mh_bw    = (2.0f * state_bytes) / (mh_us * 1e-6f) / 1e9f;

    CHECK_CUDA(cudaFree(d_state_ref));
    CHECK_CUDA(cudaFree(d_state_fused));
    CHECK_CUDA(cudaFree(d_state_mh));
    CHECK_CUDA(cudaFree(d_state_2k));
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k_buf));
    CHECK_CUDA(cudaFree(d_v_buf));
    CHECK_CUDA(cudaFree(d_out));

    BenchResult r;
    r.ref_us = ref_us; r.fused_us = fused_us; r.mh_us = mh_us; r.twokernel_us = twokernel_us;
    r.state_bytes = state_bytes;
    r.ref_bw_gbps = ref_bw; r.fused_bw_gbps = fused_bw; r.mh_bw_gbps = mh_bw;
    return r;
}

int main() {
    int dims[]     = {64, 128};
    int heads[]    = {8, 16, 32, 64};

    printf("%-8s %-8s %-8s | %-10s %-10s %-10s %-10s | %-10s %-10s %-10s | %-8s %-8s\n",
           "d_k", "d_v", "heads",
           "ref(us)", "fused(us)", "mh(us)", "2kern(us)",
           "ref_bw", "fused_bw", "mh_bw",
           "spdup_f", "spdup_m");
    printf("%s\n", std::string(120, '-').c_str());

    for (int d : dims) {
        for (int h : heads) {
            BenchResult r = run_bench(d, d, h);
            float spdup_f = r.ref_us / r.fused_us;
            float spdup_m = r.ref_us / r.mh_us;
            printf("%-8d %-8d %-8d | %-10.2f %-10.2f %-10.2f %-10.2f | %-10.2f %-10.2f %-10.2f | %-8.2f %-8.2f\n",
                   d, d, h,
                   r.ref_us, r.fused_us, r.mh_us, r.twokernel_us,
                   r.ref_bw_gbps, r.fused_bw_gbps, r.mh_bw_gbps,
                   spdup_f, spdup_m);
        }
    }

    return 0;
}
