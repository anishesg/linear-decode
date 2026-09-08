#include "reference.cuh"
#include "fused_decode.cuh"
#include "multi_head.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

#define CHECK_CUDA(expr) do {                                         \
    cudaError_t _e = (expr);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e));                              \
        exit(1);                                                      \
    }                                                                 \
} while (0)

static uint64_t rng_state = 0xdeadbeef12345678ULL;
static float next_rand() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)(rng_state & 0xFFFFFF) / (float)0xFFFFFF;
}

static float rand_uniform(float lo, float hi) {
    return lo + (hi - lo) * next_rand();
}

// Xavier uniform initialization: range = sqrt(6 / (fan_in + fan_out))
static void xavier_init(std::vector<float>& v, int fan_in, int fan_out) {
    float scale = sqrtf(6.0f / (fan_in + fan_out));
    for (auto& x : v) x = rand_uniform(-scale, scale);
}

static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return (denom > 1e-10f) ? dot / denom : 1.0f;
}

static float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float e = 0;
    for (size_t i = 0; i < a.size(); i++) e = std::max(e, fabsf(a[i] - b[i]));
    return e;
}

static float rel_mae(const std::vector<float>& a, const std::vector<float>& b) {
    float num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) {
        num += fabsf(a[i] - b[i]);
        den += fabsf(a[i]) + 1e-8f;
    }
    return num / den;
}

struct TestConfig {
    int d_k, d_v, num_heads;
    DecayType decay_type;
    float decay_val;
    const char* name;
};

static bool run_test(const TestConfig& tc) {
    const int STEPS = 16;
    RecurrentConfig cfg;
    cfg.d_k = tc.d_k; cfg.d_v = tc.d_v; cfg.num_heads = tc.num_heads;
    cfg.decay_type = tc.decay_type; cfg.decay_val = tc.decay_val;

    int H = tc.num_heads, dk = tc.d_k, dv = tc.d_v;
    size_t state_elems = (size_t)H * dk * dv;
    size_t q_elems     = (size_t)H * dk;
    size_t v_elems     = (size_t)H * dv;
    size_t gate_elems  = (tc.decay_type == DecayType::gla) ? (size_t)H * dk * dv : 0;

    // Allocate host buffers
    std::vector<float> h_state_ref(state_elems), h_state_fused(state_elems),
                        h_state_mh(state_elems);
    std::vector<__half> h_q(q_elems), h_k(q_elems), h_v(v_elems);
    std::vector<float>  h_gate(gate_elems);
    std::vector<float>  h_out_ref(H * dv), h_out_fused(H * dv), h_out_mh(H * dv);
    std::vector<__half> d_out_ref_h(H * dv), d_out_fused_h(H * dv), d_out_mh_h(H * dv);

    // Xavier init states
    xavier_init(h_state_ref, dk, dv);
    h_state_fused = h_state_ref;
    h_state_mh    = h_state_ref;

    // Device allocations
    float  *d_state_ref, *d_state_fused, *d_state_mh, *d_gate;
    __half *d_q, *d_k, *d_v, *d_out_ref, *d_out_fused, *d_out_mh;
    CHECK_CUDA(cudaMalloc(&d_state_ref,   state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_state_fused, state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_state_mh,    state_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q,  q_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_k,  q_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_v,  v_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_out_ref,   H * dv * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_out_fused, H * dv * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_out_mh,    H * dv * sizeof(__half)));
    d_gate = nullptr;
    if (gate_elems > 0) {
        CHECK_CUDA(cudaMalloc(&d_gate, gate_elems * sizeof(float)));
    }

    CHECK_CUDA(cudaMemcpy(d_state_ref,   h_state_ref.data(),   state_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_state_fused, h_state_fused.data(), state_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_state_mh,    h_state_mh.data(),    state_elems * sizeof(float), cudaMemcpyHostToDevice));

    bool passed = true;
    float worst_cos = 1.0f, worst_mae = 0.0f, worst_rel = 0.0f;

    for (int step = 0; step < STEPS && passed; step++) {
        // Random inputs
        for (auto& x : h_q) x = __float2half(rand_uniform(-1.0f, 1.0f));
        for (auto& x : h_k) x = __float2half(rand_uniform(-1.0f, 1.0f));
        for (auto& x : h_v) x = __float2half(rand_uniform(-1.0f, 1.0f));
        for (auto& x : h_gate) x = rand_uniform(-2.0f, 2.0f);

        CHECK_CUDA(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(__half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_k, h_k.data(), q_elems * sizeof(__half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_v, h_v.data(), v_elems * sizeof(__half), cudaMemcpyHostToDevice));
        if (d_gate) {
            CHECK_CUDA(cudaMemcpy(d_gate, h_gate.data(), gate_elems * sizeof(float), cudaMemcpyHostToDevice));
        }

        reference_decode_step(d_state_ref,   d_q, d_k, d_v, d_gate, d_out_ref,   cfg);
        fused_decode_step    (d_state_fused,  d_q, d_k, d_v, d_gate, d_out_fused, cfg);
        multi_head_decode_step(d_state_mh,    d_q, d_k, d_v, d_gate, d_out_mh,    cfg);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy outputs back
        CHECK_CUDA(cudaMemcpy(d_out_ref_h.data(),   d_out_ref,   H*dv*sizeof(__half), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(d_out_fused_h.data(), d_out_fused, H*dv*sizeof(__half), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(d_out_mh_h.data(),    d_out_mh,    H*dv*sizeof(__half), cudaMemcpyDeviceToHost));

        // Convert to float for comparison
        std::vector<float> ref_f(H*dv), fused_f(H*dv), mh_f(H*dv);
        for (int i = 0; i < H*dv; i++) {
            ref_f[i]   = __half2float(d_out_ref_h[i]);
            fused_f[i] = __half2float(d_out_fused_h[i]);
            mh_f[i]    = __half2float(d_out_mh_h[i]);
        }

        float cos_f = cosine_similarity(ref_f, fused_f);
        float cos_m = cosine_similarity(ref_f, mh_f);
        float mae_f = max_abs_error(ref_f, fused_f);
        float mae_m = max_abs_error(ref_f, mh_f);
        float rel_f = rel_mae(ref_f, fused_f);
        float rel_m = rel_mae(ref_f, mh_f);

        worst_cos = std::min({worst_cos, cos_f, cos_m});
        worst_mae = std::max({worst_mae, mae_f, mae_m});
        worst_rel = std::max({worst_rel, rel_f, rel_m});

        if (cos_f < 0.999f || cos_m < 0.999f) {
            printf("  FAIL step %d: cos_fused=%.5f cos_mh=%.5f\n", step, cos_f, cos_m);
            passed = false;
        }
    }

    // Also compare final states
    std::vector<float> s_ref(state_elems), s_fused(state_elems), s_mh(state_elems);
    CHECK_CUDA(cudaMemcpy(s_ref.data(),   d_state_ref,   state_elems*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(s_fused.data(), d_state_fused, state_elems*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(s_mh.data(),    d_state_mh,    state_elems*sizeof(float), cudaMemcpyDeviceToHost));

    float state_mae_f = max_abs_error(s_ref, s_fused);
    float state_mae_m = max_abs_error(s_ref, s_mh);

    if (passed) {
        printf("  PASS: cos>=%.4f  out_mae<=%.4e  rel_mae<=%.4e  state_mae_f=%.4e  state_mae_m=%.4e\n",
               worst_cos, worst_mae, worst_rel, state_mae_f, state_mae_m);
    }

    CHECK_CUDA(cudaFree(d_state_ref));
    CHECK_CUDA(cudaFree(d_state_fused));
    CHECK_CUDA(cudaFree(d_state_mh));
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_out_ref));
    CHECK_CUDA(cudaFree(d_out_fused));
    CHECK_CUDA(cudaFree(d_out_mh));
    if (d_gate) CHECK_CUDA(cudaFree(d_gate));

    return passed;
}

int main() {
    TestConfig configs[] = {
        {32,  32,  4,  DecayType::none,   1.0f,  "dk32_dv32_h4_none"},
        {32,  32,  16, DecayType::fixed,  0.99f, "dk32_dv32_h16_fixed"},
        {64,  64,  4,  DecayType::none,   1.0f,  "dk64_dv64_h4_none"},
        {64,  64,  16, DecayType::fixed,  0.95f, "dk64_dv64_h16_fixed"},
        {64,  64,  32, DecayType::gla,    0.0f,  "dk64_dv64_h32_gla"},
        {128, 128, 4,  DecayType::none,   1.0f,  "dk128_dv128_h4_none"},
        {128, 128, 4,  DecayType::fixed,  0.99f, "dk128_dv128_h4_fixed"},
        {128, 128, 4,  DecayType::gla,    0.0f,  "dk128_dv128_h4_gla"},
        {64,  128, 16, DecayType::fixed,  0.9f,  "dk64_dv128_h16_fixed"},
        {32,  64,  32, DecayType::gla,    0.0f,  "dk32_dv64_h32_gla"},
    };

    int total = (int)(sizeof(configs) / sizeof(configs[0]));
    int passed = 0;
    for (int i = 0; i < total; i++) {
        printf("Test %d/%d: %s\n", i+1, total, configs[i].name);
        bool ok = run_test(configs[i]);
        if (ok) passed++;
        else printf("  FAILED\n");
    }

    printf("\n%d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
