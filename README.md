# linear-decode

Fused recurrent linear attention decode: shared-memory-resident matrix state with gated rank-1 updates and warp-cooperative query contraction.

## Problem

Linear attention models (GLA, RetNet, RWKV-6, DeltaNet, Based) maintain a d_k x d_v matrix state S that is updated at every decode step:

```
S_t = decay * S_{t-1} + k_t outer v_t   (rank-1 gated update)
o_t = S_t @ q_t                          (query contraction)
```

The bottleneck in existing implementations is the global memory round-trip for S. When the update and contraction are separate kernel launches (as in the Triton-based flash-linear-attention library), the full d_k x d_v state must be written to global memory after the update phase and re-read before the contraction phase. At d=64 this is 16KB fp32 per head per step; at d=128 it is 64KB. For 32 heads at 100 tokens/s, this is roughly 200 GB/s of avoidable memory traffic.

## Approach

This library fuses update and contraction into a single CUDA kernel with the state S resident in shared memory throughout.

### Shared memory feasibility

| d_k | d_v | State size (fp32) | Heads fitting in 164KB (A100) |
|-----|-----|-------------------|-------------------------------|
| 32  | 32  | 4 KB              | 41                            |
| 64  | 64  | 16 KB             | 10                            |
| 128 | 128 | 64 KB             | 2                             |
| 64  | 128 | 32 KB             | 5                             |

A100 shared memory per SM is 164KB when configured for maximum smem. The fused kernel requests d_k*d_v*4 bytes for the state plus overhead for k, v, q staging buffers.

### Gated rank-1 update (outer_product.cuh)

One warp per ceil(d_k / num_warps) rows:
- Lane 0 of each warp broadcasts k[row] via `__shfl_sync`
- Each lane covers d_v/32 columns in strided pattern, computing decay * S[row][col] + k[row] * v[col]
- Three decay modes via template: none, fixed scalar exp(-gamma), and data-dependent GLA sigmoid gate

### Query contraction (state_query.cuh)

One warp per ceil(d_v / num_warps) output columns:
- Each warp computes `o[j] = sum_i S[i][j] * q[i]` for its assigned j values
- Partial sums accumulate in fp32 registers across d_k rows
- Final reduction via `__shfl_xor_sync` across lanes sharing the same output index

### Multi-head persistence (multi_head.cu)

A single thread block processes multiple heads without intermediate global stores:
- `heads_per_group = floor(smem_limit / (d_k * d_v * 4))`
- If all heads fit, one launch processes all heads in sequence
- Otherwise, launches stream head groups: load states, process, store, advance

## Comparison with flash-linear-attention

flash-linear-attention (Triton) provides excellent chunk-scan kernels for parallel prefill but the recurrent decode path uses Python-level orchestration between update and contraction Triton kernels, with no explicit shared-memory state residency between them. This library provides the complementary CUDA path for autoregressive decode.

## Target models

- **GLA** (Gated Linear Attention): data-dependent per-element gate, sigmoid(g) computed from learned projection
- **RetNet**: fixed exponential decay per head, exp(-gamma)
- **RWKV-6**: per-channel decay with learned time-mix parameters
- **DeltaNet**: beta-normalized rank-1 updates (future extension)
- **Based**: linear approximation kernels with exact recurrent decode

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./test_correctness
./bench_latency
```

Requires CUDA 11.8+ and an sm_80+ GPU (A100, A30, RTX 3090/4090, H100).

## Shared memory budget

The fused kernel uses:
- State: d_k * d_v * 4 bytes (fp32)
- k vector: d_k * 4 bytes
- v vector: d_v * 4 bytes
- q vector: d_k * 4 bytes
- Output staging: d_v * 4 bytes

Total for d=128: 64KB + 3*512B ~ 65.5KB per head group.
