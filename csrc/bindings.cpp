#include <torch/extension.h>
#include <cuda_fp16.h>
#include "fused_decode.cuh"
#include "multi_head.cuh"
#include "reference.cuh"

namespace {

void validate_inputs(
    const torch::Tensor& state,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int d_k, int d_v, int num_heads)
{
    TORCH_CHECK(state.dtype() == torch::kFloat32,
                "state must be float32, got ", state.dtype());
    TORCH_CHECK(q.dtype() == torch::kFloat16 && k.dtype() == torch::kFloat16
                && v.dtype() == torch::kFloat16,
                "q, k, v must be float16");
    TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q, k, v must be contiguous");
    TORCH_CHECK(state.dim() == 3 && state.size(0) == num_heads
                && state.size(1) == d_k && state.size(2) == d_v,
                "state must be [H, d_k, d_v]");
    TORCH_CHECK(q.dim() == 2 && q.size(0) == num_heads && q.size(1) == d_k,
                "q must be [H, d_k]");
    TORCH_CHECK(k.dim() == 2 && k.size(0) == num_heads && k.size(1) == d_k,
                "k must be [H, d_k]");
    TORCH_CHECK(v.dim() == 2 && v.size(0) == num_heads && v.size(1) == d_v,
                "v must be [H, d_v]");
}

// Returns (updated_state, output) pair.
std::pair<torch::Tensor, torch::Tensor> fused_recurrent_step(
    torch::Tensor state,          // [H, d_k, d_v] float32, modified in-place
    const torch::Tensor& q,       // [H, d_k] float16
    const torch::Tensor& k,       // [H, d_k] float16
    const torch::Tensor& v,       // [H, d_v] float16
    int    decay_type_int,
    float  decay_val,
    const c10::optional<torch::Tensor>& gate)
{
    int H  = (int)state.size(0);
    int dk = (int)state.size(1);
    int dv = (int)state.size(2);
    validate_inputs(state, q, k, v, dk, dv, H);

    RecurrentConfig cfg;
    cfg.d_k = dk; cfg.d_v = dv; cfg.num_heads = H;
    cfg.decay_type = static_cast<DecayType>(decay_type_int);
    cfg.decay_val  = decay_val;

    const float* gate_ptr = nullptr;
    if (gate.has_value()) {
        TORCH_CHECK(gate.value().dtype() == torch::kFloat32, "gate must be float32");
        TORCH_CHECK(gate.value().is_contiguous(), "gate must be contiguous");
        gate_ptr = gate.value().data_ptr<float>();
    }

    auto out = torch::empty({H, dv}, q.options());

    fused_decode_step(
        state.data_ptr<float>(),
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        gate_ptr,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        cfg);

    return {state, out};
}

std::pair<torch::Tensor, torch::Tensor> multi_head_recurrent_step(
    torch::Tensor state,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int    decay_type_int,
    float  decay_val,
    const c10::optional<torch::Tensor>& gate)
{
    int H  = (int)state.size(0);
    int dk = (int)state.size(1);
    int dv = (int)state.size(2);
    validate_inputs(state, q, k, v, dk, dv, H);

    RecurrentConfig cfg;
    cfg.d_k = dk; cfg.d_v = dv; cfg.num_heads = H;
    cfg.decay_type = static_cast<DecayType>(decay_type_int);
    cfg.decay_val  = decay_val;

    const float* gate_ptr = nullptr;
    if (gate.has_value()) {
        TORCH_CHECK(gate.value().dtype() == torch::kFloat32, "gate must be float32");
        TORCH_CHECK(gate.value().is_contiguous(), "gate must be contiguous");
        gate_ptr = gate.value().data_ptr<float>();
    }

    auto out = torch::empty({H, dv}, q.options());

    multi_head_decode_step(
        state.data_ptr<float>(),
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        gate_ptr,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        cfg);

    return {state, out};
}

std::pair<torch::Tensor, torch::Tensor> reference_recurrent_step(
    torch::Tensor state,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int    decay_type_int,
    float  decay_val,
    const c10::optional<torch::Tensor>& gate)
{
    int H  = (int)state.size(0);
    int dk = (int)state.size(1);
    int dv = (int)state.size(2);
    validate_inputs(state, q, k, v, dk, dv, H);

    RecurrentConfig cfg;
    cfg.d_k = dk; cfg.d_v = dv; cfg.num_heads = H;
    cfg.decay_type = static_cast<DecayType>(decay_type_int);
    cfg.decay_val  = decay_val;

    const float* gate_ptr = nullptr;
    if (gate.has_value()) {
        TORCH_CHECK(gate.value().dtype() == torch::kFloat32, "gate must be float32");
        TORCH_CHECK(gate.value().is_contiguous(), "gate must be contiguous");
        gate_ptr = gate.value().data_ptr<float>();
    }

    auto out = torch::empty({H, dv}, q.options());

    reference_decode_step(
        state.data_ptr<float>(),
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        gate_ptr,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        cfg);

    return {state, out};
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_recurrent_step", &fused_recurrent_step,
          "Fused decode step: smem-resident state, gated rank-1 update, query contraction",
          py::arg("state"), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("decay_type"), py::arg("decay_val"), py::arg("gate") = py::none());

    m.def("multi_head_recurrent_step", &multi_head_recurrent_step,
          "Multi-head persistent decode step: grouped head processing with smem partitioning",
          py::arg("state"), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("decay_type"), py::arg("decay_val"), py::arg("gate") = py::none());

    m.def("reference_recurrent_step", &reference_recurrent_step,
          "Sequential reference decode step: one block per head, global memory state",
          py::arg("state"), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("decay_type"), py::arg("decay_val"), py::arg("gate") = py::none());
}
