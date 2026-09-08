"""
Python-level tests for RecurrentState.
Requires a CUDA device and the extension to be installed:
    pip install -e .
"""
import math
import pytest
import torch

# Skip entire module if extension not available or no CUDA
try:
    from linear_decode import RecurrentState, DecayType
    import linear_decode._C  # noqa: F401
    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False

pytestmark = pytest.mark.skipif(
    not HAS_KERNELS or not torch.cuda.is_available(),
    reason="linear_decode extension not installed or no CUDA device",
)

DEVICE = "cuda"


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (a_f @ b_f / (a_f.norm() * b_f.norm() + 1e-10)).item()


def make_inputs(num_heads: int, d_k: int, d_v: int):
    q = torch.randn(num_heads, d_k, dtype=torch.float16, device=DEVICE)
    k = torch.randn(num_heads, d_k, dtype=torch.float16, device=DEVICE)
    v = torch.randn(num_heads, d_v, dtype=torch.float16, device=DEVICE)
    return q, k, v


def test_step_vs_reference_cosine():
    """step() and reference_step() must agree to cosine similarity > 0.999 over 32 steps."""
    d_k, d_v, H = 64, 64, 16

    rs_fused = RecurrentState.from_config(d_k, d_v, H, DecayType.FIXED, 0.99, DEVICE)
    rs_ref   = RecurrentState.from_config(d_k, d_v, H, DecayType.FIXED, 0.99, DEVICE)

    for _ in range(32):
        q, k, v = make_inputs(H, d_k, d_v)
        out_fused = rs_fused.step(q, k, v)
        out_ref   = rs_ref.reference_step(q, k, v)

        cos = cosine_similarity(out_fused, out_ref)
        assert cos > 0.999, f"cosine similarity {cos:.5f} < 0.999"


def test_state_changes_after_steps():
    """State must be non-trivially different after 32 steps vs initial zeros."""
    d_k, d_v, H = 64, 64, 16

    rs = RecurrentState.from_config(d_k, d_v, H, DecayType.FIXED, 0.99, DEVICE)
    initial_state = rs.state_tensor().clone()

    for _ in range(32):
        q, k, v = make_inputs(H, d_k, d_v)
        rs.step(q, k, v)

    final_state = rs.state_tensor()
    diff = (final_state - initial_state).abs().max().item()
    assert diff > 1e-4, f"State barely changed after 32 steps: max_diff={diff}"


def test_dtype_rejection():
    """step() must raise TypeError for non-float16 inputs."""
    d_k, d_v, H = 32, 32, 4

    rs = RecurrentState.from_config(d_k, d_v, H, device=DEVICE)

    q_bad = torch.randn(H, d_k, dtype=torch.float32, device=DEVICE)
    k     = torch.randn(H, d_k, dtype=torch.float16, device=DEVICE)
    v     = torch.randn(H, d_v, dtype=torch.float16, device=DEVICE)

    with pytest.raises(TypeError):
        rs.step(q_bad, k, v)

    q     = torch.randn(H, d_k, dtype=torch.float16, device=DEVICE)
    k_bad = torch.randn(H, d_k, dtype=torch.float32, device=DEVICE)
    with pytest.raises(TypeError):
        rs.step(q, k_bad, v)

    v_bad = torch.randn(H, d_v, dtype=torch.float32, device=DEVICE)
    with pytest.raises(TypeError):
        rs.step(q, k, v_bad)
