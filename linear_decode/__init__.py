"""
linear_decode: fused recurrent linear attention decode.

RecurrentState manages a float32 [H, d_k, d_v] state tensor and dispatches
decode steps to the fused CUDA kernels.

Usage:
    from linear_decode import RecurrentState, DecayType

    rs = RecurrentState.from_config(d_k=64, d_v=64, num_heads=16,
                                    decay_type=DecayType.FIXED, decay_val=0.99,
                                    device="cuda")
    q = torch.randn(16, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(16, 64, dtype=torch.float16, device="cuda")
    v = torch.randn(16, 64, dtype=torch.float16, device="cuda")
    out = rs.step(q, k, v)   # [H, d_v] float16
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Optional

import torch

try:
    from linear_decode import _C as _kernels
    _HAS_KERNELS = True
except ImportError:
    _HAS_KERNELS = False


class DecayType(IntEnum):
    NONE   = 0
    FIXED  = 1
    GLA    = 2
    RETNET = 3


class RecurrentState:
    """
    Manages float32 matrix state [H, d_k, d_v] for recurrent linear attention decode.

    Dispatch order: multi_head persistent kernel > fused single-head kernel > error.
    reference_step() uses the sequential reference kernel for validation.
    """

    def __init__(
        self,
        state: torch.Tensor,
        decay_type: DecayType = DecayType.NONE,
        decay_val: float = 1.0,
    ) -> None:
        assert state.dtype == torch.float32, "state must be float32"
        assert state.dim() == 3, "state must be [H, d_k, d_v]"
        assert state.is_contiguous(), "state must be contiguous"
        self._state    = state
        self.decay_type = decay_type
        self.decay_val  = float(decay_val)

    @classmethod
    def from_config(
        cls,
        d_k: int,
        d_v: int,
        num_heads: int,
        decay_type: DecayType = DecayType.NONE,
        decay_val: float = 1.0,
        device: str = "cuda",
    ) -> "RecurrentState":
        state = torch.zeros(num_heads, d_k, d_v, dtype=torch.float32, device=device)
        return cls(state, decay_type, decay_val)

    @property
    def num_heads(self) -> int:
        return self._state.size(0)

    @property
    def d_k(self) -> int:
        return self._state.size(1)

    @property
    def d_v(self) -> int:
        return self._state.size(2)

    def reset(self) -> None:
        self._state.zero_()

    def _validate_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        for name, t in (("q", q), ("k", k), ("v", v)):
            if t.dtype != torch.float16:
                raise TypeError(
                    f"{name} must be float16, got {t.dtype}"
                )
        assert q.shape == (self.num_heads, self.d_k), \
            f"q must be [{self.num_heads}, {self.d_k}], got {list(q.shape)}"
        assert k.shape == (self.num_heads, self.d_k), \
            f"k must be [{self.num_heads}, {self.d_k}], got {list(k.shape)}"
        assert v.shape == (self.num_heads, self.d_v), \
            f"v must be [{self.num_heads}, {self.d_v}], got {list(v.shape)}"

    def step(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one decode step using the multi-head persistent kernel.

        Mutates internal state in-place. Returns float16 output [H, d_v].
        """
        if not _HAS_KERNELS:
            raise RuntimeError("linear_decode._C not available; run: pip install -e .")
        self._validate_qkv(q, k, v)
        self._state, out = _kernels.multi_head_recurrent_step(
            self._state,
            q.contiguous(), k.contiguous(), v.contiguous(),
            int(self.decay_type), self.decay_val, gate,
        )
        return out

    def reference_step(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one decode step using the sequential reference kernel.

        For validation; not performance-optimized.
        """
        if not _HAS_KERNELS:
            raise RuntimeError("linear_decode._C not available; run: pip install -e .")
        self._validate_qkv(q, k, v)
        self._state, out = _kernels.reference_recurrent_step(
            self._state,
            q.contiguous(), k.contiguous(), v.contiguous(),
            int(self.decay_type), self.decay_val, gate,
        )
        return out

    def state_tensor(self) -> torch.Tensor:
        return self._state
