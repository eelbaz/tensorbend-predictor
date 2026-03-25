"""Cross-layer expert predictor for MoE models.

Two-layer MLP that predicts which experts the router will select,
using hidden states BEFORE the MoE gate plus the previous layer's
routing decision as input. Exploits 50-60% cross-layer expert overlap.

Architecture per layer:
    Linear(hidden_dim + num_experts → predictor_dim) → BatchNorm1d → GELU → Dropout → Linear(predictor_dim → num_experts)
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertPredictor(nn.Module):
    """Single-layer expert predictor.

    Input: hidden_state [hidden_dim] concat prev_routing [num_experts]
    Output: expert logits [num_experts]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        predictor_dim: int = 256,
        use_prev_routing: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.use_prev_routing = use_prev_routing
        in_dim = hidden_dim + (num_experts if use_prev_routing else 0)
        self.fc1 = nn.Linear(in_dim, predictor_dim)
        self.bn = nn.BatchNorm1d(predictor_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(predictor_dim, num_experts)

    def forward(
        self, x: torch.Tensor, prev_routing: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.use_prev_routing:
            if prev_routing is not None:
                x = torch.cat([x, prev_routing], dim=-1)
            else:
                zeros = torch.zeros(
                    x.shape[0], self.num_experts, device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, zeros], dim=-1)
        h = self.fc1(x)
        h = self.bn(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return self.fc2(h)

    @torch.no_grad()
    def predict_topk(
        self,
        x: torch.Tensor,
        k: int = 8,
        overprovision: int = 4,
        prev_routing: torch.Tensor | None = None,
    ) -> list[int]:
        if x.dim() == 1:
            x = x.view(1, -1)
        logits = self(x, prev_routing)
        _, topk = logits.topk(k + overprovision, dim=-1)
        return topk[0].tolist()


class ExpertPredictorSet:
    """Per-layer collection of expert predictors with cross-layer routing state."""

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_experts: int,
        predictor_dim: int = 256,
        device: str = "cpu",
        use_prev_routing: bool = True,
    ):
        self.predictors = nn.ModuleList(
            [
                ExpertPredictor(hidden_dim, num_experts, predictor_dim, use_prev_routing)
                for _ in range(num_layers)
            ]
        ).to(device).eval()
        self.device = device
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._trained = False
        self._prev_routing: torch.Tensor | None = None

    def predict(
        self, layer_idx: int, hidden_states: torch.Tensor,
        top_k: int = 8, overprovision: int = 4,
    ) -> list[int]:
        x = hidden_states.to(self.device).float()
        return self.predictors[layer_idx].predict_topk(
            x, top_k, overprovision, prev_routing=self._prev_routing
        )

    def update_routing(self, expert_ids: list[int]):
        r = torch.zeros(1, self.num_experts, device=self.device)
        for eid in expert_ids:
            if 0 <= eid < self.num_experts:
                r[0, eid] = 1.0
        self._prev_routing = r

    def reset_routing(self):
        self._prev_routing = None

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.predictors.state_dict(),
                "num_layers": self.num_layers,
                "trained": self._trained,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.predictors.load_state_dict(data["state_dict"])
        self._trained = data.get("trained", True)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.predictors.parameters())

    @property
    def size_mb(self) -> float:
        return sum(p.nbytes for p in self.predictors.parameters()) / 1e6
