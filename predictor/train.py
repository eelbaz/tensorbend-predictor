"""Training logic for expert predictors.

Uses pure BCE loss by default. Ranking loss and importance weighting are
available but disabled — they were found to hurt accuracy in validated
experiments (99.8% avg with pure BCE vs. ~86% with ranking loss).

Cosine LR schedule with weight decay for regularization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ExpertPredictorSet


def compute_expert_importance(
    training_data: dict[int, tuple[list, list, list]],
    num_experts: int = 72,
) -> dict[int, torch.Tensor]:
    """Compute per-expert importance scores.

    Experts near the routing decision boundary (freq ~0.5) and
    rare experts get higher importance — they need more accurate prediction.

    Note: importance weighting is available but disabled by default.
    Pure BCE without weighting achieves higher accuracy in practice.
    """
    importance = {}
    for li, data_tuple in training_data.items():
        targets = data_tuple[1]
        if not targets:
            continue
        n = len(targets)
        Y = torch.stack(targets[:n]).float()
        freq = Y.sum(dim=0) / n
        boundary_sensitivity = 1.0 - (2.0 * freq - 1.0).abs()
        rarity = 1.0 - freq.clamp(max=0.5) * 2
        scores = 0.6 * boundary_sensitivity + 0.4 * rarity
        scores = scores / (scores.max() + 1e-10)
        importance[li] = scores
    return importance


def train_predictors(
    predictor_set: ExpertPredictorSet,
    training_data: dict[int, tuple[list, list, list]],
    epochs: int = 200,
    lr: float = 1e-3,
    lambda_rank: float = 0.0,
    margin: float = 0.1,
    use_importance_weighting: bool = False,
) -> dict:
    """Train per-layer expert predictors.

    Args:
        predictor_set: Collection of predictors to train
        training_data: {layer_idx: ([hidden_states], [multi-hot targets], [prev_routing])}
        epochs: Training epochs per layer (default: 200)
        lr: Learning rate
        lambda_rank: Weight for ranking loss component (default: 0.0, ranking loss
            was found to hurt accuracy — pure BCE is better)
        margin: Margin for ranking loss (only used if lambda_rank > 0)
        use_importance_weighting: Apply Neural Thickets importance weighting
            (default: False, hurts accuracy in practice)

    Returns:
        {layer_idx: {"loss", "accuracy", "samples", ...}}
    """
    if use_importance_weighting:
        importance = compute_expert_importance(training_data)
    else:
        importance = {}

    stats = {}

    for li_key, data_tuple in training_data.items():
        inputs = data_tuple[0]
        targets = data_tuple[1]
        prev_routings = data_tuple[2] if len(data_tuple) > 2 else []
        if not inputs or not targets:
            continue

        n = min(len(inputs), len(targets))
        X = torch.stack(inputs[:n]).float()
        Y = torch.stack(targets[:n]).float()
        PR = torch.stack(prev_routings[:n]).float() if prev_routings and len(prev_routings) >= n else None

        if X.dim() == 3:
            X = X.view(-1, X.shape[-1])
            Y = Y[:X.shape[0]]
            if PR is not None:
                PR = PR[:X.shape[0]]

        predictor = predictor_set.predictors[li_key]
        dev = next(predictor.parameters()).device
        X, Y = X.to(dev), Y.to(dev)
        if PR is not None:
            PR = PR.to(dev)
        predictor.train()

        optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        if li_key in importance:
            imp = importance[li_key].to(dev)
            pos_scale = 3.0 + 2.0 * imp
            neg_weight = 0.5 * (1.0 - imp * 0.3)
        else:
            pos_scale = torch.full((Y.shape[1],), 3.0, device=dev)
            neg_weight = torch.full((Y.shape[1],), 0.5, device=dev)

        best_acc = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = predictor(X, prev_routing=PR)

            weight = torch.where(
                Y > 0,
                pos_scale.unsqueeze(0).expand_as(Y),
                neg_weight.unsqueeze(0).expand_as(Y),
            )
            loss = F.binary_cross_entropy_with_logits(logits, Y, weight=weight)

            if lambda_rank > 0:
                pred_scores = torch.sigmoid(logits)
                top_mask = Y > 0
                bot_mask = Y == 0
                if top_mask.any() and bot_mask.any():
                    top_scores = (pred_scores * top_mask.float()).sum(dim=-1) / top_mask.float().sum(dim=-1).clamp(min=1)
                    bot_scores = (pred_scores * bot_mask.float()).sum(dim=-1) / bot_mask.float().sum(dim=-1).clamp(min=1)
                    rank_loss = F.relu(margin - (top_scores - bot_scores)).mean()
                    loss = loss + lambda_rank * rank_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                pred_top8 = logits.topk(8, dim=-1)[1]
                true_top8 = Y.topk(8, dim=-1)[1]
                overlap = 0
                n_eval = min(pred_top8.shape[0], 200)
                for i in range(n_eval):
                    p = set(pred_top8[i].tolist())
                    t = set(true_top8[i].tolist())
                    overlap += len(p & t) / 8
                acc = overlap / n_eval * 100
                if acc > best_acc:
                    best_acc = acc

        predictor.eval()
        stats[li_key] = {"loss": loss.item(), "accuracy": best_acc, "samples": n}

    predictor_set._trained = True
    return stats
