"""Losses permitted by the strict advisor-DOCX training contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _validate_heatmap_pair(predicted: Tensor, target: Tensor) -> None:
    if predicted.ndim != 4 or predicted.shape != target.shape:
        raise ValueError(
            "Predicted and target heatmaps must share [B,K,H,W], got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if not torch.is_floating_point(predicted) or not torch.is_floating_point(target):
        raise TypeError("Heatmaps must use floating dtypes")
    if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("Heatmaps contain NaN or Inf")


def strict_supervised_heatmap_mse(
    predicted: Tensor,
    target: Tensor,
    valid_keypoints: Tensor,
) -> Tensor:
    """Return only masked Gaussian heatmap MSE; no coordinate or JS term exists."""

    _validate_heatmap_pair(predicted, target)
    if valid_keypoints.shape != predicted.shape[:2] or valid_keypoints.dtype != torch.bool:
        raise ValueError("valid_keypoints must be boolean [B,K]")
    valid = valid_keypoints.to(device=predicted.device)
    per_keypoint = (predicted - target).square().mean(dim=(-1, -2))
    count = valid.sum()
    if int(count.detach().cpu()) == 0:
        raise ValueError("A supervised logical batch has no valid keypoints")
    return torch.where(valid, per_keypoint, torch.zeros_like(per_keypoint)).sum() / count


@dataclass(frozen=True)
class StrictPseudoLoss:
    loss: Tensor
    accepted_keypoints: int
    candidate_keypoints: int

    @property
    def acceptance_rate(self) -> float:
        if self.candidate_keypoints == 0:
            return 0.0
        return self.accepted_keypoints / self.candidate_keypoints


def strict_pseudo_heatmap_mse(
    predicted: Tensor,
    target: Tensor,
    accepted_keypoints: Tensor,
    *,
    valid_pixels: Tensor | None = None,
) -> StrictPseudoLoss:
    """MSE to aligned teacher heatmaps for accepted keypoints only."""

    _validate_heatmap_pair(predicted, target)
    if accepted_keypoints.shape != predicted.shape[:2] or accepted_keypoints.dtype != torch.bool:
        raise ValueError("accepted_keypoints must be boolean [B,K]")
    accepted = accepted_keypoints.to(device=predicted.device)
    squared = (predicted - target).square()
    if valid_pixels is None:
        per_keypoint = squared.mean(dim=(-1, -2))
    else:
        if valid_pixels.ndim != 4 or valid_pixels.shape[0] != predicted.shape[0]:
            raise ValueError("valid_pixels must have shape [B,1,H,W] or [B,K,H,W]")
        if valid_pixels.shape[1] not in (1, predicted.shape[1]):
            raise ValueError("valid_pixels channel count must be one or K")
        if tuple(valid_pixels.shape[-2:]) != tuple(predicted.shape[-2:]):
            raise ValueError("valid_pixels spatial shape differs from heatmaps")
        mask = valid_pixels.to(device=predicted.device, dtype=predicted.dtype)
        mask = mask.expand(-1, predicted.shape[1], -1, -1)
        denominator = mask.sum(dim=(-1, -2)).clamp_min(1.0)
        per_keypoint = (squared * mask).sum(dim=(-1, -2)) / denominator
    accepted_count = int(accepted.sum().detach().cpu())
    candidate_count = accepted.numel()
    if accepted_count:
        loss = torch.where(accepted, per_keypoint, torch.zeros_like(per_keypoint)).sum()
        # The DOCX uses a fixed 1/(3*N_u) denominator.  Rejected keypoints
        # contribute zero; the scale must not be renormalized by acceptance.
        loss = loss / candidate_count
    else:
        loss = predicted.sum() * 0.0
    return StrictPseudoLoss(
        loss=loss,
        accepted_keypoints=accepted_count,
        candidate_keypoints=candidate_count,
    )


__all__ = [
    "StrictPseudoLoss",
    "strict_pseudo_heatmap_mse",
    "strict_supervised_heatmap_mse",
]
