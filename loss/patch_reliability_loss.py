import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchReliabilityLoss(nn.Module):
    """Supervise visible-body and umbrella occupancy on annotated images."""

    def __init__(self, umbrella_weight=1.0):
        super().__init__()
        self.umbrella_weight = umbrella_weight

    def forward(self, logits, targets, mask_valid):
        if logits.shape != targets.shape or logits.ndim != 3 or logits.shape[-1] != 2:
            raise ValueError('Patch logits and targets must have the same [B, N, 2] shape')
        if mask_valid.shape != logits.shape[:1] or mask_valid.dtype != torch.bool:
            raise ValueError('Patch annotation validity must have shape [B] and boolean dtype')

        valid_logits = logits[mask_valid].float()
        valid_targets = targets[mask_valid].float()
        losses = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, reduction='none')
        # Summing an empty tensor retains a zero gradient to the head.
        channel_loss = losses.sum(dim=(0, 1)) / max(1, valid_logits.shape[0] * logits.shape[1])
        return channel_loss[0] + self.umbrella_weight * channel_loss[1], channel_loss.detach()
