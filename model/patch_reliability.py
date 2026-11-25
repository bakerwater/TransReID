import math

import torch
from torch import nn
from torch.nn import functional as F


class PatchReliabilityHead(nn.Module):
    """Predict visible body and umbrella occupancy logits in patch order."""

    def __init__(self, in_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, patch_tokens):
        return self.mlp(self.norm(patch_tokens))


@torch.no_grad()
def masks_to_patch_targets(body_mask, umbrella_mask, patch_projection):
    """Convert [B, 1, H, W] masks to [B, N, 2] occupancy targets."""
    if body_mask.shape != umbrella_mask.shape or body_mask.ndim != 4 or body_mask.shape[1] != 1:
        raise ValueError('Body and umbrella masks must have the same [B, 1, H, W] shape.')

    masks = torch.cat((body_mask, umbrella_mask), dim=1).float()
    kernel_size = patch_projection.kernel_size
    padding = patch_projection.padding
    if patch_projection.padding_mode != 'zeros':
        masks = F.pad(masks, patch_projection._reversed_padding_repeated_twice,
                      mode=patch_projection.padding_mode)
        padding = 0

    counting_kernel = masks.new_ones((2, 1, *kernel_size))
    with torch.autocast(device_type=masks.device.type, enabled=False):
        foreground_pixels = F.conv2d(
            masks, counting_kernel, stride=patch_projection.stride, padding=padding,
            dilation=patch_projection.dilation, groups=2,
        )
    occupancy = foreground_pixels / math.prod(kernel_size)
    return occupancy.flatten(2).transpose(1, 2)
