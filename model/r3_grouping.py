import torch
from torch import nn


class ReliabilityGuidedGrouping(nn.Module):
    """Pool raster-ordered patch tokens with four learned queries."""

    def __init__(self, dim=768, reliability_floor=0.05):
        super().__init__()
        if not 0 < reliability_floor <= 1:
            raise ValueError('Group reliability floor must be in (0, 1].')
        self.reliability_floor = reliability_floor
        self.scale = dim ** -0.5
        self.group_queries = nn.Parameter(torch.empty(4, dim))
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_projection = nn.Linear(dim, dim, bias=False)
        self.value_projection = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.group_queries, std=0.5)
        nn.init.eye_(self.query_projection.weight)
        nn.init.eye_(self.value_projection.weight)

    def forward(self, patch_tokens, body_reliability):
        queries = self.query_projection(self.group_queries)
        keys = self.key_projection(patch_tokens)
        values = self.value_projection(patch_tokens)
        affinities = torch.matmul(queries, keys.transpose(1, 2)) * self.scale
        log_prior = (body_reliability.float().clamp_min(self.reliability_floor) + 1e-6).log()
        attention = (affinities.float() + log_prior.unsqueeze(1)).softmax(dim=-1)
        local_features = torch.bmm(attention.to(values.dtype), values)
        return local_features, attention
