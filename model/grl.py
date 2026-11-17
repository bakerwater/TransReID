import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def forward(self, x, alpha=1.0):
        return GradientReversalFunction.apply(x, alpha)
