from __future__ import annotations

import torch
from torch import nn

try:
    from captum.attr import IntegratedGradients as _CaptumIG
except Exception:
    _CaptumIG = None


class IntegratedGradients:
    def __init__(self, model: nn.Module, steps: int = 50, use_captum: bool = True):
        self.model = model.eval()
        self.steps = steps
        self.use_captum = use_captum and (_CaptumIG is not None)
        if self.use_captum:

            def fwd(x):
                return self.model(x)

            self.ig = _CaptumIG(fwd)

    @torch.no_grad()
    def _norm(self, A: torch.Tensor) -> torch.Tensor:
        A = A - A.min()
        denom = A.max().clamp(min=1e-12)
        return A / denom

    def generate(
        self, x: torch.Tensor, target_class: int, baseline: torch.Tensor | None = None
    ) -> torch.Tensor:
        if baseline is None:
            baseline = torch.zeros_like(x)
        if self.use_captum:
            attributions = self.ig.attribute(
                inputs=x, baselines=baseline, target=target_class, n_steps=self.steps
            )
        else:
            alphas = torch.linspace(0, 1, steps=self.steps, device=x.device).view(-1, 1, 1, 1)
            path = baseline + alphas * (x - baseline)
            path.requires_grad_(True)
            logits = self.model(path)
            score = logits[:, target_class].sum()
            grads = torch.autograd.grad(score, path, retain_graph=False)[0]
            attributions = (x - baseline) * grads.mean(dim=0, keepdim=True)
        sal = attributions.abs().sum(dim=1, keepdim=True)
        return self._norm(sal)
