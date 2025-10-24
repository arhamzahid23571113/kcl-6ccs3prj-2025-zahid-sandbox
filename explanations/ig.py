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

    @torch.no_grad()
    def _norm(self, A: torch.Tensor) -> torch.Tensor:
        A = A - A.min()
        eps = torch.tensor(torch.finfo(A.dtype).eps, device=A.device, dtype=A.dtype)
        return A / A.max().clamp(min=eps)

    def generate(
        self, x: torch.Tensor, target_class: int, baseline: torch.Tensor | None = None
    ) -> torch.Tensor:
        if baseline is None:
            baseline = torch.zeros_like(x)

        use_captum_now = self.use_captum and (x.device.type != "mps")
        if use_captum_now:
            ig = _CaptumIG(lambda t: self.model(t))
            attributions = ig.attribute(
                inputs=x, baselines=baseline, target=target_class, n_steps=self.steps
            )
        else:
            alphas = torch.linspace(
                0, 1, steps=self.steps, device=x.device, dtype=x.dtype
            ).view(-1, 1, 1, 1)
            path = baseline + alphas * (x - baseline)
            path.requires_grad_(True)
            logits = self.model(path)
            score = logits[:, target_class].sum()
            grads = torch.autograd.grad(
                score, path, retain_graph=False, create_graph=False
            )[0]
            attributions = (x - baseline) * grads.mean(dim=0, keepdim=True)

        sal = attributions.abs().sum(dim=1, keepdim=True)
        return self._norm(sal)
