from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RISE:
    def __init__(
        self, model: nn.Module, n_masks: int = 2000, s: int = 7, p: float = 0.5, batch: int = 64
    ):
        self.model = model.eval()
        self.n_masks = n_masks
        self.s = s
        self.p = p
        self.batch = batch

    @torch.no_grad()
    def _generate_masks(self, H: int, W: int, device) -> torch.Tensor:
        m = (torch.rand(self.n_masks, 1, self.s, self.s, device=device) < self.p).float()
        m = F.interpolate(m, size=(H + 2, W + 2), mode="bilinear", align_corners=False)
        r_h = torch.randint(0, 2, (self.n_masks,), device=device)
        r_w = torch.randint(0, 2, (self.n_masks,), device=device)
        masks = torch.zeros(self.n_masks, 1, H, W, device=device)
        for i in range(self.n_masks):
            masks[i, :, :, :] = m[i, :, r_h[i] : r_h[i] + H, r_w[i] : r_w[i] + W]
        mean = masks.view(self.n_masks, -1).mean(dim=1).view(-1, 1, 1, 1).clamp(min=1e-6)
        masks = masks / mean
        return masks

    @torch.no_grad()
    def generate(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        device = x.device
        _, C, H, W = x.shape
        masks = self._generate_masks(H, W, device)
        sal = torch.zeros(1, 1, H, W, device=device)
        for i in range(0, self.n_masks, self.batch):
            mb = masks[i : i + self.batch]
            xm = x * mb
            logits = self.model(xm)
            scores = logits.softmax(dim=-1)[:, target_class]
            w = scores.view(-1, 1, 1, 1)
            sal += (w * mb).sum(dim=0, keepdim=True)
        sal = sal - sal.min()
        sal = sal / sal.max().clamp(min=1e-12)
        return sal
