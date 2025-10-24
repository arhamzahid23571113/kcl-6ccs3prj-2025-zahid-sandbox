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
        small = (torch.rand(self.n_masks, 1, self.s, self.s, device=device) < self.p).float()
        masks = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
        cell_h = max(1, H // self.s)
        cell_w = max(1, W // self.s)
        sh_h = torch.randint(0, cell_h, (self.n_masks,), device=device)
        sh_w = torch.randint(0, cell_w, (self.n_masks,), device=device)
        h_idx = (torch.arange(H, device=device).view(1, 1, H, 1) + sh_h.view(-1, 1, 1, 1)) % H
        masks = masks.gather(2, h_idx.expand(self.n_masks, 1, H, W))
        w_idx = (torch.arange(W, device=device).view(1, 1, 1, W) + sh_w.view(-1, 1, 1, 1)) % W
        masks = masks.gather(3, w_idx.expand(self.n_masks, 1, H, W))
        mean = masks.flatten(2).mean(dim=2, keepdim=True).clamp(min=1e-6)
        masks = masks / mean.view(self.n_masks, 1, 1, 1)
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
            probs = logits.softmax(dim=1)[:, target_class]
            w = probs.view(-1, 1, 1, 1)
            sal += (w * mb).sum(dim=0, keepdim=True)
        sal = sal - sal.min()
        sal = sal / sal.max().clamp(min=1e-12)
        return sal
