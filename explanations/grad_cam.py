from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.target_layer = target_layer or self._find_last_conv(model)
        self._acts = None
        self._grads = None
        self._h_fwd = self.target_layer.register_forward_hook(self._hook_fwd)
        self._h_bwd = self.target_layer.register_full_backward_hook(self._hook_bwd)

    def _find_last_conv(self, m: nn.Module) -> nn.Module:
        last = None
        for mod in m.modules():
            if isinstance(mod, nn.Conv2d):
                last = mod
        if last is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM.")
        return last

    def _hook_fwd(self, module, inp, out):
        self._acts = out.detach()

    def _hook_bwd(self, module, grad_input, grad_output):
        self._grads = grad_output[0].detach()

    @torch.no_grad()
    def _norm(self, cam: torch.Tensor) -> torch.Tensor:
        cam = cam - cam.min()
        eps = torch.tensor(
            torch.finfo(cam.dtype).eps, device=cam.device, dtype=cam.dtype
        )
        return cam / cam.max().clamp(min=eps)

    def generate(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, target_class].sum()
        score.backward(retain_graph=False)

        acts = self._acts
        grads = self._grads
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        with torch.no_grad():
            cam = self._norm(cam)
        return cam

    def close(self):
        self._h_fwd.remove()
        self._h_bwd.remove()
