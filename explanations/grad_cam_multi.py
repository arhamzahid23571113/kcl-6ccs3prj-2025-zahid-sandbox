from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMMulti:
    """
    Grad-CAM that does ONE forward pass and multiple backward passes
    (one per target class) to reuse cached activations.

    Usage:
        gcm = GradCAMMulti(model)
        cams = gcm.generate_multi(x_norm, targets=[1,2,3])  # list of [1,1,H,W]
    """

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
        self._acts = out

    def _hook_bwd(self, module, grad_input, grad_output):
        # grad_output: tuple with grad wrt module output
        self._grads = grad_output[0]

    @torch.no_grad()
    def _norm(self, cam: torch.Tensor) -> torch.Tensor:
        cam = cam - cam.min()
        denom = cam.max().clamp(min=1e-12)
        return cam / denom

    def generate_multi(
        self, x_norm: torch.Tensor, targets: Iterable[int]
    ) -> List[torch.Tensor]:
        """
        x_norm: [1,3,H,W] already normalized (no grad)
        targets: iterable of class indices
        Returns: list of cams (each [1,1,H,W], on same device as x_norm)
        """
        x = x_norm.detach()  # no grad on inputs (we only need grads wrt layer)
        # One forward for all targets
        logits = self.model(x)
        cams: List[torch.Tensor] = []
        # For multiple backprops: retain_graph for all but the last
        targets = list(targets)
        for i, t in enumerate(targets):
            self.model.zero_grad(set_to_none=True)
            score = logits[:, t].sum()
            score.backward(retain_graph=(i < len(targets) - 1))

            acts = self._acts
            grads = self._grads
            # global-average pooling over spatial dims
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            with torch.no_grad():
                cam = self._norm(cam)
            cams.append(cam.detach())

        return cams

    def close(self):
        self._h_fwd.remove()
        self._h_bwd.remove()
