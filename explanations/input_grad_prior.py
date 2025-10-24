from __future__ import annotations

import contextlib

import torch

# avoid magic numbers
_BATCHED_NDIMS = 4
_SINGLETON_BS = 1


def _minmax_norm(a: torch.Tensor) -> torch.Tensor:
    a = a - a.min()
    return a / a.max().clamp(min=1e-12)


def sumtarget_inputgrad(
    model: torch.nn.Module,
    x_norm: torch.Tensor,  # [1,3,H,W] normalized, on device
    targets: list[int],  # classes to push toward
    use_amp: bool = True,
) -> torch.Tensor:  # [1,1,H,W] saliency in [0,1]
    assert x_norm.ndim == _BATCHED_NDIMS and x_norm.size(0) == _SINGLETON_BS
    dev_type = x_norm.device.type

    # AMP is only enabled on CUDA; MPS/CPU fall back to fp32
    amp_enabled = use_amp and (dev_type == "cuda")
    amp_dtype = torch.float16

    x = x_norm.detach().clone().requires_grad_(True)

    ctx = (
        torch.autocast("cuda", dtype=amp_dtype)
        if amp_enabled
        else contextlib.nullcontext()
    )
    with ctx:
        logits = model(x)
        score = logits[:, targets].sum()  # single scalar = sum over target logits

    model.zero_grad(set_to_none=True)
    score.backward()

    sal = x.grad.abs().sum(dim=1, keepdim=True)  # |∂score/∂x|, channel-aggregated
    return _minmax_norm(sal)
