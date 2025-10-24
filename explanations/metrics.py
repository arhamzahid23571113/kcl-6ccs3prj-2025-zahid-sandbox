from __future__ import annotations

import numpy as np
import torch

NCHW_DIMS = 4
CHANNEL_DIM = 1
SINGLE_CHANNEL = 1


def topk_mask(sal: torch.Tensor, k_frac: float) -> torch.Tensor:
    assert sal.ndim == NCHW_DIMS and sal.shape[CHANNEL_DIM] == SINGLE_CHANNEL
    H, W = sal.shape[-2:]
    k = max(1, int(round(k_frac * H * W)))
    flat = sal.view(1, -1)
    thresh = torch.topk(flat, k, dim=1).values.min()
    return sal >= thresh


def iou_at_k(a: torch.Tensor, b: torch.Tensor, k_frac: float = 0.1) -> float:
    ma = topk_mask(a, k_frac).float()
    mb = topk_mask(b, k_frac).float()
    inter = (ma * mb).sum().item()
    union = (ma + mb).clamp(max=1).sum().item()
    return float(inter / max(union, 1e-12))


def _rankdata_torch(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    order = torch.argsort(flat)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, order.numel() + 1, device=x.device, dtype=torch.float32)
    return ranks.view_as(x)


def spearman_r(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = _rankdata_torch(a)
    rb = _rankdata_torch(b)
    ra = (ra - ra.mean()) / (ra.std(unbiased=False) + 1e-12)
    rb = (rb - rb.mean()) / (rb.std(unbiased=False) + 1e-12)
    return float((ra * rb).mean().item())


@torch.no_grad()
def deletion_auc(  # noqa: PLR0913
    x: torch.Tensor,
    sal: torch.Tensor,
    model,
    target_class: int,
    steps: int = 20,
    replace_value: float = 0.0,
) -> float:
    C, H, W = x.shape[1:]
    flat_sal = sal.view(-1)
    order = torch.argsort(flat_sal, descending=True)

    xs = x.clone()
    probs = []
    for t in range(steps + 1):
        logits = model(xs)
        p = logits.softmax(dim=1)[0, target_class].item()
        probs.append(p)
        if t == steps:
            break
        start = t * (H * W // steps)
        end = min((t + 1) * (H * W // steps), H * W)
        idx = order[start:end]
        hh = (idx // W).long()
        ww = (idx % W).long()
        for h, w in zip(hh, ww, strict=False):
            xs[0, :, h, w] = replace_value

    xs_axis = np.linspace(0.0, 1.0, steps + 1)
    auc = float(np.trapz(y=np.array(probs), x=xs_axis))
    return auc
