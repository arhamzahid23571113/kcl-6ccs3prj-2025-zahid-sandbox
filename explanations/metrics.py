from __future__ import annotations

import torch

_SAL_NDIMS = 4
_SAL_CH = 1


def topk_mask(sal: torch.Tensor, k_frac: float) -> torch.Tensor:
    assert sal.ndim == _SAL_NDIMS and sal.shape[1] == _SAL_CH
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
    ranks[order] = torch.arange(
        1, order.numel() + 1, device=x.device, dtype=torch.float32
    )
    return ranks.view_as(x)


def spearman_r(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(dtype=torch.float32)
    b = b.to(dtype=torch.float32)
    ra = _rankdata_torch(a)
    rb = _rankdata_torch(b)
    eps = torch.tensor(
        torch.finfo(torch.float32).eps, device=a.device, dtype=torch.float32
    )
    ra = (ra - ra.mean()) / (ra.std(unbiased=False) + eps)
    rb = (rb - rb.mean()) / (rb.std(unbiased=False) + eps)
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
    assert (
        x.ndim == _SAL_NDIMS
        and sal.ndim == _SAL_NDIMS
        and x.shape[-2:] == sal.shape[-2:]
    )
    _, _, H, W = x.shape
    Npix = H * W
    k_step = max(1, Npix // steps)

    flat = sal.view(-1)
    order = torch.argsort(flat, descending=True)
    invrank = torch.empty_like(order)
    invrank[order] = torch.arange(Npix, device=order.device)
    invrank = invrank.view(1, 1, H, W)

    ks = torch.arange(0, steps + 1, device=x.device) * k_step
    ks = torch.clamp(ks, max=Npix)
    masks = (invrank < ks.view(-1, 1, 1, 1)).to(x.dtype)

    X = x.expand(steps + 1, -1, -1, -1)
    X = X * (1.0 - masks) + replace_value * masks

    logits = model(X)
    probs = logits.softmax(dim=1)[:, target_class].float()

    probs_cpu = probs.detach().to("cpu")
    xs_axis = torch.linspace(
        0.0, 1.0, steps + 1, device=probs_cpu.device, dtype=probs_cpu.dtype
    )
    auc = torch.trapz(probs_cpu, xs_axis).item()
    return float(auc)
