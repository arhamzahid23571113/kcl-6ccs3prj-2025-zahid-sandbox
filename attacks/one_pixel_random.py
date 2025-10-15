from __future__ import annotations

import random

import torch

# CIFAR-10 normalization used in eval
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _to_norm(rgb01):
    # map [0,1] per channel -> normalized space
    return [
        (c - m) / s
        for c, (m, s) in zip(rgb01, zip(CIFAR10_MEAN, CIFAR10_STD, strict=False), strict=False)
    ]


@torch.no_grad()
def one_pixel_random_attack(
    model: torch.nn.Module,
    x: torch.Tensor,  # shape [3,32,32], normalized
    y_true: int,
    trials: int = 2000,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, bool, int]:
    """
    Return (x_adv, success, iters_used). Randomly tries single-pixel changes.
    success=True if predicted class != y_true.
    """
    if rng is None:
        rng = random.Random(1337)
    device = next(model.parameters()).device
    x = x.clone().to(device)

    # initial pred
    logits0 = model(x.unsqueeze(0))
    pred0 = int(logits0.argmax(dim=1).item())
    if pred0 != y_true:  # already misclassified
        return x, True, 0

    H = x.shape[1]
    W = x.shape[2]
    for t in range(1, trials + 1):
        i = rng.randrange(H)
        j = rng.randrange(W)
        rgb01 = [rng.random(), rng.random(), rng.random()]
        rgbn = _to_norm(rgb01)

        x_try = x.clone()
        x_try[0, i, j] = rgbn[0]
        x_try[1, i, j] = rgbn[1]
        x_try[2, i, j] = rgbn[2]

        logits = model(x_try.unsqueeze(0))
        pred = int(logits.argmax(dim=1).item())
        if pred != y_true:
            return x_try, True, t

    return x, False, trials
