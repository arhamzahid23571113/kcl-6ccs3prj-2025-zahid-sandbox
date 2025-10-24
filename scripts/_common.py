# scripts/_common.py
from __future__ import annotations

import random
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Subset

MEAN = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
STD = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32)


def get_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    return "mps" if torch.backends.mps.is_available() else "cpu"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)


def read_indices_file(path: str | None) -> list[int] | None:
    if not path:
        return None
    out: list[int] = []
    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            out.append(int(s))
    return out


def cifar10_preprocess(batch_01: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(batch_01.device, dtype=batch_01.dtype)[:, None, None]
    std = STD.to(batch_01.device, dtype=batch_01.dtype)[:, None, None]
    return (batch_01 - mean) / std


def cifar10_mean_std(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return MEAN.to(device).view(1, 3, 1, 1), STD.to(device).view(1, 3, 1, 1)


def load_cifar10_model(device: str, name: str = "cifar10_resnet20") -> torch.nn.Module:
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", name, pretrained=True, verbose=False
    )
    return model.eval().to(device).to(memory_format=torch.channels_last)


def iter_first_correct(
    ds: torch.utils.data.Dataset,
    model: torch.nn.Module,
    device: str,
    limit: int,
    batch_size: int,
    num_workers: int,
    indices: list[int] | None,
) -> Iterable[tuple[int, torch.Tensor, int]]:
    """
    Yield (ds_idx, img_01_cpu, true_label) for the first `limit` correctly classified samples.
    """
    pin_memory = device.startswith("cuda")
    persistent = num_workers > 0
    wrapped = Subset(ds, indices) if indices else ds
    mean = MEAN.to(device).view(1, 3, 1, 1)
    std = STD.to(device).view(1, 3, 1, 1)

    dl = DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    yielded = 0
    base_offset = 0
    with torch.inference_mode():
        for bx01, by in dl:
            if yielded >= limit:
                break
            bx01_dev = bx01.to(device)
            logits = model(
                ((bx01_dev - mean) / std).to(memory_format=torch.channels_last)
            )
            preds = logits.argmax(dim=1)
            for i in range(bx01.size(0)):
                if yielded >= limit:
                    break
                ds_idx = indices[base_offset + i] if indices else (base_offset + i)
                if preds[i].item() == by[i].item():
                    yield (ds_idx, bx01[i].cpu(), int(by[i].item()))
                    yielded += 1
            base_offset += bx01.size(0)
