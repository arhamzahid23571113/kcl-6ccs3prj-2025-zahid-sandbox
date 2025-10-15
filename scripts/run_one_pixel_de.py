# scripts/run_one_pixel_de.py
from __future__ import annotations

import argparse
import csv
import os
import subprocess

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from attacks.one_pixel_de import AttackParams, OnePixelDEAttack


def get_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    return "mps" if torch.backends.mps.is_available() else "cpu"


def cifar10_preprocess(batch_01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=batch_01.dtype, device=batch_01.device)[
        :, None, None
    ]
    std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=batch_01.dtype, device=batch_01.device)[
        :, None, None
    ]
    return (batch_01 - mean) / std


def load_model(device: str) -> torch.nn.Module:
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True, verbose=False
    )
    model.eval().to(device)
    return model


def git_commit_short() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def read_indices_file(path: str | None) -> list[int] | None:
    if not path:
        return None
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def main() -> None:  # noqa: PLR0915  (CLI: many args/steps by nature)
    ap = argparse.ArgumentParser(
        description="Differential Evolution One-Pixel Attack (paper-faithful)"
    )
    ap.add_argument("--mode", choices=["untargeted", "targeted"], default="untargeted")
    ap.add_argument("--target-label", type=int, default=None)
    ap.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of correctly classified test images to attack",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--indices-file",
        type=str,
        default=None,
        help="Optional file with dataset indices (one per line)",
    )
    ap.add_argument("--pop", type=int, default=400)
    ap.add_argument("--max-gens", type=int, default=100)
    ap.add_argument("--F", type=float, default=0.5)
    ap.add_argument("--Cr", type=float, default=0.9)
    ap.add_argument("--es-target", type=float, default=0.90)
    ap.add_argument("--es-nonadv", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default="results/attacks/de1px_smoke.csv")
    args = ap.parse_args()

    device = get_device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Dataset: ToTensor() -> 0..1, no normalization here
    base_ds = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    indices = read_indices_file(args.indices_file)
    ds = Subset(base_ds, indices) if indices else base_ds  # noqa: SIM108

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    model = load_model(device)
    params = AttackParams(
        pop_size=args.pop,
        F=args.F,
        Cr=args.Cr,
        max_gens=args.max_gens,
        es_target=args.es_target,
        es_nonadv=args.es_nonadv,
        seed=args.seed,
        device=device,
    )
    attacker = OnePixelDEAttack(
        model=model, preprocess=cifar10_preprocess, device=device, params=params
    )

    # Prefilter: collect first `limit` correctly-classified (ds_idx, img, label)
    eligible: list[tuple[int, torch.Tensor, int]] = []
    with torch.no_grad():
        batch_start = 0
        for batch_x01, batch_y in dl:
            if len(eligible) >= args.limit:
                break
            batch_x01_dev = batch_x01.to(device)  # avoid overwriting loop var (fixes PLW2901)
            logits = model(cifar10_preprocess(batch_x01_dev))
            preds = logits.argmax(dim=1)
            for i in range(batch_x01_dev.size(0)):
                if len(eligible) >= args.limit:
                    break
                ds_idx = indices[batch_start + i] if indices else (batch_start + i)
                if preds[i].item() == batch_y[i].item():
                    # store CPU 0..1 tensor for the attacker
                    eligible.append((ds_idx, batch_x01[i].cpu(), int(batch_y[i].item())))
            batch_start += batch_x01_dev.size(0)

    if not eligible:
        print("No eligible images found; check model/dataset.")
        return

    commit = git_commit_short()

    fieldnames = [
        "image_id",
        "ds_idx",
        "true_label",
        "pred_before",
        "conf_before",
        "mode",
        "target_label",
        "success",
        "adv_label",
        "adv_conf",
        "gens_used",
        "queries",
        "x",
        "y",
        "r",
        "g",
        "b",
        "pop_size",
        "F",
        "Cr",
        "max_gens",
        "es_target",
        "es_nonadv",
        "seed",
        "commit",
        "device",
    ]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_idx, (ds_idx, img_01_cpu, true_label) in enumerate(tqdm(eligible, desc="DE-1px")):
            res = attacker.run(
                img_01_cpu, true_label, mode=args.mode, target_label=args.target_label
            )

            writer.writerow(
                {
                    "image_id": row_idx,
                    "ds_idx": ds_idx,
                    "true_label": true_label,
                    "pred_before": res.pred_before,
                    "conf_before": f"{res.conf_before:.6f}",
                    "mode": args.mode,
                    "target_label": (args.target_label if args.mode == "targeted" else ""),
                    "success": int(res.success),
                    "adv_label": res.adv_label,
                    "adv_conf": f"{res.adv_conf:.6f}",
                    "gens_used": res.gens_used,
                    "queries": res.queries,  # = pop * (1 + gens_used)
                    "x": res.best_tuple[0],
                    "y": res.best_tuple[1],
                    "r": f"{res.best_tuple[2]:.6f}",
                    "g": f"{res.best_tuple[3]:.6f}",
                    "b": f"{res.best_tuple[4]:.6f}",
                    "pop_size": args.pop,
                    "F": args.F,
                    "Cr": args.Cr,
                    "max_gens": args.max_gens,
                    "es_target": args.es_target,
                    "es_nonadv": args.es_nonadv,
                    "seed": args.seed,
                    "commit": commit,
                    "device": device,
                }
            )

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
