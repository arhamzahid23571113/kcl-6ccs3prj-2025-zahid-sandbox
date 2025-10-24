# scripts/run_one_pixel_random.py
from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


def get_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    return "mps" if torch.backends.mps.is_available() else "cpu"


def cifar10_preprocess(batch_01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        [0.4914, 0.4822, 0.4465], dtype=batch_01.dtype, device=batch_01.device
    )[:, None, None]
    std = torch.tensor(
        [0.2470, 0.2435, 0.2616], dtype=batch_01.dtype, device=batch_01.device
    )[:, None, None]
    return (batch_01 - mean) / std


def load_model(device: str) -> torch.nn.Module:
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_resnet20",
        pretrained=True,
        verbose=False,
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


@torch.no_grad()
def top1(model: torch.nn.Module, x_norm: torch.Tensor) -> tuple[int, float]:
    logits = model(x_norm)
    probs = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()


def read_indices_file(path: str | None) -> list[int] | None:
    if not path:
        return None
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def main() -> None:  # noqa: PLR0912, PLR0915  (CLI with branches/steps)
    ap = argparse.ArgumentParser(
        description="Random one-pixel attack (box-constrained)"
    )
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
    ap.add_argument("--max-iters", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default="results/attacks/opix_random_smoke.csv")
    args = ap.parse_args()

    device = get_device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = random.Random(args.seed)

    # Dataset: 0..1 tensors
    base_ds = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    indices = read_indices_file(args.indices_file)
    ds = Subset(base_ds, indices) if indices else base_ds  # noqa: SIM108
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False
    )

    model = load_model(device)

    # Prefilter: collect first `limit` correctly-classified (ds_idx, img, label)
    eligible: list[tuple[int, torch.Tensor, int]] = []
    with torch.no_grad():
        batch_start = 0
        for batch_x01, batch_y in dl:
            if len(eligible) >= args.limit:
                break
            batch_x01_dev = batch_x01.to(
                device
            )  # avoid overwriting loop var (fixes PLW2901)
            logits = model(cifar10_preprocess(batch_x01_dev))
            preds = logits.argmax(dim=1)
            for i in range(batch_x01_dev.size(0)):
                if len(eligible) >= args.limit:
                    break
                ds_idx = indices[batch_start + i] if indices else (batch_start + i)
                if preds[i].item() == batch_y[i].item():
                    # store CPU 0..1 tensor for the runner
                    eligible.append(
                        (ds_idx, batch_x01[i].cpu(), int(batch_y[i].item()))
                    )
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

        for row_idx, (ds_idx, img_01_cpu, true_label) in enumerate(
            tqdm(eligible, desc="Random-1px")
        ):
            img_01 = img_01_cpu.to(device)
            pred0, conf0 = top1(model, cifar10_preprocess(img_01[None, ...]))

            H, W = img_01.shape[1:]
            success = False
            iters_used = 0
            adv_label = pred0
            adv_conf = conf0
            x_sel = y_sel = None
            r_sel = g_sel = b_sel = None

            if pred0 != true_label:
                success = True
                queries = 1
            else:
                for t in range(1, args.max_iters + 1):
                    i = rng.randrange(H)
                    j = rng.randrange(W)
                    rgb01 = torch.tensor(
                        [rng.random(), rng.random(), rng.random()],
                        device=device,
                        dtype=torch.float32,
                    )

                    x_try = img_01.clone()
                    x_try[:, i, j] = rgb01  # write 0..1; preprocessing normalizes
                    pred_t, conf_t = top1(model, cifar10_preprocess(x_try[None, ...]))

                    if pred_t != true_label:
                        success = True
                        iters_used = t
                        adv_label = pred_t
                        adv_conf = conf_t
                        x_sel, y_sel = j, i  # (x,y) = (col,row)
                        r_sel, g_sel, b_sel = rgb01.tolist()
                        break

                if not success:
                    iters_used = args.max_iters
                queries = 1 + iters_used  # 1 initial + trials used

            writer.writerow(
                {
                    "image_id": row_idx,
                    "ds_idx": ds_idx,
                    "true_label": true_label,
                    "pred_before": pred0,
                    "conf_before": f"{conf0:.6f}",
                    "mode": "untargeted",
                    "target_label": "",
                    "success": int(success),
                    "adv_label": adv_label,
                    "adv_conf": f"{adv_conf:.6f}",
                    "gens_used": 0,  # not applicable
                    "queries": queries,
                    "x": (x_sel if x_sel is not None else ""),
                    "y": (y_sel if y_sel is not None else ""),
                    "r": (f"{r_sel:.6f}" if r_sel is not None else ""),
                    "g": (f"{g_sel:.6f}" if g_sel is not None else ""),
                    "b": (f"{b_sel:.6f}" if b_sel is not None else ""),
                    "pop_size": "",
                    "F": "",
                    "Cr": "",
                    "max_gens": "",
                    "es_target": "",
                    "es_nonadv": "",
                    "seed": args.seed,
                    "commit": commit,
                    "device": device,
                }
            )

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
