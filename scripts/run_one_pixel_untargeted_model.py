from __future__ import annotations

import argparse
import csv
import os
import subprocess

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from attacks.one_pixel_de import AttackParams, OnePixelDEAttack
from scripts._common import (
    cifar10_preprocess,
    get_device,
    load_cifar10_model,
    read_indices_file,
    set_seeds,
)


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


def reconstruct_adv_01(
    img_01: torch.Tensor, best_tuple: tuple[int, int, float, float, float]
) -> torch.Tensor:
    _, h, w = img_01.shape
    x = max(0, min(w - 1, int(round(best_tuple[0]))))
    y = max(0, min(h - 1, int(round(best_tuple[1]))))
    r, g, b = float(best_tuple[2]), float(best_tuple[3]), float(best_tuple[4])

    adv = img_01.clone()
    adv[:, y, x] = torch.tensor([r, g, b], dtype=adv.dtype)
    return adv


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Untargeted one-pixel DE runner with explicit model selection"
    )
    ap.add_argument("--model-name", default="cifar10_resnet20")
    ap.add_argument("--indices-file")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--dataset-root", default="./data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-adv-dir", default=None)

    ap.add_argument("--pop", type=int, default=400)
    ap.add_argument("--max-gens", type=int, default=100)
    ap.add_argument("--F", type=float, default=0.5)
    ap.add_argument("--Cr", type=float, default=0.9)
    ap.add_argument("--es-target", type=float, default=0.9)
    ap.add_argument("--es-nonadv", type=float, default=0.05)

    args = ap.parse_args()

    device = get_device(args.device)
    set_seeds(args.seed)

    base_ds = datasets.CIFAR10(
        root=args.dataset_root, train=False, download=True, transform=T.ToTensor()
    )
    indices = read_indices_file(args.indices_file)
    ds = Subset(base_ds, indices) if indices else base_ds

    pin_memory = str(device).startswith("cuda")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = load_cifar10_model(device, name=args.model_name)
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
        model=model,
        preprocess=cifar10_preprocess,
        device=device,
        params=params,
    )

    eligible: list[tuple[int, torch.Tensor, int]] = []
    with torch.inference_mode():
        batch_start = 0
        for batch_x01, batch_y in dl:
            bx = batch_x01.to(device)
            preds = model(cifar10_preprocess(bx)).argmax(1)

            for i in range(bx.size(0)):
                ds_idx = indices[batch_start + i] if indices else (batch_start + i)
                if preds[i].item() == batch_y[i].item():
                    eligible.append(
                        (ds_idx, batch_x01[i].cpu(), int(batch_y[i].item()))
                    )
                    if args.limit is not None and len(eligible) >= args.limit:
                        break

            batch_start += bx.size(0)
            if args.limit is not None and len(eligible) >= args.limit:
                break

    if not eligible:
        raise SystemExit("No eligible images found for the requested subset.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.save_adv_dir:
        os.makedirs(args.save_adv_dir, exist_ok=True)

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
        "model_name",
    ]

    success_count = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_id, (ds_idx, img_01_cpu, true_label) in enumerate(
            tqdm(eligible, desc=f"DE-1px-untargeted-{args.model_name}")
        ):
            res = attacker.run(
                img_01_cpu,
                true_label,
                mode="untargeted",
                target_label=None,
            )

            if res.success:
                success_count += 1
                if args.save_adv_dir:
                    adv_01 = reconstruct_adv_01(img_01_cpu, res.best_tuple)
                    adv_norm = cifar10_preprocess(adv_01.unsqueeze(0)).squeeze(0)
                    torch.save(
                        {"x": adv_norm.cpu()},
                        os.path.join(args.save_adv_dir, f"{ds_idx}.pt"),
                    )

            writer.writerow(
                {
                    "image_id": row_id,
                    "ds_idx": ds_idx,
                    "true_label": true_label,
                    "pred_before": res.pred_before,
                    "conf_before": f"{res.conf_before:.6f}",
                    "mode": "untargeted",
                    "target_label": "",
                    "success": int(res.success),
                    "adv_label": res.adv_label,
                    "adv_conf": f"{res.adv_conf:.6f}",
                    "gens_used": res.gens_used,
                    "queries": res.queries,
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
                    "model_name": args.model_name,
                }
            )

    print(f"Saved: {args.out}")
    if args.save_adv_dir:
        print(f"Adversarial tensors saved to: {args.save_adv_dir}")
    print(f"Eligible images attacked: {len(eligible)}")
    print(f"Successful attacks: {success_count}")
    print(f"ASR on eligible subset: {success_count / len(eligible):.4f}")


if __name__ == "__main__":
    main()
