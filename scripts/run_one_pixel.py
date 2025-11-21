from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from attacks.one_pixel_de import AttackParams, OnePixelDEAttack
from attacks.one_pixel_de_guided import OnePixelDERISEGuided, RiseGuideParams
from attacks.one_pixel_de_mt import MTParams, OnePixelDEMultiTarget
from explanations.grad_cam_multi import GradCAMMulti
from explanations.input_grad_prior import sumtarget_inputgrad
from scripts._common import (
    MEAN,
    STD,
    cifar10_preprocess,
    get_device,
    iter_first_correct,
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


def _reconstruct_adv_01(
    img_01: torch.Tensor, best_tuple: tuple[int, int, float, float, float]
) -> torch.Tensor:
    _, H, W = img_01.shape
    x = max(0, min(W - 1, int(round(best_tuple[0]))))
    y = max(0, min(H - 1, int(round(best_tuple[1]))))
    r, g, b = float(best_tuple[2]), float(best_tuple[3]), float(best_tuple[4])
    adv = img_01.clone()
    adv[:, y, x] = torch.tensor([r, g, b], dtype=adv.dtype)
    return adv


# -----------------------
# Modes
# -----------------------
def run_untargeted_or_guided(args, device: str) -> None:
    base_ds = datasets.CIFAR10(
        root=args.dataset_root, train=False, download=True, transform=T.ToTensor()
    )
    indices = read_indices_file(args.indices_file)
    ds = Subset(base_ds, indices) if indices else base_ds

    pin_memory = device.startswith("cuda")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = load_cifar10_model(device)
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

    if args.mode == "guided":
        guide = RiseGuideParams(
            temperature=args.guide_temp,
            mutate_prob=args.guide_mutate_prob,
            masks=args.rise_masks,
            s=args.rise_s,
            p=args.rise_p,
            batch=args.rise_batch,
        )
        attacker = OnePixelDERISEGuided(
            model=model,
            preprocess=cifar10_preprocess,
            device=device,
            params=params,
            guide=guide,
        )
    else:
        attacker = OnePixelDEAttack(
            model=model, preprocess=cifar10_preprocess, device=device, params=params
        )

    # prefilter
    eligible: list[tuple[int, torch.Tensor, int]] = []
    with torch.no_grad():
        batch_start = 0
        for batch_x01, batch_y in dl:
            if len(eligible) >= args.limit:
                break
            bx = batch_x01.to(device)
            preds = model(cifar10_preprocess(bx)).argmax(1)
            for i in range(bx.size(0)):
                if len(eligible) >= args.limit:
                    break
                ds_idx = indices[batch_start + i] if indices else (batch_start + i)
                if preds[i].item() == batch_y[i].item():
                    eligible.append(
                        (ds_idx, batch_x01[i].cpu(), int(batch_y[i].item()))
                    )
            batch_start += bx.size(0)

    if not eligible:
        print("No eligible images found.")
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
    if args.mode == "guided":
        fieldnames += [
            "guide_temp",
            "guide_mutate_prob",
            "rise_masks",
            "rise_s",
            "rise_p",
            "rise_batch",
        ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.save_adv_dir:
        os.makedirs(args.save_adv_dir, exist_ok=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row_id, (ds_idx, img_01_cpu, true_label) in enumerate(
            tqdm(eligible, desc=f"DE-1px-{args.mode}")
        ):
            res = attacker.run(
                img_01_cpu,
                true_label,
                mode=("targeted" if args.target_label is not None else "untargeted"),
                target_label=args.target_label,
            )
            if args.save_adv_dir and res.success:
                adv_01 = _reconstruct_adv_01(img_01_cpu, res.best_tuple)
                adv_norm = cifar10_preprocess(adv_01.unsqueeze(0)).squeeze(0)
                torch.save(
                    {"x": adv_norm.cpu()},
                    os.path.join(args.save_adv_dir, f"{ds_idx}.pt"),
                )

            row = {
                "image_id": row_id,
                "ds_idx": ds_idx,
                "true_label": true_label,
                "pred_before": res.pred_before,
                "conf_before": f"{res.conf_before:.6f}",
                "mode": ("targeted" if args.target_label is not None else "untargeted"),
                "target_label": (
                    args.target_label if args.target_label is not None else ""
                ),
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
            }
            if args.mode == "guided":
                row.update(
                    {
                        "guide_temp": args.guide_temp,
                        "guide_mutate_prob": args.guide_mutate_prob,
                        "rise_masks": args.rise_masks,
                        "rise_s": args.rise_s,
                        "rise_p": args.rise_p,
                        "rise_batch": args.rise_batch,
                    }
                )
            w.writerow(row)

    print(f"Saved: {args.out}")
    if args.save_adv_dir:
        print(f"Adversarial tensors saved to: {args.save_adv_dir}")


def run_targeted_sweep(args, device: str) -> None:
    """Per-target loop (vanilla DE) — your old sweep, but cleaner."""
    base_ds = datasets.CIFAR10(
        root=args.dataset_root, train=False, download=True, transform=T.ToTensor()
    )
    indices = read_indices_file(args.indices_file)

    model = load_cifar10_model(device)
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

    eligible = list(
        iter_first_correct(
            ds=base_ds,
            model=model,
            device=device,
            limit=args.limit,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            indices=indices,
        )
    )
    if not eligible:
        print("No eligible images found.")
        return

    commit = git_commit_short()
    fieldnames = [
        "image_id",
        "ds_idx",
        "true_label",
        "target_label",
        "pred_before",
        "conf_before",
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
        "seed",
        "commit",
        "device",
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.save_adv_dir:
        os.makedirs(args.save_adv_dir, exist_ok=True)

    rng = random.Random(args.seed)

    def pick_targets(y_true: int) -> list[int]:
        pool = [c for c in range(10) if c != y_true]
        if args.targets_mode == "all" or args.targets_per_image >= len(pool):
            return pool
        return rng.sample(pool, k=max(1, args.targets_per_image))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row_id, (ds_idx, img_01_cpu, y_true) in enumerate(
            tqdm(eligible, desc="DE-1px-targeted-sweep")
        ):
            for tgt in pick_targets(y_true):
                res = attacker.run(
                    img_01_cpu, y_true, mode="targeted", target_label=tgt
                )
                if args.save_adv_dir and res.success:
                    adv_01 = _reconstruct_adv_01(img_01_cpu, res.best_tuple)
                    adv_norm = cifar10_preprocess(adv_01.unsqueeze(0)).squeeze(0)
                    torch.save(
                        {"x": adv_norm.cpu()},
                        os.path.join(args.save_adv_dir, f"{ds_idx}_{tgt}.pt"),
                    )
                w.writerow(
                    {
                        "image_id": row_id,
                        "ds_idx": ds_idx,
                        "true_label": y_true,
                        "target_label": tgt,
                        "pred_before": res.pred_before,
                        "conf_before": f"{res.conf_before:.6f}",
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
                        "seed": args.seed,
                        "commit": commit,
                        "device": device,
                    }
                )
    print(f"Saved: {args.out}")
    if args.save_adv_dir:
        print(f"Adversarial tensors saved to: {args.save_adv_dir}")


def run_targeted_mt_prior(args, device: str, prior: str) -> None:
    """Vectorized multi-target with a prior (Grad-CAM or fast sum-grad)."""
    base_ds = datasets.CIFAR10(
        root=args.dataset_root, train=False, download=True, transform=T.ToTensor()
    )
    indices = read_indices_file(args.indices_file)
    model = load_cifar10_model(device)

    params = MTParams(
        pop_size=args.pop,
        max_gens=args.max_gens,
        F=args.F,
        Cr=args.Cr,
        es_target=args.es_target,
        xy_topk=args.xy_topk,
        xy_mutate_prob=args.xy_mutate_prob,
        seed=args.seed,
        device=device,
        use_amp=(device == "cuda"),
    )
    attacker = OnePixelDEMultiTarget(
        model=model, device=device, mean=MEAN, std=STD, params=params
    )

    eligible = list(
        iter_first_correct(
            ds=base_ds,
            model=model,
            device=device,
            limit=args.limit,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            indices=indices,
        )
    )
    if not eligible:
        print("No eligible images found.")
        return

    cam_provider = GradCAMMulti(model) if prior == "gradcam" else None

    fields = [
        "ds_idx",
        "true_label",
        "pred_before",
        "conf_before",
        "targets",
        "hit_mask",
        "gens_used",
        "queries",
        "best_xy_rgb_per_target",
        "adv_labels_per_target",
        "adv_confs_per_target",
        "pop",
        "max_gens",
        "F",
        "Cr",
        "es_target",
        "xy_topk",
        "xy_mutate_prob",
        "seed",
        "device",
        "prior",
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.save_adv_dir:
        os.makedirs(args.save_adv_dir, exist_ok=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ds_idx, x01_cpu, y_true in tqdm(
            eligible, desc=f"DE-1px-targeted-mt-{prior}"
        ):
            x01 = x01_cpu.to(device)
            mean = MEAN.to(device).view(3, 1, 1)
            std = STD.to(device).view(3, 1, 1)
            x_norm = (x01 - mean) / std
            targets = [c for c in range(10) if c != y_true]

            if prior == "fastprior":
                sal = sumtarget_inputgrad(
                    model, x_norm.unsqueeze(0), targets, use_amp=(device == "cuda")
                )
                cams = [sal for _ in targets]
            else:
                cams = cam_provider.generate_multi(
                    x_norm.unsqueeze(0), targets=targets
                )  # list of [1,1,H,W]

            _, _, H, W = cams[0].shape
            coord_pool, weights_per_tgt = attacker.build_coord_pool(
                H=H, W=W, per_target_cam=cams, topk=params.xy_topk
            )
            res = attacker.run(
                img01=x01,
                true_label=y_true,
                target_labels=targets,
                coord_pool=coord_pool,
                coord_weights=weights_per_tgt,
                mode="targeted",
            )

            if args.save_adv_dir and res.success_mask.any():
                t_first = int(torch.nonzero(res.success_mask, as_tuple=False)[0].item())
                xpix, ypix, r, g, b = res.best_tuple_per_tgt[t_first]
                adv01 = x01.clone()
                adv01[:, ypix, xpix] = torch.tensor([r, g, b], device=adv01.device)
                adv_norm = ((adv01 - mean) / std).cpu()
                torch.save(
                    {"x": adv_norm}, os.path.join(args.save_adv_dir, f"{ds_idx}.pt")
                )

            w.writerow(
                {
                    "ds_idx": ds_idx,
                    "true_label": y_true,
                    "pred_before": res.pred_before,
                    "conf_before": f"{res.conf_before:.6f}",
                    "targets": "|".join(map(str, targets)),
                    "hit_mask": "|".join(
                        ["1" if b else "0" for b in res.success_mask.tolist()]
                    ),
                    "gens_used": res.gens_used,
                    "queries": res.queries,
                    "best_xy_rgb_per_target": "|".join(
                        [
                            f"({x},{y},{r:.4f},{g:.4f},{b:.4f})"
                            for (x, y, r, g, b) in res.best_tuple_per_tgt
                        ]
                    ),
                    "adv_labels_per_target": "|".join(map(str, res.adv_label_per_tgt)),
                    "adv_confs_per_target": "|".join(
                        [f"{c:.6f}" for c in res.adv_conf_per_tgt]
                    ),
                    "pop": params.pop_size,
                    "max_gens": params.max_gens,
                    "F": params.F,
                    "Cr": params.Cr,
                    "es_target": params.es_target,
                    "xy_topk": params.xy_topk,
                    "xy_mutate_prob": params.xy_mutate_prob,
                    "seed": params.seed,
                    "device": params.device,
                    "prior": prior,
                }
            )
    print(f"Saved: {args.out}")
    if args.save_adv_dir:
        print(f"Adversarial tensors saved to: {args.save_adv_dir}")


# -----------------------
# CLI
# -----------------------
def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description="Unified runner for one-pixel DE variants")
    sub = ap.add_subparsers(dest="mode", required=True)

    # Shared knobs
    def add_shared(a: argparse.ArgumentParser) -> None:
        a.add_argument("--indices-file", type=str, default=None)
        a.add_argument("--limit", type=int, default=200)
        a.add_argument("--batch-size", type=int, default=64)
        a.add_argument("--num-workers", type=int, default=2)
        a.add_argument("--dataset-root", type=str, default="./data")
        a.add_argument("--device", type=str, default=None)
        a.add_argument("--seed", type=int, default=1337)

    # Untargeted / Guided
    p_untargeted = sub.add_parser("untargeted", help="Vanilla DE (paper-faithful)")
    add_shared(p_untargeted)
    p_untargeted.add_argument(
        "--out", type=str, default="results/attacks/de1px_untargeted.csv"
    )
    p_untargeted.add_argument(
        "--save-adv-dir",
        "--save-adv-tensors-dir",
        dest="save_adv_dir",
        type=str,
        default=None,
    )
    p_untargeted.add_argument(
        "--target-label", type=int, default=None
    )  # allow forced target

    for p in [p_untargeted]:
        p.add_argument("--pop", type=int, default=400)
        p.add_argument("--max-gens", type=int, default=100)
        p.add_argument("--F", type=float, default=0.5)
        p.add_argument("--Cr", type=float, default=0.9)
        p.add_argument("--es-target", type=float, default=0.90)
        p.add_argument("--es-nonadv", type=float, default=0.05)

    p_guided = sub.add_parser("guided", help="RISE-guided DE (black-box)")
    add_shared(p_guided)
    p_guided.add_argument(
        "--out", type=str, default="results/attacks/de1px_rise_guided.csv"
    )
    p_guided.add_argument(
        "--save-adv-dir",
        "--save-adv-tensors-dir",
        dest="save_adv_dir",
        type=str,
        default=None,
    )
    p_guided.add_argument("--target-label", type=int, default=None)
    p_guided.add_argument("--pop", type=int, default=400)
    p_guided.add_argument("--max-gens", type=int, default=50)
    p_guided.add_argument("--F", type=float, default=0.5)
    p_guided.add_argument("--Cr", type=float, default=0.9)
    p_guided.add_argument("--es-target", type=float, default=0.90)
    p_guided.add_argument("--es-nonadv", type=float, default=0.05)
    p_guided.add_argument("--guide-temp", type=float, default=2.0)
    p_guided.add_argument("--guide-mutate-prob", type=float, default=0.0)
    p_guided.add_argument("--rise-masks", type=int, default=500)
    p_guided.add_argument("--rise-s", type=int, default=7)
    p_guided.add_argument("--rise-p", type=float, default=0.5)
    p_guided.add_argument("--rise-batch", type=int, default=64)

    # Targeted sweep (vanilla per-target)
    p_sweep = sub.add_parser("targeted-sweep", help="Per-target loop with vanilla DE")
    add_shared(p_sweep)
    p_sweep.add_argument(
        "--out", type=str, default="results/attacks/de1px_targeted_sweep.csv"
    )
    p_sweep.add_argument(
        "--save-adv-dir",
        "--save-adv-tensors-dir",
        dest="save_adv_dir",
        type=str,
        default=None,
    )
    p_sweep.add_argument("--pop", type=int, default=400)
    p_sweep.add_argument("--max-gens", type=int, default=50)
    p_sweep.add_argument("--F", type=float, default=0.5)
    p_sweep.add_argument("--Cr", type=float, default=0.9)
    p_sweep.add_argument("--es-target", type=float, default=0.90)
    p_sweep.add_argument("--es-nonadv", type=float, default=0.05)
    p_sweep.add_argument("--targets-mode", choices=["all", "random"], default="all")
    p_sweep.add_argument("--targets-per-image", type=int, default=9)

    # Targeted multi-target with priors
    def add_mt_prior(name: str, help_text: str, default_out: str, prior: str):
        p = sub.add_parser(name, help=help_text)
        add_shared(p)
        p.add_argument("--out", type=str, default=default_out)
        p.add_argument(
            "--save-adv-dir",
            "--save-adv-tensors-dir",
            dest="save_adv_dir",
            type=str,
            default=None,
        )
        p.add_argument("--pop", type=int, default=192)
        p.add_argument("--max-gens", type=int, default=40)
        p.add_argument("--F", type=float, default=0.5)
        p.add_argument("--Cr", type=float, default=0.9)
        p.add_argument("--es-target", type=float, default=0.90)
        p.add_argument("--xy-topk", type=int, default=32)
        p.add_argument("--xy-mutate-prob", type=float, default=0.10)
        p.set_defaults(_prior=prior)
        return p

    add_mt_prior(
        "targeted-mt-prior",
        "Multi-target vectorized DE with Grad-CAM prior",
        "results/attacks/de1px_targeted_mt_prior.csv",
        "gradcam",
    )
    add_mt_prior(
        "fastprior",
        "Multi-target DE with ultra-fast sum-gradient prior",
        "results/attacks/de1px_targeted_fastprior.csv",
        "fastprior",
    )

    args = ap.parse_args()
    set_seeds(args.seed)
    device = get_device(args.device)

    if args.mode in {"untargeted", "guided"}:
        return run_untargeted_or_guided(args, device)
    if args.mode == "targeted-sweep":
        return run_targeted_sweep(args, device)
    if args.mode in {"targeted-mt-prior", "fastprior"}:
        return run_targeted_mt_prior(args, device, prior=args._prior)
    raise SystemExit("Unknown mode")


if __name__ == "__main__":
    main()
