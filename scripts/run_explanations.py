from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import torchvision as tv
import torchvision.transforms as T

from explanations.grad_cam import GradCAM
from explanations.ig import IntegratedGradients
from explanations.metrics import deletion_auc, iou_at_k, spearman_r
from explanations.rise import RISE

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

CSV_FIELDS = [
    "ds_idx",
    "true",
    "pred_clean",
    "pred_adv",
    "iou10_gc",
    "iou10_ig",
    "iou10_rise",
    "rho_gc",
    "rho_ig",
    "rho_rise",
    "del_auc_gc_clean",
    "del_auc_ig_clean",
    "del_auc_rise_clean",
    "del_auc_gc_adv",
    "del_auc_ig_adv",
    "del_auc_rise_adv",
]


def get_device(arg: str | None = None) -> torch.device:
    def pick(name: str) -> torch.device:
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if arg in {"cuda", "mps", "cpu"}:
        return pick(arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    m = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", model_name, pretrained=True, verbose=False
    )
    m.eval().to(device)
    return m.to(memory_format=torch.channels_last)


def get_dataset(root: str) -> tv.datasets.CIFAR10:
    tfm = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return tv.datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)


def load_adv_tensor(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["x"]


def read_indices(path: str | None, n: int) -> list[int]:
    if not path:
        return list(range(n))
    out = []
    with open(path) as f:
        for s in f:
            s = s.strip()
            if not s or s.startswith("#"):
                continue
            out.append(int(s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute explanations + metrics before/after adversarials"
    )
    ap.add_argument("--model", default="cifar10_resnet20")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--saliency-device", default=None)
    ap.add_argument("--indices-file", default=None)
    ap.add_argument("--adv-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-frac", type=float, default=0.1)
    ap.add_argument("--ig-steps", type=int, default=50)
    ap.add_argument("--rise-masks", type=int, default=2000)
    ap.add_argument("--rise-s", type=int, default=7)
    ap.add_argument("--rise-p", type=float, default=0.5)
    ap.add_argument("--batchsize", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seeds(args.seed)
    dev_forward = get_device(args.device)
    dev_sal = (
        get_device(args.saliency_device)
        if args.saliency_device
        else (torch.device("cpu") if dev_forward.type == "mps" else dev_forward)
    )

    model_fwd = load_model(args.model, dev_forward)
    model_sal = model_fwd if dev_sal == dev_forward else load_model(args.model, dev_sal)

    ds = get_dataset(args.data_root)
    indices = read_indices(args.indices_file, len(ds))

    gc = GradCAM(model_sal)
    ig = IntegratedGradients(model_sal, steps=args.ig_steps)
    rise = RISE(
        model_sal,
        n_masks=args.rise_masks,
        s=args.rise_s,
        p=args.rise_p,
        batch=args.batchsize,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for ds_idx in indices:
            try:
                x_norm, y = ds[ds_idx]
                x_norm = x_norm.unsqueeze(0)
                with torch.inference_mode():
                    pred_clean = int(model_fwd(x_norm.to(dev_forward)).argmax(1).item())
                target_class = pred_clean
                x_s = x_norm.to(dev_sal)
                gc_clean = gc.generate(x_s, target_class)
                ig_clean = ig.generate(x_s, target_class)
                rise_clean = rise.generate(x_s, target_class)
                row = {
                    "ds_idx": ds_idx,
                    "true": int(y),
                    "pred_clean": pred_clean,
                    "pred_adv": "",
                    "iou10_gc": "",
                    "iou10_ig": "",
                    "iou10_rise": "",
                    "rho_gc": "",
                    "rho_ig": "",
                    "rho_rise": "",
                    "del_auc_gc_clean": deletion_auc(
                        x_norm.to(dev_forward),
                        gc_clean.to(dev_forward),
                        model_fwd,
                        target_class,
                    ),
                    "del_auc_ig_clean": deletion_auc(
                        x_norm.to(dev_forward),
                        ig_clean.to(dev_forward),
                        model_fwd,
                        target_class,
                    ),
                    "del_auc_rise_clean": deletion_auc(
                        x_norm.to(dev_forward),
                        rise_clean.to(dev_forward),
                        model_fwd,
                        target_class,
                    ),
                    "del_auc_gc_adv": "",
                    "del_auc_ig_adv": "",
                    "del_auc_rise_adv": "",
                }
                if args.adv_dir:
                    adv_path = Path(args.adv_dir) / f"{ds_idx}.pt"
                    if adv_path.exists():
                        x_adv_norm = load_adv_tensor(adv_path).unsqueeze(0)
                        with torch.inference_mode():
                            pred_adv = int(
                                model_fwd(x_adv_norm.to(dev_forward)).argmax(1).item()
                            )
                        gc_adv = gc.generate(x_adv_norm.to(dev_sal), target_class)
                        ig_adv = ig.generate(x_adv_norm.to(dev_sal), target_class)
                        rise_adv = rise.generate(x_adv_norm.to(dev_sal), target_class)
                        row.update(
                            {
                                "pred_adv": pred_adv,
                                "iou10_gc": iou_at_k(
                                    gc_clean, gc_adv, k_frac=args.k_frac
                                ),
                                "iou10_ig": iou_at_k(
                                    ig_clean, ig_adv, k_frac=args.k_frac
                                ),
                                "iou10_rise": iou_at_k(
                                    rise_clean, rise_adv, k_frac=args.k_frac
                                ),
                                "rho_gc": spearman_r(gc_clean, gc_adv),
                                "rho_ig": spearman_r(ig_clean, ig_adv),
                                "rho_rise": spearman_r(rise_clean, rise_adv),
                                "del_auc_gc_adv": deletion_auc(
                                    x_adv_norm.to(dev_forward),
                                    gc_adv.to(dev_forward),
                                    model_fwd,
                                    target_class,
                                ),
                                "del_auc_ig_adv": deletion_auc(
                                    x_adv_norm.to(dev_forward),
                                    ig_adv.to(dev_forward),
                                    model_fwd,
                                    target_class,
                                ),
                                "del_auc_rise_adv": deletion_auc(
                                    x_adv_norm.to(dev_forward),
                                    rise_adv.to(dev_forward),
                                    model_fwd,
                                    target_class,
                                ),
                            }
                        )
                w.writerow(row)
            except Exception as e:
                print(f"[run_explanations] WARN: ds_idx={ds_idx} failed: {e}")
    gc.close()
    print(f"Done. Wrote {out_path}")


if __name__ == "__main__":
    main()
