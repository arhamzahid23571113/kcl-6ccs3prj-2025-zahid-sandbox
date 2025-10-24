from __future__ import annotations

import argparse
import csv
import os
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


def get_device(arg: str | None = None) -> torch.device:
    if arg and arg in ("cuda", "mps", "cpu"):
        if arg == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if arg == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_name: str, device):
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True)
    model.to(device).eval()
    return model


def get_dataset(data_root: str):
    tfm = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = tv.datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm)
    return ds


def load_adv_image(path: Path):
    payload = torch.load(path, map_location="cpu")
    return payload["x"]


def save_row(w, rowdict):
    if w.fieldnames is None:
        w.fieldnames = list(rowdict.keys())
        w.writeheader()
    w.writerow(rowdict)


def main():  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cifar10_resnet20")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--indices-file", default=None)
    ap.add_argument("--adv-dir", default=None, help="Dir with '{ds_idx}.pt' tensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-frac", type=float, default=0.1)
    ap.add_argument("--ig-steps", type=int, default=50)
    ap.add_argument("--rise-masks", type=int, default=2000)
    ap.add_argument("--rise-s", type=int, default=7)
    ap.add_argument("--rise-p", type=float, default=0.5)
    ap.add_argument("--batchsize", type=int, default=64)
    args = ap.parse_args()

    device = get_device(args.device)
    model = load_model(args.model, device)
    ds = get_dataset(args.data_root)

    if args.indices_file:
        with open(args.indices_file) as f:
            indices = [int(x.strip()) for x in f if x.strip()]
    else:
        indices = list(range(len(ds)))

    gc = GradCAM(model)
    ig = IntegratedGradients(model, steps=args.ig_steps)
    rise = RISE(model, n_masks=args.rise_masks, s=args.rise_s, p=args.rise_p, batch=args.batchsize)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=None)
        for ds_idx in indices:
            x, y = ds[ds_idx]
            x = x.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x)
                pred_clean = int(logits.argmax(dim=1).item())

            target_class = pred_clean

            gc_clean = gc.generate(x, target_class)
            ig_clean = ig.generate(x, target_class)
            rise_clean = rise.generate(x, target_class)

            row = {"ds_idx": ds_idx, "true": int(y), "pred_clean": pred_clean}

            if args.adv_dir:
                adv_path = Path(args.adv_dir) / f"{ds_idx}.pt"
                if adv_path.exists():
                    x_adv = load_adv_image(adv_path).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits_adv = model(x_adv)
                        pred_adv = int(logits_adv.argmax(dim=1).item())
                    gc_adv = gc.generate(x_adv, target_class)
                    ig_adv = ig.generate(x_adv, target_class)
                    rise_adv = rise.generate(x_adv, target_class)

                    row.update(
                        {
                            "pred_adv": pred_adv,
                            "iou10_gc": iou_at_k(gc_clean, gc_adv, k_frac=args.k_frac),
                            "iou10_ig": iou_at_k(ig_clean, ig_adv, k_frac=args.k_frac),
                            "iou10_rise": iou_at_k(rise_clean, rise_adv, k_frac=args.k_frac),
                            "rho_gc": spearman_r(gc_clean, gc_adv),
                            "rho_ig": spearman_r(ig_clean, ig_adv),
                            "rho_rise": spearman_r(rise_clean, rise_adv),
                            "del_auc_gc_clean": deletion_auc(x, gc_clean, model, target_class),
                            "del_auc_gc_adv": deletion_auc(x_adv, gc_adv, model, target_class),
                            "del_auc_ig_clean": deletion_auc(x, ig_clean, model, target_class),
                            "del_auc_ig_adv": deletion_auc(x_adv, ig_adv, model, target_class),
                            "del_auc_rise_clean": deletion_auc(x, rise_clean, model, target_class),
                            "del_auc_rise_adv": deletion_auc(x_adv, rise_adv, model, target_class),
                        }
                    )
                else:
                    row.update(
                        {
                            "pred_adv": "",
                            "iou10_gc": "",
                            "iou10_ig": "",
                            "iou10_rise": "",
                            "rho_gc": "",
                            "rho_ig": "",
                            "rho_rise": "",
                            "del_auc_gc_clean": deletion_auc(x, gc_clean, model, target_class),
                            "del_auc_ig_clean": deletion_auc(x, ig_clean, model, target_class),
                            "del_auc_rise_clean": deletion_auc(x, rise_clean, model, target_class),
                        }
                    )
            else:
                row.update(
                    {
                        "del_auc_gc_clean": deletion_auc(x, gc_clean, model, target_class),
                        "del_auc_ig_clean": deletion_auc(x, ig_clean, model, target_class),
                        "del_auc_rise_clean": deletion_auc(x, rise_clean, model, target_class),
                    }
                )

            if w.fieldnames is None:
                w.fieldnames = list(row.keys())
                w.writeheader()
            w.writerow(row)

    gc.close()
    print(f"Done. Wrote {args.out}")


if __name__ == "__main__":
    main()
