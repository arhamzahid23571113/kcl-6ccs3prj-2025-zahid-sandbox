import argparse
import csv
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# CIFAR-10 channel stats used by most pretrained CIFAR models
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_device(pref="mps"):
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_name: str, device):
    """
    Uses torch.hub to fetch a CIFAR-10 pretrained model.
    Good options: cifar10_resnet20, cifar10_resnet32, cifar10_resnet56.
    """
    try:
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", model_name, pretrained=True, verbose=False
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load '{model_name}' from torch.hub (need internet once). "
            f"Try a different model_name or check your network. Original error: {e}"
        ) from e
    model.eval().to(device)
    return model


def get_loader(root: str, batch_size: int, num_images: int | None):
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    if num_images is not None and num_images < len(ds):
        idx = list(range(len(ds)))
        random.Random(1337).shuffle(idx)
        ds = Subset(ds, idx[:num_images])
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)


@torch.no_grad()
def evaluate(model, loader, device, out_csv: str):
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    base_ds = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
    classes = getattr(base_ds, "classes", [str(i) for i in range(10)])

    total, correct, nll_sum = 0, 0, 0.0
    t0 = time.time()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label_idx", "label", "pred_idx", "pred", "correct", "confidence"])
        image_id = 0
        for imgs, labels in loader:
            x = imgs.to(device)
            y = labels.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            corrects = (preds == y).to(torch.float32)

            # simple NLL (with log-softmax) for reporting
            nll_sum += torch.nn.functional.nll_loss(
                torch.log(probs + 1e-12), y, reduction="sum"
            ).item()

            for i in range(x.size(0)):
                li = y[i].item()
                pi = preds[i].item()
                w.writerow(
                    [
                        image_id,
                        li,
                        classes[li],
                        pi,
                        classes[pi],
                        int(li == pi),
                        float(confs[i].item()),
                    ]
                )
                image_id += 1
            total += x.size(0)
            correct += int(corrects.sum().item())

    dt = time.time() - t0
    acc = correct / total
    nll = nll_sum / total
    return {"total": total, "correct": correct, "acc": acc, "nll": nll, "time_s": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", help="CIFAR-10 root folder")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-images", type=int, default=100, help="subset size (None=all 10k)")
    ap.add_argument(
        "--model-name",
        default="cifar10_resnet20",
        help="torch.hub model: e.g., cifar10_resnet20/32/56",
    )
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--out", default="results/tables/baseline_cifar10.csv")
    args = ap.parse_args()

    device = get_device(args.device)
    model = load_model(args.model_name, device)
    loader = get_loader(
        args.data_root, args.batch_size, None if args.num_images < 0 else args.num_images
    )
    metrics = evaluate(model, loader, device, args.out)

    print(f"[OK] Saved per-image results → {args.out}")
    print(
        f"total={metrics['total']} | acc={metrics['acc']:.4f} | nll={metrics['nll']:.3f} | time_s={metrics['time_s']:.1f}"
    )


if __name__ == "__main__":
    main()
