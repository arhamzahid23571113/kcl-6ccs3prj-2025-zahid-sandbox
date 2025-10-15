import argparse
import csv
import random
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from attacks.one_pixel_random import CIFAR10_MEAN, CIFAR10_STD, one_pixel_random_attack


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_model(model_name, device):
    m = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True, verbose=False)
    m.eval().to(device)
    return m


def make_loader(root, num_images):
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    idx = list(range(len(ds)))
    random.Random(1337).shuffle(idx)
    ds = Subset(ds, idx[:num_images]) if num_images > 0 else ds
    return ds


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--model-name", default="cifar10_resnet20")
    ap.add_argument("--num-images", type=int, default=100)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--out", default="results/attacks/one_pixel_random.csv")
    args = ap.parse_args()

    device = get_device()
    model = load_model(args.model_name, device)
    ds = make_loader(args.data_root, args.num_images)

    success = 0
    total = 0
    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "true", "pred_before", "pred_after", "success", "iters_used"])
        for k, (xb, yb) in enumerate(
            tqdm(DataLoader(ds, batch_size=1, shuffle=False), total=len(ds))
        ):
            x = xb[0].to(device)
            y = int(yb[0].item())
            pred_before = int(model(x.unsqueeze(0)).argmax(dim=1).item())
            if pred_before != y:
                # skip already-wrong images to evaluate real attack success
                continue
            total += 1
            x_adv, ok, iters_used = one_pixel_random_attack(model, x, y, trials=args.trials)
            pred_after = int(model(x_adv.unsqueeze(0)).argmax(dim=1).item())
            success += int(ok)
            w.writerow([k, y, pred_before, pred_after, int(ok), iters_used])
            f.flush()
            f.flush()

    dt = time.time() - t0
    rate = success / max(total, 1)
    print(
        f"[one-pixel-random] total={total} success={success} rate={rate:.3f} time_s={dt:.1f} out={args.out}"
    )


if __name__ == "__main__":
    main()
