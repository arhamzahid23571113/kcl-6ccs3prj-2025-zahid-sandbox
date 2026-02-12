from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

IOU_COLS = ["iou10_gc", "iou10_ig", "iou10_rise"]
RHO_COLS = ["rho_gc", "rho_ig", "rho_rise"]
DELTA_COLS = ["del_auc_gc_delta", "del_auc_ig_delta", "del_auc_rise_delta"]


def success_idx_untargeted(df: pd.DataFrame) -> set[int]:
    m = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int) == 1
    return set(df.loc[m, "ds_idx"].astype(int).tolist())


def success_idx_fastprior(df: pd.DataFrame) -> set[int]:
    s = df["hit_mask"].astype(str)
    m = s.str.contains(r"(?:^|\|)1(?:\||$)", na=False)
    return set(df.loc[m, "ds_idx"].astype(int).tolist())


def plot_success_rates(atk_u: Path, atk_f: Path, tag: str) -> None:
    df_u = pd.read_csv(atk_u)
    df_f = pd.read_csv(atk_f)

    su = pd.to_numeric(df_u["success"], errors="coerce").fillna(0).astype(int) == 1
    sf = df_f["hit_mask"].astype(str).str.contains(r"(?:^|\|)1(?:\||$)", na=False)

    rates = pd.Series({"untargeted": float(su.mean()), "fastprior": float(sf.mean())})

    plt.figure()
    rates.plot(kind="bar")
    plt.ylabel("Success rate (image-wise)")
    plt.ylim(0, max(0.45, rates.max() * 1.2))
    plt.title(f"{tag} attacks: success rate on eligible images (n={len(df_u)})")
    plt.tight_layout()
    out = OUTDIR / f"compare_{tag}_success_rate.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Wrote", out)


def plot_metric_bars(
    title: str, means: pd.DataFrame, rows: list[str], outname: str
) -> None:
    missing = [r for r in rows if r not in means.index]
    if missing:
        raise KeyError(f"Missing rows in means: {missing}")

    plt.figure()
    ax = means.loc[rows].plot(kind="bar")
    ax.set_title(title)
    ax.set_ylabel("Mean (successful attacks only)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = OUTDIR / outname
    plt.savefig(out, dpi=200)
    plt.close()
    print("Wrote", out)


def run(n: int) -> None:
    tag = f"{n}"

    expl_u = ROOT / "results" / "explanations" / f"before_after_{n}_untargeted_gpu.csv"
    expl_f = ROOT / "results" / "explanations" / f"before_after_{n}_fastprior_gpu.csv"
    atk_u = ROOT / "results" / "attacks" / f"de1px_untargeted_{n}.csv"
    atk_f = ROOT / "results" / "attacks" / f"de1px_targeted_fastprior_{n}.csv"

    for p in [expl_u, expl_f, atk_u, atk_f]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    df_atk_u = pd.read_csv(atk_u)
    df_atk_f = pd.read_csv(atk_f)
    idx_u = success_idx_untargeted(df_atk_u)
    idx_f = success_idx_fastprior(df_atk_f)

    df_expl_u = pd.read_csv(expl_u)
    df_expl_f = pd.read_csv(expl_f)

    su = df_expl_u[df_expl_u["ds_idx"].isin(idx_u)].copy()
    sf = df_expl_f[df_expl_f["ds_idx"].isin(idx_f)].copy()

    metrics = (
        IOU_COLS
        + RHO_COLS
        + [
            "del_auc_gc_clean",
            "del_auc_gc_adv",
            "del_auc_ig_clean",
            "del_auc_ig_adv",
            "del_auc_rise_clean",
            "del_auc_rise_adv",
        ]
    )
    means = pd.DataFrame(
        index=metrics, columns=["untargeted", "fastprior"], dtype=float
    )

    for col in metrics:
        means.loc[col, "untargeted"] = pd.to_numeric(su[col], errors="coerce").mean()
        means.loc[col, "fastprior"] = pd.to_numeric(sf[col], errors="coerce").mean()

    means.loc["del_auc_gc_delta"] = (
        means.loc["del_auc_gc_adv"] - means.loc["del_auc_gc_clean"]
    )
    means.loc["del_auc_ig_delta"] = (
        means.loc["del_auc_ig_adv"] - means.loc["del_auc_ig_clean"]
    )
    means.loc["del_auc_rise_delta"] = (
        means.loc["del_auc_rise_adv"] - means.loc["del_auc_rise_clean"]
    )

    plot_success_rates(atk_u, atk_f, tag)
    plot_metric_bars(
        f"Explanation stability (IoU@10) on successful attacks ({tag})",
        means,
        IOU_COLS,
        f"compare_{tag}_iou10.png",
    )
    plot_metric_bars(
        f"Explanation stability (Spearman ρ) on successful attacks ({tag})",
        means,
        RHO_COLS,
        f"compare_{tag}_spearman.png",
    )
    plot_metric_bars(
        f"Faithfulness shift (Deletion AUC Δ = adv − clean) on successful attacks ({tag})",
        means,
        DELTA_COLS,
        f"compare_{tag}_deletion_delta.png",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, choices=[1000, 10000], required=True)
    args = ap.parse_args()
    run(args.n)


if __name__ == "__main__":
    main()
