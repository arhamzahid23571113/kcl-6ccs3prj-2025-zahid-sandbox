from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

EXPL_U = ROOT / "results" / "explanations" / "before_after_10000_untargeted_gpu.csv"
EXPL_F = ROOT / "results" / "explanations" / "before_after_10000_fastprior_gpu.csv"
ATK_U = ROOT / "results" / "attacks" / "de1px_untargeted_10000.csv"
ATK_F = ROOT / "results" / "attacks" / "de1px_targeted_fastprior_10000.csv"

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


def plot_success_rates() -> None:
    atk_u = pd.read_csv(ATK_U)
    atk_f = pd.read_csv(ATK_F)

    su = pd.to_numeric(atk_u["success"], errors="coerce").fillna(0).astype(int) == 1
    sf = atk_f["hit_mask"].astype(str).str.contains(r"(?:^|\|)1(?:\||$)", na=False)

    rates = pd.Series({"untargeted": float(su.mean()), "fastprior": float(sf.mean())})

    plt.figure()
    rates.plot(kind="bar")
    plt.ylabel("Success rate (image-wise)")
    plt.ylim(0, max(0.45, rates.max() * 1.2))
    plt.title("10k attacks: success rate on eligible images (n=9213)")
    plt.tight_layout()
    out = OUTDIR / "compare_10k_success_rate.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Wrote", out)


def plot_metric_bars(
    title: str, means: pd.DataFrame, rows: list[str], outname: str
) -> None:
    # Here, means has index=metric names and columns=['untargeted','fastprior'].
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


def main() -> None:
    atk_u = pd.read_csv(ATK_U)
    atk_f = pd.read_csv(ATK_F)
    idx_u = success_idx_untargeted(atk_u)
    idx_f = success_idx_fastprior(atk_f)

    expl_u = pd.read_csv(EXPL_U)
    expl_f = pd.read_csv(EXPL_F)

    su = expl_u[expl_u["ds_idx"].isin(idx_u)].copy()
    sf = expl_f[expl_f["ds_idx"].isin(idx_f)].copy()

    # Build a compact means table with metrics as rows and runs as columns
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

    # Add deletion deltas (adv - clean)
    means.loc["del_auc_gc_delta"] = (
        means.loc["del_auc_gc_adv"] - means.loc["del_auc_gc_clean"]
    )
    means.loc["del_auc_ig_delta"] = (
        means.loc["del_auc_ig_adv"] - means.loc["del_auc_ig_clean"]
    )
    means.loc["del_auc_rise_delta"] = (
        means.loc["del_auc_rise_adv"] - means.loc["del_auc_rise_clean"]
    )

    plot_success_rates()
    plot_metric_bars(
        "Explanation stability (IoU@10) on successful attacks",
        means,
        IOU_COLS,
        "compare_10k_iou10.png",
    )
    plot_metric_bars(
        "Explanation stability (Spearman ρ) on successful attacks",
        means,
        RHO_COLS,
        "compare_10k_spearman.png",
    )
    plot_metric_bars(
        "Faithfulness shift (Deletion AUC Δ = adv − clean) on successful attacks",
        means,
        DELTA_COLS,
        "compare_10k_deletion_delta.png",
    )


if __name__ == "__main__":
    main()
