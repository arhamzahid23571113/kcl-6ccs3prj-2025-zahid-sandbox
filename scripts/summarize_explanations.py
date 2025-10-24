# scripts/summarize_explanations.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _num(s: pd.Series) -> pd.Series:
    """Convert a Series to numeric, coercing errors to NaN."""
    return pd.to_numeric(s, errors="coerce")


def _mm(series: pd.Series) -> Tuple[float, float]:
    """Mean/median with NaN tolerance (returns np.nan on empty)."""
    arr = _num(series).to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanmedian(arr))


def _fmt(x: float) -> str:
    """Format floats; show em dash for NaN."""
    return "—" if (x is None or np.isnan(x)) else f"{x:.3f}"


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(
        description="Summarize before/after explanation metrics"
    )
    ap.add_argument("--expl", required=True, help="Explanations CSV (before_after)")
    ap.add_argument("--adv-dir", default=None, help="Directory containing {ds_idx}.pt")
    ap.add_argument("--attacks", default=None, help="DE 1px CSV (with 'success' col)")
    ap.add_argument("--out", default=None, help="Optional markdown output path")
    ap.add_argument("--seed", type=int, default=1337)  # reserved for future sampling
    args = ap.parse_args()

    df = pd.read_csv(args.expl)
    n_total = len(df)

    # Rows that actually have adversarial results recorded (pred_adv not NaN)
    has_adv = df["pred_adv"].notna()
    n_adv = int(has_adv.sum())

    adv_files = None
    if args.adv_dir:
        adv_files = len(list(Path(args.adv_dir).glob("*.pt")))

    # IoU / Spearman on adv-only rows (if any)
    metrics: dict[str, dict[str, float]] = {}
    adv_subset = df.loc[has_adv]
    for m in ["iou10_gc", "iou10_ig", "iou10_rise", "rho_gc", "rho_ig", "rho_rise"]:
        mean, med = _mm(adv_subset[m] if n_adv > 0 else pd.Series([], dtype=float))
        metrics[m] = {"mean": mean, "median": med}

    # Deletion AUC: clean (all rows) vs adv (adv-only) + delta (adv - clean per-row on adv subset)
    deltas: dict[str, dict[str, float]] = {}
    for tag in ["gc", "ig", "rise"]:
        c_mean, c_med = _mm(df[f"del_auc_{tag}_clean"])
        if n_adv > 0:
            a_mean, a_med = _mm(adv_subset[f"del_auc_{tag}_adv"])
            delta = _num(adv_subset[f"del_auc_{tag}_adv"]) - _num(
                adv_subset[f"del_auc_{tag}_clean"]
            )
            d_mean = float(np.nanmean(delta.to_numpy())) if len(delta) else float("nan")
            d_med = (
                float(np.nanmedian(delta.to_numpy())) if len(delta) else float("nan")
            )
        else:
            a_mean = a_med = d_mean = d_med = float("nan")

        deltas[tag] = {
            "clean_mean": c_mean,
            "clean_median": c_med,
            "adv_mean": a_mean,
            "adv_median": a_med,
            "delta_mean": d_mean,
            "delta_median": d_med,
        }

    # Optional: success rate and overlap agreement from attack CSV
    success_rate = None
    agree_advcount = None
    if args.attacks and Path(args.attacks).exists():
        atk = pd.read_csv(args.attacks)
        if "success" in atk.columns:
            success_rate = float(_num(atk["success"]).mean())
            if "ds_idx" in atk.columns:
                succ = atk.loc[_num(atk["success"]) == 1, "ds_idx"]
                succ = _num(succ).dropna().astype(int)
                expl_has = set(df.loc[has_adv, "ds_idx"].astype(int).tolist())
                agree_advcount = int(sum(int(s) in expl_has for s in succ))

    # Build markdown
    md: list[str] = []
    md.append("# Explanations summary\n")
    md.append(f"- Total rows: **{n_total}**")
    md.append(f"- Rows with adversarial present: **{n_adv}**")
    if adv_files is not None:
        md.append(f"- Adversarial tensor files: **{adv_files}**")
    if success_rate is not None and not np.isnan(success_rate):
        md.append(f"- Attack success rate (from DE CSV): **{success_rate*100:.1f}%**")
    if agree_advcount is not None:
        md.append(f"- Successful attacks with matching adv row: **{agree_advcount}**")

    md.append("\n## IoU@k & Spearman (adv-only)\n")
    for m, vals in metrics.items():
        md.append(
            f"- **{m}**: mean **{_fmt(vals['mean'])}**, median **{_fmt(vals['median'])}**"
        )

    md.append("\n## Deletion AUC\n")
    for tag, vals in deltas.items():
        md.append(
            f"- **{tag.upper()}**: clean mean **{_fmt(vals['clean_mean'])}**, "
            f"adv mean **{_fmt(vals['adv_mean'])}**, Δ (adv−clean) mean **{_fmt(vals['delta_mean'])}** "
            f"(medians: clean **{_fmt(vals['clean_median'])}**, adv **{_fmt(vals['adv_median'])}**, "
            f"Δ **{_fmt(vals['delta_median'])}**)"
        )

    out_text = "\n".join(md) + "\n"
    print(out_text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_text)


if __name__ == "__main__":
    main()
