# scripts/summarize_explanations.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _mm(series: pd.Series) -> tuple[float, float]:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return float(np.nanmean(arr)), float(np.nanmedian(arr))


def main():  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--expl", required=True, help="explanations CSV (before_after)")
    ap.add_argument("--adv-dir", default=None, help="dir containing {ds_idx}.pt")
    ap.add_argument("--attacks", default=None, help="de1px CSV with success column")
    ap.add_argument("--out", default=None, help="optional markdown output path")
    args = ap.parse_args()

    df = pd.read_csv(args.expl)
    n_total = len(df)
    has_adv = df["pred_adv"].notna()
    n_adv = int(has_adv.sum())

    adv_files = None
    if args.adv_dir:
        adv_files = len(list(Path(args.adv_dir).glob("*.pt")))

    metrics: dict[str, dict[str, float]] = {}
    for m in ["iou10_gc", "iou10_ig", "iou10_rise", "rho_gc", "rho_ig", "rho_rise"]:
        mean, med = _mm(df.loc[has_adv, m])
        metrics[m] = {"mean": mean, "median": med}

    deltas: dict[str, dict[str, float]] = {}
    for tag in ["gc", "ig", "rise"]:
        c_mean, c_med = _mm(df[f"del_auc_{tag}_clean"])
        a_mean, a_med = _mm(df.loc[has_adv, f"del_auc_{tag}_adv"])
        delta = pd.to_numeric(
            df.loc[has_adv, f"del_auc_{tag}_adv"], errors="coerce"
        ) - pd.to_numeric(df.loc[has_adv, f"del_auc_{tag}_clean"], errors="coerce")
        d_mean = float(np.nanmean(delta.to_numpy()))
        d_med = float(np.nanmedian(delta.to_numpy()))
        deltas[tag] = {
            "clean_mean": c_mean,
            "clean_median": c_med,
            "adv_mean": a_mean,
            "adv_median": a_med,
            "delta_mean": d_mean,
            "delta_median": d_med,
        }

    success_rate = None
    agree_advcount = None
    if args.attacks and Path(args.attacks).exists():
        atk = pd.read_csv(args.attacks)
        if "success" in atk.columns:
            success_rate = float(pd.to_numeric(atk["success"], errors="coerce").mean())
            if "ds_idx" in atk.columns:
                succ = atk.loc[pd.to_numeric(atk["success"], errors="coerce") == 1, "ds_idx"]
                succ = pd.to_numeric(succ, errors="coerce").dropna().astype(int)
                expl_has = set(df.loc[has_adv, "ds_idx"].astype(int).tolist())
                agree_advcount = int(sum(int(s) in expl_has for s in succ))

    md: list[str] = []
    md.append("# Explanations summary\n")
    md.append(f"- Total rows: **{n_total}**")
    md.append(f"- Rows with adversarial present: **{n_adv}**")
    if adv_files is not None:
        md.append(f"- Adversarial tensor files: **{adv_files}**")
    if success_rate is not None:
        md.append(f"- Attack success rate (from DE CSV): **{success_rate*100:.1f}%**")
    if agree_advcount is not None:
        md.append(f"- Successful attacks with matching adv row: **{agree_advcount}**")
    md.append("\n## IoU@k & Spearman (adv-only)\n")
    for m, vals in metrics.items():
        md.append(f"- **{m}**: mean **{vals['mean']:.3f}**, median **{vals['median']:.3f}**")
    md.append("\n## Deletion AUC\n")
    for tag, vals in deltas.items():
        md.append(
            f"- **{tag.upper()}**: clean mean **{vals['clean_mean']:.3f}**, "
            f"adv mean **{vals['adv_mean']:.3f}**, Δ (adv-clean) mean **{vals['delta_mean']:.3f}** "
            f"(medians: clean **{vals['clean_median']:.3f}**, adv **{vals['adv_median']:.3f}**, "
            f"Δ **{vals['delta_median']:.3f}**)"
        )
    out_text = "\n".join(md) + "\n"
    print(out_text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_text)


if __name__ == "__main__":
    main()
