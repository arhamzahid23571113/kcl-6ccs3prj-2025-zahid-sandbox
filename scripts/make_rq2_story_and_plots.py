#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = [
    (
        "untargeted",
        "results/explanations/before_after_10000_untargeted_gpu.csv",
        "results/attacks/de1px_untargeted_10000.csv",
    ),
    (
        "fastprior",
        "results/explanations/before_after_10000_fastprior_gpu.csv",
        "results/attacks/de1px_targeted_fastprior_10000.csv",
    ),
    (
        "rise-guided",
        "results/explanations/before_after_10000_rise_guided_gpu.csv",
        "results/attacks/de1px_rise_guided_10000.csv",
    ),
]

OUT_DIR_PLOTS = Path("results/plots")
OUT_DIR_EXPL = Path("results/explanations")
OUT_DIR_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_DIR_EXPL.mkdir(parents=True, exist_ok=True)

KEY_CANDIDATES = ["ds_idx", "image_id", "idx", "index", "i", "image_idx"]


def pick_key(df: pd.DataFrame) -> str:
    for c in KEY_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]


def adv_mask_from_expl(df: pd.DataFrame) -> pd.Series:
    adv_cols = [c for c in df.columns if c.endswith("_adv")]
    if adv_cols:
        return df[adv_cols].notna().any(axis=1)
    if "pred_adv" in df.columns:
        return df["pred_adv"].notna()
    return pd.Series([False] * len(df), index=df.index)


def parse_hit_mask_cell(x) -> list[int]:
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in x]
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []
    for fn in (json.loads, ast.literal_eval):
        try:
            v = fn(s)
            if isinstance(v, (list, tuple)):
                return [int(float(t)) for t in v]
        except Exception:
            pass
    if all(ch in "01" for ch in s) and len(s) > 1:
        return [int(ch) for ch in s]
    s2 = s.strip("[](){} ")
    parts = [
        p for p in s2.replace("|", ",").replace(";", ",").split(",") if p.strip() != ""
    ]
    out = []
    for p in parts:
        t = p.strip().lower()
        if t in ("true", "t", "yes", "y"):
            out.append(1)
        elif t in ("false", "f", "no", "n"):
            out.append(0)
        else:
            try:
                out.append(int(float(t)))
            except Exception:
                pass
    return out


def compute_asr_and_cost(attacks_csv: str) -> dict:
    a = pd.read_csv(attacks_csv)
    key = pick_key(a)

    def safe_num(s):
        return pd.to_numeric(s, errors="coerce")

    cost_cols = [
        c
        for c in a.columns
        if c.lower()
        in ("queries", "gens_used", "iters_used", "steps_used", "nfev", "fevals")
    ]

    if "success" in a.columns:
        succ = safe_num(a["success"]).fillna(0) > 0
        denom = a[key].nunique()
        succ_images = int(a.loc[succ, key].nunique())
        out = {
            "attack_mode": "numeric_success",
            "denom_images": int(denom),
            "succ_images": int(succ_images),
            "asr_image": float(succ_images / denom) if denom else 0.0,
        }
        for c in cost_cols:
            v = safe_num(a[c])
            out[f"{c}_median_all"] = float(v.median()) if v.notna().any() else np.nan
            out[f"{c}_median_succ"] = (
                float(v[succ].median()) if v[succ].notna().any() else np.nan
            )
        return out

    if "hit_mask" in a.columns:
        masks = a["hit_mask"].apply(parse_hit_mask_cell)
        img_hit = masks.apply(lambda m: int(any(int(x) > 0 for x in m)))
        denom = a[key].nunique()
        succ_images = int(img_hit.sum())

        total_hits = int(sum(sum(int(x) > 0 for x in m) for m in masks))
        total_targets = int(sum(len(m) for m in masks))
        target_asr = (total_hits / total_targets) if total_targets else 0.0

        out = {
            "attack_mode": "hit_mask",
            "denom_images": int(denom),
            "succ_images": int(succ_images),
            "asr_image": float(succ_images / denom) if denom else 0.0,
            "target_hits": int(total_hits),
            "target_total": int(total_targets),
            "asr_target": float(target_asr),
        }
        for c in cost_cols:
            v = safe_num(a[c])
            out[f"{c}_median_all"] = float(v.median()) if v.notna().any() else np.nan
            out[f"{c}_median_succ"] = (
                float(v[img_hit.astype(bool)].median())
                if v[img_hit.astype(bool)].notna().any()
                else np.nan
            )
        return out

    return {
        "attack_mode": "unknown",
        "denom_images": int(a[key].nunique()),
        "succ_images": np.nan,
        "asr_image": np.nan,
    }


def summarize_expl(expl_csv: str) -> dict:
    e = pd.read_csv(expl_csv)
    adv = adv_mask_from_expl(e)
    out = {"expl_rows": int(len(e)), "expl_adv_rows": int(adv.sum())}

    for m in ["iou10_gc", "iou10_ig", "iou10_rise", "rho_gc", "rho_ig", "rho_rise"]:
        if m in e.columns:
            v = pd.to_numeric(e.loc[adv, m], errors="coerce")
            out[f"{m}_mean_adv"] = float(v.mean())
            out[f"{m}_median_adv"] = float(v.median())

    for meth in ["gc", "ig", "rise"]:
        ccol = f"del_auc_{meth}_clean"
        acol = f"del_auc_{meth}_adv"
        if ccol in e.columns:
            vc = pd.to_numeric(e[ccol], errors="coerce")
            out[f"{ccol}_mean"] = float(vc.mean())
            out[f"{ccol}_median"] = float(vc.median())
        if acol in e.columns:
            va = pd.to_numeric(e.loc[adv, acol], errors="coerce")
            out[f"{acol}_mean_adv"] = float(va.mean())
            out[f"{acol}_median_adv"] = float(va.median())
        if ccol in e.columns and acol in e.columns:
            out[f"del_delta_{meth}_mean"] = out.get(
                f"{acol}_mean_adv", np.nan
            ) - out.get(f"{ccol}_mean", np.nan)
            out[f"del_delta_{meth}_median"] = out.get(
                f"{acol}_median_adv", np.nan
            ) - out.get(f"{ccol}_median", np.nan)
    return out


def bar_plot(title, labels, values, out_path, y_label=""):
    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.title(title)
    if y_label:
        plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def grouped_bar_plot(title, run_labels, series, out_path, y_label=""):
    plt.figure(figsize=(10, 5))
    x = np.arange(len(run_labels))
    n = len(series)
    width = 0.8 / max(n, 1)
    for i, (name, vals) in enumerate(series.items()):
        plt.bar(x + (i - (n - 1) / 2) * width, vals, width=width, label=name)
    plt.xticks(x, run_labels)
    plt.title(title)
    if y_label:
        plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    rows = []
    for name, expl_csv, attacks_csv in RUNS:
        a = compute_asr_and_cost(attacks_csv)
        e = summarize_expl(expl_csv)
        rows.append({"run": name, **a, **e})

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR_EXPL / "rq2_metrics_table.csv"
    df.to_csv(out_csv, index=False)

    run_labels = df["run"].tolist()

    bar_plot(
        "RQ2: Image-level attack success rate (ASR)",
        run_labels,
        (df["asr_image"] * 100).tolist(),
        OUT_DIR_PLOTS / "rq2_asr_image_percent.png",
        y_label="ASR (%)",
    )

    if "queries_median_succ" in df.columns:
        bar_plot(
            "RQ2: Median queries (successful images)",
            run_labels,
            df["queries_median_succ"].fillna(np.nan).tolist(),
            OUT_DIR_PLOTS / "rq2_queries_median_success.png",
            y_label="Queries (median, succ)",
        )

    groups = [
        (
            "IoU@10 (adv-only mean)",
            ["iou10_gc_mean_adv", "iou10_ig_mean_adv", "iou10_rise_mean_adv"],
            "rq2_iou10_adv_mean.png",
            "IoU@10 (mean)",
        ),
        (
            "Spearman rho (adv-only mean)",
            ["rho_gc_mean_adv", "rho_ig_mean_adv", "rho_rise_mean_adv"],
            "rq2_rho_adv_mean.png",
            "rho (mean)",
        ),
        (
            "Deletion AUC Δ mean (adv-clean)",
            ["del_delta_gc_mean", "del_delta_ig_mean", "del_delta_rise_mean"],
            "rq2_del_auc_delta_mean.png",
            "Δ AUC (mean)",
        ),
    ]

    for title, keys, fname, ylabel in groups:
        series = {}
        for k in keys:
            if k in df.columns:
                series[k.replace("_mean_adv", "").replace("del_delta_", "Δ_")] = df[
                    k
                ].tolist()
        if series:
            grouped_bar_plot(
                title, run_labels, series, OUT_DIR_PLOTS / fname, y_label=ylabel
            )

    story_path = OUT_DIR_EXPL / "story_rq2_guidance_tradeoff.md"
    with open(story_path, "w") as f:
        f.write("# RQ2 story: guidance vs untargeted vs fastprior (10k)\n\n")
        f.write("Auto-generated from attack CSVs + before/after explanation CSVs.\n\n")
        f.write("## Key headline\n")
        f.write(
            "- **Fastprior trades ASR for stability**: lower image-level ASR but higher explanation stability among successful adversarials.\n"
        )
        f.write(
            "- **RISE-guided keeps ASR ~the same as untargeted**, but can shift explanation stability (especially RISE-based stability).\n\n"
        )

        f.write("## Numbers (from `rq2_metrics_table.csv`)\n\n")
        show_cols = [
            "run",
            "asr_image",
            "succ_images",
            "denom_images",
            "queries_median_succ",
            "gens_used_median_succ",
            "expl_rows",
            "expl_adv_rows",
            "iou10_gc_mean_adv",
            "iou10_ig_mean_adv",
            "iou10_rise_mean_adv",
            "rho_gc_mean_adv",
            "rho_ig_mean_adv",
            "rho_rise_mean_adv",
            "del_delta_gc_mean",
            "del_delta_ig_mean",
            "del_delta_rise_mean",
            "asr_target",
            "target_hits",
            "target_total",
        ]
        cols = [c for c in show_cols if c in df.columns]
        view = df[cols].copy()
        if "asr_image" in view.columns:
            view["asr_image"] = (view["asr_image"] * 100).round(2)
        if "asr_target" in view.columns:
            view["asr_target"] = (view["asr_target"] * 100).round(3)
        f.write(view.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Artifacts\n")
        f.write("- Metrics table: `results/explanations/rq2_metrics_table.csv`\n")
        f.write("- Story: `results/explanations/story_rq2_guidance_tradeoff.md`\n")
        f.write("- Plots in `results/plots/` (rq2_*.png)\n")

    print("Wrote:")
    print(" -", out_csv)
    print(" -", story_path)
    print(" - plots in", OUT_DIR_PLOTS)


if __name__ == "__main__":
    main()
