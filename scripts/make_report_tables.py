#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final_pack"
OUT_DIR = ROOT / "report" / "tables" / "final_pack"


def fmt_int(x):
    if pd.isna(x):
        return ""
    return f"{int(float(x)):,}"


def fmt_float(x, nd=3):
    if pd.isna(x):
        return ""
    return f"{float(x):.{nd}f}"


def write_tex(path: Path, tex: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")


def table_attack_summary() -> str:
    p = FINAL / "tables" / "attack_summary.csv"
    df = pd.read_csv(p)

    df_out = pd.DataFrame(
        {
            "Run": df["run"],
            "Eligible": df["n_images"].map(fmt_int),
            "Success": df["n_success_images"].map(fmt_int),
            "ASR": df["success_rate_images"].map(lambda x: fmt_float(x, 3)),
            "Median queries (succ)": df["median_cost_success"].map(fmt_int),
            "Median queries (all)": df["median_cost_all"].map(fmt_int),
        }
    )

    tab = df_out.to_latex(index=False, escape=False, column_format="lrrrrr")
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{RQ2 attack outcomes on the eligible CIFAR-10 subset (9{,}213 images): image-wise success rate (ASR) and query cost.}\n"
        "\\label{tab:rq2_attack_summary}\n"
        f"{tab}\n"
        "\\end{table}\n"
    )


def table_rq2_stability_means() -> str:
    p = FINAL / "rq2_metrics_table.csv"
    df = pd.read_csv(p)

    df["run"] = df["run"].astype(str)
    keep = (
        df["run"]
        .str.lower()
        .isin(["untargeted", "fastprior", "rise-guided", "rise_guided"])
    )
    df = df[keep].copy()

    cols = [
        ("IoU@10 GC", "iou10_gc_mean_adv"),
        ("IoU@10 IG", "iou10_ig_mean_adv"),
        ("IoU@10 RISE", "iou10_rise_mean_adv"),
        ("Spearman GC", "rho_gc_mean_adv"),
        ("Spearman IG", "rho_ig_mean_adv"),
        ("Spearman RISE", "rho_rise_mean_adv"),
    ]

    out = {"Run": df["run"]}
    for name, c in cols:
        out[name] = df[c].map(lambda x: fmt_float(x, 3)) if c in df.columns else ""

    df_out = pd.DataFrame(out)
    tab = df_out.to_latex(
        index=False, escape=False, column_format="l" + "r" * (len(df_out.columns) - 1)
    )

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        "\\begin{minipage}{\\linewidth}\n"
        "\\caption{RQ2 explanation stability on successful adversarial examples (means). Higher IoU@10 and Spearman indicate more stable explanations.}\n"
        "\\label{tab:rq2_stability_means}\n"
        f"{tab}\n"
        "\\end{minipage}}\n"
        "\\end{table}\n"
    )


def table_rq2_del_delta_means() -> str:
    p = FINAL / "rq2_metrics_table.csv"
    df = pd.read_csv(p)

    df["run"] = df["run"].astype(str)
    keep = (
        df["run"]
        .str.lower()
        .isin(["untargeted", "fastprior", "rise-guided", "rise_guided"])
    )
    df = df[keep].copy()

    cols = [
        (r"$\Delta$DelAUC GC", "del_delta_gc_mean"),
        (r"$\Delta$DelAUC IG", "del_delta_ig_mean"),
        (r"$\Delta$DelAUC RISE", "del_delta_rise_mean"),
    ]

    out = {"Run": df["run"]}
    for name, c in cols:
        out[name] = df[c].map(lambda x: fmt_float(x, 3)) if c in df.columns else ""

    df_out = pd.DataFrame(out)
    tab = df_out.to_latex(index=False, escape=False, column_format="lrrr")

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{RQ2 faithfulness shift under attack: mean change in deletion AUC (adversarial minus clean) on successful adversarial examples. More negative indicates a larger drop.}\n"
        "\\label{tab:rq2_del_delta}\n"
        f"{tab}\n"
        "\\end{table}\n"
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_tex(OUT_DIR / "rq2_attack_summary.tex", table_attack_summary())
    write_tex(OUT_DIR / "rq2_stability_means.tex", table_rq2_stability_means())
    write_tex(OUT_DIR / "rq2_del_delta_means.tex", table_rq2_del_delta_means())

    print(f"[ok] wrote tables to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
