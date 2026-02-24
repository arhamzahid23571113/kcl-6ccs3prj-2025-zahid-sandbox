#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

FINAL_DIR = RESULTS / "final_pack"
FIG_DIR = FINAL_DIR / "figures"
TAB_DIR = FINAL_DIR / "tables"

PLOTS_DIR = RESULTS / "plots"
ATTACKS_DIR = RESULTS / "attacks"
EXPL_DIR = RESULTS / "explanations"
LOGS_DIR = RESULTS / "logs"


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[warn] missing: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] failed reading {path}: {e}")
        return None


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> list[Path]:
    copied: list[Path] = []
    if not src_dir.exists():
        return copied
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.glob(pattern)):
        out = dst_dir / p.name
        shutil.copy2(p, out)
        copied.append(out)
    return copied


def _write_csv_and_md(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md = tabulate(df, headers="keys", tablefmt="github", showindex=False)
    md_path.write_text(md + "\n", encoding="utf-8")


def _render_csv_as_md(csv_path: Path, md_path: Path) -> bool:
    df = _safe_read_csv(csv_path)
    if df is None or df.empty:
        return False
    md = tabulate(df, headers="keys", tablefmt="github", showindex=False)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md + "\n", encoding="utf-8")
    return True


def _pick_image_id_col(df: pd.DataFrame) -> str | None:
    prefs = ["image_id", "idx", "index", "id"]
    cols = {c.lower(): c for c in df.columns}
    for p in prefs:
        if p in cols:
            return cols[p]
    return None


def _pick_cost_col(df: pd.DataFrame) -> str | None:
    prefs = [
        "queries",
        "n_queries",
        "evals",
        "n_evals",
        "num_evals",
        "model_calls",
        "calls",
        "iters_used",
        "iters",
        "gens_used",
        "gens",
    ]
    cols = {c.lower(): c for c in df.columns}
    for p in prefs:
        if p in cols:
            return cols[p]
    return None


def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) != 0
    sl = s.astype(str).str.lower().fillna("")
    return sl.isin(["1", "true", "t", "yes", "y"]) | sl.str.contains(r"\btrue\b")


def _parse_hit_mask_cell(x: object) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip().lower()
    if "true" in s:
        return True
    return re.search(r"(?<!\d)1(?!\d)", s) is not None


def _infer_row_success(df: pd.DataFrame) -> pd.Series:
    cols = {c.lower(): c for c in df.columns}

    for cand in ["success", "is_success", "fooled", "is_fooled", "hit", "is_hit"]:
        if cand in cols:
            return _as_bool_series(df[cols[cand]])

    for cand in [
        "n_hit",
        "n_hits",
        "hit_count",
        "hits",
        "n_success",
        "n_success_targets",
    ]:
        if cand in cols:
            s = pd.to_numeric(df[cols[cand]], errors="coerce").fillna(0).astype(float)
            return s > 0

    if "hit_mask" in cols:
        return df[cols["hit_mask"]].apply(_parse_hit_mask_cell)

    pred_before = None
    pred_after = None
    for cand in ["pred_before", "y_pred_before", "pred0", "pred_clean"]:
        if cand in cols:
            pred_before = cols[cand]
            break
    for cand in ["pred_after", "y_pred_after", "pred1", "pred_adv", "adv_pred"]:
        if cand in cols:
            pred_after = cols[cand]
            break
    if pred_before and pred_after:
        pb = pd.to_numeric(df[pred_before], errors="coerce")
        pa = pd.to_numeric(df[pred_after], errors="coerce")
        return pb.notna() & pa.notna() & (pb.astype(int) != pa.astype(int))

    return pd.Series([False] * len(df))


@dataclass
class AttackSummaryRow:
    run: str
    mode: str
    n_rows: int
    n_images: int
    n_success_images: int
    success_rate_images: float
    cost_col: str | None
    mean_cost_success: float | None
    median_cost_success: float | None
    mean_cost_all: float | None
    median_cost_all: float | None
    notes: str | None = None


def summarize_attack_csv(path: Path, run_name: str) -> AttackSummaryRow | None:
    df = _safe_read_csv(path)
    if df is None or df.empty:
        return None

    img_col = _pick_image_id_col(df)
    cost_col = _pick_cost_col(df)

    if img_col is None:
        df["_image_id_tmp"] = np.arange(len(df))
        img_col = "_image_id_tmp"

    df["_row_success"] = _infer_row_success(df)

    any_success_by_image = df.groupby(img_col, dropna=False)["_row_success"].any()
    n_images = int(any_success_by_image.shape[0])
    n_success_images = int(any_success_by_image.sum())
    sr = float(n_success_images / n_images) if n_images else 0.0

    mean_cost_success = median_cost_success = mean_cost_all = median_cost_all = None
    if cost_col is not None:
        cost = pd.to_numeric(df[cost_col], errors="coerce")
        if cost.notna().sum() > 0:
            mean_cost_all = float(cost.mean())
            median_cost_all = float(cost.median())
            succ_cost = cost[df["_row_success"]]
            if succ_cost.notna().sum() > 0:
                mean_cost_success = float(succ_cost.mean())
                median_cost_success = float(succ_cost.median())

    return AttackSummaryRow(
        run=run_name,
        mode="untargeted",
        n_rows=int(len(df)),
        n_images=n_images,
        n_success_images=n_success_images,
        success_rate_images=sr,
        cost_col=cost_col,
        mean_cost_success=mean_cost_success,
        median_cost_success=median_cost_success,
        mean_cost_all=mean_cost_all,
        median_cost_all=median_cost_all,
        notes=None,
    )


def rise_guided_from_rq2_metrics(rq2_csv: Path) -> AttackSummaryRow | None:
    df = _safe_read_csv(rq2_csv)
    if df is None or df.empty:
        return None

    cols = {c.lower(): c for c in df.columns}
    if "run" not in cols:
        return None

    # Row label is "rise-guided" in your table.
    run_col = cols["run"]
    row = df[
        df[run_col]
        .astype(str)
        .str.lower()
        .isin(["rise-guided", "rise_guided", "riseguided"])
    ].copy()
    if row.empty:
        return None
    row = row.iloc[0]

    def _get(name: str) -> float | None:
        if name not in cols:
            return None
        v = pd.to_numeric(pd.Series([row[cols[name]]]), errors="coerce").iloc[0]
        return None if pd.isna(v) else float(v)

    denom = _get("denom_images")
    succ = _get("succ_images")
    asr = _get("asr_image")
    q_med_all = _get("queries_median_all")
    q_med_succ = _get("queries_median_succ")

    if denom is None or succ is None or asr is None:
        return None

    return AttackSummaryRow(
        run="rise-guided",
        mode="untargeted",
        n_rows=int(denom),
        n_images=int(denom),
        n_success_images=int(succ),
        success_rate_images=float(asr),
        cost_col="queries",
        mean_cost_success=None,
        median_cost_success=q_med_succ,
        mean_cost_all=None,
        median_cost_all=q_med_all,
        notes="derived from rq2_metrics_table (medians only)",
    )


def _plot_bar(
    values: dict[str, float], out_path: Path, title: str, ylabel: str
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(values.keys())
    ys = [values[k] for k in labels]
    plt.figure()
    plt.bar(labels, ys)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> int:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[str] = []
    manifest.append("# Final Results Pack\n")
    manifest.append("Frozen, report-ready figures/tables.\n")

    # Copy canonical artifacts
    copied: list[Path] = []
    canonical = [
        EXPL_DIR / "summary_compare_10000_untargeted_fastprior_rise_guided_gpu_v4.md",
        EXPL_DIR / "rq2_metrics_table.csv",
        EXPL_DIR / "story_rq2_guidance_tradeoff.md",
        RESULTS / "experiments_gpu_10k_overview.md",
        ATTACKS_DIR / "de1px_untargeted_10000.csv",
        ATTACKS_DIR / "de1px_targeted_fastprior_10000.csv",
        LOGS_DIR / "de1px_rise_guided_10000.log",
        LOGS_DIR / "explanations_10000_rise_guided_gpu.log",
    ]
    for f in canonical:
        dst = FINAL_DIR / f.name
        if _copy_if_exists(f, dst):
            copied.append(dst)

    manifest.append("\n## Copied canonical artifacts\n")
    for p in copied:
        manifest.append(f"- `{p.relative_to(ROOT)}`\n")

    # Copy plots
    copied_plots = _copy_glob(PLOTS_DIR, "rq2_*.png", FIG_DIR)
    manifest.append("\n## Copied plots\n")
    if copied_plots:
        for p in copied_plots:
            manifest.append(f"- `{p.relative_to(ROOT)}`\n")
    else:
        manifest.append("- (no rq2_*.png found)\n")

    # Render canonical RQ2 metrics table to MD
    rq2_csv = EXPL_DIR / "rq2_metrics_table.csv"
    if rq2_csv.exists():
        _copy_if_exists(rq2_csv, FINAL_DIR / rq2_csv.name)
        _render_csv_as_md(rq2_csv, TAB_DIR / "rq2_metrics_table.md")
        manifest.append(
            f"\n- `{(TAB_DIR / 'rq2_metrics_table.md').relative_to(ROOT)}`\n"
        )

    # Attack summary: untargeted + fastprior from their attack CSVs; rise-guided from rq2_metrics_table
    rows: list[AttackSummaryRow] = []
    r1 = summarize_attack_csv(
        ATTACKS_DIR / "de1px_untargeted_10000.csv", run_name="untargeted"
    )
    if r1:
        rows.append(r1)
    r2 = summarize_attack_csv(
        ATTACKS_DIR / "de1px_targeted_fastprior_10000.csv", run_name="fastprior"
    )
    if r2:
        rows.append(r2)

    rg = rise_guided_from_rq2_metrics(rq2_csv)
    if rg:
        rows.append(rg)

    if rows:
        adf = pd.DataFrame([r.__dict__ for r in rows])
        cols = [
            "run",
            "mode",
            "n_rows",
            "n_images",
            "n_success_images",
            "success_rate_images",
            "cost_col",
            "mean_cost_success",
            "median_cost_success",
            "mean_cost_all",
            "median_cost_all",
            "notes",
        ]
        adf = adf[[c for c in cols if c in adf.columns]]
        _write_csv_and_md(
            adf, TAB_DIR / "attack_summary.csv", TAB_DIR / "attack_summary.md"
        )
        manifest.append(f"- `{(TAB_DIR / 'attack_summary.md').relative_to(ROOT)}`\n")

        try:
            sr = {row.run: float(row.success_rate_images) for row in rows}
            _plot_bar(
                sr,
                FIG_DIR / "asr_by_run.png",
                "Attack success rate (image-wise)",
                "ASR",
            )
            manifest.append(f"- `{(FIG_DIR / 'asr_by_run.png').relative_to(ROOT)}`\n")
        except Exception as e:
            print(f"[warn] could not plot ASR bar: {e}")

    # Claims → evidence map
    claims: list[str] = []
    claims.append("# Claims → Evidence Map\n")
    claims.append("Each claim links to a frozen table/figure in this pack.\n")

    claims.append("\n## RQ2 — Guidance trade-off\n")
    claims.append("- **Claim:** Guidance changes ASR vs cost trade-off.\n")
    claims.append(
        f"  - Evidence: `{(TAB_DIR / 'rq2_metrics_table.md').relative_to(ROOT)}` and `results/final_pack/figures/rq2_*.png`.\n"
    )
    claims.append(
        "- **Claim:** Guidance changes explanation stability metrics among successful adversarials.\n"
    )
    claims.append(
        f"  - Evidence: `{(TAB_DIR / 'rq2_metrics_table.md').relative_to(ROOT)}` and `{(FINAL_DIR / 'summary_compare_10000_untargeted_fastprior_rise_guided_gpu_v4.md').relative_to(ROOT)}`.\n"
    )

    claims.append(
        "\n## RQ1 — Explanation stability under successful one-pixel attacks\n"
    )
    claims.append(
        "- **Claim:** Explanation stability changes under successful one-pixel perturbations (method-dependent).\n"
    )
    claims.append(
        "  - Evidence: `results/explanations/summary_before_after_10000_untargeted_gpu.md` (and corresponding plots you choose for the report).\n"
    )

    (FINAL_DIR / "CLAIMS_EVIDENCE.md").write_text("".join(claims), encoding="utf-8")
    (FINAL_DIR / "README.md").write_text("".join(manifest), encoding="utf-8")

    print(f"[ok] wrote final pack to: {FINAL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
