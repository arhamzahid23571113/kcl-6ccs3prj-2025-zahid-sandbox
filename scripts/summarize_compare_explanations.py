#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPL_DIR = ROOT / "results" / "explanations"

KNOWN_METRIC_COLS = [
    "iou10_gc",
    "iou10_ig",
    "iou10_rise",
    "rho_gc",
    "rho_ig",
    "rho_rise",
    "del_auc_gc_clean",
    "del_auc_ig_clean",
    "del_auc_rise_clean",
    "del_auc_gc_adv",
    "del_auc_ig_adv",
    "del_auc_rise_adv",
]

IDX_CANDIDATES = [
    "ds_idx",
    "image_id",
    "idx",
    "index",
    "cifar_idx",
    "dataset_idx",
]


@dataclass(frozen=True)
class RunSpec:
    label: str
    expl_csv: Path
    attacks_csv: Optional[Path]


def _resolve(p: str) -> Path:
    return Path(p).expanduser().resolve()


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p.resolve())


def parse_run(s: str) -> RunSpec:
    parts = [p.strip() for p in s.split(":")]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("Expected: label:expl_csv[:attacks_csv]")
    label = parts[0]
    expl = _resolve(parts[1])
    attacks = _resolve(parts[2]) if len(parts) == 3 else None
    if not label:
        raise argparse.ArgumentTypeError("Label cannot be empty")
    if not expl.exists():
        raise argparse.ArgumentTypeError(f"Explanations CSV not found: {expl}")
    if attacks is not None and not attacks.exists():
        raise argparse.ArgumentTypeError(f"Attacks CSV not found: {attacks}")
    return RunSpec(label=label, expl_csv=expl, attacks_csv=attacks)


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {path}") from e


def pick_idx_col(df: pd.DataFrame) -> Optional[str]:
    for c in IDX_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if re.search(
            r"(ds|data|dataset)?_?idx$|^idx$|image_?id|^index$", c, re.IGNORECASE
        ):
            return c
    return None


def infer_attack_success_mask(df: pd.DataFrame) -> tuple[pd.Series, str]:
    # 1) direct success cols
    for col in ("success", "attack_success", "adv_success"):
        if col in df.columns:
            m = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int) == 1
            return m, f"success from `{col} == 1`"

    # 2) fastprior / multi-target encoding: hit_mask like "0|0|1|0|..."
    if "hit_mask" in df.columns:
        s = df["hit_mask"].astype(str)
        # match a standalone 1 between separators, to be safe
        m = s.str.contains(r"(?:^|\|)1(?:\||$)", na=False)
        return m, "success from `hit_mask` contains a 1"

    # 3) multi-target style: n_success_targets > 0
    for col in (
        "n_success_targets",
        "num_success_targets",
        "success_targets",
        "n_succ_targets",
    ):
        if col in df.columns:
            m = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
            return m, f"success from `{col} > 0`"

    # 4) any_success / image_success style
    for col in ("any_success", "image_success", "has_success"):
        if col in df.columns:
            v = df[col]
            if v.dtype == bool:
                return v, f"success from `{col}` (bool)"
            m = pd.to_numeric(v, errors="coerce").fillna(0).astype(int) == 1
            return m, f"success from `{col} == 1`"

    # 5) prediction change fallback
    if "pred_before" in df.columns and "pred_after" in df.columns:
        m = df["pred_before"].astype(str) != df["pred_after"].astype(str)
        return m, "success from `pred_before != pred_after`"
    if "pred_clean" in df.columns and "pred_adv" in df.columns:
        m = df["pred_clean"].astype(str) != df["pred_adv"].astype(str)
        return m, "success from `pred_clean != pred_adv`"

    m = pd.Series([False] * len(df), index=df.index)
    return m, "could not infer success (no known columns)"


def pick_query_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("queries", "iters_used", "num_queries", "nfev", "evals"):
        if c in df.columns:
            return c
    return None


def qstats(x: pd.Series) -> dict[str, float]:
    v = pd.to_numeric(x, errors="coerce").dropna()
    if len(v) == 0:
        return {}
    return {
        "median": float(np.median(v)),
        "p90": float(np.quantile(v, 0.90)),
        "mean": float(np.mean(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def summarize_attacks(label: str, df: pd.DataFrame) -> dict[str, object]:
    succ, succ_rule = infer_attack_success_mask(df)
    out: dict[str, object] = {
        "run": label,
        "n_rows": int(len(df)),
        "n_success": int(succ.sum()),
        "success_rate": float(succ.mean()) if len(df) else None,
        "success_rule": succ_rule,
    }
    qcol = pick_query_col(df)
    out["query_col"] = qcol
    if qcol is not None:
        out["queries_all"] = qstats(df[qcol])
        out["queries_success"] = qstats(df.loc[succ, qcol]) if succ.any() else {}
    else:
        out["queries_all"] = {}
        out["queries_success"] = {}
    return out


def summarize_explanations(
    label: str, df_expl: pd.DataFrame, success_idx: Optional[set]
) -> dict[str, object]:
    idx_col = pick_idx_col(df_expl)

    if success_idx is not None and idx_col is not None:
        mask = df_expl[idx_col].isin(success_idx)
        rule = f"success rows from attack CSV via `{idx_col}` join"
    else:
        if "pred_clean" in df_expl.columns and "pred_adv" in df_expl.columns:
            mask = df_expl["pred_clean"].astype(str) != df_expl["pred_adv"].astype(str)
            rule = "fallback: success from `pred_clean != pred_adv`"
        else:
            mask = pd.Series([False] * len(df_expl), index=df_expl.index)
            rule = "fallback: could not infer success in explanations"
    df_s = df_expl.loc[mask].copy()

    out: dict[str, object] = {
        "run": label,
        "n_total": int(len(df_expl)),
        "n_adv": int(len(df_s)),
        "success_rule": rule,
    }

    for col in KNOWN_METRIC_COLS:
        if col in df_s.columns:
            v = pd.to_numeric(df_s[col], errors="coerce").dropna()
            if len(v):
                out[col] = float(v.mean())

    for m in ("gc", "ig", "rise"):
        c_clean = f"del_auc_{m}_clean"
        c_adv = f"del_auc_{m}_adv"
        if c_clean in df_s.columns and c_adv in df_s.columns:
            a = pd.to_numeric(df_s[c_adv], errors="coerce")
            b = pd.to_numeric(df_s[c_clean], errors="coerce")
            d = (a - b).dropna()
            if len(d):
                out[f"del_auc_{m}_delta"] = float(d.mean())

    return out


def to_md_table(rows: list[dict[str, object]], headers: list[str]) -> str:
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        if v is None:
            return ""
        return str(v)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(h)) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare explanation runs (filter explanations by attack success when provided)."
    )
    ap.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="label:expl_csv[:attacks_csv]",
    )
    ap.add_argument("--out", type=str, default=None, help="Markdown output path")
    args = ap.parse_args()

    runs: list[RunSpec] = args.run
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    expl_rows: list[dict[str, object]] = []
    atk_rows: list[dict[str, object]] = []

    for r in runs:
        df_expl = read_csv(r.expl_csv)

        success_idx: Optional[set] = None
        if r.attacks_csv is not None:
            df_atk = read_csv(r.attacks_csv)
            atk_rows.append(summarize_attacks(r.label, df_atk))

            atk_idx_col = pick_idx_col(df_atk)
            succ_mask, _ = infer_attack_success_mask(df_atk)
            if atk_idx_col is not None:
                success_idx = set(df_atk.loc[succ_mask, atk_idx_col].tolist())

        expl_rows.append(summarize_explanations(r.label, df_expl, success_idx))

    md: list[str] = []
    md.append("# Explanation comparison")
    md.append("")
    md.append(f"_Generated: {now}_")
    md.append("")
    md.append("## Runs")
    md.append("")
    for r in runs:
        md.append(f"- **{r.label}**")
        md.append(f"  - explanations: `{_rel(r.expl_csv)}`")
        if r.attacks_csv is not None:
            md.append(f"  - attacks: `{_rel(r.attacks_csv)}`")
    md.append("")

    base_headers = [
        "run",
        "n_total",
        "n_adv",
        "iou10_gc",
        "iou10_ig",
        "iou10_rise",
        "rho_gc",
        "rho_ig",
        "rho_rise",
        "del_auc_gc_clean",
        "del_auc_gc_adv",
        "del_auc_gc_delta",
        "del_auc_ig_clean",
        "del_auc_ig_adv",
        "del_auc_ig_delta",
        "del_auc_rise_clean",
        "del_auc_rise_adv",
        "del_auc_rise_delta",
        "success_rule",
    ]
    present = {k for row in expl_rows for k in row.keys()}
    expl_headers = [h for h in base_headers if h in present]

    md.append("## Aggregate explanation metrics (means over successful attacks)")
    md.append("")
    md.append(to_md_table(expl_rows, expl_headers))
    md.append("")

    if atk_rows:
        atk_headers = [
            "run",
            "n_rows",
            "n_success",
            "success_rate",
            "query_col",
            "success_rule",
        ]
        md.append("## Attack CSV summary")
        md.append("")
        md.append(to_md_table(atk_rows, atk_headers))
        md.append("")
        for row in atk_rows:
            qc = row.get("query_col")
            if qc:
                qa = row.get("queries_all", {})
                qs = row.get("queries_success", {})
                md.append(f"### {row.get('run')}: `{qc}` stats")
                if qa:
                    md.append(
                        f"- all: median={qa.get('median'):.2f}, p90={qa.get('p90'):.2f}, mean={qa.get('mean'):.2f}, min={qa.get('min'):.2f}, max={qa.get('max'):.2f}"
                    )
                if qs:
                    md.append(
                        f"- success: median={qs.get('median'):.2f}, p90={qs.get('p90'):.2f}, mean={qs.get('mean'):.2f}, min={qs.get('min'):.2f}, max={qs.get('max'):.2f}"
                    )
                md.append("")

    out_path = (
        _resolve(args.out) if args.out else (DEFAULT_EXPL_DIR / "summary_compare.md")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
