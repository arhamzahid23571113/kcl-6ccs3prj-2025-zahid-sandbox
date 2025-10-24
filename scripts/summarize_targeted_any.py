# scripts/summarize_targeted_any.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(x: float) -> str:
    return "—" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize targeted sweep with 'at least one target' metric"
    )
    ap.add_argument(
        "--csv", required=True, help="CSV from run_one_pixel_de_targeted_sweep.py"
    )
    ap.add_argument(
        "--out", default=None, help="Optional path to write markdown summary"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Per-image metric: success if ANY target succeeded
    any_success = df.groupby("ds_idx")["success"].max().astype(bool)
    overall = float(any_success.mean()) if len(any_success) else float("nan")

    # Per-target label success rate (among attempted)
    per_tgt = (
        df.groupby("target_label")["success"].mean().sort_index()
        if "target_label" in df
        else pd.Series(dtype=float)
    )

    # Attempts per image
    attempts = df.groupby("ds_idx")["target_label"].nunique()
    attempts_desc = attempts.describe() if len(attempts) else pd.Series(dtype=float)

    lines: list[str] = []
    lines.append("# Targeted sweep summary\n")
    lines.append(f"- CSV: `{args.csv}`")
    lines.append(f"- Images covered: **{len(any_success)}**")
    if len(attempts_desc):
        lines.append(
            f"- Attempts per image (unique targets): mean **{attempts_desc['mean']:.2f}**, "
            f"min **{attempts_desc['min']:.0f}**, max **{attempts_desc['max']:.0f}**"
        )
    lines.append(f"- **'At least one target' success rate:** **{_fmt(overall*100)}**\n")

    lines.append("## Per-target success rate\n")
    if len(per_tgt):
        for tgt, rate in per_tgt.items():
            lines.append(f"- target `{int(tgt)}`: **{_fmt(rate*100)}**")
    else:
        lines.append("_No per-target results available._")

    md = "\n".join(lines) + "\n"
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)


if __name__ == "__main__":
    main()
