from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXPL_DIR = ROOT / "results" / "explanations"

CSV_FILES = {
    "vanilla": EXPL_DIR / "before_after_200.csv",
    "guided": EXPL_DIR / "before_after_200_guided.csv",
    "fastprior": EXPL_DIR / "before_after_200_fastprior.csv",
}

METRIC_COLS = [
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


def load_attack_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {path}")
    return pd.read_csv(path)


def summarise_attack(name: str, df: pd.DataFrame) -> dict:
    # "Successful" attacks = prediction changed
    adv_mask = df["pred_clean"] != df["pred_adv"]

    out: dict[str, float | int | str] = {
        "attack": name,
        "n_total": int(len(df)),
        "n_adv": int(adv_mask.sum()),
    }

    # Compute means for each metric over successful attacks only
    adv_df = df.loc[adv_mask]
    for col in METRIC_COLS:
        if col in adv_df.columns:
            out[col] = float(adv_df[col].mean(skipna=True))
    return out


def to_markdown_table(rows: list[dict]) -> str:
    # Build a compact markdown table with key metrics
    headers = [
        "attack",
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
        "del_auc_rise_clean",
        "del_auc_rise_adv",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    def fmt(x):
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    for row in rows:
        line = []
        for h in headers:
            line.append(fmt(row.get(h, "")))
        lines.append("| " + " | ".join(line) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = []
    for name, path in CSV_FILES.items():
        df = load_attack_df(path)
        rows.append(summarise_attack(name, df))

    md_lines: list[str] = []
    md_lines.append("# Explanation comparison: vanilla vs guided vs fast-prior")
    md_lines.append("")
    md_lines.append(
        "This file summarises explanation metrics for three one-pixel attack variants "
        "over the 200-image CIFAR-10 subset."
    )
    md_lines.append("")
    md_lines.append(
        "- **vanilla**: standard DE-based one-pixel attack\n"
        "- **guided**: RISE-guided DE one-pixel attack\n"
        "- **fastprior**: multi-target DE with fast input-gradient prior (run on GPU)"
    )
    md_lines.append("")
    md_lines.append("## Aggregate metrics over successful attacks")
    md_lines.append("")
    md_lines.append(to_markdown_table(rows))
    md_lines.append("")

    out_path = EXPL_DIR / "summary_compare_v_g_f.md"
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Done. Wrote {out_path}")


if __name__ == "__main__":
    main()
