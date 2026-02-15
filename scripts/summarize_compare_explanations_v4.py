#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY_CANDIDATES = ["ds_idx", "image_id", "idx", "index", "i", "image_idx"]
SUCCESS_CANDIDATES = ["success", "target_success", "succ", "is_success", "hit"]
HIT_MASK_CANDIDATES = ["hit_mask", "hits", "target_hits", "hitmask"]


def _load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Expected CSV not found: {p}")
    return pd.read_csv(p)


def _pick_key(df: pd.DataFrame) -> str:
    for c in KEY_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]


def _adv_mask(df: pd.DataFrame) -> pd.Series:
    for c in ["has_adv", "adv_present", "is_adv", "adv_exists"]:
        if c in df.columns:
            return df[c].astype(bool)

    for c in ["adv_path", "adv_file", "adv_tensor", "adv_filepath"]:
        if c in df.columns:
            s = df[c].astype(str)
            return df[c].notna() & (s != "") & (s != "nan")

    adv_cols = [c for c in df.columns if c.endswith("_adv") or c.startswith("adv_")]
    if adv_cols:
        return df[adv_cols].notna().any(axis=1)

    return pd.Series([False] * len(df), index=df.index)


def _count_pt(adv_dir: str | None) -> int | None:
    if not adv_dir:
        return None
    p = Path(adv_dir)
    if not p.exists():
        return None
    return sum(1 for _ in p.glob("*.pt"))


def _parse_hit_mask(x) -> list[int]:
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in list(x)]
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []

    # JSON / python literal list
    for fn in (json.loads, ast.literal_eval):
        try:
            v = fn(s)
            if isinstance(v, (list, tuple)):
                return [int(float(t)) for t in v]
        except Exception:
            pass

    # "0|1|0" or "0,1,0"
    s2 = s.strip("[](){} ")
    parts = [
        p for p in s2.replace("|", ",").replace(";", ",").split(",") if p.strip() != ""
    ]
    out: list[int] = []
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


def _pick_success_col(df: pd.DataFrame) -> str | None:
    for c in SUCCESS_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _pick_hitmask_col(df: pd.DataFrame) -> str | None:
    for c in HIT_MASK_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _image_level_success_from_attacks(att: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Return per-image success boolean Series indexed by image key, plus metadata."""
    img_col = _pick_key(att)
    meta = {"att_img_col": img_col, "succ_col": None, "mode": None}

    succ_col = _pick_success_col(att)
    if succ_col is not None:
        s = pd.to_numeric(att[succ_col], errors="coerce").fillna(0)
        per_img = (
            att.assign(_succ=(s > 0).astype(int))
            .groupby(img_col)["_succ"]
            .max()
            .astype(bool)
        )
        meta.update({"succ_col": succ_col, "mode": "numeric_success"})
        return per_img, meta

    hit_col = _pick_hitmask_col(att)
    if hit_col is not None:
        masks = att[hit_col].apply(_parse_hit_mask)
        per_row = masks.apply(lambda m: int(any(int(v) != 0 for v in m)))
        per_img = att.assign(_succ=per_row).groupby(img_col)["_succ"].max().astype(bool)

        # also compute target-wise stats
        total_hits = masks.apply(lambda m: sum(int(v) != 0 for v in m)).sum()
        total_targets = masks.apply(len).sum()
        meta.update(
            {
                "succ_col": hit_col,
                "mode": "hit_mask",
                "total_hits": int(total_hits),
                "total_targets": int(total_targets),
            }
        )
        return per_img, meta

    # No success info
    per_img = att.groupby(img_col).size()
    meta.update({"mode": "none"})
    return pd.Series([False] * len(per_img), index=per_img.index), meta


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _find_col(df: pd.DataFrame, must_contain: list[str]) -> str | None:
    cols = list(df.columns)
    low = {c: c.lower() for c in cols}
    for c in cols:
        name = low[c]
        if all(tok in name for tok in must_contain):
            return c
    return None


def _summ_iou_rho(df: pd.DataFrame, adv: pd.Series) -> dict:
    out = {}
    metrics = ["iou10_gc", "iou10_ig", "iou10_rise", "rho_gc", "rho_ig", "rho_rise"]
    adv_df = df.loc[adv]
    for m in metrics:
        if m in df.columns:
            v = _num(adv_df[m]).dropna()
            if len(v) > 0:
                out[m] = {"mean": float(v.mean()), "median": float(v.median())}
    return out


def _summ_del_auc(df: pd.DataFrame, adv: pd.Series) -> dict:
    out = {}
    for method in ["gc", "ig", "rise"]:
        clean_col = _find_col(df, ["del", "auc", method, "clean"]) or _find_col(
            df, ["del", "auc", method, "before"]
        )
        adv_col = _find_col(df, ["del", "auc", method, "adv"]) or _find_col(
            df, ["del", "auc", method, "after"]
        )
        if not clean_col or not adv_col:
            continue

        clean_all = _num(df[clean_col]).dropna()
        adv_only = _num(df.loc[adv, adv_col]).dropna()

        # delta per-adv-row where both exist
        a = _num(df.loc[adv, adv_col])
        c = _num(df.loc[adv, clean_col])
        delta = (a - c).dropna()

        if len(clean_all) == 0 or len(adv_only) == 0 or len(delta) == 0:
            continue

        out[method.upper()] = {
            "clean_mean": float(clean_all.mean()),
            "adv_mean": float(adv_only.mean()),
            "delta_mean": float(delta.mean()),
            "clean_med": float(clean_all.median()),
            "adv_med": float(adv_only.median()),
            "delta_med": float(delta.median()),
        }
    return out


def summarize_run(
    name: str, expl_path: str, attacks_path: str | None, adv_dir: str | None
) -> dict:
    df = _load_csv(expl_path)
    key = _pick_key(df)
    adv = _adv_mask(df)

    run = {
        "name": name,
        "expl_path": expl_path,
        "rows_total": int(len(df)),
        "rows_adv": int(adv.sum()),
        "key": key,
        "adv_files": _count_pt(adv_dir),
        "iou_rho": _summ_iou_rho(df, adv),
        "del_auc": _summ_del_auc(df, adv),
        "asr": None,
        "succ_images": None,
        "denom_images": None,
        "succ_match_advrow": None,
        "asr_meta": None,
    }

    if attacks_path:
        att = _load_csv(attacks_path)
        per_img_succ, meta = _image_level_success_from_attacks(att)
        run["asr_meta"] = meta
        run["succ_images"] = int(per_img_succ.sum())
        run["denom_images"] = int(per_img_succ.shape[0])
        run["asr"] = (
            float(run["succ_images"]) / float(run["denom_images"])
            if run["denom_images"]
            else None
        )

        # match “successful images” with “images that have an adv row in expl”
        expl_has_adv = df.assign(_adv=adv).groupby(key)["_adv"].max().astype(bool)
        common = per_img_succ.index.intersection(expl_has_adv.index)
        matched = (per_img_succ.loc[common] & expl_has_adv.loc[common]).sum()
        run["succ_match_advrow"] = int(matched)

    return run


def _fmt(x, nd=3):
    if x is None:
        return "NA"
    return f"{x:.{nd}f}"


def write_md(runs: list[dict], out_path: str):
    lines = ["# Compare explanation metrics (multi-run)\n"]
    for r in runs:
        lines.append(f"## {r['name']}")
        lines.append(f"- Explanations CSV: `{r['expl_path']}`")
        lines.append(f"- Total rows: **{r['rows_total']}**")
        lines.append(f"- Rows with adversarial present: **{r['rows_adv']}**")
        if r["adv_files"] is not None:
            lines.append(f"- Adversarial tensor files: **{r['adv_files']}**")

        if r["asr"] is not None:
            meta = r["asr_meta"] or {}
            extra = ""
            if meta.get("mode") == "hit_mask":
                extra = f"  (target-wise: {meta.get('total_hits',0)}/{meta.get('total_targets',0)} = {_fmt((meta.get('total_hits',0)/meta.get('total_targets',1)), nd=4)})"
            lines.append(
                f"- Attack success rate (image-level): **{_fmt(100*r['asr'], nd=1)}%**  "
                f"(succ **{r['succ_images']}** / denom **{r['denom_images']}**) {extra}"
            )
            lines.append(
                f"- Join column used for matching: `{r['key']}` (att img col `{meta.get('att_img_col')}`, succ col `{meta.get('succ_col')}`, mode `{meta.get('mode')}`)"
            )
            lines.append(
                f"- Successful attacks with matching adv row: **{r['succ_match_advrow']}**"
            )

        lines.append("\n### IoU@k & Spearman (adv-only)")
        if not r["iou_rho"]:
            lines.append("_No iou/rho columns found._\n")
        else:
            lines.append("| metric | mean | median |")
            lines.append("|---|---:|---:|")
            for m, s in r["iou_rho"].items():
                lines.append(f"| {m} | {_fmt(s['mean'])} | {_fmt(s['median'])} |")

        lines.append("\n### Deletion AUC")
        if not r["del_auc"]:
            lines.append("_No deletion AUC columns found._\n")
        else:
            lines.append(
                "| method | clean mean | adv mean | Δ mean (adv-clean) | clean med | adv med | Δ med |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for method, s in r["del_auc"].items():
                lines.append(
                    f"| {method} | {_fmt(s['clean_mean'])} | {_fmt(s['adv_mean'])} | {_fmt(s['delta_mean'])} | "
                    f"{_fmt(s['clean_med'])} | {_fmt(s['adv_med'])} | {_fmt(s['delta_med'])} |"
                )
        lines.append("")

    # deltas vs baseline (first run)
    base = runs[0]
    lines.append("## Deltas vs baseline\n")
    lines.append(f"Baseline: **{base['name']}**\n")
    lines.append(
        "| run | ΔASR (pp) | Δiou10_gc | Δiou10_ig | Δiou10_rise | Δrho_gc | Δrho_ig | Δrho_rise |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in runs[1:]:

        def d(m):
            if m not in base["iou_rho"] or m not in r["iou_rho"]:
                return None
            return r["iou_rho"][m]["mean"] - base["iou_rho"][m]["mean"]

        dasr = None
        if base["asr"] is not None and r["asr"] is not None:
            dasr = 100.0 * (r["asr"] - base["asr"])

        lines.append(
            f"| {r['name']} | {_fmt(dasr, nd=1)} | "
            f"{_fmt(d('iou10_gc'))} | {_fmt(d('iou10_ig'))} | {_fmt(d('iou10_rise'))} | "
            f"{_fmt(d('rho_gc'))} | {_fmt(d('rho_ig'))} | {_fmt(d('rho_rise'))} |"
        )

    Path(out_path).write_text("\n".join(lines))
    print(f"Wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Compare explanation CSVs across multiple runs (handles hit_mask for fastprior)"
    )
    ap.add_argument("--u-name", default="untargeted")
    ap.add_argument("--u-expl", required=True)
    ap.add_argument("--u-attacks")
    ap.add_argument("--u-adv-dir")

    ap.add_argument("--f-name", default="fastprior")
    ap.add_argument("--f-expl", required=True)
    ap.add_argument("--f-attacks")
    ap.add_argument("--f-adv-dir")

    ap.add_argument("--g-name", default="guided")
    ap.add_argument("--g-expl", required=True)
    ap.add_argument("--g-attacks")
    ap.add_argument("--g-adv-dir")

    ap.add_argument("--out", required=True)

    args = ap.parse_args()

    runs = [
        summarize_run(args.u_name, args.u_expl, args.u_attacks, args.u_adv_dir),
        summarize_run(args.f_name, args.f_expl, args.f_attacks, args.f_adv_dir),
        summarize_run(args.g_name, args.g_expl, args.g_attacks, args.g_adv_dir),
    ]
    write_md(runs, args.out)


if __name__ == "__main__":
    main()
