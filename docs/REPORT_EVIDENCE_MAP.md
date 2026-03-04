# Report Evidence Map (frozen from results/final_pack)

This file maps each **claim** to the exact **table/figure** that supports it, and where it should appear in the report.

## Canonical frozen artefacts (do not change)
- Folder: `results/final_pack/`
- Claims map: `results/final_pack/CLAIMS_EVIDENCE.md`
- Core tables:
  - `results/final_pack/tables/attack_summary.md`
  - `results/final_pack/tables/rq2_metrics_table.md`
- Core figures (RQ2):
  - `report/figures/final_pack/rq2_asr_image_percent.png`
  - `report/figures/final_pack/rq2_queries_median_success.png`
  - `report/figures/final_pack/rq2_iou10_adv_mean.png`
  - `report/figures/final_pack/rq2_rho_adv_mean.png`
  - `report/figures/final_pack/rq2_del_auc_delta_mean.png`
  - (optional overview) `report/figures/final_pack/asr_by_run.png`

## RQ2 story (what you will write)
### Claim RQ2-C1: Guidance changes the ASR vs cost trade-off
**Where:** Evaluation → RQ2 → “Attack effectiveness & cost”

**Evidence:**
- Table: `results/final_pack/tables/attack_summary.md`
- Figures:
  - `report/figures/final_pack/rq2_asr_image_percent.png`
  - `report/figures/final_pack/rq2_queries_median_success.png`
- Detailed metrics table: `results/final_pack/tables/rq2_metrics_table.md`

### Claim RQ2-C2: Guidance changes explanation stability among *successful* adversarials
**Where:** Evaluation → RQ2 → “Explanation stability under successful attacks”

**Evidence:**
- Detailed metrics table: `results/final_pack/tables/rq2_metrics_table.md`
- Figures:
  - `report/figures/final_pack/rq2_iou10_adv_mean.png`
  - `report/figures/final_pack/rq2_rho_adv_mean.png`
  - `report/figures/final_pack/rq2_del_auc_delta_mean.png`
- Narrative summary: `results/final_pack/summary_compare_10000_untargeted_fastprior_rise_guided_gpu_v4.md`

## RQ1 anchor (use untargeted baseline)
### Claim RQ1-C1: Successful one-pixel perturbations change explanation stability (method-dependent)
**Where:** Evaluation → RQ1

**Evidence:**
- `results/final_pack/summary_before_after_10000_untargeted_gpu.md`
- `results/final_pack/before_after_10000_untargeted_gpu.csv`

## Numbers to quote
From `results/final_pack/tables/attack_summary.md`:
- Untargeted: 3,477 / 9,213  (ASR 0.3774)
- Fastprior:  1,310 / 9,213  (ASR 0.1422)
- RISE-guided: 3,477 / 9,213 (ASR 0.3774), median-success queries 6,400 (rq2 metrics table)
