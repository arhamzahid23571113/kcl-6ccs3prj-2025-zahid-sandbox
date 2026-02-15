# Compare explanation metrics (multi-run)

## untargeted
- Explanations CSV: `results/explanations/before_after_10000_untargeted_gpu.csv`
- Total rows: **10000**
- Rows with adversarial present: **3477**
- Attack success rate (image-level): **37.7%**  (succ **3477** / denom **9213**)
- Join column used for matching: `ds_idx` (att img col `ds_idx`, succ col `success`, mode `numeric_success`)
- Successful attacks with matching adv row: **3477**

### IoU@k & Spearman (adv-only)
| metric | mean | median |
|---|---:|---:|
| iou10_gc | 0.326 | 0.308 |
| iou10_ig | 0.485 | 0.478 |
| iou10_rise | 0.322 | 0.308 |
| rho_gc | 0.695 | 0.760 |
| rho_ig | 0.842 | 0.849 |
| rho_rise | 0.477 | 0.503 |

### Deletion AUC
| method | clean mean | adv mean | Δ mean (adv-clean) | clean med | adv med | Δ med |
|---|---:|---:|---:|---:|---:|---:|
| GC | 0.305 | 0.132 | -0.103 | 0.276 | 0.076 | -0.077 |
| IG | 0.250 | 0.190 | -0.027 | 0.200 | 0.127 | -0.025 |
| RISE | 0.257 | 0.093 | -0.095 | 0.200 | 0.032 | -0.073 |

## fastprior
- Explanations CSV: `results/explanations/before_after_10000_fastprior_gpu.csv`
- Total rows: **10000**
- Rows with adversarial present: **1310**
- Attack success rate (image-level): **14.2%**  (succ **1310** / denom **9213**)   (target-wise: 1482/82917 = 0.0179)
- Join column used for matching: `ds_idx` (att img col `ds_idx`, succ col `hit_mask`, mode `hit_mask`)
- Successful attacks with matching adv row: **1310**

### IoU@k & Spearman (adv-only)
| metric | mean | median |
|---|---:|---:|
| iou10_gc | 0.517 | 0.534 |
| iou10_ig | 0.532 | 0.511 |
| iou10_rise | 0.367 | 0.360 |
| rho_gc | 0.839 | 0.921 |
| rho_ig | 0.866 | 0.873 |
| rho_rise | 0.525 | 0.548 |

### Deletion AUC
| method | clean mean | adv mean | Δ mean (adv-clean) | clean med | adv med | Δ med |
|---|---:|---:|---:|---:|---:|---:|
| GC | 0.305 | 0.157 | -0.045 | 0.276 | 0.101 | -0.025 |
| IG | 0.250 | 0.181 | -0.015 | 0.200 | 0.128 | -0.011 |
| RISE | 0.257 | 0.109 | -0.048 | 0.195 | 0.054 | -0.029 |

## rise-guided
- Explanations CSV: `results/explanations/before_after_10000_rise_guided_gpu.csv`
- Total rows: **10000**
- Rows with adversarial present: **3477**
- Adversarial tensor files: **3477**
- Attack success rate (image-level): **37.7%**  (succ **3477** / denom **9213**)
- Join column used for matching: `ds_idx` (att img col `ds_idx`, succ col `success`, mode `numeric_success`)
- Successful attacks with matching adv row: **3477**

### IoU@k & Spearman (adv-only)
| metric | mean | median |
|---|---:|---:|
| iou10_gc | 0.325 | 0.299 |
| iou10_ig | 0.484 | 0.478 |
| iou10_rise | 0.296 | 0.283 |
| rho_gc | 0.693 | 0.767 |
| rho_ig | 0.842 | 0.849 |
| rho_rise | 0.445 | 0.467 |

### Deletion AUC
| method | clean mean | adv mean | Δ mean (adv-clean) | clean med | adv med | Δ med |
|---|---:|---:|---:|---:|---:|---:|
| GC | 0.305 | 0.132 | -0.103 | 0.276 | 0.075 | -0.078 |
| IG | 0.250 | 0.190 | -0.027 | 0.199 | 0.129 | -0.025 |
| RISE | 0.256 | 0.098 | -0.089 | 0.195 | 0.034 | -0.070 |

## Deltas vs baseline

Baseline: **untargeted**

| run | ΔASR (pp) | Δiou10_gc | Δiou10_ig | Δiou10_rise | Δrho_gc | Δrho_ig | Δrho_rise |
|---|---:|---:|---:|---:|---:|---:|---:|
| fastprior | -23.5 | 0.190 | 0.047 | 0.045 | 0.144 | 0.024 | 0.048 |
| rise-guided | 0.0 | -0.001 | -0.000 | -0.026 | -0.002 | -0.000 | -0.032 |
