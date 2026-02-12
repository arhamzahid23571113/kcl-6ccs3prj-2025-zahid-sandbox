# Explanation comparison

_Generated: 2026-02-12 00:32:42_

## Runs

- **untargeted_1k**
  - explanations: `results/explanations/before_after_1000_untargeted_gpu.csv`
  - attacks: `results/attacks/de1px_untargeted_1000.csv`
- **fastprior_1k**
  - explanations: `results/explanations/before_after_1000_fastprior_gpu.csv`
  - attacks: `results/attacks/de1px_targeted_fastprior_1000.csv`

## Aggregate explanation metrics (means over successful attacks)

| run | n_total | n_adv | iou10_gc | iou10_ig | iou10_rise | rho_gc | rho_ig | rho_rise | del_auc_gc_clean | del_auc_gc_adv | del_auc_gc_delta | del_auc_ig_clean | del_auc_ig_adv | del_auc_ig_delta | del_auc_rise_clean | del_auc_rise_adv | del_auc_rise_delta | success_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| untargeted_1k | 1000 | 355 | 0.3371 | 0.4792 | 0.3336 | 0.7116 | 0.8434 | 0.4848 | 0.2357 | 0.1371 | -0.0986 | 0.2229 | 0.1950 | -0.0279 | 0.2023 | 0.1001 | -0.1022 | success rows from attack CSV via `ds_idx` join |
| fastprior_1k | 1000 | 148 | 0.5327 | 0.5203 | 0.3748 | 0.8479 | 0.8645 | 0.5380 | 0.2057 | 0.1557 | -0.0501 | 0.1847 | 0.1821 | -0.0026 | 0.1648 | 0.1070 | -0.0578 | success rows from attack CSV via `ds_idx` join |

## Attack CSV summary

| run | n_rows | n_success | success_rate | query_col | success_rule |
| --- | --- | --- | --- | --- | --- |
| untargeted_1k | 917 | 355 | 0.3871 | queries | success from `success == 1` |
| fastprior_1k | 917 | 148 | 0.1614 | queries | success from `hit_mask` contains a 1 |

### untargeted_1k: `queries` stats
- all: median=40400.00, p90=40400.00, mean=31677.21, min=400.00, max=40400.00
- success: median=4400.00, p90=40400.00, mean=17868.17, min=400.00, max=40400.00

### fastprior_1k: `queries` stats
- all: median=70848.00, p90=70848.00, mean=70848.00, min=70848.00, max=70848.00
- success: median=70848.00, p90=70848.00, mean=70848.00, min=70848.00, max=70848.00
