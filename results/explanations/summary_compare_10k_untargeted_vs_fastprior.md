# Explanation comparison

_Generated: 2026-02-12 00:27:42_

## Runs

- **untargeted**
  - explanations: `results/explanations/before_after_10000_untargeted_gpu.csv`
  - attacks: `results/attacks/de1px_untargeted_10000.csv`
- **fastprior**
  - explanations: `results/explanations/before_after_10000_fastprior_gpu.csv`
  - attacks: `results/attacks/de1px_targeted_fastprior_10000.csv`

## Aggregate explanation metrics (means over successful attacks)

| run | n_total | n_adv | iou10_gc | iou10_ig | iou10_rise | rho_gc | rho_ig | rho_rise | del_auc_gc_clean | del_auc_gc_adv | del_auc_gc_delta | del_auc_ig_clean | del_auc_ig_adv | del_auc_ig_delta | del_auc_rise_clean | del_auc_rise_adv | del_auc_rise_delta | success_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| untargeted | 10000 | 3477 | 0.3260 | 0.4849 | 0.3223 | 0.6949 | 0.8420 | 0.4771 | 0.2351 | 0.1318 | -0.1033 | 0.2168 | 0.1898 | -0.0270 | 0.1874 | 0.0928 | -0.0946 | success rows from attack CSV via `ds_idx` join |
| fastprior | 10000 | 1310 | 0.5165 | 0.5320 | 0.3671 | 0.8389 | 0.8657 | 0.5250 | 0.2019 | 0.1568 | -0.0451 | 0.1963 | 0.1810 | -0.0153 | 0.1565 | 0.1089 | -0.0476 | success rows from attack CSV via `ds_idx` join |

## Attack CSV summary

| run | n_rows | n_success | success_rate | query_col | success_rule |
| --- | --- | --- | --- | --- | --- |
| untargeted | 9213 | 3477 | 0.3774 | queries | success from `success == 1` |
| fastprior | 9213 | 1310 | 0.1422 | queries | success from `hit_mask` contains a 1 |

### untargeted: `queries` stats
- all: median=40400.00, p90=40400.00, mean=32373.30, min=400.00, max=40400.00
- success: median=6000.00, p90=40400.00, mean=19154.67, min=400.00, max=40400.00

### fastprior: `queries` stats
- all: median=70848.00, p90=70848.00, mean=70848.00, min=70848.00, max=70848.00
- success: median=70848.00, p90=70848.00, mean=70848.00, min=70848.00, max=70848.00
