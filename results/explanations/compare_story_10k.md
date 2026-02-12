# 10k comparison: untargeted DE vs fastprior multi-target

## Attack success (image-wise, eligible images only)
- Untargeted DE: 3477 / 9213 = 37.74%
- Fastprior: 1310 / 9213 = 14.22%

## Explanation stability on successful attacks
(Values are means over successful attacks only; success defined by attack CSV and joined via ds_idx.)

| run | n_total | n_adv | iou10_gc | iou10_ig | iou10_rise | rho_gc | rho_ig | rho_rise | del_auc_gc_clean | del_auc_gc_adv | del_auc_gc_delta | del_auc_ig_clean | del_auc_ig_adv | del_auc_ig_delta | del_auc_rise_clean | del_auc_rise_adv | del_auc_rise_delta | success_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| untargeted | 10000 | 3477 | 0.3260 | 0.4849 | 0.3223 | 0.6949 | 0.8420 | 0.4771 | 0.2351 | 0.1318 | -0.1033 | 0.2168 | 0.1898 | -0.0270 | 0.1874 | 0.0928 | -0.0946 | success rows from attack CSV via `ds_idx` join |
| fastprior | 10000 | 1310 | 0.5165 | 0.5320 | 0.3671 | 0.8389 | 0.8657 | 0.5250 | 0.2019 | 0.1568 | -0.0451 | 0.1963 | 0.1810 | -0.0153 | 0.1565 | 0.1089 | -0.0476 | success rows from attack CSV via `ds_idx` join |


## Interpretation (draft bullets)
- Fastprior successes exhibit higher attribution overlap/stability (IoU@10 and Spearman) than untargeted successes.
- Both methods reduce deletion-AUC after attack (negative Δ), indicating a drop in faithfulness under deletion; the drop is smaller on fastprior successes.
- Overall: guiding search with a prior changes which adversarial examples you find (lower ASR but “more stable” explanations among successes).
