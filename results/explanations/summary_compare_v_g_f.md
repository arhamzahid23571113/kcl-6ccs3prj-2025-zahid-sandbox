# Explanation comparison: vanilla vs guided vs fast-prior

This file summarises explanation metrics for three one-pixel attack variants over the 200-image CIFAR-10 subset.

- **vanilla**: standard DE-based one-pixel attack
- **guided**: RISE-guided DE one-pixel attack
- **fastprior**: multi-target DE with fast input-gradient prior (run on GPU)

## Aggregate metrics over successful attacks

| attack | n_total | n_adv | iou10_gc | iou10_ig | iou10_rise | rho_gc | rho_ig | rho_rise | del_auc_gc_clean | del_auc_gc_adv | del_auc_rise_clean | del_auc_rise_adv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla | 200 | 200 | 0.3846 | 0.4883 | 0.3078 | 0.7632 | 0.8478 | 0.4264 | 0.3066 | 0.1186 | 0.2568 | 0.0773 |
| guided | 200 | 200 | 0.3640 | 0.4699 | 0.2329 | 0.7393 | 0.8410 | 0.3740 | 0.3066 | 0.1082 | 0.2514 | 0.1011 |
| fastprior | 200 | 192 | 0.4526 | 0.4626 | 0.4190 | 0.7789 | 0.8314 | 0.6190 | 0.3102 | 0.0713 | 0.2349 | 0.1016 |
