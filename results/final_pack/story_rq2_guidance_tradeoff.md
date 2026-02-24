# RQ2 story: guidance vs untargeted vs fastprior (10k)

Auto-generated from attack CSVs + before/after explanation CSVs.

## Key headline
- **Fastprior trades ASR for stability**: lower image-level ASR but higher explanation stability among successful adversarials.
- **RISE-guided keeps ASR ~the same as untargeted**, but can shift explanation stability (especially RISE-based stability).

## Numbers (from `rq2_metrics_table.csv`)

| run         |   asr_image |   succ_images |   denom_images |   queries_median_succ |   gens_used_median_succ |   expl_rows |   expl_adv_rows |   iou10_gc_mean_adv |   iou10_ig_mean_adv |   iou10_rise_mean_adv |   rho_gc_mean_adv |   rho_ig_mean_adv |   rho_rise_mean_adv |   del_delta_gc_mean |   del_delta_ig_mean |   del_delta_rise_mean |   asr_target |   target_hits |   target_total |
|:------------|------------:|--------------:|---------------:|----------------------:|------------------------:|------------:|----------------:|--------------------:|--------------------:|----------------------:|------------------:|------------------:|--------------------:|--------------------:|--------------------:|----------------------:|-------------:|--------------:|---------------:|
| untargeted  |       37.74 |          3477 |           9213 |                  6000 |                      14 |       10000 |            3477 |            0.326027 |            0.484914 |              0.322344 |          0.694871 |          0.842024 |            0.47715  |           -0.173026 |          -0.0602526 |             -0.164727 |      nan     |           nan |            nan |
| fastprior   |       14.22 |          1310 |           9213 |                 70848 |                      40 |       10000 |            1310 |            0.51651  |            0.531976 |              0.36714  |          0.838862 |          0.865739 |            0.525033 |           -0.148028 |          -0.0689581 |             -0.148369 |        1.787 |          1482 |          82917 |
| rise-guided |       37.74 |          3477 |           9213 |                  6400 |                      15 |       10000 |            3477 |            0.325462 |            0.484462 |              0.296216 |          0.692717 |          0.841958 |            0.445417 |           -0.172794 |          -0.0595488 |             -0.158782 |      nan     |           nan |            nan |

## Artifacts
- Metrics table: `results/explanations/rq2_metrics_table.csv`
- Story: `results/explanations/story_rq2_guidance_tradeoff.md`
- Plots in `results/plots/` (rq2_*.png)
