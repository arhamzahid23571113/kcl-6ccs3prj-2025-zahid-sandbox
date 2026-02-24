# GPU 10k experiment overview

## Attacks
- `results/attacks/de1px_untargeted_10000.csv`
- `results/attacks/de1px_targeted_fastprior_10000.csv`
- `results/attacks/summary_de1px_targeted_fastprior_10000.md`

## Explanations
- `results/explanations/before_after_10000_untargeted_gpu.csv`
- `results/explanations/before_after_10000_fastprior_gpu.csv`
- `results/explanations/summary_before_after_10000_untargeted_gpu.md`
- `results/explanations/summary_before_after_10000_fastprior_gpu.md`

## Key numbers
- Correctly classified images: 9213 / 10000
- Untargeted DE: 3477 / 9213 successes ≈ 37.7% (image-wise)
- Fastprior targeted multi-target DE:
  - 1310 / 9213 images with ≥1 successful target ≈ 14.2%
  - 1482 / 82917 successful targets ≈ 1.8%
  - Mean successful targets per successful image ≈ 1.13
