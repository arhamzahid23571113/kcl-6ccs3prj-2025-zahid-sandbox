# ResNet-32 validation summary

Subset: first 2000 CIFAR-10 test images

Clean accuracy: 0.9295
Eligible images: 1872
Successful one-pixel attacks: 579
Adversarial explanation rows: 579
ASR on eligible subset: 0.309295
Median queries (all eligible): 40400
Median queries (successful only): 3600

Stability means on successful adversarial examples:
- IoU@10 GC: 0.267932
- IoU@10 IG: 0.453890
- IoU@10 RISE: 0.312526
- Spearman GC: 0.643802
- Spearman IG: 0.808784
- Spearman RISE: 0.445432

Deletion AUC deltas on successful adversarial examples (adv - clean):
- GC mean: -0.125404
- GC median: -0.092594
- IG mean: -0.026517
- IG median: -0.023416
- RISE mean: -0.083298
- RISE median: -0.063696

See:
- results/validation/eval_resnet32_2000.csv
- results/validation/de1px_untargeted_resnet32_2000.csv
- results/validation/before_after_resnet32_2000.csv
- results/validation/de1px_untargeted_resnet32_2000.log
- results/validation/explanations_resnet32_2000.log
