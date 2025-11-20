# Explanations summary

- Total rows: **200**
- Rows with adversarial present: **67**
- Adversarial tensor files: **67**
- Attack success rate (from DE CSV): **36.8%**
- Successful attacks with matching adv row: **67**

## IoU@k & Spearman (adv-only)

- **iou10_gc**: mean **0.364**, median **0.351**
- **iou10_ig**: mean **0.470**, median **0.457**
- **iou10_rise**: mean **0.233**, median **0.229**
- **rho_gc**: mean **0.739**, median **0.804**
- **rho_ig**: mean **0.841**, median **0.849**
- **rho_rise**: mean **0.374**, median **0.388**

## Deletion AUC

- **GC**: clean mean **0.307**, adv mean **0.108**, Δ (adv−clean) mean **-0.118** (medians: clean **0.264**, adv **0.075**, Δ **-0.083**)
- **IG**: clean mean **0.256**, adv mean **0.186**, Δ (adv−clean) mean **-0.011** (medians: clean **0.197**, adv **0.161**, Δ **-0.015**)
- **RISE**: clean mean **0.251**, adv mean **0.101**, Δ (adv−clean) mean **-0.066** (medians: clean **0.202**, adv **0.030**, Δ **-0.059**)
