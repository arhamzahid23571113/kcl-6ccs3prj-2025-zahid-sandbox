# Explanations summary

- Total rows: **200**
- Rows with adversarial present: **68**
- Adversarial tensor files: **68**
- Attack success rate (from DE CSV): **37.4%**
- Successful attacks with matching adv row: **68**

## IoU@k & Spearman (adv-only)

- **iou10_gc**: mean **0.385**, median **0.382**
- **iou10_ig**: mean **0.488**, median **0.489**
- **iou10_rise**: mean **0.308**, median **0.291**
- **rho_gc**: mean **0.763**, median **0.822**
- **rho_ig**: mean **0.848**, median **0.852**
- **rho_rise**: mean **0.426**, median **0.464**

## Deletion AUC

- **GC**: clean mean **0.307**, adv mean **0.119**, Δ (adv-clean) mean **-0.108** (medians: clean **0.264**, adv **0.069**, Δ **-0.059**)
- **IG**: clean mean **0.253**, adv mean **0.197**, Δ (adv-clean) mean **-0.002** (medians: clean **0.204**, adv **0.170**, Δ **-0.012**)
- **RISE**: clean mean **0.257**, adv mean **0.077**, Δ (adv-clean) mean **-0.116** (medians: clean **0.208**, adv **0.019**, Δ **-0.082**)
