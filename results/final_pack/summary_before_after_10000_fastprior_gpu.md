# Explanations summary

- Total rows: **10000**
- Rows with adversarial present: **1310**

## IoU@k & Spearman (adv-only)

- **iou10_gc**: mean **0.517**, median **0.534**
- **iou10_ig**: mean **0.532**, median **0.511**
- **iou10_rise**: mean **0.367**, median **0.360**
- **rho_gc**: mean **0.839**, median **0.921**
- **rho_ig**: mean **0.866**, median **0.873**
- **rho_rise**: mean **0.525**, median **0.548**

## Deletion AUC

- **GC**: clean mean **0.305**, adv mean **0.157**, Δ (adv−clean) mean **-0.045** (medians: clean **0.276**, adv **0.101**, Δ **-0.025**)
- **IG**: clean mean **0.250**, adv mean **0.181**, Δ (adv−clean) mean **-0.015** (medians: clean **0.200**, adv **0.128**, Δ **-0.011**)
- **RISE**: clean mean **0.257**, adv mean **0.109**, Δ (adv−clean) mean **-0.048** (medians: clean **0.195**, adv **0.054**, Δ **-0.029**)
