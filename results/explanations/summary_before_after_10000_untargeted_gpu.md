# Explanations summary

- Total rows: **10000**
- Rows with adversarial present: **3477**

## IoU@k & Spearman (adv-only)

- **iou10_gc**: mean **0.326**, median **0.308**
- **iou10_ig**: mean **0.485**, median **0.478**
- **iou10_rise**: mean **0.322**, median **0.308**
- **rho_gc**: mean **0.695**, median **0.760**
- **rho_ig**: mean **0.842**, median **0.849**
- **rho_rise**: mean **0.477**, median **0.503**

## Deletion AUC

- **GC**: clean mean **0.305**, adv mean **0.132**, Δ (adv−clean) mean **-0.103** (medians: clean **0.276**, adv **0.076**, Δ **-0.077**)
- **IG**: clean mean **0.250**, adv mean **0.190**, Δ (adv−clean) mean **-0.027** (medians: clean **0.200**, adv **0.127**, Δ **-0.025**)
- **RISE**: clean mean **0.257**, adv mean **0.093**, Δ (adv−clean) mean **-0.095** (medians: clean **0.200**, adv **0.032**, Δ **-0.073**)
