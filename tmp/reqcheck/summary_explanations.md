# Explanations summary

- Total rows: **6**
- Rows with adversarial present: **2**
- Adversarial tensor files: **2**
- Attack success rate (from DE CSV): **33.3%**
- Successful attacks with matching adv row: **2**

## IoU@k & Spearman (adv-only)

- **iou10_gc**: mean **0.256**, median **0.256**
- **iou10_ig**: mean **0.687**, median **0.687**
- **iou10_rise**: mean **0.153**, median **0.153**
- **rho_gc**: mean **0.848**, median **0.848**
- **rho_ig**: mean **0.884**, median **0.884**
- **rho_rise**: mean **0.191**, median **0.191**

## Deletion AUC

- **GC**: clean mean **0.320**, adv mean **0.075**, Δ (adv−clean) mean **-0.221** (medians: clean **0.309**, adv **0.075**, Δ **-0.221**)
- **IG**: clean mean **0.408**, adv mean **0.098**, Δ (adv−clean) mean **-0.018** (medians: clean **0.435**, adv **0.098**, Δ **-0.018**)
- **RISE**: clean mean **0.240**, adv mean **0.071**, Δ (adv−clean) mean **0.018** (medians: clean **0.143**, adv **0.071**, Δ **0.018**)
