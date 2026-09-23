# Improving One-Pixel Attacks on Image Classifiers

This repository contains the source code, report sources, experiment outputs, and supporting scripts for **Improving One-Pixel Attacks on Image Classifiers**.

The project studies black-box one-pixel adversarial attacks on image classifiers, with two main research questions:

1. **RQ1:** What effect do one-pixel adversarial perturbations have on explanation methods?
2. **RQ2:** Can responsibility-map guidance improve one-pixel attack behaviour, measured in terms of success, query cost, and explanation-side properties?

The repository preserves both the source-code submission and later research artefacts.

## Repository contents

Important directories and files:

- `attacks/` — reusable implementations of differential-evolution one-pixel attack variants
- `explanations/` — explanation methods and utilities
- `scripts/` — command-line entry points for evaluation, attacks, explanations, summaries, and exports
- `docs/` — supporting notes and project documentation
- `report/`, `appendix/` — LaTeX sources and report figures
- `results/` — tracked outputs from earlier runs (large; not regenerated automatically)
- `tests/` — focused metrics regression tests
- `requirements.txt` — Python dependencies
- `pyproject.toml` — project tooling/configuration
- `Makefile` — convenience commands

## Data and other generated files

The CIFAR-10 dataset is downloaded locally on first use and is excluded from Git. Some result tensors, plots, tables, and LaTeX sources are already tracked. This repository contains more than 5,000 tracked files under `results/`, so a clone may be sizable. Built report PDFs are excluded.

## Environment setup

A local setup route is:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

This project was developed primarily with Python on macOS, with larger GPU-backed runs carried out in a CUDA environment.

## Main workflow

The core software workflow is:

1. evaluate the baseline model and identify the eligible subset
2. run one-pixel attack variants
3. compute explanations on clean/adversarial pairs
4. aggregate outputs into summaries and exported tables/plots

The implementation is organised as a reproducible research artefact rather than as a single script. Attack runs, explanation runs, summaries, and exports are kept as separate stages connected through stable file outputs.

## Key script entry points

Main scripts:

- `scripts/eval_cifar10.py`
- `scripts/run_one_pixel.py`
- `scripts/run_explanations.py`
- `scripts/summarize_explanations.py`
- `scripts/summarize_compare_explanations.py`
- `scripts/summarize_compare_explanations_v4.py`
- `scripts/make_rq2_story_and_plots.py`
- `scripts/make_report_tables.py`

## Data

The project uses CIFAR-10.

If the dataset is not already present locally, the relevant scripts will download CIFAR-10 automatically into a local `data/` directory when run.

## Example usage

Evaluate the pretrained CIFAR-10 model on a small subset:

    python scripts/eval_cifar10.py --limit 100

Run the untargeted one-pixel attack on a small subset:

    python scripts/run_one_pixel.py untargeted --limit 100 --out results/attacks/de1px_untargeted_smoke.csv

Compute explanations for a run:

    python scripts/run_explanations.py --out results/explanations/before_after_smoke.csv --adv-dir results/adv_tensors/smoke_run

## Outputs

When the scripts are run, they create output directories such as:

- `data/`
- `results/attacks/`
- `results/explanations/`
- `results/plots/`
- `results/tables/`

Some earlier generated artefacts are tracked; new outputs are written to the selected paths.

## Verification

The project was developed with several practical checks in mind:

- baseline sanity checks before large-scale runs
- smoke tests before longer attack jobs
- fixed seeds for frozen runs
- logged configurations for final experiments
- modular scripts with stable intermediate file outputs

Run the focused, data-free metric checks with `python -m unittest discover -s tests -p 'test_*.py'` after installing dependencies. Full experiment validation requires downloading CIFAR-10 and pretrained model weights and may require GPU time.

**Metric compatibility:** `topk_mask` now selects exactly the requested number of pixels when saliency values tie, and Spearman correlation uses average ranks for ties. The tracked result files and report figures were produced before this correction. Regenerate explanations, summaries, and plots before using the corrected metrics to compare against those historical results; do not interpret the old plots as newly recomputed.

## Notes

The written dissertation and appendix also have source files in this repository.

## Licence / attribution note

This repository contains original project code together with references to third-party libraries, benchmark datasets, and cited research papers. External dependencies and cited methods remain the intellectual property of their respective authors and maintainers.
