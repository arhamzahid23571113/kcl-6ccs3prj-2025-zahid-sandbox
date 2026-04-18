# Improving One-Pixel Attacks on Image Classifiers

This ZIP contains the source code and supporting project scripts for the project **Improving One-Pixel Attacks on Image Classifiers**.

The project studies black-box one-pixel adversarial attacks on image classifiers, with two main research questions:

1. **RQ1:** What effect do one-pixel adversarial perturbations have on explanation methods?
2. **RQ2:** Can responsibility-map guidance improve one-pixel attack behaviour, measured in terms of success, query cost, and explanation-side properties?

This source-code submission contains the implemented Python code, configuration files, and supporting notes needed to understand and run the software artefact.

## What is included in this ZIP

Important directories and files:

- `attacks/` — reusable implementations of differential-evolution one-pixel attack variants
- `explanations/` — explanation methods and utilities
- `scripts/` — command-line entry points for evaluation, attacks, explanations, summaries, and exports
- `docs/` — supporting notes and project documentation
- `src/` — source directory placeholder
- `tests/` — tests / validation placeholder
- `requirements.txt` — Python dependencies
- `pyproject.toml` — project tooling/configuration
- `Makefile` — convenience commands

## What is not included in this ZIP

To keep the source-code submission small and focused, this ZIP does **not** include large artefacts such as:

- downloaded datasets
- experiment results
- logs
- frozen result packs
- LaTeX report sources
- appendix sources
- built PDFs

The **main report PDF** and **appendix PDF** are submitted separately.

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

These generated artefacts are not included in this ZIP.

## Verification

The project was developed with several practical checks in mind:

- baseline sanity checks before large-scale runs
- smoke tests before longer attack jobs
- fixed seeds for frozen runs
- logged configurations for final experiments
- modular scripts with stable intermediate file outputs

## Notes

This ZIP is intended to provide the implemented source code for the computing artefact. The written dissertation/report and appendix are separate submission components.

## Licence / attribution note

This repository contains original project code together with references to third-party libraries, benchmark datasets, and cited research papers. External dependencies and cited methods remain the intellectual property of their respective authors and maintainers.
