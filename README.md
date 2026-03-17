# Improving One-Pixel Attacks on Image Classifiers

This repository contains the implementation, experiment artefacts, report sources, and supporting appendix material for the project **Improving One-Pixel Attacks on Image Classifiers**.

The project studies black-box one-pixel adversarial attacks on image classifiers, with two main research questions:

1. **RQ1:** What effect do one-pixel adversarial perturbations have on explanation methods?
2. **RQ2:** Can responsibility-map guidance improve one-pixel attack behaviour, measured in terms of success, query cost, and explanation-side properties?

The implementation is organised as a reproducible research artefact rather than as a single script. Attack runs, explanation runs, summaries, plots, and report-ready tables are kept as separate stages connected through stable file outputs.

## Repository overview

Important top-level directories:

- `attacks/` — reusable implementations of differential-evolution one-pixel attack variants
- `explanations/` — explanation methods and utilities
- `scripts/` — command-line entry points for evaluation, attacks, explanations, summaries, and report exports
- `results/` — experiment outputs, summaries, plots, logs, and the frozen results pack
- `report/` — LaTeX source for the main report and imported report-ready figures/tables
- `appendix/` — LaTeX source for the appendix / supporting PDF
- `docs/` — project notes and planning material
- `data/` — local dataset storage
- `tests/` — tests and validation-related material

## Environment

This project was developed primarily with Python on macOS and with larger GPU-backed runs on a CUDA environment.

A local setup route is:

    cd ~/Projects/kcl-6ccs3prj-2025
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

If you already use `direnv`, entering the repository may activate the environment automatically through `.envrc`.

## Dependencies

Python dependencies are listed in `requirements.txt`.

Project tooling/configuration files include:
- `pyproject.toml`
- `.pre-commit-config.yaml`

## Data

The project uses CIFAR-10.
The local data directory is `data/`.

## Main artefact structure

The core workflow is:
1. evaluate the baseline model and identify the eligible subset
2. run one-pixel attack variants
3. compute explanations on clean/adversarial pairs
4. aggregate outputs into summaries, plots, and report-ready tables
5. import frozen artefacts into the report and appendix

## Key script entry points

- `scripts/eval_cifar10.py`
- `scripts/run_one_pixel.py`
- `scripts/run_explanations.py`
- `scripts/summarize_explanations.py`
- `scripts/summarize_compare_explanations.py`
- `scripts/summarize_compare_explanations_v4.py`
- `scripts/make_rq2_story_and_plots.py`
- `scripts/make_report_tables.py`

## Frozen final results pack

All report-facing final numbers are taken from `results/final_pack/`.

Important files include:
- `results/final_pack/README.md`
- `results/final_pack/CLAIMS_EVIDENCE.md`
- `results/final_pack/experiments_gpu_10k_overview.md`
- `results/final_pack/story_rq2_guidance_tradeoff.md`
- `results/final_pack/summary_compare_10000_untargeted_fastprior_rise_guided_gpu_v4.md`

## Report-ready assets

The report imports generated assets rather than manually transcribed numbers.

Important imported assets are stored in:
- `report/figures/final_pack/`
- `report/tables/final_pack/`

## Building the main report

    cd ~/Projects/kcl-6ccs3prj-2025/report
    latexmk -pdf -halt-on-error main.tex

Built PDF:
- `report/build/main.pdf`

## Building the appendix PDF

    cd ~/Projects/kcl-6ccs3prj-2025/appendix
    latexmk -pdf -outdir=build -halt-on-error main.tex

Built PDF:
- `appendix/build/main.pdf`

## Reproducibility route for submitted materials

    cd ~/Projects/kcl-6ccs3prj-2025
    source .venv/bin/activate
    cd report
    latexmk -pdf -halt-on-error main.tex
    cd ../appendix
    latexmk -pdf -outdir=build -halt-on-error main.tex

## Key reported outputs

Examples of final reported artefacts include:
- `results/final_pack/de1px_untargeted_10000.csv`
- `results/final_pack/de1px_targeted_fastprior_10000.csv`
- `results/final_pack/before_after_10000_untargeted_gpu.csv`
- `results/final_pack/before_after_10000_fastprior_gpu.csv`
- `results/final_pack/before_after_10000_rise_guided_gpu.csv`
- `results/final_pack/rq2_metrics_table.csv`

Examples of report-ready exports include:
- `report/tables/final_pack/rq2_attack_summary.tex`
- `report/tables/final_pack/rq2_stability_means.tex`
- `report/tables/final_pack/rq2_del_delta_means.tex`

## Verification and sanity checking

The project was developed with several practical checks in mind:
- baseline sanity checks before large-scale runs
- smoke tests before long attack jobs
- fixed seeds for frozen runs
- logged configurations for final experiments
- frozen artefacts used by the report instead of manual number copying

## Suggested minimal submission components

A clean source-code submission should normally contain:
- `attacks/`
- `explanations/`
- `scripts/`
- `src/`
- `tests/`
- `requirements.txt`
- `pyproject.toml`
- `README.md`

## Licence / attribution note

This repository contains original project code together with references to third-party libraries, benchmark datasets, and cited research papers. External dependencies and cited methods remain the intellectual property of their respective authors and maintainers.
