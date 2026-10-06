# TaskFit-AI

A task-aware algorithm for evaluating the fitness of African health datasets for AI applications.

Reproducible pipeline for the article by R. O. Tamanja, A. L. Yussif, and A. S. Yussif, University of Cape Coast.

## Data

All datasets are openly downloadable. Copies are included in `inputs/`.

| File                           | Source        | Task                                         |
| ------------------------------ | ------------- | -------------------------------------------- |
| `SAheart.RData`                | ElemStatLearn | Coronary heart disease, South Africa         |
| `gambia.rda`, `gambia.csv`     | geoR          | Malaria parasitaemia, The Gambia             |
| `thornton_hiv.csv`             | Rdatasets     | HIV result collection and positivity, Malawi |
| `ebola_sierraleone_2014.RData` | outbreaks     | Ebola laboratory confirmation, Sierra Leone  |

`gambia.csv` was exported from `gambia.rda` using R because `.rda` is not supported by pyreadr.

## Installation and Usage

```bash
python -m pip install -r requirements.txt
cd src
python run_all.py
```

Runtime: approximately 45 minutes on one CPU core.

## Scripts

| Script               | Purpose                                                     |
| -------------------- | ----------------------------------------------------------- |
| `tasks.py`           | Standardised tasks and subgroup definitions                 |
| `taskfit.py`         | TaskFit dimensions, scoring, baselines, and degradation     |
| `run_experiments.py` | Combined- and single-factor experiments                     |
| `analysis.py`        | Validity, unfit detection, diagnosis, and statistical tests |
| `analysis2.py`       | Sensitivity analyses and METRIC-inspired baseline           |
| `figures.py`         | Generates Figures 1–4                                       |

## Reproducibility

**Seed:** `2026`

Instance seeds are derived using CRC32 of task and setting names.
