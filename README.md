# TaskFit-AI: A Task-Aware Algorithm for Evaluating the Fitness of African Health Datasets for AI Applications

Reproducible pipeline for the article by R. O. Tamanja, A. L. Yussif, and A. S. Yussif (University of Cape Coast).

## Data (all openly downloadable, no registration; copies in `inputs/`)
| File | Source | Task |
|---|---|---|
| `SAheart.RData` | ElemStatLearn, https://github.com/cran/ElemStatLearn | coronary heart disease, South Africa |
| `gambia.rda`, `gambia.csv` | geoR, https://github.com/cran/geoR | malaria parasitaemia, The Gambia |
| `thornton_hiv.csv` | Rdatasets (causaldata), https://github.com/vincentarelbundock/Rdatasets | HIV result collection and HIV positivity, Malawi |
| `ebola_sierraleone_2014.RData` | outbreaks, https://github.com/reconverse/outbreaks | Ebola laboratory confirmation, Sierra Leone |
`gambia.csv` was exported from `gambia.rda` with R (`write.csv`), because the .rda format is not readable by pyreadr.

## Run
```bash
python -m pip install -r requirements.txt      # Python 3.12
cd src && python run_all.py                    # about 45 minutes on one core
```

| Script | Purpose | Article |
|---|---|---|
| `tasks.py` | five standardised tasks with subgroups and natural groupings | Section IV-A, Table I |
| `taskfit.py` | the seven dimensions, TaskFit score, baselines (incl. METRIC-inspired profile), degradation engine | Section III, Eqs. (1)-(12) |
| `run_experiments.py` | 1,920 combined-degradation instances in 13 settings; 1,650 single-factor instances | Section IV-B, Table III |
| `analysis.py` | validity, unfit detection, secondary outcomes, diagnosis, real shift, paired and permutation tests | Section V, Tables IV, V, VI |
| `analysis2.py` | aggregation sensitivity, label-estimator sensitivity against true noise, METRIC-inspired baseline | Section V-A, V-D, Table VIII |
| `figures.py` | Figs. 1-4 at print size | |

Seed: 2026 (instance seeds derived with CRC32 of task and setting names).
