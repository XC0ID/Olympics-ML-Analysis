# Architecture

## Data flow
```
data/raw/
  SummerSD.csv  ──┐
  WinterSD.csv  ──┼──► loader.py ──► cleaner.py ──► builder.py
  CountriesSD.csv ┘                                      │
                                                         ▼
                                                   features.csv
                                                         │
                                          ┌──────────────┼──────────────┐
                                          ▼              ▼              ▼
                                    regression.py  classification.py  clustering.py
                                          │              │              │
                                          ▼              ▼              ▼
                                   regression_rf.pkl  clf_rf.pkl   kmeans.pkl
                                          │              │              │
                                          └──────────────┼──────────────┘
                                                         ▼
                                                   metrics.json
                                                   visualizations/
```

## Folder structure
```
src/data/        — loading, cleaning, validation
src/features/    — feature engineering, selection
src/models/      — regression, classification, clustering
src/evaluation/  — metrics, plots
src/utils/       — helpers, constants
scripts/         — runnable train/predict/evaluate
config/          — all hyperparameters in YAML
models/trained/  — saved .pkl files
results/         — metrics and charts
```

## Configuration
All settings live in `config/config.yaml`.
All hyperparameters live in `config/model_config.yaml`.
No hardcoded values in source code.