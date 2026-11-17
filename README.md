apnea-ecg-ml/
├── data/
│   ├── raw/            # PhysioNet files (NOT committed)
│   └── processed/      # features.parquet etc.
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_eda_preprocessing.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation_error_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py         # load record, load labels, list records
│   └── features.py     # RR cleaning, HRV features per minute
├── reports/
│   └── figures/        # saved plots for report/slides
├── requirements.txt
├── .gitignore
└── README.md