# PCOS Diagnostic Dashboard

An interactive Streamlit dashboard for exploring Polycystic Ovary Syndrome (PCOS) data, assessing diagnostic risk, and understanding feature importance through machine learning.

**GitHub:** https://github.com/AustinJSimmons/pcos-diagnostic-dashboard

## Overview

Built on a clinical dataset of 541 patients (177 PCOS, 364 controls) spanning 41 features — hormonal assays, anthropometric measurements, ultrasound findings, and self-reported symptoms.

**Key results (5-fold stratified CV, NB07):**

| Model | Full AUC | Low-Cost AUC |
|---|---|---|
| XGBoost | 0.9623 ± 0.012 | 0.8750 ± 0.018 |
| Random Forest | 0.9610 ± 0.012 | 0.8891 ± 0.024 |
| LASSO | 0.9551 ± 0.009 | 0.8853 ± 0.024 |
| Ridge | 0.9505 ± 0.004 | 0.8803 ± 0.027 |
| Logistic Regression | 0.9419 ± 0.003 | 0.8778 ± 0.027 |

All pairwise differences are non-significant (Wilcoxon signed-rank, Bonferroni-corrected).

## Dashboard Features

- **Phenotype Explorer** — K-Means clustering (k=2) on PCOS patients reveals Metabolic and Hyperandrogenic subtypes
- **Risk Calculator** — Logistic Regression and Random Forest for personalized risk scoring; full clinical or non-invasive (symptom/vitals only) model
- **Feature Impact** — Feature importance from LR coefficients and mutual information; head-to-head comparison of all five models

## Quickstart

### Requirements
- Python 3.8+
- All dependencies in `requirements.txt`

### Installation

```bash
git clone https://github.com/AustinJSimmons/pcos-diagnostic-dashboard.git
cd pcos-diagnostic-dashboard

python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Data setup

The raw data files are not committed to this repository. Place the following files in `data/raw/` before running:

- `PCOS_data_without_infertility.xlsx` — main clinical dataset (541 patients, 45 columns)
- `PCOS_infertility.xlsx` — infertility subset

Then generate the cleaned dataset by running `notebooks/01-data-cleaning.ipynb`. This produces `data/processed/cleaned_data.csv`, which all other notebooks and the dashboard depend on.

### Run the dashboard

```bash
# macOS/Linux (installs deps if needed, then launches)
./run_dashboard.sh

# Manual
source venv/bin/activate
streamlit run app/Home.py
```

Dashboard opens at `http://localhost:8501`.

### Run tests

```bash
source venv/bin/activate
python -m pytest tests/test_dashboard.py -v
```

## Reproducing the analysis

Run notebooks in order. Each notebook reads `data/processed/cleaned_data.csv` and writes outputs to `data/processed/`.

| Notebook | Purpose |
|---|---|
| `01-data-cleaning.ipynb` | Cleans raw data, recalculates BMI/WHR/FSH-LH ratio, produces `cleaned_data.csv` |
| `02-eda-pca.ipynb` | EDA with t-tests, chi-square tests, PCA on 27 continuous features |
| `03-xgboost-shap.ipynb` | Tuned XGBoost, SHAP feature importance, threshold optimisation, outlier sensitivity |
| `04-eda-classifier.ipynb` | Mutual information feature selection across all 41 features; fast food confounding analysis |
| `05-clustering.ipynb` | K-Means phenotype clustering on PCOS-only cohort |
| `06-random-forest.ipynb` | Random Forest classifier with cross-validation |
| `07-model-comparison.ipynb` | All five models with identical CV splits, Wilcoxon signed-rank significance testing |
| `08-screening-context-comparison.ipynb` | Sensitivity, specificity, PPV, NPV, calibration, DCA, model ranking for screening context |

## Project structure

```
├── app/
│   ├── Home.py                     # Landing page
│   ├── styles.py                   # Shared CSS and matplotlib theme
│   └── pages/
│       ├── 1_Phenotype_Explorer.py
│       ├── 2_Risk_Calculator.py
│       └── 3_Feature_Impact.py
├── data/
│   ├── raw/                        
│   └── processed/                  # cleaned_data.csv and notebook outputs
├── notebooks/                      # Analysis notebooks (run in order 01–08)
├── tests/
│   └── test_dashboard.py
├── requirements.txt
├── run_dashboard.sh
└── run_dashboard.bat
```

## Data Citation

Kottarathil, P. (2020). *Polycystic ovary syndrome (PCOS)* [Dataset]. Kaggle.
https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos

Original data collected from 10 different hospitals across Kerala, India.

## AI Disclosure

Claude Code (Anthropic) was used during development of this project to:
- Template and debug the Streamlit dashboard UI
- Help debug model integration and data pipeline issues
- Help conceptualise analytical sections across the notebooks

All analysis, modelling decisions, interpretation of results, and written conclusions are the work of the project authors. AI assistance was used as a development tool, not as a substitute for analytical judgement.