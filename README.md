# Heart Disease Classification

A clean, portfolio-ready machine learning project for predicting heart disease using the **Cleveland Heart Disease dataset**.

This repository compares multiple models on both a **baseline processed dataset** and a **feature-engineered dataset**, then selects the strongest approach based on validation and test performance.

## Highlights
- End-to-end binary classification pipeline
- Comparison of raw vs. feature-engineered features
- Multiple classic ML models plus ensemble learning
- Modular Python code organized under `src/`
- Reproducible structure ready for GitHub and CV showcase

## Best Result
- **Best model:** Stacking Ensemble
- **Dataset:** Feature-engineered
- **Validation Accuracy:** `0.9333`
- **Test Accuracy:** `0.9032`
- **Validation F1-score:** `0.9333`
- **Test F1-score:** `0.9032`

## Problem Statement
The goal is to classify whether a patient is likely to have heart disease based on medical indicators such as age, chest pain type, cholesterol, maximum heart rate, exercise-induced angina, and related clinical features.

Target label:
- `0` -> no heart disease
- `1` -> heart disease present

## Models
### Required models
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)
- Decision Tree
- K-Means
- Stacking Ensemble

### Additional models
- Logistic Regression
- Random Forest
- Extra Trees
- Support Vector Machine (RBF)
- Support Vector Machine (Linear)

## Repository Structure
```text
heart-disease-classification/
├── data/
│   ├── raw/
│   │   └── cleveland.csv
│   └── processed/
│       ├── raw_train.csv
│       ├── raw_val.csv
│       ├── raw_test.csv
│       ├── fe_train.csv
│       ├── fe_val.csv
│       └── fe_test.csv
├── outputs/
├── src/
│   └── heart_disease/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── feature_engineering.py
│       ├── modeling.py
│       └── visualization.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to Run
From the project root:
```bash
python main.py
```

## Output Files
After running the project, the `outputs/` folder will contain:
- `model_results.csv`
- `required_models_test_accuracy.png`
- `all_models_test_accuracy.png`
- `best_model_confusion_matrix.png`
- `summary.json`

## Notes
- The original `cleveland.csv` file has no header row, so column names are assigned manually in the loader.
- The feature engineering pipeline adds ratio features and an age bin, then selects the top features using mutual information.
- If you want a strict academic version, prioritize the 5 required models. If you want the strongest benchmark, include the additional models.

## CV-Friendly Summary
Built a modular machine learning project for heart disease classification using Python, Pandas, Scikit-learn, and Matplotlib. Implemented preprocessing, feature engineering, model comparison, and ensemble learning, achieving **90.32% test accuracy** with a stacking model.
