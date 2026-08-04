# Stroke Prediction - Machine Learning II Project (Group O)

Predicting stroke occurrence from demographic, lifestyle, and medical attributes using two supervised classification methods — **Support Vector Machines (SVM)** and **Logistic Regression (Ridge/Lasso)** — with a focus on model comparison and interpretable machine learning (XAI).

## Overview

This project was completed for the *Machine Learning II* course. It analyzes the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle to predict whether a patient is likely to experience a stroke, based on 10 demographic, lifestyle, and medical predictors.

Because this is a medical screening problem, the analysis prioritizes **sensitivity** (catching true stroke cases) over raw accuracy, since a missed diagnosis is far more costly than a false alarm.

## Repository Structure

```
Stroke_Prediction_ML-II/
│
├── README.md
├── .gitignore
│
├── data/
│   └── healthcare-dataset-stroke-data.csv
│
├── report/
│   └── ML2_Report_Group_O.pdf
│
└── analysis/
    └── stroke_prediction_analysis.Rmd
```

| File | Description |
|---|---|
| `analysis/stroke_prediction_analysis.Rmd` | Full R Markdown analysis: data exploration, preprocessing, model training, evaluation, and interpretability |
| `report/ML2_Report_Group_O.pdf` | Final written report submitted for grading |
| `data/healthcare-dataset-stroke-data.csv` | Raw dataset (5,110 patient records) |

## Dataset

- **Source:** [Kaggle – Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Observations:** 5,110 patients
- **Target:** `stroke` (binary: 0 = No Stroke, 1 = Stroke) — highly imbalanced (~5% positive class)

| Variable | Type | Description |
|---|---|---|
| gender | nominal | Patient gender |
| age | continuous | Age in years |
| hypertension | binary | Presence of hypertension |
| heart_disease | binary | Presence of heart disease |
| ever_married | nominal | Marital status |
| work_type | nominal | Employment type |
| Residence_type | nominal | Urban / Rural |
| avg_glucose_level | continuous | Average blood glucose level |
| bmi | continuous | Body Mass Index |
| smoking_status | nominal | Smoking behavior |

### Key preprocessing steps
- `bmi` converted from character to numeric; 201 missing values imputed with the **median** (BMI is right-skewed, so median avoids distortion by extreme values)
- Categorical variables one-hot encoded via `model.matrix`
- Class imbalance handled with **weighted models** (inverse class-frequency weights)
- Stratified **60/20/20 train/validation/test split**

## Methods

Two ML methods were compared, one linear (ML1) and one from the ML2 curriculum:

1. **Support Vector Machine (SVM)** — linear and radial kernels, hyperparameter (`cost`) tuned via 5-fold cross-validation
2. **Logistic Regression** — baseline, full model, and Ridge/Lasso-regularized variants (`glmnet`)

### Model comparison (validation set)

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Linear SVM (2 features) | 0.718 | 0.90 | 0.709 | 0.874 |
| Linear SVM (4 features) | 0.711 | 0.94 | — | 0.872 |
| Linear SVM (all features) | 0.708 | 0.94 | 0.697 | 0.869 |
| **Linear SVM (all features + CV)** | **0.713** | **0.94** | **0.702** | **0.868** |
| Radial SVM (all features) | 0.728 | 0.88 | — | 0.845 |
| Radial SVM (all features + CV) | 0.841 | 0.24 | 0.871 | 0.700 |
| Full Logistic Regression | 0.733 | 0.90 | 0.724 | 0.869 |
| Ridge Logistic Regression | 0.725 | 0.92 | 0.725 | 0.870 |
| **Lasso Logistic Regression** | **0.725** | **0.92** | **0.715** | **0.870** |

**Final selected models:**
- **Linear SVM (All Features + Cross-Validation)** — highest sensitivity (0.94), the safest choice for a medical screening context.
- **Lasso Logistic Regression** — best overall balance of accuracy, sensitivity, and interpretability, with automatic feature selection.

> Note: The Radial SVM (all features + CV) achieved the highest raw accuracy (0.841) but is clinically unsafe — it misses ~76% of true stroke cases (sensitivity 0.24), illustrating why accuracy alone is a poor metric for imbalanced medical data.

## Interpretable Machine Learning (XAI)

Both global and local explanation methods were applied to compare how the two models make decisions:

- **Global feature importance:** SVM coefficient weights and Permutation Feature Importance (PFI)
- **Feature effects:** Partial Dependence Plots (PDP) and Accumulated Local Effects (ALE)
- **Local/individual explanations:** SHAP values (via `iml` for SVM, `DALEX`/`DALEXtra` for Lasso)

**Key findings:**
- **Age** is by far the strongest predictor for both models — stroke risk stays low until ~55 years, then rises sharply (a non-linear "threshold" effect confirmed by both PDP and ALE).
- **Average glucose level** is the second most influential predictor.
- Work type and smoking status contribute moderately.
- BMI, hypertension, heart disease, gender, and residence type have comparatively small effects.
- SHAP explanations for individual patients confirm age and glucose level as the dominant drivers of predicted risk.

## Requirements

This project uses R with the following packages:

```r
install.packages(c(
  "caret", "pROC", "dplyr", "e1071", "iml", "glmnet",
  "MASS", "DALEX", "DALEXtra", "localModel", "gridExtra"
))
```

## Reproducing the Analysis

1. Clone the repository
2. Open `analysis/stroke_prediction_analysis.Rmd` in RStudio
3. Install the required packages (above)
4. Knit the document — the `.Rmd` reads the dataset via a relative path (`../data/healthcare-dataset-stroke-data.csv`), so it must be knit from within the `analysis/` folder to resolve correctly

```r
setwd("path/to/stroke-prediction-ml2/analysis")
rmarkdown::render("stroke_prediction_analysis.Rmd")
```

## Report Structure

The full write-up (`report/ML2_Report_Group_O.pdf`) covers:
1. Introduction and descriptive data analysis
2. Mathematical overview of SVM and Logistic Regression/Lasso
3. Model fitting process, hyperparameter tuning, and evaluation
4. XAI-based interpretation and comparison of both models
5. Graphical presentation of results
6. Bibliography

## References

- Kaggle Stroke Prediction Dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- `iml`, `DALEX`, `DALEXtra`, `localModel` R packages for model-agnostic interpretability
- `glmnet` for Ridge/Lasso regularized regression
- `e1071` for SVM implementation

## Authors

Group O — Machine Learning II, Berliner Hochschule für Technik (BHT)
