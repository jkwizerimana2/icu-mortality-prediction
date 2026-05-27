# ICU Mortality Prediction

This project builds and evaluates machine learning models to predict in-hospital mortality for ICU patients using demographic, admission, APACHE, vital sign, laboratory, and comorbidity variables.

The workflow compares regularized logistic regression and XGBoost models, then explores class weighting, threshold tuning, ROC/precision-recall performance, and SHAP-based model interpretation.

## Project Files

- `icuproject.ipynb` - Jupyter/Colab notebook containing data preparation, exploratory summary tables, modeling, evaluation, threshold tuning, and SHAP interpretation.
- `icu_mortality_prediction_report.pdf` - PDF export of the notebook workflow and results.
- `requirements.txt` - Python package dependencies needed to run the notebook locally.

## Research Question

Can routinely available ICU admission and early clinical variables be used to predict in-hospital mortality, and how do model choice and class imbalance handling affect sensitivity for death prediction?

## Data Source

The notebook expects a raw ICU mortality dataset named:

```text
training_v2.csv
```

The source data are not included in this repository. The notebook currently loads the file from a Google Drive path used during the original analysis. To reproduce the analysis, place your authorized copy of `training_v2.csv` in an accessible location and update the `pd.read_csv(...)` path in the notebook.

The target variable is `hospital_death`, renamed in the notebook as `Death`.

## Methods

The notebook performs the following steps:

1. Loads the raw ICU dataset.
2. Selects and renames clinically relevant variables into readable labels.
3. Removes variables with very high missingness: `pH`, `PaO2`, `PaCO2`, `Bilirubin`, `Albumin`, and `Urine Output`.
4. Removes highly correlated numeric variables using a correlation threshold, including `Post-Operative` and `GCS Motor`.
5. Builds a compact Table 1 stratified by survival status.
6. Splits data into stratified training and test sets.
7. Applies preprocessing with mean imputation and scaling for numeric variables, plus most-frequent imputation and one-hot encoding for categorical variables.
8. Fits LASSO logistic regression, weighted LASSO logistic regression, tuned XGBoost, and weighted tuned XGBoost.
9. Evaluates models using classification reports, ROC AUC, precision-recall curves, and threshold-specific operating points.
10. Uses SHAP values to interpret the weighted XGBoost model.

## Model Results

The test set contained 9,172 observations, including 792 deaths and 8,380 survivors.

| Model | ROC AUC | Death precision | Death recall | Death F1 |
| --- | ---: | ---: | ---: | ---: |
| LASSO logistic regression | 0.859 | 0.61 | 0.16 | 0.26 |
| Weighted LASSO logistic regression | 0.859 | 0.25 | 0.77 | 0.38 |
| Tuned XGBoost | 0.895 | 0.69 | 0.29 | 0.41 |
| Weighted tuned XGBoost | 0.891 | 0.28 | 0.79 | 0.42 |

The best overall discrimination came from tuned XGBoost with ROC AUC of 0.895. The weighted XGBoost model had slightly lower AUC but much higher recall for mortality, which may be more useful when the priority is identifying high-risk patients.

## Threshold Tuning

The notebook also evaluates custom classification thresholds instead of relying only on the default 0.50 threshold.

| Operating point | Threshold | Recall | Precision | False positive rate |
| --- | ---: | ---: | ---: | ---: |
| Weighted XGBoost, F2-max | 0.6350 | 0.699 | 0.361 | 0.117 |
| Weighted XGBoost, recall >= 0.75 | 0.5596 | 0.750 | 0.310 | 0.158 |
| Weighted LASSO, F2-max | 0.4872 | 0.793 | 0.245 | 0.231 |
| Weighted LASSO, recall >= 0.75 | 0.5232 | 0.750 | 0.259 | 0.203 |

These results show the expected clinical tradeoff: increasing mortality recall identifies more patients who die, but also increases false positives and lowers precision.

## How to Run

Clone the repository, create a Python environment, and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook icuproject.ipynb
```

If running in Google Colab, mount Google Drive and edit the notebook's `pd.read_csv(...)` path so it points to your authorized copy of `training_v2.csv`.

## Reproducibility Notes

- The raw dataset is intentionally excluded from the repository.
- The notebook uses randomized hyperparameter search with `random_state=42` where specified.
- XGBoost tuning can take several minutes because the notebook evaluates 30 randomized parameter combinations with 3-fold cross-validation.
- The notebook includes SHAP plots and model evaluation plots as notebook outputs.
- Before final publication, consider rerunning the notebook from a clean runtime so every output reflects a fresh execution.

## Limitations

This project is for statistical and machine learning practice only. It is not a validated clinical decision support system. Model performance is based on a held-out test split from the same dataset, so external validation would be required before any real-world use. The class imbalance also means that accuracy alone is misleading; recall, precision, calibration, and clinical cost tradeoffs should be considered together.

## Author

Jean D'Amour Kwizerimana

