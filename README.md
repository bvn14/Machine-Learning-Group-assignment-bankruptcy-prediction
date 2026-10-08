# Bankruptcy Prediction – Polish Companies

Group assignment for **DA2111 – Statistical and Machine Learning**
Department of Decision Science, Faculty of Business, University of Moratuwa (Semester 4)

## Problem
Can we classify a company as **bankrupt** or **non-bankrupt** from its financial ratios, so that investors, creditors and management can act early?

- **Task:** binary classification
- **Target:** `class` (0 = not bankrupt, 1 = bankrupt)

## Dataset
[Polish Companies Bankruptcy Data](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data) (UCI ML Repository).
The five yearly files were combined: **43,405 companies × 64 financial ratios** (`Attr1`–`Attr64`: profitability, liquidity, leverage, efficiency).
Only about **4.8%** of companies are bankrupt (2,091 vs 41,314), so the data is heavily imbalanced.

## Workflow
1. **Load** the five ARFF files and merge them.
2. **Explore:** shape, types, summary statistics, skewness (all features highly skewed).
3. **Preprocess**
   - Missing values: median imputation
   - Duplicates: checked and removed
   - Outliers: IQR rule (rows removed) – leaves ~2,200 rows
   - Skewness: `log1p` transform on features with skew > 1
   - Scaling: `StandardScaler`
4. **EDA:** histograms, box plots, class-balance table, correlation heatmap.
5. **Split:** 70% train / 15% validation / 15% test (stratified).
6. **Class balancing:** SMOTE on the training set only.
7. **Models:** Logistic Regression (baseline), Decision Tree, Random Forest, XGBoost.
8. **Tuning:** `RandomizedSearchCV` (100 iterations, 5-fold CV, ROC-AUC scoring), then retrain.
9. **Evaluate:** accuracy, precision, recall, F1, ROC-AUC on validation and on the final test set.

## Results (final test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.889 | 0.267 | 0.800 | 0.400 | 0.861 |
| Decision Tree | 0.960 | 0.550 | 0.733 | 0.629 | 0.880 |
| Random Forest | 0.966 | 0.833 | 0.333 | 0.476 | 0.899 |
| **XGBoost** | **0.975** | 0.818 | 0.600 | **0.692** | **0.942** |

**Best model: XGBoost.** Top features were profit on operating activities / financial expenses, retained earnings / total assets, EBIT / total assets and gross profit / total assets – i.e. profitability and financial stability.

## Limitations
- IQR outlier removal discarded ~95% of rows, including most bankrupt companies, so the test set holds only ~15 bankrupt cases. Results are therefore noisy.
- The scaler was fitted before the train/test split, which leaks information from the test set.
- Tuned models did not always beat the untuned ones on the validation set.

**Possible improvements:** cap (winsorize) outliers instead of deleting rows, fit scaling inside a pipeline, use cross-validation on the full data, tune for recall/PR-AUC, and try class weights instead of SMOTE.

## Repository structure
```
├── README.md
├── requirements.txt
├── data/                  # instructions to download the dataset
├── notebooks/
│   ├── group_assignment.ipynb
│   └── group_assignment_notebook.pdf
└── report/
    └── ML_Final_Report.pdf
```

## Run it
```bash
pip install -r requirements.txt
jupyter notebook notebooks/group_assignment.ipynb
```

## Infrastructure
Google Colab (free tier, ~12.7 GB RAM), accessed from group members' laptops.
