
---

## Approach

### 1. Data Understanding and Cleaning
- Identified `TotalCharges` stored as string instead of float due to 11 blank values
- All 11 blank values corresponded to customers with `tenure = 0` (not yet billed) - filled with 0
- No duplicate rows confirmed before dropping `customerID`

### 2. Exploratory Data Analysis
- Class imbalance confirmed: 73.5% No churn / 26.5% Churn
- Key findings from categorical features:
  - `Contract` is the strongest predictor: Month-to-month customers churn at 43% vs 3% for Two-year contracts
  - `InternetService`: Fiber optic customers churn at 42%, DSL at 19%, no internet at 7%
  - `PaymentMethod`: Electronic check customers churn at 45% vs 15-20% for other methods
  - `gender` and `PhoneService` show no meaningful difference in churn rate
- Key findings from numeric features:
  - Newer customers (0-6 months) churn at 52.9% vs 5.3% for long-term customers (48-72 months)
  - Higher `MonthlyCharges` associated with higher churn

### 3. Statistical Testing
- **Chi-square + Cramér's V** for categorical features: confirmed `gender` and `PhoneService` have no statistically significant relationship with Churn (p > 0.05)
- **Mann-Whitney U**: `tenure` shows the strongest effect (r = 0.367, medium)
- **Spearman correlation**: confirmed multicollinearity between `TotalCharges` and `tenure` (r = 0.890)

### 4. Feature Engineering
- Dropped `TotalCharges` (multicollinearity: r = 0.890 with tenure)
- Dropped `gender` and `PhoneService` (p > 0.05, no predictive value)
- Binary encoding for Yes/No columns
- Ordinal encoding for `Contract` (Month-to-month=0, One year=1, Two year=2)
- One-Hot Encoding for nominal categorical columns
- Created `tenure_group` (0-6m, 6-12m, 12-24m, 24-48m, 48-72m) based on observed churn rate changes
- Created `num_services` (sum of 6 binary service columns) - customers with 1-2 services churn at 46%/36% vs 5.3% for 6 services
- Stratified 80/20 train/test split (Churn rate preserved at 26.5% in both sets)

### 5. Modelling

**Imbalance handling:** Compared 3 techniques using Logistic Regression and 5-fold CV:
- Baseline: F1 = 0.591
- SMOTE: F1 = 0.610 (initial 0.797 was due to data leakage - fixed by applying SMOTE inside CV pipeline)
- `class_weight="balanced"`: F1 = 0.633 - selected for all models

**Hyperparameter tuning:** Optuna with early stopping (patience=15) and 5-fold cross-validation

| Model | CV F1 | Trials |
|---|---|---|
| Logistic Regression | 0.636 | 28 |
| Decision Tree | 0.620 | 17 |
| Random Forest | 0.638 | 15 |
| XGBoost | 0.636 | 18 |

### 6. Final Evaluation on Test Set

| Model | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.622 | 0.514 | 0.786 | 0.842 |
| Decision Tree | 0.621 | 0.514 | 0.783 | 0.816 |
| Random Forest | 0.640 | 0.542 | 0.783 | 0.844 |
| **XGBoost** | **0.633** | **0.523** | **0.802** | **0.843** |

**XGBoost selected** as the best model based on Recall (0.802) - the most business-relevant metric, since missing a churner is more costly than sending an unnecessary retention offer.

**Confusion Matrix (XGBoost):**
- True Positives: 300 (churners correctly identified)
- False Negatives: 74 (churners missed)
- False Positives: 274 (non-churners wrongly flagged)
- True Negatives: 761

### 7. Interpretation (Feature Importance and SHAP)

Top predictors (XGBoost feature importance):
1. `Contract` (27.0%) - Month-to-month contracts strongly increase churn
2. `InternetService_Fiber optic` (16.8%) - Fiber optic customers churn significantly more
3. `InternetService_No` (7.9%) - No internet customers almost never churn
4. `PaymentMethod_Electronic check` (4.5%)
5. `tenure_group` (4.4%) - engineered feature

SHAP confirms directionality: short tenure, month-to-month contracts, Fiber optic internet and high monthly charges all increase churn probability.

---

## Business Recommendations

1. **Focus retention on new customers (0-12 months):** 47.4% of new customers churn. Invest in onboarding and early incentives.
2. **Incentivize long-term contracts:** Month-to-month churn at 43% vs 3% for two-year. Offer discounts to switch.
3. **Review Fiber optic pricing and quality:** 42% churn rate suggests pricing or quality issues.
4. **Upsell services:** Customers with 3+ services churn at under 20%. Encourage add-ons early.
5. **Investigate Electronic check:** 45% churn rate vs 15-20% for other methods.

> These recommendations are based on correlations. Causality requires further investigation (e.g. A/B testing).

---

## Limitations and Next Steps

- Dataset is a static snapshot - no temporal patterns captured
- Causality cannot be confirmed without A/B testing
- Threshold tuning could improve the precision/recall trade-off
- A cost-benefit analysis with real retention offer costs would sharpen the business value

---

## Tech Stack

`Python` `pandas` `NumPy` `scikit-learn` `XGBoost` `Optuna` `SHAP` `matplotlib` `seaborn` `scipy`

---

## Author

**Evdoxia Lamprinou**
- LinkedIn: [evdoxia-lamprinou](https://linkedin.com/in/evdoxia-lamprinou)
- Kaggle: [evdoxialamprinou](https://www.kaggle.com/evdoxialamprinou)
