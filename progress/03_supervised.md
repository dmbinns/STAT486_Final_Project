# Deliverable 3: Supervised Modeling Checkpoint

## 1) Problem Context and Research Question

This project analyzes the Utah fitness business market by combining Yelp Fusion API business data with U.S. Census demographic data. The central question is: **can neighborhood socio-economic characteristics (income, education) and business category predict customer engagement — measured by review count — for fitness businesses in Utah?**

---

## 2) Supervised Models Implemented

Both models predict `log_review_count` (log-transformed review count), using an 80/20 train/test split with 5-fold cross-validation for tuning. All preprocessing (imputation, scaling, one-hot encoding) is handled inside a `sklearn` Pipeline to prevent data leakage.

| Model | Type | Key Hyperparameters Explored | Best Params | CV RMSE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| Ridge Regression | Regularized linear regression (baseline) | `alpha`: [0.1, 1.0, 10.0, 100.0] | `alpha = 10.0` | 1.011 | 0.878 | 0.047 |
| Random Forest | Ensemble tree-based regressor | `n_estimators`: [100, 200]; `max_depth`: [None, 10, 20]; `min_samples_split`: [2, 5] | depth=10, split=2, trees=100 | 0.797 | 0.734 | 0.334 |

**Features used:** `rating`, `median_income`, `pct_bachelors`, `total_pop`, `median_age`, `distance`, `category` (one-hot encoded)

**Validation setup:** `GridSearchCV` with 5-fold CV on the training set; final evaluation on a held-out test set (20%).

---

## 3) Model Comparison and Selection

Random Forest outperformed Ridge Regression on every metric — lower test RMSE (0.734 vs. 0.878) and substantially higher R² (0.334 vs. 0.047). The gap is large: Ridge explains almost none of the variance in log review count, suggesting that the relationships between features and engagement are non-linear and involve interactions that a linear model cannot capture.

The most notable challenge was the low overall explanatory power of the feature set. Even the Random Forest explains only ~33% of variance, which aligns with the EDA finding that demographic variables (income, education) have near-zero correlation with review count. Most of the signal comes from `rating` and `distance`, not neighborhood socio-economics.

Overfitting was a minor concern for Random Forest — the CV RMSE (0.797) is higher than the test RMSE (0.734), indicating reasonable but not perfect generalization. Constraining `max_depth` to 10 helped prevent deeper overfitting.

---

## 4) Explainability and Interpretability

Feature importances from the best Random Forest model are shown below:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `rating` | 0.556 |
| 2 | `distance` | 0.106 |
| 3 | `pct_bachelors` | 0.080 |
| 4 | `median_income` | 0.053 |
| 5 | `category_gyms` | 0.045 |
| 6 | `total_pop` | 0.039 |
| 7 | `median_age` | 0.033 |
| 8 | `category_yoga` | 0.012 |
| 9 | `category_healthtrainers` | 0.008 |
| 10 | `category_climbing` | 0.007 |

**Interpretation:** `rating` dominates at 55.6% of total importance — businesses with higher star ratings tend to accumulate more reviews, consistent with the positive feedback loop where satisfied customers are more likely to leave reviews. `distance` (from the search centroid) ranks second, suggesting that businesses closer to population centers see more engagement. Neighborhood income and education contribute modestly (combined ~13%), confirming the EDA finding that demographics are not strong drivers of fitness business engagement on their own. Gym category accounts for ~4.5%, reflecting that general gyms attract broader audiences than boutique categories like yoga or pilates.

---

## 5) Final Takeaways

The supervised learning analysis reveals that **business-level attributes — particularly quality (`rating`) and location accessibility (`distance`) — are far more predictive of customer engagement than neighborhood demographics.** Income and education level have limited predictive value, challenging the initial hypothesis that wealthier, more educated neighborhoods would produce more engaged fitness consumers.

Random Forest is the preferred model for this task due to its ability to capture non-linear relationships and feature interactions. However, the moderate R² (0.334) indicates that important drivers of review count — such as marketing, brand recognition, or time in business — are not captured in the current feature set. Future work should consider incorporating business age or social media presence as additional predictors.

In answer to the research question: neighborhood socio-economics have a **weak and indirect** influence on fitness business popularity in Utah. Category matters at the margins (gyms outperform boutique formats), but the strongest predictor of how many reviews a business accumulates is simply how good it is.
