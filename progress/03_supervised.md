# Deliverable 3: Supervised Modeling Checkpoint

## 1) Problem Context and Research Question

This project analyzes the Utah fitness business market by combining Yelp Fusion API data with U.S. Census ACS demographics. The central question is: **can neighborhood characteristics and business type predict whether a fitness business will thrive in Utah?**

We define *thriving* as: open on Yelp AND rating ≥ 4.0 AND review_count ≥ 10. We also model a continuous `success_score` = 0.5 × normalized(rating) + 0.5 × normalized(log_reviews).

---

## 2) Supervised Models Implemented

All preprocessing (median imputation, standard scaling, one-hot encoding) is inside a `sklearn Pipeline` to prevent data leakage. GridSearchCV with 5-fold CV is used for hyperparameter tuning. `random_state=42` throughout.

### Task A: Binary Classifier — Will this business thrive?

**Dataset:** 164 businesses with sufficient review history (101 thriving / 63 not thriving)
**Split:** 80/20 stratified

| Model | Best Params | AUC | Accuracy |
|---|---|---|---|
| Logistic Regression | C=0.1 | 0.923 | 82.1% |
| **Random Forest** | depth=None, trees=200, split=2 | **0.927** | **87.9%** |
| Gradient Boosting | lr=0.05, depth=3, trees=100 | 0.892 | 84.8% |

**Winner: Random Forest** (AUC=0.927, Acc=87.9%)

Top features (Random Forest classifier):
1. `category_group_general_gym` — General gyms thrive at far higher rates than boutique formats
2. `category_group_mind_body` — Yoga/pilates studios also perform well
3. `income_per_competitor` — Higher income relative to competition density is strongly predictive
4. `market_gap` — Underserved markets with large prime-age populations favor thriving
5. `median_income` — Wealthier neighborhoods support fitness business survival

### Task B: Regressor — How successful will an open business be?

**Dataset:** 1,080 open businesses (continuous success_score target)
**Split:** 80/20

| Model | Best Params | RMSE | R² |
|---|---|---|---|
| Ridge | alpha=100 | 0.278 | 0.220 |
| Random Forest | depth=5, trees=100 | 0.279 | 0.215 |
| **Gradient Boosting** | lr=0.05, depth=3, trees=200 | **0.263** | **0.244** |

**Winner: Gradient Boosting** (R²=0.244, RMSE=0.263)

Top features (Gradient Boosting regressor):
1. `competition_3km` — More nearby competitors → lower success scores
2. `median_age` — Older neighborhoods correlate with higher engagement
3. `pct_bachelors` — Education level predicts willingness to engage with fitness
4. `market_gap` — Underserved markets show higher success scores
5. `median_income` — Income remains a significant predictor

---

## 3) Model Comparison and Selection

For the classification task, Random Forest narrowly leads on AUC (0.927 vs. 0.923 for LR). The AUC gap is small but RF also outperforms on accuracy (87.9% vs. 82.1%), making it the preferred classifier.

For regression, Gradient Boosting achieves the best R² (0.244) and RMSE (0.263). The moderate R² reflects inherent noise in Yelp engagement data — some thriving businesses simply haven't been discovered yet, while some non-thriving businesses have inflated review counts from early hype.

---

## 4) Explainability

The classifier's most important features confirm two key hypotheses:
1. **Business type matters most** — general gyms have broader market appeal than boutique formats
2. **Competition-adjusted income is the key environmental signal** — a wealthy neighborhood with few competitors is the ideal location

The regressor shifts emphasis toward **competition density and demographics**, suggesting that for businesses already open, the local market environment explains variance in how well they do rather than their category.

---

## 5) Final Takeaways

- Random Forest (AUC=0.927) can meaningfully predict whether a Utah fitness business will thrive, well above chance
- Business category (general gym vs. mind-body vs. other) is the single strongest signal
- After category, the best location signal is `income_per_competitor` — places where potential income exceeds the competitive supply
- R² of 0.244 on success_score indicates that not all variance is captured; business-specific factors (marketing, management quality, age of business) not available via Yelp would likely improve the model substantially

Figures: `thriving_model/figures/03a_roc_curves.png`, `03a_feature_importance.png`, `03b_regression_comparison.png`, `03b_feature_importance.png`
