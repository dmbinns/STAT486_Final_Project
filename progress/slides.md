# Where Will a Gym Thrive in Utah?
### STAT 486 Final Project Presentation
**Aubrey Coons-Call, Daniela Binns, and Sedona | April 2026**

---

## Slide 1: Problem & Motivation

**Research Question:**  
Can neighborhood characteristics and business category predict whether a fitness business will thrive in Utah?

**Why it matters:**
- Utah's fitness industry is growing — but most gyms fail within 5 years
- Entrepreneurs need data-driven location guidance, not just intuition
- Yelp + Census data provides a rich, real-world testbed

**What "thriving" means:**  
Open on Yelp + Rating ≥ 4.0 + Review count ≥ 10

---

## Slide 2: Data

**Sources:**
- Yelp Fusion API — 1,081 Utah fitness businesses
  - Rating, review count, category, price tier, coordinates
  - 7 categories: gyms, yoga, pilates, crossfit, martial arts, rock climbing, personal trainers
- U.S. Census ACS 2022 (5-year) — ZCTA-level demographics
  - Median income, home value, education, age distribution, population

**Derived features:**
- Competition density (businesses within 1km and 3km haversine radius)
- `market_gap` = prime-age pop / (gyms in zip + 1)
- `income_per_competitor` = median income / (3km competitors + 1)
- `gym_density_per_1k` = gyms per 1,000 residents

**Coverage:** 84xxx Utah zip codes with population ≥ 5,000

---

## Slide 3: EDA Highlights

**Target distribution (classifier subset, n=164):**
- 101 thriving (61.6%) / 63 not thriving (38.4%)

**Key patterns:**
- General gyms thrive at ~70% vs. ~45% for boutique studios
- High `income_per_competitor` strongly associated with thriving
- Competition density (3km) negatively correlated with success_score (r ≈ −0.28)
- Demographic features (income, education) show moderate positive correlation

*[See figures/02_competition_income_vs_status.png and 02_correlation_matrix_v2.png]*

---

## Slide 4: Supervised Models — Classifier

**Task:** Predict `is_thriving` (binary)  
**Data:** 164 businesses, 80/20 stratified split, 5-fold GridSearchCV

| Model | AUC | Accuracy |
|---|---|---|
| Logistic Regression | 0.923 | 82.1% |
| **Random Forest** | **0.927** | **87.9%** |
| Gradient Boosting | 0.892 | 84.8% |

**Top features (Random Forest):**
1. `category_group_general_gym` — general gyms thrive most
2. `category_group_mind_body` — yoga/pilates also perform well
3. `income_per_competitor` — best location signal
4. `market_gap` — underserved markets favor success

*[See figures/03a_roc_curves.png and 03a_feature_importance.png]*

---

## Slide 5: Supervised Models — Regressor

**Task:** Predict `success_score` = 0.5×norm(rating) + 0.5×norm(log_reviews)  
**Data:** 1,080 open businesses, 80/20 split

| Model | RMSE | R² |
|---|---|---|
| Ridge | 0.278 | 0.220 |
| Random Forest | 0.279 | 0.215 |
| **Gradient Boosting** | **0.263** | **0.244** |

**Top regressor features:**
1. `competition_3km` — saturation hurts success scores
2. `median_age` — older neighborhoods = higher engagement
3. `pct_bachelors` — education predicts willingness to engage

**R² = 0.244** — meaningful signal, but important unmeasured factors remain (business age, marketing, management quality)

*[See figures/03b_regression_comparison.png and 03b_feature_importance.png]*

---

## Slide 6: K-Means Market Segmentation

**Task:** Cluster Utah zip codes into market archetypes  
**Method:** K-Means on 8 zip-level features, K=8 selected by silhouette score

**Key market archetypes discovered:**
- **Affluent suburban** — high income, low competition → thriving rate 80–100%
- **Dense urban core** — high competition despite large population → thriving rate ~35%
- **High-gap rural** — underserved but thin consumer base → mixed results
- **Mid-density moderate** — median performance across all metrics

**Geographic pattern:** Clear spatial clustering along Wasatch Front with distinct suburban vs. urban segments

*[See figures/03c_utah_clusters_map.png and 03c_elbow_silhouette.png]*

---

## Slide 7: Connection Between Methods

**Supervised → Clustering alignment:**  
- `income_per_competitor` = top classifier feature  
- Clusters defined by this ratio show dramatically different thriving rates (33% to 100%)  
- Clustering validates and *explains* the supervised signal: it's not just income or competition alone — it's the ratio

**Clustering → Actionable insight:**  
- A new gym should target: affluent suburban zip codes with low existing competition  
- Salt Lake City proper = avoid (dense urban cluster, lowest thriving rates)  
- Lehi, Draper, South Jordan, Saratoga Springs area = optimal market entry zones

---

## Slide 8: Limitations & Next Steps

**Limitations:**
- Yelp API does not return closed businesses — cannot model failure directly
- Business age/longevity unavailable at scale (BBB, Wayback, Yelp Reviews API all failed)
- `is_thriving` threshold (4.0 stars, 10 reviews) is a proxy — not ground truth
- R² = 0.244 suggests important unmeasured variables

**Next steps:**
- Incorporate business age from Utah Division of Corporations licensing data
- Add Google Maps data for a second opinion on open/closed status
- Try DBSCAN for geographic density-aware clustering (handles outliers better)
- Build a Streamlit tool that scores any zip code for a given business category

---

## Slide 9: Conclusion

> Fitness businesses in Utah thrive when they operate in the right *category* in the right *market context* — specifically, high-income neighborhoods where competition hasn't yet saturated demand.

- **Random Forest classifier achieves AUC = 0.927** — strong predictive signal for thriving/not thriving
- **Business category dominates** over raw demographics
- **Income per competitor** is the single most useful location signal
- **K-Means reveals 8 distinct market archetypes** that explain why geographic location matters beyond just income or density alone

**For a new gym owner:** open a general gym in an affluent suburban Utah zip with few existing competitors and a large 20–44-year-old population.

---

*Figures and reproducibility: see `thriving_model/figures/` and `README.md`*
