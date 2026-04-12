# Where Will a Gym Thrive in Utah?

**STAT 486 Final Project — Aubrey Coons & Dylan Binns**

A data science pipeline that predicts whether a fitness business will thrive in Utah using Yelp business data, U.S. Census demographics, and machine learning.

---

## Research Question

> *Can neighborhood socioeconomic characteristics and business category predict whether a fitness business (gym, yoga studio, CrossFit box, etc.) will thrive in Utah?*

A business is considered **thriving** if it is currently open, has a Yelp rating ≥ 4.0, and has at least 10 reviews — a composite signal of quality and community engagement.

---

## Key Results

| Task | Best Model | Metric |
|---|---|---|
| Binary classifier (will it thrive?) | Random Forest | AUC = **0.927**, Accuracy = **87.9%** |
| Regressor (how successful?) | Gradient Boosting | R² = **0.244**, RMSE = **0.263** |
| Market segmentation | K-Means | K = **8** clusters, silhouette = 0.35+ |

**Top predictive signals:**
1. **Business category** — general gyms thrive most; boutique studios are mixed
2. **Income per competitor** — wealthy neighborhoods with few competitors are the ideal location
3. **Market gap** — prime-age population relative to existing gym supply
4. **Competition at 3km radius** — saturated local markets suppress individual business success

---

## Project Structure

```
STAT486_Final_Project/
├── thriving_model/              # Main analysis (v2 — revised research question)
│   ├── 01_data_collection.ipynb # Yelp API + Census data pull
│   ├── 02_feature_engineering.ipynb  # Feature derivation + target engineering
│   ├── run_models.py            # All modeling: classifier, regressor, clustering
│   ├── data/
│   │   ├── utah_fitness_v2.csv  # 1,081 Utah fitness businesses (main dataset)
│   │   ├── features_classifier.csv  # 164 businesses with enough history
│   │   └── features_regressor.csv   # 1,080 open businesses
│   └── figures/                 # All output plots
│       ├── 01_data_overview.png
│       ├── 02_competition_income_vs_status.png
│       ├── 02_correlation_matrix_v2.png
│       ├── 03a_roc_curves.png
│       ├── 03a_feature_importance.png
│       ├── 03b_regression_comparison.png
│       ├── 03b_feature_importance.png
│       ├── 03c_elbow_silhouette.png
│       └── 03c_utah_clusters_map.png
├── progress/                    # Deliverable write-ups
│   ├── 01_proposal.md
│   ├── 02_eda.md
│   ├── 03_supervised.md
│   └── 04_unsupervised.md
├── demo.ipynb                   # Clean end-to-end demo notebook
├── requirements.txt
├── README.md
└── progress/
    ├── slides.md                # Presentation slides (Markdown format)
    └── ...
```

---

## How to Reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Yelp API key and Census API key

Create a `.env` file or export environment variables:

```bash
export YELP_API_KEY="your_yelp_key_here"
export CENSUS_API_KEY="your_census_key_here"
```

Census API keys are free at: https://api.census.gov/data/key_signup.html
Yelp Fusion API keys are free at: https://docs.developer.yelp.com/

### 3. Run in order

```bash
# Step 1: Collect data (Yelp + Census) — creates thriving_model/data/utah_fitness_v2.csv
# Run thriving_model/01_data_collection.ipynb in Jupyter

# Step 2: Engineer features — creates features_classifier.csv and features_regressor.csv
# Run thriving_model/02_feature_engineering.ipynb in Jupyter

# Step 3: Train all models and generate figures
cd thriving_model/
python run_models.py
```

All figures are saved to `thriving_model/figures/`. All `random_state=42`.

### 4. View the demo

Open `demo.ipynb` for a clean, self-contained walkthrough of the full pipeline using the pre-collected data.

---

## Data Sources

| Source | Description | Access |
|---|---|---|
| [Yelp Fusion API](https://docs.developer.yelp.com/) | Business name, rating, review count, category, coordinates, open/closed status | Free tier (500 calls/day) |
| [U.S. Census ACS 5-Year (2022)](https://www.census.gov/data/developers/data-sets/acs-5year.html) | ZCTA-level demographics: income, home value, age distribution, education, population | Public, free API |

**Coverage:** 1,081 fitness businesses across 84xxx Utah zip codes with population ≥ 5,000.

**Categories searched:** gyms, yoga, pilates, crossfit, martialarts, rockclimbing, personal trainers

**Note:** Yelp's API returns only currently-listed businesses — permanently closed businesses are not surfaced. All `is_closed` values in the dataset are 0 by construction. The `is_thriving` target uses rating/review thresholds as a proxy for business health.

---

## Features Used

| Feature | Source | Description |
|---|---|---|
| `category_group` | Yelp | Grouped category: general_gym, mind_body, martial_arts, personal_training, climbing_outdoor, other |
| `price` | Yelp | Price tier (1–4, imputed with median) |
| `median_income` | Census ACS | Median household income by ZCTA |
| `median_home_value` | Census ACS | Median home value by ZCTA |
| `median_age` | Census ACS | Median age by ZCTA |
| `pct_bachelors` | Census ACS | % population with bachelor's degree |
| `pct_prime_gym_age` | Census ACS | % population aged 20–44 |
| `total_pop` | Census ACS | Total population by ZCTA |
| `competition_1km` | Derived (haversine) | Fitness businesses within 1km radius |
| `competition_3km` | Derived (haversine) | Fitness businesses within 3km radius |
| `market_gap` | Derived | `pct_prime_gym_age × total_pop / (gyms_in_zip + 1)` |
| `gym_density_per_1k` | Derived | Gyms per 1,000 residents in zip |
| `income_per_competitor` | Derived | Median income / (competition_3km + 1) |

---

## Limitations

- Yelp does not return closed businesses — we cannot model business failure directly
- Business age/longevity data was not obtainable at scale (BBB, Wayback Machine, and Yelp Reviews API all had prohibitive limitations)
- R² of 0.244 on success_score suggests important unmeasured factors (marketing, management, franchise affiliation)
- K-Means assumes spherical clusters; Salt Lake City proper is an outlier on competition density
- Yelp review counts conflate time-in-business with popularity — older businesses accumulate reviews regardless of quality
