# Deliverable 4: Unsupervised Learning Checkpoint

## 1) Research Question (Revised)

Can we identify distinct **fitness market segments** across Utah zip codes, and do some segments produce significantly higher rates of thriving businesses? A thriving business is defined as one that is open, has a Yelp rating ≥ 4.0, and has accumulated at least 10 reviews — a composite proxy for quality and community engagement.

---

## 2) Unsupervised Method: K-Means Clustering

We applied K-Means clustering to **zip-code-level** market profiles rather than individual businesses, treating each zip code as an observation. Features used for clustering:

| Feature | Description |
|---|---|
| `median_income` | Median household income (Census ACS) |
| `median_home_value` | Median home value (Census ACS) |
| `pct_prime_gym_age` | % of population aged 20–44 |
| `pct_bachelors` | % with bachelor's degree or higher |
| `total_pop` | Total population |
| `gyms_in_zip` | Number of fitness businesses per zip |
| `market_gap` | Prime-age population / (gyms + 1) |
| `avg_rating` | Average Yelp rating across businesses in zip |

All features were standardized with `StandardScaler` before clustering. We tested K = 2–8, evaluating both within-cluster inertia (elbow method) and silhouette score. The silhouette method selected **K = 8** as optimal.

---

## 3) Results

The elbow and silhouette plots are saved in `thriving_model/figures/03c_elbow_silhouette.png`.

The final 8 clusters reveal meaningful market archetypes. Thriving rates (% of businesses meeting all three thresholds) varied substantially across clusters — from ~33% in the most saturated, lower-income zip codes to 100% in high-income, low-competition segments. The geographic cluster map (`03c_utah_clusters_map.png`) shows clear spatial patterns: high-income Wasatch Front suburbs form distinct clusters separate from rural and mid-density corridors.

Key segment types identified:

- **High-gap, low-competition rural**: Low saturation, strong potential but thin consumer base
- **Affluent suburban**: High income + education + low competition → highest thriving rates
- **Dense urban core**: High competition density suppresses individual business success despite large population
- **Mid-density moderate income**: Median performance across the board

---

## 4) Connection to Supervised Learning

Cluster membership was merged back into the full dataset. Cluster identity ranked among the more informative contextual variables — zip codes in the "affluent suburban" cluster had significantly higher probabilities of housing a thriving business, consistent with the classifier's finding that `income_per_competitor` is a top-3 feature. Clustering thus provides interpretable market segments that complement the black-box predictive models.

---

## 5) Limitations

- Zip-code aggregation loses within-zip heterogeneity (e.g., one wealthy neighborhood inside a low-income zip)
- K-Means assumes spherical clusters and is sensitive to outliers; Salt Lake City proper is a strong outlier on competition density
- "Thriving" is defined from Yelp data alone — businesses with few reviews may be thriving offline
- No temporal data: we cannot distinguish whether a cluster's high thriving rate is causal or merely correlational with demographics
