# Deliverable 2: Research Question and Exploratory Data Analysis (EDA)

## 1) Research Question and Dataset Overview

### Main Research Question
> **How do neighborhood socio-economic factors (income and education) and gym specialization (categories like Yoga vs. Martial Arts) influence the popularity and consumer engagement of fitness businesses in Utah?**

### Dataset Summary
This project integrates two primary data sources to analyze the Utah fitness market:
1.  **Demographic Data (U.S. Census Bureau):** ZCTA-level data providing median household income, educational attainment, and population counts for Utah.
2.  **Business Data (Yelp Fusion API):** Detailed business metrics for five specific fitness categories: **Gyms, Yoga, Pilates, Health Trainers, and Martial Arts**. This includes location coordinates, star ratings, and review counts.

### Data Citations
* **U.S. Census Bureau.** (2024). *American Community Survey 5-year estimates (2018-2022)*. Accessed via the `census` Python library.
* **Yelp Inc.** (2026). *Yelp Fusion API Business Search*. Retrieved from [https://www.yelp.com/developers](https://www.yelp.com/developers).

### Legal and Ethical Considerations
* **Terms of Use:** All data was retrieved through official API channels adhering to rate limits and attribution requirements. 
* **PII & Ethics:** The dataset contains no personally identifiable information (PII). All business data is public commercial information. No ethical concerns were identified as the analysis focuses on aggregate market trends rather than individual user behavior.

---

## 2) Data Description and Variables

### Key Variables
| Variable | Type | Description |
| :--- | :--- | :--- |
| `review_count` | Numeric | **Target Variable:** Total Yelp reviews (proxy for business volume/popularity). |
| `median_income`| Numeric | Median household income of the gym's zip code. |
| `category` | Categorical | The primary business niche (Yoga, Pilates, Martial Arts, etc.). |
| `rating` | Numeric | Average star rating (1.0 to 5.0). |
| `distance` | Numeric | The distance (in meters) from the center of the search area. |

### Preprocessing Steps
1.  **Filtering:** Restricted search to Utah-specific zip codes.
2.  **Missing Value Handling:** The `price` variable was excluded due to a **98% missingness rate**. Business closure status (`is_closed`) was removed as a target because 100% of the retrieved sample was active.
3.  **Standardization:** Zip codes were converted to strings to ensure a clean join between Census and Yelp data frames.
4.  **Feature Engineering:** Created a `main_cat` variable to simplify Yelp’s nested category lists into a single primary label.

---

## 3) Summary Statistics

### Numeric Variables

* **Sample Size:** N = 1,812 observations

* **Target Variable (`review_count`):**
  Mean = 5.17, Standard Deviation = 16.06.
  The distribution is **highly right-skewed**, with a median of 0 and a maximum of 220. This indicates that most businesses receive very few reviews, while a small number of businesses receive substantially higher engagement.

* **Rating:**
  Mean = 2.04, SD = 2.25. The median rating is 0, suggesting that many businesses have no recorded ratings, while those with ratings tend to be concentrated near the upper end (close to 5.0).

* **Median Income:**
  Mean = $96,539 (SD = $25,525), ranging from $41,964 to $171,151, indicating substantial socioeconomic variation across zip codes.

* **Population (`total_pop`):**
  Mean = 35,895 (SD = 16,692), showing variation in the size of communities where businesses are located.

* **Education (`pct_bachelors`):**
  Mean = 16.92% (SD = 5.44%), with values ranging from about 2% to 30%, reflecting variation in educational attainment.

---

### Categorical Variables

* **Category Frequency Distribution:**
  The dataset contains **87 unique business categories**, though many categories have very low counts (some appearing only once). This suggests a **highly imbalanced categorical distribution**, with a long tail of niche fitness services.

---

### Interpretation

The summary statistics reveal several important patterns:

* The **extreme right skew** and high variance in `review_count` indicate that business popularity is highly uneven, making it a challenging but meaningful target variable.
* A large number of businesses have **zero reviews and zero ratings**, which may reflect newly established businesses or missing Yelp engagement data.
* Socioeconomic variables such as income and education show **substantial variation**, supporting their inclusion as predictors.
* The categorical variable is **high-dimensional and sparse**, which may require grouping or feature engineering before modeling.

## 4) Visual Exploration

### Visualization 1: Popularity vs. Neighborhood Wealth
**[Insert your Scatter Plot of median_income vs. review_count here]**
* **Description:** A scatter plot with a trend line showing the relationship between zip code income and gym reviews.
* **Relevance:** This addresses if "Success" (engagement) is concentrated in high-income areas or if fitness popularity is independent of neighborhood wealth.

### Visualization 2: Engagement Levels by Gym Category
**[Insert your Boxplot of category vs. review_count here]**
* **Description:** A boxplot showing the distribution of review counts across the five specialized categories.
* **Relevance:** This helps determine if certain niches (like Yoga) naturally generate more public engagement than others (like Martial Arts), which is critical for my predictive model.

---

## 5) Challenges and Reflection

### Challenges Faced
A significant challenge was the discovery that **business closure data (`is_closed`) and price data were essentially non-existent** in the current Yelp API pull for Utah. Initially, I intended to predict business failure, but the lack of "closed" gyms in the search results made this unfeasible. 

### Current Concerns
My primary concern is the **skewness of the target variable (`review_count`)**. Most gyms have fewer than 20 reviews, while a few have hundreds. I will likely need to apply a **Log Transformation** to the target variable before modeling to ensure that outliers do not disproportionately bias the regression results.











