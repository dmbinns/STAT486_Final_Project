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
* **Sample Size:** N = [Insert your total row count here]
* **Target Variable (`review_count`):** Mean: [X], SD: [Y]. (Note: Distribution is right-skewed).
* **Demographics:** Median Income ranges from $[Min] to $[Max] across sampled zip codes.

### Categorical Variables
* **Category Frequency Distribution:**
    * Gyms: [Count]
    * Yoga: [Count]
    * Pilates: [Count]
    * [List others...]

### Interpretation
The summary statistics indicate a high variance in `review_
