# Used Car Pricing Intelligence — eBay Kleinanzeigen

Turning a messy, real-world used-car marketplace dataset into pricing intelligence for dealers, resellers, buyers, and platforms.

## Project Overview

The used car market is fragmented, price-sensitive, and shaped by brand perception, condition, and age. This project cleans the eBay Kleinanzeigen used car dataset — unclean, inconsistent, and full of real-world anomalies — and turns it into actionable pricing intelligence, including a working price prediction tool.

## Business Objectives

- Establish reliable pricing benchmarks across car brands
- Identify which brands retain value best over time
- Understand how mileage and vehicle age impact resale value
- Detect and remove misleading listings (outliers, errors)
- Build a simple pricing estimator for real-time decision-making

## Key Business Questions & Answers

**1. Which car brands consistently command higher resale prices?**
Porsche is priced at a premium throughout its full range. Brands like Audi, BMW, and Mercedes-Benz are largely mainstream-priced with a smaller high-end tail pulling their averages up.

**2. How does mileage affect price depreciation across different brands?**
Mileage sensitivity varies meaningfully by brand. Measured as €-lost-per-km (using a full regression slope across all mileage brackets, not just endpoint comparison — an initial endpoint-based method gave misleading results for brands with thin low-mileage samples), premium brands (MINI, BMW, Mercedes-Benz, Audi) lose more value per kilometer in absolute terms than mainstream brands (Mitsubishi, Renault, Honda, Toyota).

**3. What is the typical depreciation curve based on vehicle age?**
Price peaks around 1 year old, declines steeply through the first decade, bottoms out around 15-20 years, then rises modestly past 30 years — driven by a narrow segment of well-preserved, sporty, premium-brand vehicles (coupes/convertibles, Porsche/Mercedes-Benz, no unrepaired damage) rather than old cars broadly.

**4. Which brands hold value best (low depreciation over time)?**
Smart and MINI are the strongest resale investments, combining high value retention with low price volatility (a combined "investment score" weighting retention 60% / stability 40%). Premium brand status alone does not guarantee superior resale performance — several premium brands are riskier resale bets than their reputation suggests.

**5. What price range should a seller expect given a car's brand, mileage, and age?**
Answered directly by the pricing model (see below), deployed as both an Excel lookup tool and a Streamlit app.

## Data Cleaning

- Translated German column names/values to English, standardized to snake_case
- Converted price and mileage from text to numeric
- Filled missing categorical values with an explicit `unknown`/`other` category rather than dropping rows, preserving missingness as a signal
- Restricted registration year to 1966–2016 (1966 = earliest legitimate model year based on baseline premium-brand history research; 2016 = year the dataset was collected)
- Capped price (€500–€200,000) and engine power (60–700 PS), replacing outliers with the column median
- **Data quality fix:** found and removed 737 listings across the dataset with `registration_year = 2016` (appearing "0 years old") but 80,000+ km already on the odometer — a physical impossibility, most likely caused by a missing registration year silently defaulting to the crawl year. Left uncorrected, this distorted brand-level depreciation figures (most visibly for Peugeot, whose near-new price looked artificially low before the fix).

## Segment & Value Retention Analysis

Brands were grouped into data-driven price tiers (Luxury / Premium / Mainstream, based on price quantiles among the highest-volume brands) and compared across four dimensions: value retention, depreciation rate, resale demand, and price stability. No single tier wins on every measure — Luxury is the most stable and in-demand but retains the smallest share of its value over time; Mainstream retains the largest share but is the least predictable to price.

## Pricing Model

Four regression approaches were compared — Linear Regression, Lasso, Ridge, and ElasticNet — using brand, mileage, vehicle age, transmission type, and damage status as features (matching the brief's usability requirements; broader features explored during EDA, like engine power, were intentionally left out to keep the tool simple for a seller/buyer to use).

| Model | R² | MAE (€) | RMSE (€) |
|---|---|---|---|
| Linear Regression | 0.438 | 3,442.53 | 6,046.34 |
| Lasso | 0.438 | 3,441.66 | 6,046.62 |
| Ridge | 0.438 | 3,442.09 | 6,046.48 |
| ElasticNet | 0.438 | 3,441.58 | 6,046.65 |

All four perform virtually identically — a VIF check found only moderate (not severe) multicollinearity between mileage and age, and ElasticNet's cross-validated mixing parameter converged to pure Lasso, confirming Ridge's regularization added no benefit here.

**Lasso was chosen as the final model**: same accuracy as the others, but it zeroed out an uninformative feature (`transmission_type_manual`) entirely, giving a simpler, more interpretable model without sacrificing performance.

Final test-set performance: **R² = 0.405, MAE = €3,606.97, RMSE = €6,808.89**.

## Pricing Intelligence Tool

Deployed as a **Streamlit app** (`app.py`), per the brief's "usable by dealers, buyers, and platforms" requirement — live predictions from the trained Lasso model, with an estimated price range based on model RMSE.

## Business Impact

**Dealers:** Pricing should be brand-aware, not flat. Premium brands need steeper mileage/age-based markdowns than mainstream brands; the pricing tool gives a fast, evidence-based starting point instead of gut-feel pricing.

**Buyers:** The steepest depreciation happens in a car's first 10 years — buyers optimizing for value should look at the 8–15 year range, where price has stabilized. Mileage should be weighed differently by brand tier when comparing listings.

**Marketplaces:** The data quality issues found here (mislabeled registration years, extreme placeholder prices) point to a concrete fix — validating registration year against mileage at listing creation would catch a meaningful share of bad data at the source.

**Investors/Resellers:** Smart and MINI are the strongest resale investments in this dataset, not the most expensive brands — real weight should go to retention and stability, not badge prestige alone.

## Limitations

- The pricing model's feature set is deliberately narrow to match the brief's usability scope; adding features like engine power or vehicle type would likely improve accuracy at the cost of requiring more input from the user.
- Some brand-level findings, particularly for lower-volume brands, are directional rather than statistically robust given limited sample sizes in certain age/mileage brackets.
- The dataset reflects 2015–2016 market conditions; absolute price levels are not adjusted for inflation or current market conditions.

## Repository Contents

- `Used_Cars_Project_cleaned.ipynb` — full cleaning, EDA, segment analysis, value retention analysis, and modeling
- `app.py` — Streamlit pricing estimator
- `lasso_model.pkl`, `scaler.pkl`, `model_columns.pkl`, `numeric_cols.pkl`, `brand_list.pkl` — saved model artifacts used by `app.py`

## Running the App

```bash
pip install streamlit pandas scikit-learn joblib
streamlit run app.py
```
Ensure the five `.pkl` files are in the same folder as `app.py`.
