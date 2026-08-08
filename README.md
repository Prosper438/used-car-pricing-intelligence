# Used Car Price Estimator

Pricing intelligence for the **eBay Kleinanzeigen** used car market — built for dealers,
resellers, and platforms who need a fast, data-driven price reference instead of guesswork.

**[Live App](#)** · **[Notebook](#)**

## Project Overview

Germany's used car market is fragmented and price-sensitive, driven heavily by brand
perception, vehicle condition, and age. This project turns ~370,000 raw eBay Kleinanzeigen
listings into a working pricing tool, plus a descriptive analysis of what actually drives
price in this market.

- **Predictive tool:** a Lasso Regression model estimating a fair price from brand, mileage,
  age, transmission type, damage status, vehicle type, and engine power — deployed as an
  interactive Streamlit app.
- **Descriptive analysis:** brand-level value retention rankings, depreciation curves, and
  price-stability analysis across market segments.

## The Data Cleaning Problem

The raw dataset had real, marketplace-typical quality issues that had to be resolved before
any modeling could happen:

- Registration years outside plausible bounds (kept 1966–2016 only; 1,807 invalid rows dropped)
- Prices with obvious data-entry errors (bounded to €500–€200,000)
- Engine power (PS) values below 60 or above 700 (replaced with the column median)
- 737 rows with a registration year suggesting a brand-new car but mileage of 80,000+ km —
  a clear mislabeling pattern, removed before the value-retention analysis
- High-cardinality `brand` and `model` columns handled via grouping (rare brands bucketed
  into "Other") rather than dropped

## Modeling

Linear Regression, Ridge, Lasso, and ElasticNet were compared; all performed similarly
(R² ≈ 0.44) with the initial feature set. Adding `power_ps` and `vehicle_type` back into the
model meaningfully improved performance (R² 0.405 → 0.552, RMSE €6,808.89 → €5,492.83).
`fuel_type` was considered but excluded after a VIF check showed severe multicollinearity
(VIF in the 25–46 range for `fuel_type_petrol`/`fuel_type_diesel`).

**Final model:** Lasso Regression, trained on brand (grouped), mileage, vehicle age,
transmission type, unrepaired damage status, vehicle type, and engine power.

## Prediction Range

The app originally displayed a flat `estimate ± RMSE` range, which came out far too wide
(~±55% of the estimate) and applied the same margin to every car regardless of context. This
was replaced with a **bootstrap prediction interval**: 200 Lasso models trained on resampled
versions of the training data, with their coefficients saved. At prediction time, the new
car's features run through all 200 saved coefficient sets, and the 2.5th/97.5th percentile of
the resulting spread becomes the range — tightening for common car profiles and widening for
rarer ones, instead of one blanket margin for everything.

## Repository Structure

```
├── app.py                          # Streamlit app
├── Used_Cars_Project.ipynb         # Full analysis: cleaning, EDA, modeling, evaluation
├── lasso_model.pkl                 # Trained Lasso model
├── scaler.pkl                      # Fitted StandardScaler
├── model_columns.pkl               # Column order expected by the model
├── numeric_cols.pkl                # Which columns get scaled
├── brand_list.pkl                  # Dropdown options for the app
├── vehicle_type_list.pkl           # Dropdown options for the app
├── boot_coefs.pkl                  # 200 bootstrap coefficient sets (prediction range)
├── boot_intercepts.pkl             # 200 bootstrap intercepts (prediction range)
├── requirements.txt
└── README.md
```

## Running Locally

```bash
git clone <repo-url>
cd used-car-price-estimator
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
statsmodels
```

## Limitations

- R² of 0.552 means the model explains roughly half of price variance — a meaningful chunk
  of price is driven by factors not captured here (equipment/trim level, exact model variant,
  cosmetic condition, seller type, regional demand).
- The dataset reflects the German used car market circa 2016; absolute price levels won't
  transfer directly to current-day or other-market pricing without recalibration.
- The bootstrap prediction range reflects model uncertainty, not a guarantee — it should be
  used as guidance alongside other pricing signals, not a fixed valuation.
