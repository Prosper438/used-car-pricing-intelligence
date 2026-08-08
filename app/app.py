import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗", layout="centered")

# ---------------------------------------------------------------
# Load model artifacts (must be in the same folder as this file)
# Paths are built relative to this script's own location, not the
# current working directory - this matters because Streamlit Cloud
# runs the app with the working directory set to the repo root,
# not the folder the script lives in.
# ---------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(APP_DIR, "lasso_model.pkl"))
    scaler = joblib.load(os.path.join(APP_DIR, "scaler.pkl"))
    model_columns = joblib.load(os.path.join(APP_DIR, "model_columns.pkl"))
    numeric_cols = joblib.load(os.path.join(APP_DIR, "numeric_cols.pkl"))
    brand_list = joblib.load(os.path.join(APP_DIR, "brand_list.pkl"))
    vehicle_type_list = joblib.load(os.path.join(APP_DIR, "vehicle_type_list.pkl"))
    # New: bootstrap coefficient sets, used to build a data-driven prediction
    # range instead of a flat +/- RMSE band
    boot_coefs = joblib.load(os.path.join(APP_DIR, "boot_coefs.pkl"))
    boot_intercepts = joblib.load(os.path.join(APP_DIR, "boot_intercepts.pkl"))
    return (model, scaler, model_columns, numeric_cols, brand_list,
            vehicle_type_list, boot_coefs, boot_intercepts)

try:
    (model, scaler, model_columns, numeric_cols, brand_list,
     vehicle_type_list, boot_coefs, boot_intercepts) = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please place lasso_model.pkl, scaler.pkl, "
        "model_columns.pkl, numeric_cols.pkl, brand_list.pkl, "
        "vehicle_type_list.pkl, boot_coefs.pkl, and boot_intercepts.pkl "
        "in the same folder as this app."
    )
    st.stop()

# ---------------------------------------------------------------
# Prediction function (same logic as the notebook)
# ---------------------------------------------------------------
def predict_price_with_range(brand, kilometer, vehicle_age, transmission_type,
                              unrepaired_damage, vehicle_type, power_ps):
    """Returns (point_estimate, lower, upper). The range comes from running
    the new car's features through 200 bootstrap-trained Lasso coefficient
    sets and taking the 2.5th/97.5th percentile of the resulting spread --
    this tightens for common car profiles and widens for rare/edge-case
    ones, rather than applying one flat +/- RMSE margin to every car.
    """
    new_car = pd.DataFrame({
        "brand_grouped": [brand],
        "kilometer": [kilometer],
        "vehicle_age": [vehicle_age],
        "transmission_type": [transmission_type],
        "unrepaired_damage": [unrepaired_damage],
        "vehicle_type": [vehicle_type],
        "power_ps": [power_ps],
    })
    new_car_encoded = pd.get_dummies(
        new_car, columns=["brand_grouped", "transmission_type", "unrepaired_damage", "vehicle_type"]
    )
    new_car_encoded = new_car_encoded.reindex(columns=model_columns, fill_value=0)
    new_car_encoded[numeric_cols] = scaler.transform(new_car_encoded[numeric_cols])

    x = new_car_encoded.values[0]
    point_estimate = model.predict(new_car_encoded)[0]

    boot_predictions = boot_coefs @ x + boot_intercepts
    lower = np.percentile(boot_predictions, 2.5)
    upper = np.percentile(boot_predictions, 97.5)

    return point_estimate, lower, upper

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🚗 Used Car Price Estimator")
st.caption(
    "Pricing intelligence for the eBay Kleinanzeigen used car market — "
    "for dealers, resellers, and buyers."
)

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", brand_list)
    vehicle_age = st.slider("Vehicle Age (years)", min_value=0, max_value=50, value=5)
    transmission_type = st.selectbox(
        "Transmission Type", ["manual", "automatic", "unknown"]
    )
    vehicle_type = st.selectbox("Vehicle Type", vehicle_type_list)

with col2:
    kilometer = st.select_slider(
        "Mileage (km)",
        options=[5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000,
                 80000, 90000, 100000, 125000, 150000],
        value=100000,
    )
    unrepaired_damage = st.selectbox(
        "Unrepaired Damage", ["no", "yes", "unknown"]
    )
    power_ps = st.slider("Engine Power (PS)", min_value=60, max_value=700, value=150)

st.divider()

if st.button("Estimate Price", type="primary", use_container_width=True):
    predicted, low, high = predict_price_with_range(
        brand=brand,
        kilometer=kilometer,
        vehicle_age=vehicle_age,
        transmission_type=transmission_type,
        unrepaired_damage=unrepaired_damage,
        vehicle_type=vehicle_type,
        power_ps=power_ps,
    )
    low = max(0, low)

    st.metric("Estimated Price", f"€{predicted:,.0f}")
    st.write(f"**Expected range:** €{low:,.0f} – €{high:,.0f}")

    st.caption(
        "Estimate based on a Lasso regression model trained on the eBay "
        "Kleinanzeigen used car dataset. Range reflects a 95% bootstrap "
        "prediction interval and should be used as guidance, not a fixed valuation."
    )

st.divider()
st.caption(
    "Built for dealers pricing inventory, buyers evaluating listings, and "
    "platforms suggesting prices. Model: Lasso Regression | "
    "Features: brand, mileage, age, transmission, damage status, vehicle type, engine power."
)
