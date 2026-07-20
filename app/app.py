import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗", layout="centered")

# ---------------------------------------------------------------
# Load model artifacts (must be in the same folder as this file)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("lasso_model.pkl")
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    numeric_cols = joblib.load("numeric_cols.pkl")
    brand_list = joblib.load("brand_list.pkl")
    return model, scaler, model_columns, numeric_cols, brand_list

try:
    model, scaler, model_columns, numeric_cols, brand_list = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please place lasso_model.pkl, scaler.pkl, "
        "model_columns.pkl, numeric_cols.pkl, and brand_list.pkl in the same "
        "folder as this app."
    )
    st.stop()

# ---------------------------------------------------------------
# Prediction function (same logic as the notebook)
# ---------------------------------------------------------------
def predict_price(brand, kilometer, vehicle_age, transmission_type, unrepaired_damage):
    new_car = pd.DataFrame({
        "brand_grouped": [brand],
        "kilometer": [kilometer],
        "vehicle_age": [vehicle_age],
        "transmission_type": [transmission_type],
        "unrepaired_damage": [unrepaired_damage],
    })
    new_car_encoded = pd.get_dummies(
        new_car, columns=["brand_grouped", "transmission_type", "unrepaired_damage"]
    )
    new_car_encoded = new_car_encoded.reindex(columns=model_columns, fill_value=0)
    new_car_encoded[numeric_cols] = scaler.transform(new_car_encoded[numeric_cols])
    return model.predict(new_car_encoded)[0]

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

st.divider()

if st.button("Estimate Price", type="primary", use_container_width=True):
    predicted = predict_price(
        brand=brand,
        kilometer=kilometer,
        vehicle_age=vehicle_age,
        transmission_type=transmission_type,
        unrepaired_damage=unrepaired_damage,
    )

    # Use the model's known test RMSE as a rough uncertainty margin
    margin = 6808.89
    low = max(0, predicted - margin)
    high = predicted + margin

    st.metric("Estimated Price", f"€{predicted:,.0f}")
    st.write(f"**Expected range:** €{low:,.0f} – €{high:,.0f}")

    st.caption(
        "Estimate based on a Lasso regression model trained on the eBay "
        "Kleinanzeigen used car dataset. Range reflects typical model error "
        "(± RMSE) and should be used as guidance, not a fixed valuation."
    )

st.divider()
st.caption(
    "Built for dealers pricing inventory, buyers evaluating listings, and "
    "platforms suggesting prices. Model: Lasso Regression | "
    "Features: brand, mileage, age, transmission, damage status."
)
