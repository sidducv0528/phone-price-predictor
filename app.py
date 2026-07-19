import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# ---------------------------------------------------------------------------
# LOAD SAVED MODEL + PREPROCESSORS (these files must sit next to app.py)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(f"{MODEL_DIR}/phone_price_ann_model.keras")
    x_scaler = joblib.load("x_scaler.pkl")
    y_scaler = joblib.load("y_scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, x_scaler, y_scaler, feature_names

model, x_scaler, y_scaler, feature_names = load_artifacts()

# ---------------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Used Phone Price Predictor", page_icon="📱")
st.title("📱 Used Phone Resale Price Predictor")
st.write("Fill in the phone's details below and get an instant resale price estimate.")

BRAND_MODELS = {
    'Apple':   ['iPhone 11', 'iPhone 12', 'iPhone 13', 'iPhone 14', 'iPhone 15'],
    'Google':  ['Pixel 6', 'Pixel 7', 'Pixel 8'],
    'OnePlus': ['OnePlus 9', 'OnePlus 10', 'OnePlus 11', 'Nord 3'],
    'Realme':  ['Realme GT', 'Realme Narzo 60'],
    'Samsung': ['Galaxy A54', 'Galaxy S21', 'Galaxy S22', 'Galaxy S23'],
    'Vivo':    ['Vivo V27', 'Vivo X90'],
    'Xiaomi':  ['Mi 11', 'Redmi Note 12', 'Poco X5'],
}
BRANDS = list(BRAND_MODELS.keys())
CONDITIONS = ['Excellent', 'Good', 'Fair', 'Poor']
OS_TYPES = ['Android', 'iOS']

# ---------------------------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------------------------
# Brand lives OUTSIDE the form so picking a brand instantly updates
# the Model list below, before you fill in the rest of the form.
st.subheader("Phone Details")
brand = st.selectbox("Brand", BRANDS)

with st.form("phone_form"):
    col1, col2 = st.columns(2)

    with col1:
        model_name = st.selectbox("Model", BRAND_MODELS[brand])
        os_type = st.selectbox("OS Type", OS_TYPES)
        ram_gb = st.number_input("RAM (GB)", min_value=2, max_value=32, value=8)
        storage_gb = st.number_input("Storage (GB)", min_value=16, max_value=1024, value=128)
        battery_capacity = st.number_input("Battery Capacity (mAh)", min_value=1000, max_value=8000, value=4500)
        has_5g = st.selectbox("Has 5G?", ["Yes", "No"])
        original_price = st.number_input("Original Price (₹)", min_value=5000, max_value=300000, value=60000)
        purchase_year = st.number_input("Purchase Year", min_value=2015, max_value=2026, value=2024)

    with col2:
        age_months = st.number_input("Age (months)", min_value=0, max_value=120, value=18)
        condition = st.selectbox("Overall Condition", CONDITIONS)
        battery_health = st.slider("Battery Health (%)", min_value=0, max_value=100, value=85)
        market_demand_score = st.slider("Market Demand Score", min_value=0, max_value=100, value=70)
        warranty_remaining_months = st.number_input("Warranty Remaining (months)", min_value=0, max_value=36, value=6)
        box_available = st.selectbox("Original Box Available?", ["Yes", "No"])
        screen_cracked = st.selectbox("Screen Cracked?", ["No", "Yes"])
        body_damage = st.selectbox("Body Damage?", ["No", "Yes"])
        repair_history = st.selectbox("Repair History?", ["No", "Yes"])
        water_damage = st.selectbox("Water Damage?", ["No", "Yes"])

    submitted = st.form_submit_button("Predict Resale Price")

# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
def yn(val):
    return 1 if val == "Yes" else 0

if submitted:
    row = {
        "brand": brand,
        "model": model_name,
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "battery_capacity": battery_capacity,
        "os_type": os_type,
        "has_5g": yn(has_5g),
        "original_price": original_price,
        "purchase_year": purchase_year,
        "age_months": age_months,
        "condition": condition,
        "battery_health": battery_health,
        "screen_cracked": yn(screen_cracked),
        "body_damage": yn(body_damage),
        "repair_history": yn(repair_history),
        "water_damage": yn(water_damage),
        "warranty_remaining_months": warranty_remaining_months,
        "box_available": yn(box_available),
        "market_demand_score": market_demand_score,
    }

    row_df = pd.DataFrame([row])
    row_encoded = pd.get_dummies(row_df)
    # align with the exact columns the model was trained on
    row_encoded = row_encoded.reindex(columns=feature_names, fill_value=0)

    row_scaled = x_scaler.transform(row_encoded)
    pred_scaled = model.predict(row_scaled, verbose=0).ravel()
    price = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]

    st.success(f"### Estimated Resale Price: ₹{price:,.0f}")
    st.caption("This is a model estimate based on patterns in historical resale data.")
