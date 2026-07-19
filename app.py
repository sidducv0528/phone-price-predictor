"""
app.py
======
Streamlit web app for the Used Phone Resale Price Predictor.

Loads the trained model + preprocessors from models/ and serves
an interactive form that returns a live price estimate.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import tensorflow.keras as keras

MODEL_DIR = "models"

# ---------------------------------------------------------------------------
# LOAD MODEL + PREPROCESSORS (cached so this only runs once, not per click)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(f"{MODEL_DIR}/phone_price_ann_model.keras")
    x_scaler = joblib.load(f"{MODEL_DIR}/x_scaler.pkl")
    y_scaler = joblib.load(f"{MODEL_DIR}/y_scaler.pkl")
    feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
    return model, x_scaler, y_scaler, feature_names


model, x_scaler, y_scaler, feature_names = load_artifacts()

# ---------------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Used Phone Price Predictor", page_icon="📱", layout="centered")

# ---------------------------------------------------------------------------
# STYLING (single, clean CSS block)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Background + base text color */
    .stApp {
        background:
            linear-gradient(rgba(255,255,255,0.55), rgba(240,242,250,0.65)),
            url("https://images.unsplash.com/photo-1511707171634-5f897ff02aa9");
        background-size: cover;
        background-position: top center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #1c1e2b;
    }

    /* Hide Streamlit's default chrome for a cleaner, product-like look */
    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* Gradient page title */
    h1 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #0284c7, #7c3aed, #db2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.3rem;
    }

    /* Subheader */
    h3 {
        color: #2b2d42;
        font-weight: 600;
    }

    /* Body text */
    p, label, span {
        color: #2b2d42;
    }

    /* Light glassy card look for the form */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 20px;
        padding: 2.2rem;
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    /* Labels inside the form */
    div[data-testid="stForm"] label {
        color: #2b2d42 !important;
        font-weight: 500;
    }

    /* Gradient submit button */
    button[kind="formSubmit"] {
        background: linear-gradient(90deg, #0284c7, #7c3aed, #db2777) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 0.7rem 2rem !important;
        transition: all 0.2s ease-in-out;
    }
    button[kind="formSubmit"]:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.35);
    }

    /* Result box */
    div[data-testid="stAlertContainer"] {
        border-radius: 16px;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("📱 Used Phone Resale Price Predictor")
st.write("Fill in the phone's details below and get an instant resale price estimate.")

# ---------------------------------------------------------------------------
# DROPDOWN OPTIONS (must match the categories seen during training)
# ---------------------------------------------------------------------------
BRAND_MODELS = {
    "Apple":   ["iPhone 11", "iPhone 12", "iPhone 13", "iPhone 14", "iPhone 15"],
    "Google":  ["Pixel 6", "Pixel 7", "Pixel 8"],
    "OnePlus": ["OnePlus 9", "OnePlus 10", "OnePlus 11", "Nord 3"],
    "Realme":  ["Realme GT", "Realme Narzo 60"],
    "Samsung": ["Galaxy A54", "Galaxy S21", "Galaxy S22", "Galaxy S23"],
    "Vivo":    ["Vivo V27", "Vivo X90"],
    "Xiaomi":  ["Mi 11", "Redmi Note 12", "Poco X5"],
}
BRANDS = list(BRAND_MODELS.keys())
CONDITIONS = ["Excellent", "Good", "Fair", "Poor"]
OS_TYPES = ["Android", "iOS"]

# ---------------------------------------------------------------------------
# FORM
# ---------------------------------------------------------------------------
# Brand lives OUTSIDE the form so picking a brand instantly updates
# the Model dropdown before the rest of the form is filled in.
st.subheader("📋 Phone Details")
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

    submitted = st.form_submit_button("🔮 Predict Resale Price")

# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
def yn(val: str) -> int:
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

    st.success(f"### 💰 Estimated Resale Price: ₹{price:,.0f}")
    st.caption("This is a model estimate based on patterns in historical resale data.")
