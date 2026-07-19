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
            linear-gradient(rgba(5,6,10,0.88), rgba(13,15,26,0.92)),
            url("https://images.unsplash.com/photo-1511707171634-5f897ff02aa9");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #eaeaf0;
    }

    /* Hide Streamlit's default chrome for a cleaner, product-like look */
    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* Gradient page title */
    h1 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #7dd3fc, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.3rem;
    }

    /* Subheader */
    h3 {
        color: #cdd3ee;
        font-weight: 600;
    }

    /* Glassy card look for the form */
    div[data-testid="stForm"] {
        background: rgba(20, 22, 38, 0.65);
        border: 1px solid rgba(167, 139, 250, 0.25);
        border-radius: 20px;
        padding: 2.2rem;
        box-shadow: 0 12px 48px rgba(0,0,0,0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }

    /* Labels inside the form */
    div[data-testid="stForm"] label {
        color: #dcdfef !important;
        font-weight: 500;
    }

    /* Gradient submit button */
    button[kind="formSubmit"] {
        background: linear-gradient(90deg, #7dd3fc, #a78bfa, #f472b6) !important;
        color: #0d0f1a !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 0.7rem 2rem !important;
        transition: all 0.2s ease-in-out;
    }
    button[kind="formSubmit"]:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 30px rgba(167, 139, 250, 0.5);
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
CONDITIONS =
