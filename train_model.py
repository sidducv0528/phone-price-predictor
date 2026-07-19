# train_model.py
# ---------------------------------------------------------
# Used Phone Resale Price Prediction using ANN (Keras)
# This script loads the data, cleans it, trains a neural
# network, checks how well it performs, and saves the model
# so it can be used later in the Streamlit app.
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# -----------------------------
# STEP 1: Load the dataset
# -----------------------------
df = pd.read_csv("used_phone_price_prediction_1M.csv")
print("Data loaded. Shape:", df.shape)


# -----------------------------
# STEP 2: Drop columns that don't help much
# -----------------------------
# These columns were checked earlier using correlation and
# feature importance, and they barely affect resale_price.
# We keep original_price, age_months, market_demand_score,
# battery_health and condition because those matter the most.

columns_to_drop = [
    "screen_size_inches",
    "usage_hours_per_day",
    "city_tier",
    "seller_type",
    "charger_available",
    "release_year",
    "processor_score",
    "camera_score"
]

df = df.drop(columns=columns_to_drop)
print("Columns after dropping:", df.shape[1])


# -----------------------------
# STEP 3: Separate input (X) and target (y)
# -----------------------------
X = df.drop("resale_price", axis=1)
y = df["resale_price"]


# -----------------------------
# STEP 4: One-hot encode text columns
# -----------------------------
# ANN can only understand numbers, so brand/model/condition/os_type
# need to be converted into 0/1 columns.

categorical_cols = ["brand", "model", "condition", "os_type"]
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Save the column names now - we will need the EXACT same
# columns later when the app makes a single prediction.
feature_names = X_encoded.columns.tolist()


# -----------------------------
# STEP 5: Split into train / validation / test
# -----------------------------
# train = model learns from this
# val   = used during training to check progress
# test  = only used once at the end for the final honest score

X_train, X_temp, y_train, y_temp = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Train size:", X_train.shape)
print("Val size:  ", X_val.shape)
print("Test size: ", X_test.shape)


# -----------------------------
# STEP 6: Scale the data
# -----------------------------
# Neural networks train better when numbers are on a
# similar scale (mean 0, std 1), instead of huge numbers
# like original_price mixed with 0/1 flags.

x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()


# -----------------------------
# STEP 7: Build the ANN model
# -----------------------------
n_features = X_train_scaled.shape[1]

model = Sequential([
    Dense(128, activation="relu", input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dropout(0.1),

    Dense(1, activation="linear")   # 1 output = predicted price
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

model.summary()


# -----------------------------
# STEP 8: Set up callbacks
# -----------------------------
# EarlyStopping = stop training automatically once the model
#                 stops improving, so we don't waste time.
# ReduceLROnPlateau = if progress stalls, slow down the
#                     learning rate so it can fine-tune better.

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
]


# -----------------------------
# STEP 9: Train the model
# -----------------------------
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=256,
    callbacks=callbacks,
    verbose=2
)


# -----------------------------
# STEP 10: Check how good the model is
# -----------------------------
predictions_scaled = model.predict(X_test_scaled).ravel()

# convert predictions back to real rupees
predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Final Test Results ---")
print("MAE (average error in rupees):", round(mae, 2))
print("R2 score (0 to 1, higher is better):", round(r2, 4))


# -----------------------------
# STEP 11: Save the model and preprocessing tools
# -----------------------------
# We need to save these 4 things so the Streamlit app can
# load them later and make predictions without retraining.

model.save("models/phone_price_ann_model.keras")
joblib.dump(x_scaler, "models/x_scaler.pkl")
joblib.dump(y_scaler, "models/y_scaler.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")

print("\nModel and files saved successfully in the 'models' folder.")
