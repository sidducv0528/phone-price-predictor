# models/

This folder should contain your trained model files:

- phone_price_ann_model.keras
- x_scaler.pkl
- y_scaler.pkl
- feature_names.pkl

## How to get these files

**Option A — you already have them** (from running your notebook / train_model.py before):
Just copy your 4 saved files into this folder, replacing this README (or keeping it alongside).

**Option B — generate them fresh:**
1. Put your full `used_phone_price_prediction_1M.csv` in the project's root folder (next to train_model.py).
2. Run:
   ```
   python train_model.py
   ```
3. This will train the model on your full dataset and save all 4 files here automatically.

The app.py file expects these 4 files to be in this exact folder before it will run correctly.
