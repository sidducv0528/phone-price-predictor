# 📖 Data Dictionary

This document describes every feature used in the **Phone Price Prediction Using Artificial Neural Network (ANN)** project.

---

# Dataset Information

| Property        | Value                                 |
| --------------- | ------------------------------------- |
| Dataset Name    | Phone Resale Price Prediction Dataset |
| Problem Type    | Regression                            |
| Total Features  | 28                                    |
| Input Features  | 27                                    |
| Target Variable | resale_price                          |

---

# Feature Description

| Column Name                   | Data Type   | Description                                 | Example                     |
| ----------------------------- | ----------- | ------------------------------------------- | --------------------------- |
| **brand**                     | Categorical | Smartphone manufacturer                     | Apple, Samsung, Google      |
| **model**                     | Categorical | Smartphone model                            | iPhone 15, Galaxy S23       |
| **release_year**              | Integer     | Year the device was launched                | 2024                        |
| **ram_gb**                    | Integer     | RAM capacity in GB                          | 8                           |
| **storage_gb**                | Integer     | Internal storage in GB                      | 256                         |
| **screen_size_inches**        | Float       | Screen size in inches                       | 6.70                        |
| **battery_capacity**          | Integer     | Battery capacity (mAh)                      | 5000                        |
| **processor_score**           | Integer     | Processor benchmark score                   | 92                          |
| **camera_score**              | Integer     | Camera quality score                        | 88                          |
| **os_type**                   | Categorical | Mobile operating system                     | Android, iOS                |
| **has_5g**                    | Binary      | 5G support                                  | 1 = Yes, 0 = No             |
| **original_price**            | Float       | Original retail price (₹)                   | 79999                       |
| **purchase_year**             | Integer     | Year the phone was purchased                | 2023                        |
| **age_months**                | Integer     | Device age in months                        | 18                          |
| **usage_hours_per_day**       | Float       | Average daily usage                         | 5.6                         |
| **condition**                 | Categorical | Physical condition                          | Excellent, Good, Fair, Poor |
| **battery_health**            | Integer     | Battery health percentage                   | 91                          |
| **screen_cracked**            | Binary      | Screen damage                               | 1 = Yes, 0 = No             |
| **body_damage**               | Binary      | Physical body damage                        | 1 = Yes, 0 = No             |
| **repair_history**            | Binary      | Previously repaired                         | 1 = Yes, 0 = No             |
| **water_damage**              | Binary      | Water damage status                         | 1 = Yes, 0 = No             |
| **city_tier**                 | Categorical | Seller location category                    | Tier1, Tier2, Tier3         |
| **seller_type**               | Categorical | Seller category                             | Individual, Store           |
| **warranty_remaining_months** | Integer     | Remaining warranty (months)                 | 12                          |
| **box_available**             | Binary      | Original box available                      | 1 = Yes, 0 = No             |
| **charger_available**         | Binary      | Original charger available                  | 1 = Yes, 0 = No             |
| **market_demand_score**       | Integer     | Market demand score (0–100)                 | 84                          |
| **resale_price**              | Float       | Target variable: estimated resale price (₹) | 45250.75                    |

---

# Categorical Features

| Feature     | Possible Values                                                   |
| ----------- | ----------------------------------------------------------------- |
| brand       | Apple, Samsung, Google, OnePlus, Xiaomi, Vivo, Oppo, Realme, etc. |
| os_type     | Android, iOS                                                      |
| condition   | Excellent, Good, Fair, Poor                                       |
| city_tier   | Tier1, Tier2, Tier3                                               |
| seller_type | Individual, Store                                                 |

---

# Binary Features

| Feature           | Values          |
| ----------------- | --------------- |
| has_5g            | 0 = No, 1 = Yes |
| screen_cracked    | 0 = No, 1 = Yes |
| body_damage       | 0 = No, 1 = Yes |
| repair_history    | 0 = No, 1 = Yes |
| water_damage      | 0 = No, 1 = Yes |
| box_available     | 0 = No, 1 = Yes |
| charger_available | 0 = No, 1 = Yes |

---

# Target Variable

**resale_price**

The ANN model predicts the expected resale price of a smartphone based on its specifications, condition, usage, and market characteristics.

---

# Notes

* All prices are represented in **Indian Rupees (₹)**.
* Numerical features were scaled before model training.
* Categorical features were encoded during preprocessing.
* This dataset is intended for educational and portfolio purposes.
