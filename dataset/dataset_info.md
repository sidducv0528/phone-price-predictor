
# 📊 Dataset Information

## Overview

The **Phone Resale Price Prediction Dataset** is designed to estimate the resale value of smartphones based on their technical specifications, physical condition, usage history, and market-related factors.

This dataset is used to train an **Artificial Neural Network (ANN)** regression model that predicts the expected resale price of a mobile phone.

---

# Dataset Summary

| Attribute                  | Details                          |
| -------------------------- | -------------------------------- |
| **Project Name**           | Phone Price Prediction Using ANN |
| **Problem Type**           | Regression                       |
| **Target Variable**        | `resale_price`                   |
| **Machine Learning Model** | Artificial Neural Network (ANN)  |
| **Framework**              | TensorFlow & Keras               |
| **Programming Language**   | Python                           |
| **Deployment**             | Streamlit                        |

---

# Dataset Features

The dataset contains **28 columns**, including **27 input features** and **1 target variable**.

| Feature                   | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| brand                     | Smartphone manufacturer (Apple, Samsung, Google, etc.) |
| model                     | Smartphone model name                                  |
| release_year              | Year the phone was launched                            |
| ram_gb                    | RAM capacity (GB)                                      |
| storage_gb                | Internal storage (GB)                                  |
| screen_size_inches        | Display size (inches)                                  |
| battery_capacity          | Battery capacity (mAh)                                 |
| processor_score           | Processor performance score                            |
| camera_score              | Camera quality score                                   |
| os_type                   | Operating System (Android/iOS)                         |
| has_5g                    | 5G support (1 = Yes, 0 = No)                           |
| original_price            | Original retail price (₹)                              |
| purchase_year             | Year of purchase                                       |
| age_months                | Age of the phone (months)                              |
| usage_hours_per_day       | Average daily usage (hours)                            |
| condition                 | Physical condition (Excellent, Good, Fair, Poor)       |
| battery_health            | Battery health (%)                                     |
| screen_cracked            | Screen damaged (1 = Yes, 0 = No)                       |
| body_damage               | Body damaged (1 = Yes, 0 = No)                         |
| repair_history            | Previous repair history                                |
| water_damage              | Water damage (1 = Yes, 0 = No)                         |
| city_tier                 | Tier1, Tier2 or Tier3 city                             |
| seller_type               | Individual or Store                                    |
| warranty_remaining_months | Remaining warranty period                              |
| box_available             | Original box available                                 |
| charger_available         | Original charger available                             |
| market_demand_score       | Market demand score                                    |
| **resale_price**          | **Target variable (Predicted Value)**                  |

---

# Target Variable

**resale_price**

The objective of this project is to accurately predict the resale value of a smartphone based on its specifications, usage history, condition, and market demand.

---

# Data Preprocessing

Before training the model, the following preprocessing steps were performed:

* Removed duplicate records
* Checked and handled missing values
* Encoded categorical variables
* Selected relevant features
* Scaled numerical features using Scikit-learn
* Split the dataset into training and testing sets

---

# Machine Learning Workflow

```text
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Categorical Encoding
      │
      ▼
Feature Scaling
      │
      ▼
Train-Test Split
      │
      ▼
ANN Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Phone Resale Price Prediction
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow
* Keras
* Streamlit

---

# Project Objective

The primary objective of this project is to build a deep learning model capable of predicting smartphone resale prices with high accuracy.

The model considers hardware specifications, device condition, battery health, warranty status, usage behavior, and market demand to estimate the expected resale value.

---

# Repository Note

To keep this repository lightweight and GitHub-friendly, only a representative **sample dataset** is included.

The complete dataset used during model training is not included due to its large size.

---

# License

This dataset is provided solely for educational and portfolio purposes. Please ensure you have the appropriate rights or permissions before using it for commercial applications.
