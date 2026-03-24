# Sales Forecasting using Machine Learning 📈

This project builds an advanced time series forecasting system to predict future product sales using historical e-commerce data.  
It compares multiple models, performs extensive feature engineering, and selects the best configuration using a custom composite score.

---

## 🔍 Problem Statement
Accurate sales forecasting is critical for inventory planning, pricing strategies, and promotional decisions.  
This project aims to predict future product purchases by learning seasonal patterns, lagged behavior, and customer conversion dynamics.

---

## 🧠 Key Features
- Time series–aware modeling using lag and rolling window features
- Multiple models comparison (Ridge Regression vs XGBoost)
- Hyperparameter tuning with TimeSeriesSplit
- Baseline comparison using naïve lag prediction
- Multi-step future forecasting
- Product-level performance analysis and visualization

---

## ⚙️ Feature Engineering
- Lag features (configurable number of past periods)
- Rolling mean and rolling standard deviation
- Conversion rate
- Cart drop rate
- One-hot encoding for product IDs
- Iterative imputation for missing values
- Standard scaling for numerical features

---

## 🧪 Models Used
- **Ridge Regression**
- **XGBoost Regressor**

Each model is evaluated using RMSE and compared against a baseline forecast.

---

## 🏆 Model Selection Strategy
A custom composite score is used to select the best model configuration:

