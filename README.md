# Applied AI & Machine Learning Projects (Internship & NUST SINES Bootcamp)

**Repository:** `ai-ml-projects`  
**Author:** Arooj Khan (MS in Computer Science, The University of Haripur)

## Overview

A curated collection of applied Machine Learning, Data Preprocessing, Exploratory Data Analysis (EDA), and Time-Series Regression notebooks completed during my **AI/ML Engineer Internship at DevelopersHub Corporation** (Dec 2025 – Jan 2026) and the **Applied ML 360° Bootcamp at SINES, NUST Islamabad** (Aug 2025).

## Included Projects & Notebooks

### 1. DevelopersHub Corporation — AI/ML Internship (`developershub-internship/`)
- **`01_iris_eda_and_visualization.ipynb`**: Exploratory Data Analysis (EDA) on the Iris dataset using Pandas, Seaborn, and Matplotlib (species-stratified scatter plots, feature distribution histograms, and box-and-whisker outlier analysis). *(Fixed `iris.describe()` invocation bug from original notebook).*
- **`02_googl_stock_price_prediction.ipynb`**: Time-series next-day closing price prediction on historical Alphabet Inc. (`GOOGL`, 2015–2025) stock market data retrieved via `yfinance`. Uses `Open`, `High`, `Low`, and `Volume` features with chronological (non-shuffled) train/test splitting (`80/20`), `LinearRegression`, RMSE evaluation, and Actual vs. Predicted trajectory plotting. *(Fixed cell execution order so `Target = Close.shift(-1)` is computed before feature/target slicing).*

### 2. SINES, NUST Islamabad — Applied ML 360° Bootcamp (`nust-sines-ml-bootcamp/`)
- **`01_climate_pm25_time_series_forecasting.ipynb`**: End-to-end environmental time-series pipeline on `climate_data.csv` (`730` daily observations across `pm25`, `temp`, `humidity`, `wind_speed`, and `rainfall`). Implements forward-fill imputation (`.ffill()`), lag feature engineering (`pm25_lag_1`, `pm25_lag_7`), 7-day rolling mean (`pm25_rolling_7`), chronological 80/20 split (`572` train / `144` test rows), and a `scikit-learn` `Pipeline` (`ColumnTransformer` + `StandardScaler` + `LinearRegression`).
  - **Verified Test Results (from executed notebook output):**
    - **Mean Absolute Error (MAE):** `4.1525`
    - **Root Mean Squared Error (RMSE):** `5.0207`
    - **R-squared ($R^2$):** `0.6120`
- **`02_rainfall_prediction_preprocessing.ipynb`**: Missing-value imputation (mean imputation for continuous meteorological features, mode imputation for categorical city locations across Karachi, Lahore, Islamabad, and Quetta) and stratified feature preparation for rainfall prediction (`rainfall_prediction_with_missing.csv`).

## Installation

```bash
git clone https://github.com/Arooj-khan301/ai-ml-projects.git
cd ai-ml-projects
pip install -r requirements.txt
jupyter notebook
```
