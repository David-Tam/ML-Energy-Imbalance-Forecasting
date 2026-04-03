# ML-Energy-Imbalance-Forecasting

# Energy Imbalance Forecasting — Enefit Kaggle Competition

Predicting the energy imbalance of prosumers (households and businesses that both consume
and produce energy) in Estonia. Accurate forecasting reduces grid instability and
operational costs for energy providers.

**Competition:** [Predict Energy Behavior of Prosumers](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers)  
**Metric:** Mean Absolute Error (MAE)

---

## Results

| Model | Validation MAE | vs Baseline |
|---|---|---|
| Persistence Baseline (48h lag) | 99.04 | — |
| XGBoost (default) | 60.29 | −39.1% |
| LightGBM (default) | 66.07 | −33.3% |
| XGBoost (Optuna tuned) | **58.06** | **−41.4%** |

---

## Pipeline Overview

### 1 — Data Loading
Six raw data sources totalling ~1GB are loaded and inspected:

| File | Description |
|---|---|
| `train.csv` | Target variable (energy imbalance) per prosumer per hour |
| `client.csv` | Prosumer metadata — installed capacity, contract type |
| `historical_weather.csv` | Actual observed weather per station per hour |
| `forecast_weather.csv` | Predicted weather per station per forecast hour |
| `electricity_prices.csv` | Hourly electricity spot price (EUR/MWh) |
| `gas_prices.csv` | Daily gas price range (EUR/MWh) |

### 2 — Exploratory Data Analysis
- Null analysis across all 6 sources
- Target distribution (skewness, IQR outlier detection)
- Mean daily target over time — confirms strong seasonality
- Target breakdown by `is_business` and `product_type`

### 3 — Data Merging
All 6 sources merged onto the base train table using left joins.
Key design decisions:
- **Granularity matching** — daily tables joined on `date`, hourly on `datetime`
- **Weather station mapping** — lat/lon coordinates mapped to Estonian counties, then aggregated to county-level means before joining
- **Dual weather features** — historical (`hist_*`) and forecast (`fcst_*`) weather retained as separate features; the gap between them is an informative signal

### 4 — Feature Engineering
**Datetime features:** hour, month, day of week, quarter, day of year, week of year, is_weekend

**Lag features (no-lookahead design):**  
Past values of the target brought forward as input features — autoregression.
All lags start at 48 hours (2 days) to respect competition constraints and prevent temporal leakage.

| Feature | Lookback |
|---|---|
| `lag_2d` | 48 hours |
| `lag_3d` | 72 hours |
| `lag_7d` | 168 hours |
| `lag_14d` | 336 hours |

Lags computed within prosumer groups (`county`, `is_business`, `product_type`, `is_consumption`)
to prevent cross-prosumer contamination.

**Rolling features:** 7-day rolling mean and std over `lag_2d` — captures recent trend without leakage.

**Total features: 50**

### 5 — Train/Validation Split
Time-based split — no random shuffling to prevent temporal leakage.

| Set | Date Range | Rows |
|---|---|---|
| Train | 2021-09-01 → 2022-12-31 | 1,537,584 |
| Validation | 2023-01-01 → 2023-05-31 | 480,768 |

### 6 — Persistence Baseline
Predict today = 48 hours ago (`lag_2d`). Industry-standard sanity check for time series.  
**Baseline MAE: 99.04**

### 7 — Model Training
XGBoost and LightGBM trained with:
- MAE objective (`reg:absoluteerror` / `mae`)
- Early stopping (50 rounds)
- GPU acceleration (NVIDIA T4)
- Time-based validation split

### 8 — Hyperparameter Tuning (Optuna)
Bayesian optimisation using Tree-structured Parzen Estimator (TPE).
- **25 trials**, **3-fold TimeSeriesSplit CV** (temporal leakage-safe)
- `n_estimators=3000` fixed with early stopping — optimal tree count found automatically
- Search space: `learning_rate` (log scale), `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`

Best parameters found:

| Parameter | Value |
|---|---|
| `learning_rate` | 0.0243 |
| `max_depth` | 10 |
| `subsample` | 0.985 |
| `colsample_bytree` | 0.515 |
| `min_child_weight` | 7 |

### 9 — SHAP Analysis
TreeExplainer applied to the tuned XGBoost model:
- **Global summary plot** — lag features dominate; `lag_2d`, `lag_7d`, `lag_14d` are top 3
- **Waterfall plot** — local explanation for individual predictions
- High lag values push predictions upward — physically sensible autocorrelation

---

## Key Technical Decisions

**Why MAE and not RMSE?**  
The target is heavily right-skewed. RMSE squares the residual, causing the model to
disproportionately chase large outliers. MAE penalises all errors linearly, making the
model robust to extreme energy events (heatwaves, grid failures) that are real phenomena,
not noise.

**Why time-based train/val split?**  
A random split would allow the model to train on future data and validate on past data —
temporal leakage. The model would appear to perform well but fail immediately in production.

**Why lags start at 48 hours?**  
Competition rules require predictions to be made before the target is observed. The
earliest available historical target is 48 hours prior. Using shorter lags would be leakage.

**Why fix n_estimators in Optuna?**  
Tuning n_estimators as a trial parameter is inefficient — early stopping already finds
the optimal number of trees for each parameter combination. Fixing it at a high value
(3000) and letting early stopping decide is cleaner and faster.

---

## Reproduction

This notebook is designed to run on **Kaggle** with the competition dataset attached.

1. Go to the [competition page](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers)
2. Create a new notebook and attach the competition data
3. Enable GPU accelerator (T4 x2)
4. Upload and run `energy_imbalance_forecasting.ipynb`

---

## Tech Stack

`Python` `XGBoost` `LightGBM` `Optuna` `SHAP` `Pandas` `NumPy` `Seaborn` `Scikit-learn` `CUDA`
