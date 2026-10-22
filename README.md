# Crypto LOB Micro-Alpha: HFT Signal Research on Binance Futures

*Read this in Spanish: [README.es.md](README.es.md)*

Research project that builds a full pipeline — from raw Level-2 order book
data to trained models and a backtest — to test whether short-horizon
directional "micro-alpha" signals exist in Binance Futures BTCUSDT order
book snapshots, using classic FI-2010-style microstructure features and a
triple-barrier labeling scheme.

## Overview

The project downloads 25-level order book snapshots for `BTCUSDT` on
Binance Futures via [Tardis.dev](https://tardis.dev/), engineers 144
microstructure features (order book imbalance, spreads, volumes, price
levels, etc., following the FI-2010 feature family), labels each snapshot
with a triple-barrier method over several horizons (`k = 10, 20, 30, 50,
100` ticks), and trains several classifiers to predict short-term price
direction: **Logistic Regression**, **XGBoost / LightGBM**, an **MLP**
(PyTorch), and a **DeepLOB CNN-LSTM** model. Predictions are then run
through a simple backtest with transaction fees and a random-signal
control to check whether the strategy beats chance.

> ⚠️ **Disclaimer**: This is a research / educational project, not
> financial advice. Nothing here is investment advice, and past
> backtested performance does not guarantee future results.

## Pipeline

| Step | Notebook | Purpose | Output |
|---|---|---|---|
| 1 | `raw_data.ipynb` | Download raw L2 snapshots (Tardis.dev) and convert to Parquet | `df_book_10.parquet` |
| 2 | `Features_Labeling.ipynb` | Compute 144 FI-2010-style features and triple-barrier labels | `dataset_completo.parquet` |
| 3 | `Corr_limpieza.ipynb` | Drop nulls, drop features with pairwise correlation > 0.90 | `dataset_final.parquet` |
| 4 | `EDA_H0test.ipynb` | Exploratory analysis and hypothesis testing | — |
| 5 | `regresion_logistica_label_k_100_hyperopt.ipynb` | Multinomial logistic regression + Hyperopt tuning | `final_scaler_logreg.joblib` |
| 6 | `mirco_alpha_XGBoost_LightBoost.ipynb` | XGBoost / LightGBM classifiers | `final_xgb_model.pkl/.json/.ubj` |
| 7 | `mirco_alpha_MLP.ipynb` | MLP classifier (PyTorch), validation-based checkpointing | `best_val_model_pytorch.pt` |
| 8 | `DeepLOB_CNN_LSTM.ipynb` | DeepLOB-style CNN-LSTM deep learning model | — |
| 9 | `Backtesting.ipynb` | Threshold-based backtest with fees vs. a random-signal control | plots |
| 10 | `REsultados_+_otros.ipynb` | Result plots, cumulative PnL, alpha-decay curve, interactive Plotly viewer | plots |

## Repository structure

```
.
├── README.md
├── README.es.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_raw_data.ipynb
│   ├── 02_features_labeling.ipynb
│   ├── 03_corr_cleaning.ipynb
│   ├── 04_eda_h0test.ipynb
│   ├── 05_logistic_regression_hyperopt.ipynb
│   ├── 06_xgboost_lightgbm.ipynb
│   ├── 07_mlp.ipynb
│   ├── 08_deeplob_cnn_lstm.ipynb
│   ├── 09_backtesting.ipynb
│   └── 10_results_analysis.ipynb
├── models/
│   ├── final_scaler_logreg.joblib
│   ├── final_xgb_model.pkl
│   └── best_val_model_pytorch.pt
└── data/
├── binance-futures_book_snapshot_25_2026-04-01_BTCUSDT.parquet
└── (Raw/intermediate Parquet files — see "About the Data" below)
```

Using this exact layout is not mandatory, but it matches the paths referenced across this README and keeps notebooks, models, and data neatly separated (which also simplifies `.gitignore` and Git LFS rules).

## About the Data & Artifacts (Important before pushing to Git)

Data files and model artifacts in this project can be large. Keep GitHub's strict 100 MB per-file limit in mind:

| File / Artifact | Approx. Size | Status vs. GitHub 100 MB Limit |
|---|---|---|
| `dataset_completo.parquet` | ~179 MB | ❌ **Exceeds limit** — must use Git LFS or exclude |
| `binance-futures_book_snapshot_25_2026-04-01_BTCUSDT.parquet` | ~94.5 MB | ⚠️ **Near limit** (warning at 50 MB) — Git LFS recommended |
| `dataset_final.parquet` | ~73 MB | ⚠️ Within limit, but triggers GitHub's 50 MB soft warning |
| `df_book_10.parquet` | ~39 MB | ✅ Safe |
| `final_xgb_model.pkl` | ~21 MB | ✅ Safe |
| `final_scaler_logreg.joblib` | < 1 MB | ✅ Safe (Scikit-Learn `StandardScaler`) |
| `best_val_model_pytorch.pt` | ~0.6 MB | ✅ Safe |

## Installation

```bash
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

You'll also need a [Tardis.dev](https://tardis.dev/) API key to re-download
raw order book data (free tier covers a limited date range); set it as
an environment variable or pass it directly in `raw_data.ipynb`.

## Usage

Run the notebooks in the order listed in the Pipeline table above. Each
notebook looks for its input Parquet file inside a local `datasets/`
folder — create that folder and place the previous step's output there
(or adjust the paths at the top of each notebook).

## License

This project is released under the MIT License — see [LICENSE](LICENSE).

## Feature/label reference

- **Features**: 144 features per snapshot following the FI-2010 benchmark
  feature family (price/volume at 10 book levels, spreads, order book
  imbalance, mid-price, derived ratios), computed over Binance Futures
  BTCUSDT L2 snapshots.
- **Labels**: triple-barrier classification (`0` = down, `1` = neutral,
  `2` = up) computed at multiple horizons `k ∈ {10, 20, 30, 50, 100}`
  ticks; the modeling notebooks standardize on `label_k_100`.
