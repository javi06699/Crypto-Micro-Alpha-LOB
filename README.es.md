# Crypto LOB Micro-Alpha: Investigación de señales HFT en Binance Futures

*Read this in English: [README.md](README.md)*

Proyecto de investigación que construye un pipeline completo — desde
datos crudos de order book (Nivel 2) hasta modelos entrenados y un
backtest — para comprobar si existen señales direccionales de corto
plazo ("micro-alpha") en los snapshots del libro de órdenes de Binance
Futures BTCUSDT, usando características de microestructura al estilo
FI-2010 y un esquema de etiquetado por triple barrera.

## Descripción general

El proyecto descarga snapshots del libro de órdenes a 25 niveles para
`BTCUSDT` en Binance Futures a través de [Tardis.dev](https://tardis.dev/),
calcula 144 características de microestructura (imbalance del libro,
spreads, volúmenes, niveles de precio, etc., siguiendo la familia de
features FI-2010), etiqueta cada snapshot con un método de triple
barrera sobre varios horizontes (`k = 10, 20, 30, 50, 100` ticks), y
entrena varios clasificadores para predecir la dirección del precio a
corto plazo: **Regresión Logística**, **XGBoost / LightGBM**, un **MLP**
(PyTorch) y un modelo **DeepLOB CNN-LSTM**. Las predicciones se pasan
luego por un backtest simple con comisiones y un control de señal
aleatoria para verificar si la estrategia supera al azar.

> ⚠️ **Aviso**: Este es un proyecto de investigación / educativo, no es
> asesoramiento financiero. Nada de lo aquí presentado constituye una
> recomendación de inversión, y el rendimiento pasado en backtesting no
> garantiza resultados futuros.

## Pipeline

| Paso | Notebook | Propósito | Salida |
|---|---|---|---|
| 1 | `raw_data.ipynb` | Descarga snapshots crudos de L2 (Tardis.dev) y los convierte a Parquet | `df_book_10.parquet` |
| 2 | `Features_Labeling.ipynb` | Calcula 144 características estilo FI-2010 y etiquetas de triple barrera | `dataset_completo.parquet` |
| 3 | `Corr_limpieza.ipynb` | Elimina nulos y features con correlación mutua > 0.90 | `dataset_final.parquet` |
| 4 | `EDA_H0test.ipynb` | Análisis exploratorio y contraste de hipótesis | — |
| 5 | `regresion_logistica_label_k_100_hyperopt.ipynb` | Regresión logística multinomial + ajuste con Hyperopt | `final_scaler_logreg.joblib` |
| 6 | `mirco_alpha_XGBoost_LightBoost.ipynb` | Clasificadores XGBoost / LightGBM | `final_xgb_model.pkl/.json/.ubj` |
| 7 | `mirco_alpha_MLP.ipynb` | Clasificador MLP (PyTorch) con checkpoint por validación | `best_val_model_pytorch.pt` |
| 8 | `DeepLOB_CNN_LSTM.ipynb` | Modelo de deep learning CNN-LSTM al estilo DeepLOB | — |
| 9 | `Backtesting.ipynb` | Backtest por umbral con comisiones vs. control de señal aleatoria | gráficos |
| 10 | `REsultados_+_otros.ipynb` | Gráficos de resultados, PnL acumulado, curva de decaimiento del alpha, visor interactivo con Plotly | gráficos |

## Estructura del repositorio

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
└── (Parquet crudos/intermedios — ver "Sobre los datos" abajo)
```

No es obligatorio usar exactamente esta estructura, pero es la que se referencia en este README y mantiene notebooks, modelos y datos separados de forma ordenada, lo que además simplifica las reglas de `.gitignore` y Git LFS.

## Sobre los datos y artefactos (importante antes de subir el repo)

Los archivos de datos y algunos modelos de este proyecto son grandes, por lo que debes tener en cuenta los límites estrictos de GitHub (100 MB por archivo):

| Archivo / Artefacto | Tamaño aproximado | Estado respecto al límite de 100 MB de GitHub |
|---|---|---|
| `dataset_completo.parquet` | ~179 MB | ❌ **Supera el límite** — obligatorio usar Git LFS o excluirlo |
| `binance-futures_book_snapshot_25_2026-04-01_BTCUSDT.parquet` | ~94.5 MB | ⚠️ **Cerca del límite** (alerta a los 50 MB) — recomendado Git LFS |
| `dataset_final.parquet` | ~73 MB | ⚠️ Dentro del límite, pero por encima del umbral de aviso de 50 MB |
| `df_book_10.parquet` | ~39 MB | ✅ Correcto |
| `final_xgb_model.pkl` | ~21 MB | ✅ Correcto |
| `final_scaler_logreg.joblib` | < 1 MB | ✅ Correcto (escalador `StandardScaler` de Scikit-Learn) |
| `best_val_model_pytorch.pt` | ~0.6 MB | ✅ Correcto |

## Instalación

```bash
git clone [https://github.com/](https://github.com/)<tu-usuario>/<tu-repo>.git
cd <tu-repo>
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt


## Instalación

```bash
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

También necesitarás una API key de [Tardis.dev](https://tardis.dev/) para
volver a descargar datos crudos del order book (el plan gratuito cubre
un rango de fechas limitado); configúrala como variable de entorno o
pásala directamente en `raw_data.ipynb`.

## Uso

Ejecuta los notebooks en el orden indicado en la tabla del Pipeline.
Cada notebook busca su archivo Parquet de entrada dentro de una carpeta
local `datasets/` — crea esa carpeta y coloca ahí la salida del paso
anterior (o ajusta las rutas al inicio de cada notebook).

## Licencia

Este proyecto se publica bajo la Licencia MIT — ver [LICENSE](LICENSE).

## Referencia de features y labels

- **Features**: 144 características por snapshot, siguiendo la familia
  de features del benchmark FI-2010 (precio/volumen en 10 niveles del
  libro, spreads, imbalance del libro, mid-price, ratios derivados),
  calculadas sobre snapshots L2 de Binance Futures BTCUSDT.
- **Labels**: clasificación por triple barrera (`0` = baja, `1` =
  neutral, `2` = sube) calculada en varios horizontes `k ∈ {10, 20, 30,
  50, 100}` ticks; los notebooks de modelado usan `label_k_100` como
  estándar.
