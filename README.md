# PM2.5 Forecasting in the Mediterranean Basin

Master 1 mini project, Group 1.

| Team |
| --- |
| Adjaout A Yacine |
| Admane M Houssameddine |
| Amari Manel |
| Assiakhi Rania |

**Assigned models:** SARIMA, SARIMAX

The goal is to forecast daily surface PM2.5 concentrations at ground stations around the Mediterranean from
satellite and reanalysis data (aerosol optical depth, NO2, CO, O3, temperature, pressure, wind). This repository
currently covers the first milestone: data audit, exploration, time-series analysis and leakage-free preprocessing.
The models assigned to the group, SARIMA and SARIMAX, come next.

## Data

[Air Quality - Mediterranean Basin (2016-2026)](https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026/), Kaggle.

| File | Content |
| --- | --- |
| `pm25.csv` | daily PM2.5 per ground station (target, µg/m³) |
| `gee_features_daily.csv` | daily Google Earth Engine features per grid point |

License: listed as "Other (specified in description)". The data are derived from OpenAQ (CC BY 4.0),
Copernicus (open access) and NASA open data. The units of NO2, CO, O3 (mol/m²) and wind (m/s) are inferred
from the value ranges.

The raw files are not versioned. Download them and place them in `data/raw/`.

## Usage

```bash
pip install -r requirements.txt
python scripts/run_eda.py             # audit, cleaning, matching, exploration, time series
python scripts/run_preprocessing.py   # features, folds, preprocessing checks
```

Run both scripts from the project root, in this order. Together they take about 75 seconds. The random seed is
fixed at 42, so every run produces identical folds and result files.

The notebooks in `notebooks/` contain the same analyses with their results and comments. `02_timeseries.ipynb`
reads the cleaned tables, so run `run_eda.py` once before opening it.

## Structure

```
data/
  raw/                  Kaggle files (not versioned)
  processed/            cleaned and modelling tables (generated, not versioned)
notebooks/
  01_eda.ipynb          audit, cleaning, matching, exploratory analysis
  02_timeseries.ipynb   regularity, ACF/PACF, stationarity, drift, lead-lag
src/
  config.py             paths, seed, thresholds
  preprocessing.py      loading, audit, cleaning, station IDs, GEE matching, imputation, scaling, outliers
  features.py           lags, rolling statistics, differences, calendar and past-PM2.5 features
  splits.py             test period and cross-validation folds
  eda.py                exploratory analysis
  timeseries.py         time-series analysis
  utils.py              logging and figure helpers
scripts/
  run_eda.py
  run_preprocessing.py
folds/                  fold indices (JSON), shared by all models
results/                result tables (CSV)
figures/                plots
reports/                technical report
presentation/           slides
```

## Method

**Cleaning.** PM2.5 values that are missing, <= 0 or >= 500 are sentinel codes and are removed before any
analysis. Repeated readings for the same station and day are averaged.

**Station identifiers.** Some station names refer to several locations. "Barcelona", for example, covers three
stations reporting on the same days. Coordinates are clustered within each name (0.01° tolerance) to build a
reliable `station_id`.

**Matching.** Each station is assigned its nearest GEE grid point, within 0.25°, and joined on the date.
107 of 120 stations are matched.

**Time series.** PM2.5 is strongly autocorrelated from one day to the next (median lag-1 ACF of 0.65, Ljung-Box
significant at 99% of the stations tested) and stationary at most stations (ADF). This justifies temporal
validation and lagged features.

**Features.** All features use past information only and are built on a regular daily calendar per station,
so no lag crosses a missing day. Past PM2.5 is included because the task is a forecast at least one day ahead.
A second feature set without it is kept for same-day nowcasting.

**Validation.** Data from 2025-01-01 onward form the final test period and appear in no fold. Three fold sets
are saved once and reused by every model:

| Scheme | Method | Role |
| --- | --- | --- |
| temporal | `TimeSeriesSplit`, 5 folds, 1-day gap | main evaluation |
| spatial | `GroupKFold` by station, 5 folds | generalisation to unseen stations |
| standard | shuffled `KFold`, 5 folds | diagnostic only |

**Preprocessing.** Imputation, scaling and feature selection are fitted on the training fold only, inside a
scikit-learn `Pipeline`. Missing values are filled with the training median, and an indicator column flags each
filled value. KNN imputation was rejected because the gas variables are almost absent from the early folds.
Extreme PM2.5 values are kept: capping them did not consistently improve held-out error.

**Station retention.** Stations need at least 200 observed days and 30% coverage, which keeps 72 stations. The
stricter rule of 365 days and 50% coverage would keep too few.

## Outputs

| File | Content |
| --- | --- |
| `results/Exploratory_Data.csv` | audit, cleaning and matching counts, summaries, correlations, outliers, missingness |
| `results/Exploratory_temporal.csv` | timestamp quality, autocorrelation, stationarity tests, drift, lead-lag |
| `results/features.json` | feature lists and families, test start date, thresholds |
| `results/feature_selection.csv` | features retained by five selection methods |
| `folds/folds_*.json` | row indices of the temporal, spatial and standard folds |
| `data/processed/dev.csv`, `test.csv` | development and test tables for modelling |

When reading generated CSV files, pass `float_precision="round_trip"` to `pd.read_csv` to recover the exact values.

## Next steps

- SARIMA and SARIMAX per station, evaluated on the saved temporal folds
- Comparison with baselines (RMSE, MAE, R², Wilcoxon tests)
- Ablation by feature family and permutation importance
- Technical report and presentation
