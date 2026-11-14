# PM2.5 Forecasting in the Mediterranean Basin - Group 1

Master 1 mini project. Milestone 1: **data exploration and preprocessing**. Models, evaluation, explainability
and ablation come in the next milestone.

## Data
Air Quality - Mediterranean Basin (2016-2026), https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026/
- `pm25.csv`: daily PM2.5 at ground stations (target, ug/m3).
- `gee_features_daily.csv`: Google Earth Engine features (AOD, NO2, CO, O3, temperature, pressure, wind).

License: "Other (specified in description)"; the data are built from OpenAQ (CC BY 4.0), Copernicus (open access)
and NASA open data. Units of NO2/CO/O3 (mol/m2) and wind (m/s) are inferred from the value ranges.

## Install and run
```bash
pip install -r requirements.txt
# put pm25.csv and gee_features_daily.csv in data/raw/
python scripts/run_eda.py              # audit, cleaning, matching, EDA, time series   (~45 s)
python scripts/run_preprocessing.py    # features, folds, leakage-free preprocessing   (~30 s)
```
Run from the project root, in this order. Fixed seed: 42. The notebooks in `notebooks/` call the same `src/` code
and show the results with comments (run `scripts/run_eda.py` before `02_timeseries.ipynb`).

## Structure
```
data/raw/            Kaggle CSV files (not on GitHub)
data/processed/      pm25_clean.csv, data_cleaned.csv, dev.csv, test.csv (generated, not on GitHub)
notebooks/           01_eda.ipynb, 02_timeseries.ipynb
src/
  config.py          paths, SEED = 42, constants (thresholds, test start date)
  preprocessing.py   loading, audit, cleaning, station IDs, GEE matching, station retention,
                     missing values, scaling, feature selection, outliers, Pipeline
  features.py        lags, rolling statistics, differences, cyclical calendar, time since high AOD, past PM2.5
  splits.py          final test period, temporal / spatial / random folds, saved as JSON
  eda.py             data exploration (summaries, plots, outliers, correlations, target distribution)
  timeseries.py      regularity, ACF/PACF, ADF, Ljung-Box, drift, patterns, lead-lag, decision gate
  utils.py           logging of results, figure saving
scripts/
  run_eda.py            runs the exploration, writes Exploratory_Data.csv and Exploratory_temporal.csv
  run_preprocessing.py  builds the features and the folds, checks the preprocessing
folds/               folds_temporal.json, folds_spatial.json, folds_standard.json
results/             Exploratory_Data.csv, Exploratory_temporal.csv, features.json, feature_selection*.csv,
                     missingness_gas_by_{year,month,station}.csv
figures/             all plots (eda_*, ts_*, correlation_matrix, matching_map)
reports/             technical report (later)
presentation/        slides (later)
```

## Main decisions
- Invalid PM2.5 (missing, <= 0 or >= 500: sentinel codes) removed before EDA; repeated station-days averaged.
- `station_id` built by merging coordinates within 0.01 degrees per station name ("Barcelona" is three stations).
- Each station matched to its nearest GEE point within 0.25 degrees, then joined on date.
- Stations kept: at least 200 days and 30% coverage (the stricter 365 days / 50% keeps too few stations).
- Temporal validation and lag features justified by strong autocorrelation (median lag-1 ACF 0.65).
- Features use past days only; past PM2.5 is allowed because we forecast at least one day ahead
  (`FEATURES_NOWCAST` drops it for same-day nowcasting).
- Final test period: 2025-01-01 onward, never inside any fold. Folds saved once and reused by every model.
- Median imputation with missingness indicators, fitted on each training fold (KNN imputation not justified).
- StandardScaler for scale-sensitive models only; raw PM2.5 kept (capping did not help consistently).

Read the generated CSV files with `pd.read_csv(..., float_precision="round_trip")` to get exactly the same values.
