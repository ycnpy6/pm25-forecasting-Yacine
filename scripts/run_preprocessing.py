"""Preprocessing without data leakage (sections 5.3 and 5.4): station retention, features, final test period,
folds, missing values, scaling, feature selection, outliers and the reusable Pipeline. Run after run_eda.py.

Writes folds/*.json, data/processed/dev.csv and test.csv, results/features.json, results/feature_selection*.csv,
results/missingness_gas_by_*.csv, and adds the preprocessing sections to results/Exploratory_Data.csv.

    python scripts/run_preprocessing.py
"""
import importlib.metadata as im
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                                          # noqa: E402

from src import preprocessing as prep, splits                               # noqa: E402
from src.config import (DATA_PROCESSED, HIGH_AOD, MIN_COVERAGE, MIN_OBS,    # noqa: E402
                        RESULTS, SEED, TARGET, TEST_START)
from src.features import FAMILIES, FEATURES, FEATURES_NOWCAST, build_features, check_no_leakage   # noqa: E402
from src.utils import save_log                                               # noqa: E402


def section(title):
    print(f"\n{'=' * 15} {title}\n")


P = pd.read_csv(DATA_PROCESSED / "pm25_clean.csv", parse_dates=["date"], float_precision="round_trip")      # exact floats
df = pd.read_csv(DATA_PROCESSED / "data_cleaned.csv", parse_dates=["date"], float_precision="round_trip")
with open(DATA_PROCESSED / "station_ids_check.json") as f:
    spatial_cv_ok = json.load(f)["spatial_cv_ok"]
rows = []

section("6.1 Stations to retain (section 5.3.4)")
df, keep = prep.select_stations(df, prep.station_regularity(df), rows)

section("6.2 Features: lags, rolling statistics, differences, calendar (section 5.3.4)")
data = build_features(df, P)
check_no_leakage(data, P, keep)

section("6.3 Final test period and folds (section 5.4)")
dev, test = splits.split_dev_test(data)
folds = splits.make_folds(dev, spatial_cv_ok)
splits.check_folds(dev, folds)
splits.describe_folds(dev, folds)
splits.save_folds(folds)

section("6.4 Missing values (section 5.3.1)")
gm, gy, gs = prep.missingness_report(dev, FEATURES, rows)
gm.to_csv(RESULTS / "missingness_gas_by_month.csv"); gy.to_csv(RESULTS / "missingness_gas_by_year.csv"); gs.to_csv(RESULTS / "missingness_gas_by_station.csv")
prep.knn_imputer_check(dev, folds["temporal"], rows)
tr_idx, va_idx = folds["temporal"][-1]               # demonstration on the last temporal fold
X_tr, X_va = dev.loc[tr_idx, FEATURES], dev.loc[va_idx, FEATURES]
y_tr, y_va = dev.loc[tr_idx, TARGET], dev.loc[va_idx, TARGET]
prep.imputation_check(X_tr, X_va, FEATURES)

section("6.5 Scaling and feature selection, training fold only (section 5.3.2)")
sel, rec = prep.scaling_and_selection(X_tr, y_tr, FEATURES)
sel.to_csv(RESULTS / "feature_selection_scores.csv"); rec.to_csv(RESULTS / "feature_selection.csv", index=False)

section("6.6 Outliers (section 5.3.3)")
prep.outlier_treatment(dev, folds["temporal"], FEATURES, y_tr, y_va, rows)

section("6.7 Reusable preprocessing pipeline")
prep.pipeline_check(X_tr, y_tr, X_va, y_va, FEATURES)

dev.to_csv(DATA_PROCESSED / "dev.csv", index=False)
test.to_csv(DATA_PROCESSED / "test.csv", index=False)
with open(RESULTS / "features.json", "w") as f:
    json.dump({"features": FEATURES, "features_nowcast": FEATURES_NOWCAST, "families": FAMILIES, "target": TARGET,
               "test_start": TEST_START, "seed": SEED, "high_aod_threshold": HIGH_AOD,
               "station_retention": {"min_obs": MIN_OBS, "min_coverage": MIN_COVERAGE}}, f, indent=1)
save_log(rows, RESULTS / "Exploratory_Data.csv", replace=False)
print("\nsaved: folds/, data/processed/dev.csv and test.csv, results/features.json, results/feature_selection*.csv, "
      "results/missingness_gas_by_*.csv, preprocessing sections of results/Exploratory_Data.csv")
print("library versions used:", {p: im.version(p) for p in ["pandas", "numpy", "scipy", "scikit-learn", "statsmodels", "matplotlib"]})
