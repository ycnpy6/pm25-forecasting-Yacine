"""Audit, cleaning, matching to GEE, data exploration and time-series exploration.

Writes results/Exploratory_Data.csv, results/Exploratory_temporal.csv, figures/*.png and the cleaned tables
data/processed/pm25_clean.csv (all valid PM2.5 readings) and data/processed/data_cleaned.csv (matched to GEE).

    python scripts/run_eda.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import eda, preprocessing as prep, timeseries as ts   # noqa: E402
from src.config import DATA_PROCESSED, RESULTS                 # noqa: E402
from src.utils import save_log                                 # noqa: E402


def section(title):
    print(f"\n{'=' * 15} {title}\n")


eda_rows, temporal_rows = [], []

section("1. Dataset audit (section 3.1)")
P, G = prep.load_raw()
prep.dataset_description()
prep.audit_size_and_coverage(P, G, eda_rows)
prep.audit_missing_and_duplicates(P, G, eda_rows)
prep.audit_distributions(P, G)
prep.audit_independence(P, eda_rows)
prep.audit_target_leakage(G)
P, spatial_cv_ok = prep.audit_timestamps_and_ids(P, G, eda_rows)

section("2. Remove invalid PM2.5 and merge duplicates (section 5.1)")
P = prep.remove_invalid(P, eda_rows)

section("3. Match stations to GEE points (section 5.3.4)")
df = prep.match_gee(P, G, eda_rows)
df = eda.add_derived_columns(df)

section("4. Data exploration (section 5.1)")
eda.describe(df)
eda.numerical_summary(df, eda_rows)
eda.distribution_plots(df)
eda.outliers(df, eda_rows)
eda.correlations(df, eda_rows)
eda.pm25_vs_predictors(df, eda_rows)
eda.target_distribution(df, eda_rows)

section("5. Time-series exploration (section 5.2)")
daily, panelP, dailyP = ts.panels(df, P)
reg = ts.regularity(P, G, df, temporal_rows)
acf_sum, tests = ts.autocorrelation(P, panelP, dailyP, temporal_rows)
ts.drift(P, temporal_rows)
ts.patterns(P, df, temporal_rows)
ts.lead_lag(df, daily, temporal_rows)
ts.decision_gate(acf_sum, tests, reg, temporal_rows)

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(exist_ok=True)
P.to_csv(DATA_PROCESSED / "pm25_clean.csv", index=False)
df.to_csv(DATA_PROCESSED / "data_cleaned.csv", index=False)
with open(DATA_PROCESSED / "station_ids_check.json", "w") as f:
    json.dump({"spatial_cv_ok": spatial_cv_ok}, f)
save_log(eda_rows, RESULTS / "Exploratory_Data.csv")
save_log(temporal_rows, RESULTS / "Exploratory_temporal.csv")
print(f"\nsaved: data/processed/pm25_clean.csv, data/processed/data_cleaned.csv, results/Exploratory_Data.csv "
      f"({len(eda_rows)} rows), results/Exploratory_temporal.csv ({len(temporal_rows)} rows), figures/")
