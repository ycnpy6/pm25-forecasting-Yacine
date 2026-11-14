"""Paths, seed and constants shared by the whole project."""
from pathlib import Path

SEED = 42                                       # section 6.1: seed 42 everywhere

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FOLDS = ROOT / "folds"

PM_FILE = DATA_RAW / "pm25.csv"                 # ground stations (Kaggle)
GEE_FILE = DATA_RAW / "gee_features_daily.csv"  # satellite / reanalysis features (Kaggle)

TARGET = "pm25"
GAS = ["NO2", "CO", "O3"]
MET = ["temperature_celsius", "pressure_mb", "wind_u", "wind_v"]
KEY_VARS = [TARGET, "AOD"] + GAS + MET

PM25_MIN, PM25_MAX = 0, 500     # PM2.5 <= 0 or >= 500 are sentinel codes, removed before EDA
STATION_TOL = 0.01              # coordinates closer than this (deg) under one name = same station
MAX_DIST = 0.25                 # max distance (deg) between a station and its GEE point
MIN_OBS, MIN_COVERAGE = 200, 0.3    # station retention rule (decided before splitting)
HIGH_AOD = 0.25                 # fixed "high AOD day" threshold (about the top 13% of days)
TEST_START = "2025-01-01"       # final test period, outside every fold
N_FOLDS = 5
